"""Synthetic landing data for GenericBank: customers, accounts, transactions and ground truth.

Everything is vectorised numpy + pandas (no Spark, no Python loops over rows). Transactions are generated
month by month and appended to the part files, so peak memory stays around one month of data.
Money is handled as int64 grosze and only formatted to "1234.56" strings when written.

Usage: python -m genericbank.generate_data [--customers N] [--seed S]
"""
from __future__ import annotations

import argparse
import math
import shutil
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from genericbank import config

# --- Vocabularies (arrays are indexed by the integer codes used internally) ---
SEGMENTS = np.array(["RETAIL", "PREMIER"])
DIRECTIONS = np.array(["CREDIT", "DEBIT"])
CREDIT, DEBIT = range(2)
TXN_TYPES = np.array(["SALARY", "CARD_PAYMENT", "TRANSFER", "CASH_DEPOSIT", "CASH_WITHDRAWAL"])
SALARY, CARD_PAYMENT, TRANSFER, CASH_DEPOSIT, CASH_WITHDRAWAL = range(5)
CHANNELS = np.array(["MOBILE", "WEB", "BRANCH", "ATM", "POS"])
MOBILE, WEB, BRANCH, ATM, POS = range(5)
MERCHANT_CATEGORIES = np.array(
    ["GROCERY", "FUEL", "RESTAURANTS", "TRAVEL", "ONLINE", "UTILITIES", "HEALTH", "ENTERTAINMENT", ""]
)
GROCERY, FUEL, RESTAURANTS, TRAVEL, ONLINE, UTILITIES, HEALTH, ENTERTAINMENT, NO_CATEGORY = range(9)
COUNTRIES = np.array(["PL", "DE", "CZ", "SK", "LT", "ES", "IT", "GB", "US", "NL"])
PL = 0

# Voivodeships weighted by approximate population (millions)
_REGION_POPULATION = {
    "dolnośląskie": 2.9, "kujawsko-pomorskie": 2.0, "lubelskie": 2.0, "lubuskie": 1.0,
    "łódzkie": 2.4, "małopolskie": 3.4, "mazowieckie": 5.5, "opolskie": 0.95,
    "podkarpackie": 2.1, "podlaskie": 1.15, "pomorskie": 2.35, "śląskie": 4.4,
    "świętokrzyskie": 1.2, "warmińsko-mazurskie": 1.4, "wielkopolskie": 3.5, "zachodniopomorskie": 1.65,
}
REGIONS = np.array(list(_REGION_POPULATION))
REGION_WEIGHTS = np.array(list(_REGION_POPULATION.values())) / sum(_REGION_POPULATION.values())

# --- Behaviour model (shape of "normal" data; business knobs live in config.py) ---
SALARY_MEDIAN_PLN = {False: 6_500, True: 20_000}     # RETAIL, PREMIER
SALARY_SIGMA = 0.35
PAY_DAYS = np.array([1, 5, 10, 15, 25, 28])
PAY_DAY_WEIGHTS = np.array([0.20, 0.10, 0.30, 0.10, 0.15, 0.15])
PREMIER_ACTIVITY_FACTOR = 1.2
PREMIER_SPEND_FACTOR = 1.6

# Card category: (base weight, median amount PLN, lognormal sigma, share paid abroad)
_CARD_PROFILE = {
    GROCERY: (0.34, 85, 0.7, 0.01),
    FUEL: (0.12, 220, 0.4, 0.02),
    RESTAURANTS: (0.13, 75, 0.6, 0.03),
    TRAVEL: (0.04, 900, 0.9, 0.35),
    ONLINE: (0.15, 150, 0.9, 0.15),
    UTILITIES: (0.06, 250, 0.5, 0.00),
    HEALTH: (0.06, 120, 0.8, 0.00),
    ENTERTAINMENT: (0.10, 90, 0.7, 0.03),
}
CAT_WEIGHT, CAT_MEDIAN, CAT_SIGMA, CAT_FOREIGN = (np.array(col) for col in zip(*_CARD_PROFILE.values()))
CARD_VOLUME_SEASON = {11: 1.1, 12: 1.3, 1: 0.9}                  # calendar month -> card volume factor
CATEGORY_SEASON = {7: {TRAVEL: 3.0}, 8: {TRAVEL: 3.0}, 12: {ONLINE: 1.5, ENTERTAINMENT: 1.3, GROCERY: 1.1}}
ONLINE_TRAVEL_SHARE = 0.5                                         # TRAVEL card payments made online (WEB)

# Expected monthly counts per average customer; the card rate fills up to AVG_TXN_PER_CUSTOMER_PER_MONTH
TRANSFER_OUT_RATE = 1.2
TRANSFER_IN_RATE = 0.5
SAVINGS_SWEEP_PROB = 0.7          # monthly CURRENT -> SAVINGS transfer (two rows: DEBIT + CREDIT)
ATM_RATE = 0.6
CASH_DEPOSIT_RATE = 0.08
TRANSFER_FOREIGN_SHARE = 0.03
ATM_AMOUNTS_PLN = np.array([100, 200, 300, 400, 500, 800, 1000, 1500, 2000])
ATM_AMOUNT_WEIGHTS = np.array([0.10, 0.20, 0.12, 0.08, 0.20, 0.05, 0.15, 0.05, 0.05])
CASH_DEPOSIT_RANGE_PLN = (100, 5_000)

ORPHAN_ACCOUNT_BASE = 90_000_000   # "A9xxxxxxx" ids never assigned to real accounts

# --- Calendar ---
MONTHS = np.arange(np.datetime64(config.PERIOD_START, "M"), np.datetime64(config.PERIOD_END, "M") + 1)
PERIOD_START_D = np.datetime64(config.PERIOD_START, "D")
PERIOD_END_D = np.datetime64(config.PERIOD_END, "D")
PERIOD_DAYS = int((PERIOD_END_D - PERIOD_START_D).astype(int)) + 1
NAT_D = np.datetime64("NaT", "D")

# Independent random streams per stage: changing one stage does not shift the others
STREAM_BANK, STREAM_PATTERNS, STREAM_CUSTOMER_DIRT, STREAM_MONTH, STREAM_TXN_DIRT = range(1, 6)

EVENT_COLUMNS = ("cust", "acc", "ts", "amount_gr", "direction", "txn_type", "channel", "category", "country",
                 "injected")
TXN_COLUMNS = ["transaction_id", "account_id", "txn_ts", "amount", "direction", "txn_type", "channel",
               "merchant_category", "counterparty_country"]


@dataclass
class Bank:
    """Customers and their accounts as numpy arrays, indexed by customer position 0..n-1."""
    rel_start: np.ndarray       # datetime64[D]
    premier: np.ndarray         # bool
    birth_year: np.ndarray
    region: np.ndarray          # index into REGIONS
    has_salary: np.ndarray      # bool
    salary_gr: np.ndarray       # int64 grosze
    pay_day: np.ndarray         # day of month
    activity: np.ndarray        # per-customer activity factor, mean ~1
    current_acc: np.ndarray     # int64 account number
    savings_acc: np.ndarray     # int64, -1 if none
    savings_open: np.ndarray    # datetime64[D], NaT if none
    card_acc: np.ndarray
    card_open: np.ndarray

    @property
    def n(self) -> int:
        return self.rel_start.size


@dataclass
class Patterns:
    """Injected behaviour: per-customer activity changes plus pre-generated AML deposit rows."""
    churn_month: np.ndarray     # month index where the drop starts; len(MONTHS) = never
    churn_drop: np.ndarray
    dip_month: np.ndarray       # month index of the temporary dip; -1 = none
    dip_drop: np.ndarray
    injected: dict              # event arrays: structuring, decoy and noise deposits
    ground_truth: pd.DataFrame
    counts: dict

    def activity_multiplier(self, m_idx: int) -> np.ndarray:
        churned = np.where(self.churn_month <= m_idx, 1.0 - self.churn_drop, 1.0)
        dipped = np.where(self.dip_month == m_idx, 1.0 - self.dip_drop, 1.0)
        return churned * dipped


# --- Small helpers ---

def _rng(seed: int, *stream: int) -> np.random.Generator:
    return np.random.default_rng([seed, *stream])


def _ids(prefix: str, numbers: np.ndarray, width: int) -> pd.Series:
    return prefix + pd.Series(numbers).astype(str).str.zfill(width)


def _share_count(share: float, n: int) -> int:
    return 0 if share <= 0 else max(1, round(share * n))


def _month_index(d) -> int:
    return int((np.datetime64(d, "M") - MONTHS[0]).astype(int))


def _index_within_groups(sizes: np.ndarray) -> np.ndarray:
    """[2, 3] -> [0, 1, 0, 1, 2]: position of each repeated element inside its group."""
    starts = np.cumsum(sizes) - sizes
    return np.arange(sizes.sum()) - np.repeat(starts, sizes)


def _ts(dates: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    return dates.astype("datetime64[s]") + seconds.astype("timedelta64[s]")


def _lognormal_gr(rng: np.random.Generator, median_pln, sigma, size: int) -> np.ndarray:
    amount = median_pln * np.exp(sigma * rng.standard_normal(size))
    return np.maximum(100, np.round(amount * 100)).astype(np.int64)      # at least 1 PLN


def _countries(rng: np.random.Generator, foreign_share, size: int) -> np.ndarray:
    foreign = rng.random(size) < foreign_share
    return np.where(foreign, rng.integers(1, COUNTRIES.size, size), PL)


def _account(bank: Bank, acc: np.ndarray, opened: np.ndarray, cust: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """The given account if the customer has it and it is open on that date, else their CURRENT account."""
    usable = (acc[cust] >= 0) & (opened[cust] <= dates)          # NaT compares False
    return np.where(usable, acc[cust], bank.current_acc[cust])


def _events(cust, acc, ts, amount_gr, direction, txn_type, channel, category=NO_CATEGORY, country=PL,
            injected=False) -> dict:
    size = len(cust)
    values = (cust, acc, ts, amount_gr, direction, txn_type, channel, category, country, injected)
    return {col: np.broadcast_to(np.asarray(v), (size,)) for col, v in zip(EVENT_COLUMNS, values)}


def _concat(blocks: list[dict]) -> dict:
    return {col: np.concatenate([b[col] for b in blocks]) for col in EVENT_COLUMNS}


def _select(events: dict, mask: np.ndarray) -> dict:
    return {col: v[mask] for col, v in events.items()}


def _labels(vocab: np.ndarray, codes: np.ndarray, mangle: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map codes to labels; rows in `mangle` get inconsistent case / whitespace (" Retail", "mobile ")."""
    out = vocab.astype(object)[codes]
    variants = np.array([[v.lower(), v.title(), " " + v.title(), v.lower() + " "] for v in vocab], dtype=object)
    idx = np.flatnonzero(mangle)
    out[idx] = variants[codes[idx], rng.integers(0, variants.shape[1], idx.size)]
    return out


# --- Customers and accounts ---

def make_bank(n: int, rng: np.random.Generator) -> Bank:
    rel_lo, rel_hi = (np.datetime64(d, "D") for d in config.RELATIONSHIP_START_RANGE)
    rel_start = rel_lo + rng.integers(0, int((rel_hi - rel_lo).astype(int)) + 1, n)
    premier = rng.random(n) < config.PREMIER_SHARE
    by_lo, by_hi = config.BIRTH_YEAR_RANGE

    salary_median = np.where(premier, SALARY_MEDIAN_PLN[True], SALARY_MEDIAN_PLN[False])
    salary_gr = np.round(salary_median * np.exp(SALARY_SIGMA * rng.standard_normal(n)) * 100).astype(np.int64)

    # Account numbers are consecutive per customer: CURRENT, then SAVINGS, then CARD
    has_savings = rng.random(n) < config.SAVINGS_ACCOUNT_SHARE
    has_card = rng.random(n) < config.CARD_ACCOUNT_SHARE
    n_accounts = 1 + has_savings.astype(np.int64) + has_card
    first = np.cumsum(n_accounts) - n_accounts + 1

    def open_dates() -> np.ndarray:
        # Skewed towards the start of the relationship; some open during the period
        span = (PERIOD_END_D - rel_start).astype(np.int64)
        return rel_start + np.floor(rng.random(n) ** 2 * (span + 1)).astype(np.int64)

    return Bank(
        rel_start=rel_start,
        premier=premier,
        birth_year=rng.integers(by_lo, by_hi + 1, n),
        region=rng.choice(REGIONS.size, n, p=REGION_WEIGHTS),
        has_salary=rng.random(n) < config.SALARY_SHARE,
        salary_gr=salary_gr,
        pay_day=rng.choice(PAY_DAYS, n, p=PAY_DAY_WEIGHTS),
        activity=rng.gamma(2.0, 0.5, n) * np.where(premier, PREMIER_ACTIVITY_FACTOR, 1.0),
        current_acc=first,
        savings_acc=np.where(has_savings, first + 1, -1),
        savings_open=np.where(has_savings, open_dates(), NAT_D),
        card_acc=np.where(has_card, first + 1 + has_savings, -1),
        card_open=np.where(has_card, open_dates(), NAT_D),
    )


def customers_frame(bank: Bank, rng: np.random.Generator) -> pd.DataFrame:
    mangled = rng.random(bank.n) < config.DIRTY_CASE_WHITESPACE_RATE
    return pd.DataFrame({
        "customer_id": _ids("C", np.arange(1, bank.n + 1), 7),
        "segment": _labels(SEGMENTS, bank.premier.astype(np.int64), mangled, rng),
        "birth_year": bank.birth_year,
        "region": REGIONS[bank.region],
        "relationship_start": np.datetime_as_string(bank.rel_start, unit="D"),
    })


def accounts_frame(bank: Bank) -> pd.DataFrame:
    cust = np.arange(bank.n)
    has_savings, has_card = bank.savings_acc >= 0, bank.card_acc >= 0
    acc = np.concatenate([bank.current_acc, bank.savings_acc[has_savings], bank.card_acc[has_card]])
    owner = np.concatenate([cust, cust[has_savings], cust[has_card]])
    acc_type = np.concatenate([np.full(bank.n, "CURRENT"), np.full(has_savings.sum(), "SAVINGS"),
                               np.full(has_card.sum(), "CARD")])
    opened = np.concatenate([bank.rel_start, bank.savings_open[has_savings], bank.card_open[has_card]])
    order = np.argsort(acc, kind="stable")
    return pd.DataFrame({
        "account_id": _ids("A", acc[order], 8),
        "customer_id": _ids("C", owner[order] + 1, 7),
        "account_type": acc_type[order],
        "currency": "PLN",
        "open_date": np.datetime_as_string(opened[order], unit="D"),
    })


# --- Injected patterns ---

def _truth(cust: np.ndarray, pattern: str, start: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"cust": cust, "pattern": pattern, "pattern_start": np.datetime_as_string(start, unit="D")})


def _near_threshold_deposits(rng: np.random.Generator, bank: Bank, cust: np.ndarray, day: np.ndarray,
                             acc: np.ndarray | None = None) -> dict:
    """Cash deposits at 85-99% of the AML threshold on the given period days (whole PLN, business hours)."""
    threshold = float(config.AML_THRESHOLD_PLN)
    lo, hi = config.NEAR_THRESHOLD_DEPOSIT_RATIO_RANGE
    size = cust.size
    amount_gr = rng.integers(math.ceil(lo * threshold), math.floor(hi * threshold) + 1, size) * 100
    channel = np.where(rng.random(size) < 0.8, BRANCH, ATM)
    ts = _ts(PERIOD_START_D + day, rng.integers(9 * 3600, 18 * 3600, size))
    acc = bank.current_acc[cust] if acc is None else acc
    return _events(cust, acc, ts, amount_gr, CREDIT, CASH_DEPOSIT, channel, injected=True)


def make_structuring(bank: Bank, cust: np.ndarray, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """1-3 non-overlapping episodes per customer, each 3-6 near-threshold deposits within the AML window."""
    window = config.AML_WINDOW_DAYS
    ep_lo, ep_hi = config.STRUCTURING_EPISODES_RANGE
    dep_lo, dep_hi = config.STRUCTURING_DEPOSITS_RANGE

    n_ep = rng.integers(ep_lo, ep_hi + 1, cust.size)
    ep_cust = np.repeat(cust, n_ep)
    # Split the period into one segment per episode so episodes never overlap
    seg_len = PERIOD_DAYS // np.repeat(n_ep, n_ep)
    ep_start = _index_within_groups(n_ep) * seg_len + rng.integers(0, seg_len - window + 1)
    spread = (rng.random(ep_cust.size) < config.STRUCTURING_MULTI_ACCOUNT_PROB) & (bank.savings_acc[ep_cust] >= 0)
    parity = rng.integers(0, 2, ep_cust.size)

    n_dep = rng.integers(dep_lo, dep_hi + 1, ep_cust.size)
    dep_ep = np.repeat(np.arange(ep_cust.size), n_dep)
    dep_no = _index_within_groups(n_dep)
    d_cust = ep_cust[dep_ep]
    offset = np.where(dep_no == 0, 0, rng.integers(0, window, dep_ep.size))   # span <= window - 1 days
    day = ep_start[dep_ep] + offset
    # Spread episodes alternate CURRENT / SAVINGS (when the savings account is already open)
    to_savings = (spread[dep_ep] & ((dep_no + parity[dep_ep]) % 2 == 1)
                  & (bank.savings_open[d_cust] <= PERIOD_START_D + day))
    acc = np.where(to_savings, bank.savings_acc[d_cust], bank.current_acc[d_cust])

    events = _near_threshold_deposits(rng, bank, d_cust, day, acc)
    return events, _truth(ep_cust, "STRUCTURING", PERIOD_START_D + ep_start)


def make_decoy_pairs(bank: Bank, cust: np.ndarray, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Exactly 2 near-threshold deposits within the AML window: one short of AML_MIN_DEPOSITS."""
    window = config.AML_WINDOW_DAYS
    start = rng.integers(0, PERIOD_DAYS - window + 1, cust.size)
    second = start + rng.integers(0, window, cust.size)
    day = np.column_stack([start, second]).ravel()
    events = _near_threshold_deposits(rng, bank, np.repeat(cust, 2), day)
    return events, _truth(cust, "AML_DECOY", PERIOD_START_D + start)


def make_decoy_spread(bank: Bank, cust: np.ndarray, rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """3-4 near-threshold deposits over 10-20 days where any 3 consecutive ones span > AML_WINDOW_DAYS."""
    min_span = config.AML_WINDOW_DAYS + 1          # 3 deposits must cover at least this many days
    k_lo, k_hi = config.AML_DECOY_SPREAD_DEPOSITS_RANGE
    days_lo, days_hi = config.AML_DECOY_SPREAD_DAYS_RANGE
    if not (3 <= k_lo <= k_hi <= 4) or days_hi < 2 * min_span - 1:
        raise ValueError("AML_DECOY_SPREAD_* must allow 3-4 deposits with every 3 spanning > AML_WINDOW_DAYS")
    m = cust.size
    k = rng.integers(k_lo, k_hi + 1, m)
    zero = np.zeros(m, dtype=np.int64)

    # 3 deposits at 0, g, s3 with s3 >= min_span
    s3 = rng.integers(max(days_lo, min_span), days_hi + 1, m)
    g = rng.integers(1, s3)
    # 4 deposits with gaps a, b, c where a + b >= min_span and b + c >= min_span
    b = rng.integers(1, min_span // 2 + 1, m)
    base = 2 * min_span - b                        # smallest total span given b
    s4 = rng.integers(np.maximum(days_lo, base), days_hi + 1)
    extra = s4 - base
    e = rng.integers(0, extra + 1)
    a = min_span - b + e
    offsets = np.where((k == 3)[:, None],
                       np.column_stack([zero, g, s3, zero]),
                       np.column_stack([zero, a, a + b, s4]))
    span = np.where(k == 3, s3, s4)
    start = rng.integers(0, PERIOD_DAYS - span)

    keep = np.arange(4)[None, :] < k[:, None]
    day = (start[:, None] + offsets)[keep]
    events = _near_threshold_deposits(rng, bank, np.repeat(cust, k), day)
    return events, _truth(cust, "AML_DECOY", PERIOD_START_D + start)


def make_noise_deposits(bank: Bank, cust: np.ndarray, rng: np.random.Generator) -> dict:
    """One innocent near-threshold deposit per customer (never repeated, not in the ground truth)."""
    return _near_threshold_deposits(rng, bank, cust, rng.integers(0, PERIOD_DAYS, cust.size))


def make_patterns(bank: Bank, rng: np.random.Generator) -> Patterns:
    n, n_months = bank.n, MONTHS.size
    # Pattern customers need a baseline before the period starts
    cutoff = (MONTHS[0] - config.CHURN_BASELINE_MONTHS).astype("datetime64[D]")
    eligible = rng.permutation(np.flatnonzero(bank.rel_start <= cutoff))
    sizes = {
        "churn": _share_count(config.CHURN_CUSTOMER_SHARE, n),
        "temp_dip": _share_count(config.TEMP_DIP_CUSTOMER_SHARE, n),
        "structuring": _share_count(config.STRUCTURING_CUSTOMER_SHARE, n),
        "decoy_pair": _share_count(config.AML_DECOY_PAIR_SHARE, n),
        "decoy_spread": _share_count(config.AML_DECOY_SPREAD_SHARE, n),
        "noise": _share_count(config.NEAR_THRESHOLD_NOISE_SHARE, n),
    }
    if sum(sizes.values()) > eligible.size:
        raise ValueError(f"Not enough eligible customers ({eligible.size}) for patterns {sizes}")
    groups = dict(zip(sizes, np.split(eligible, np.cumsum(list(sizes.values())))))   # disjoint groups

    churn_lo, churn_hi = (_month_index(d) for d in config.CHURN_START_RANGE)
    dip_lo, dip_hi = (_month_index(d) for d in config.TEMP_DIP_MONTH_RANGE)
    if not (0 <= churn_lo <= churn_hi < n_months and 0 <= dip_lo <= dip_hi < n_months - 1):
        raise ValueError("CHURN_START_RANGE / TEMP_DIP_MONTH_RANGE must lie inside the period "
                         "(a dip needs at least one normal month after it)")

    c = groups["churn"]
    churn_month = np.full(n, n_months)
    churn_month[c] = rng.integers(churn_lo, churn_hi + 1, c.size)
    churn_drop = np.zeros(n)
    churn_drop[c] = rng.uniform(*config.CHURN_DROP_RANGE, c.size)

    d = groups["temp_dip"]
    dip_month = np.full(n, -1)
    dip_month[d] = rng.integers(dip_lo, dip_hi + 1, d.size)
    dip_drop = np.zeros(n)
    dip_drop[d] = rng.uniform(*config.TEMP_DIP_DROP_RANGE, d.size)

    structuring, structuring_truth = make_structuring(bank, groups["structuring"], rng)
    pairs, pairs_truth = make_decoy_pairs(bank, groups["decoy_pair"], rng)
    spread, spread_truth = make_decoy_spread(bank, groups["decoy_spread"], rng)
    noise = make_noise_deposits(bank, groups["noise"], rng)

    truth = pd.concat([
        _truth(c, "CHURN", MONTHS[churn_month[c]].astype("datetime64[D]")),
        _truth(d, "TEMP_DIP", MONTHS[dip_month[d]].astype("datetime64[D]")),
        structuring_truth, pairs_truth, spread_truth,
    ], ignore_index=True).sort_values(["cust", "pattern_start"], kind="stable")
    ground_truth = pd.DataFrame({
        "customer_id": _ids("C", truth["cust"].to_numpy() + 1, 7),
        "pattern": truth["pattern"].to_numpy(),
        "pattern_start": truth["pattern_start"].to_numpy(),
    })

    counts = {
        "CHURN customers": c.size,
        "TEMP_DIP customers": d.size,
        "STRUCTURING customers": groups["structuring"].size,
        "STRUCTURING episodes": len(structuring_truth),
        "STRUCTURING deposits": structuring["cust"].size,
        "AML_DECOY pair customers (2 deposits in window)": groups["decoy_pair"].size,
        "AML_DECOY spread customers (3-4 deposits, > window)": groups["decoy_spread"].size,
        "AML_DECOY deposits": pairs["cust"].size + spread["cust"].size,
        "single near-threshold noise deposits": noise["cust"].size,
    }
    return Patterns(churn_month, churn_drop, dip_month, dip_drop,
                    _concat([structuring, pairs, spread, noise]), ground_truth, counts)


# --- Monthly transactions ---

def _card_rate() -> float:
    other = (config.SALARY_SHARE + TRANSFER_OUT_RATE + TRANSFER_IN_RATE + ATM_RATE + CASH_DEPOSIT_RATE
             + 2 * config.SAVINGS_ACCOUNT_SHARE * SAVINGS_SWEEP_PROB)
    return max(1.0, config.AVG_TXN_PER_CUSTOMER_PER_MONTH - other)


def _category_weights(cal_month: int) -> np.ndarray:
    w = CAT_WEIGHT.astype(float)
    for cat, factor in CATEGORY_SEASON.get(cal_month, {}).items():
        w[cat] *= factor
    return w / w.sum()


def _poisson_customers(rng: np.random.Generator, lam: np.ndarray) -> np.ndarray:
    """One entry per event: customer index repeated by a Poisson draw of its monthly rate."""
    return np.repeat(np.arange(lam.size), rng.poisson(lam))


def make_month_transactions(bank: Bank, patterns: Patterns, m_idx: int, rng: np.random.Generator) -> dict:
    """All transactions of one calendar month as event arrays, sorted by timestamp."""
    month = MONTHS[m_idx]
    start = month.astype("datetime64[D]")
    next_start = (month + 1).astype("datetime64[D]")
    n_days = int((next_start - start).astype(int))
    cal_month = month.astype(object).month
    mult = patterns.activity_multiplier(m_idx)
    rate = bank.activity * mult
    blocks = []

    # SALARY: fixed pay day, stops once a customer churns
    pay_date = start + np.minimum(bank.pay_day, n_days) - 1
    cust = np.flatnonzero(bank.has_salary & (m_idx < patterns.churn_month))
    amount = np.round(bank.salary_gr[cust] * rng.uniform(0.97, 1.03, cust.size)).astype(np.int64)
    ts = _ts(pay_date[cust], rng.integers(6 * 3600, 9 * 3600, cust.size))
    blocks.append(_events(cust, bank.current_acc[cust], ts, amount, CREDIT, SALARY, WEB))

    # CARD_PAYMENT: category mix and volume follow the season
    cust = _poisson_customers(rng, _card_rate() * CARD_VOLUME_SEASON.get(cal_month, 1.0) * rate)
    k = cust.size
    dates = start + rng.integers(0, n_days, k)
    cat = rng.choice(CAT_WEIGHT.size, k, p=_category_weights(cal_month))
    median = CAT_MEDIAN[cat] * np.where(bank.premier[cust], PREMIER_SPEND_FACTOR, 1.0)
    amount = _lognormal_gr(rng, median, CAT_SIGMA[cat], k)
    online = (cat == ONLINE) | ((cat == TRAVEL) & (rng.random(k) < ONLINE_TRAVEL_SHARE))
    blocks.append(_events(
        cust, _account(bank, bank.card_acc, bank.card_open, cust, dates),
        _ts(dates, rng.integers(7 * 3600, 23 * 3600, k)), amount, DEBIT, CARD_PAYMENT,
        np.where(online, WEB, POS), cat, _countries(rng, CAT_FOREIGN[cat], k),
    ))

    # TRANSFER out / in (bills, rent, friends)
    for direction, lam, median in ((DEBIT, TRANSFER_OUT_RATE, 350), (CREDIT, TRANSFER_IN_RATE, 250)):
        cust = _poisson_customers(rng, lam * rate)
        k = cust.size
        median = median * np.where(bank.premier[cust], PREMIER_SPEND_FACTOR, 1.0)
        blocks.append(_events(
            cust, bank.current_acc[cust],
            _ts(start + rng.integers(0, n_days, k), rng.integers(6 * 3600, 23 * 3600, k)),
            _lognormal_gr(rng, median, 1.0, k), direction, TRANSFER,
            np.where(rng.random(k) < 0.65, MOBILE, WEB), country=_countries(rng, TRANSFER_FOREIGN_SHARE, k),
        ))

    # TRANSFER CURRENT -> SAVINGS: two rows with the same timestamp and amount
    cust = np.flatnonzero((bank.savings_acc >= 0) & (rng.random(bank.n) < SAVINGS_SWEEP_PROB * mult))
    dates = start + rng.integers(0, n_days, cust.size)
    opened = bank.savings_open[cust] <= dates
    cust, dates = cust[opened], dates[opened]
    k = cust.size
    median = np.where(bank.has_salary[cust], bank.salary_gr[cust] / 100 * 0.1, 300)
    amount = _lognormal_gr(rng, median, 0.5, k)
    ts = _ts(dates, rng.integers(6 * 3600, 23 * 3600, k))
    channel = np.where(rng.random(k) < 0.7, MOBILE, WEB)
    blocks.append(_events(cust, bank.current_acc[cust], ts, amount, DEBIT, TRANSFER, channel))
    blocks.append(_events(cust, bank.savings_acc[cust], ts, amount, CREDIT, TRANSFER, channel))

    # CASH_WITHDRAWAL: round amounts, mostly ATM
    cust = _poisson_customers(rng, ATM_RATE * rate)
    k = cust.size
    blocks.append(_events(
        cust, bank.current_acc[cust],
        _ts(start + rng.integers(0, n_days, k), rng.integers(6 * 3600, 23 * 3600, k)),
        rng.choice(ATM_AMOUNTS_PLN, k, p=ATM_AMOUNT_WEIGHTS) * 100, DEBIT, CASH_WITHDRAWAL,
        np.where(rng.random(k) < 0.9, ATM, BRANCH),
    ))

    # CASH_DEPOSIT: small, far below the AML band
    cust = _poisson_customers(rng, CASH_DEPOSIT_RATE * rate)
    k = cust.size
    dep_lo, dep_hi = CASH_DEPOSIT_RANGE_PLN
    blocks.append(_events(
        cust, bank.current_acc[cust],
        _ts(start + rng.integers(0, n_days, k), rng.integers(8 * 3600, 20 * 3600, k)),
        rng.integers(dep_lo * 100, dep_hi * 100 + 1, k), CREDIT, CASH_DEPOSIT,
        np.where(rng.random(k) < 0.6, ATM, BRANCH),
    ))

    events = _concat(blocks)
    day = events["ts"].astype("datetime64[D]")
    keep = (day >= bank.rel_start[events["cust"]]) & (day >= PERIOD_START_D) & (day <= PERIOD_END_D)
    inj = patterns.injected
    in_month = (inj["ts"] >= start.astype("datetime64[s]")) & (inj["ts"] < next_start.astype("datetime64[s]"))
    events = _concat([_select(events, keep), _select(inj, in_month)])
    return _select(events, np.argsort(events["ts"], kind="stable"))


def format_transactions(events: dict, first_txn_no: int,
                        rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray, Counter]:
    """Format events to landing strings, inject dirty data and assign each row a part file."""
    n = events["ts"].size
    injected = events["injected"]

    def rows(rate: float) -> np.ndarray:
        return rng.random(n) < rate

    orphan = rows(config.DIRTY_ORPHAN_ACCOUNT_RATE) & ~injected       # injected AML rows stay detectable
    empty = rows(config.DIRTY_EMPTY_AMOUNT_RATE) & ~injected
    comma = rows(config.DIRTY_COMMA_DECIMAL_RATE) & ~empty
    alt_ts = rows(config.DIRTY_ALT_TS_FORMAT_RATE)
    mangled = rows(config.DIRTY_CASE_WHITESPACE_RATE)

    acc = np.where(orphan, ORPHAN_ACCOUNT_BASE + rng.integers(0, 10_000_000, n), events["acc"])

    amount_gr = events["amount_gr"]
    amount = (pd.Series(amount_gr // 100).astype(str)
              + pd.Series(np.where(comma, ",", "."))
              + pd.Series(amount_gr % 100).astype(str).str.zfill(2))
    amount = amount.where(~empty, "")

    txn_ts = pd.Series(np.datetime_as_string(events["ts"], unit="s")).str.replace("T", " ", regex=False)
    txn_ts[alt_ts] = pd.Series(events["ts"][alt_ts]).dt.strftime("%d.%m.%Y %H:%M").to_numpy()

    df = pd.DataFrame({
        "transaction_id": _ids("T", first_txn_no + np.arange(n), 10),
        "account_id": _ids("A", acc, 8),
        "txn_ts": txn_ts,
        "amount": amount,
        "direction": DIRECTIONS[events["direction"]],
        "txn_type": TXN_TYPES[events["txn_type"]],
        "channel": _labels(CHANNELS, events["channel"], mangled, rng),
        "merchant_category": MERCHANT_CATEGORIES[events["category"]],
        "counterparty_country": COUNTRIES[events["country"]],
    })

    # Exact duplicates (same transaction_id); the copy gets its own part file, often a different one
    dup = rows(config.DIRTY_DUPLICATE_RATE) | (injected & rows(config.DIRTY_INJECTED_DUPLICATE_RATE))
    df = pd.concat([df, df[dup]], ignore_index=True)
    part = rng.integers(0, config.N_TXN_PART_FILES, len(df))

    stats = Counter({
        "duplicate rows": int(dup.sum()),
        "  of which injected AML deposits": int((dup & injected).sum()),
        "empty amount": int(empty.sum()),
        "comma decimal separator": int(comma.sum()),
        "dd.MM.yyyy HH:mm timestamps": int(alt_ts.sum()),
        "orphan account_id": int(orphan.sum()),
        "mangled channel": int(mangled.sum()),
    })
    return df, part, stats


# --- I/O ---

def _reset_landing_dir() -> None:
    landing, data = config.LANDING_DIR.resolve(), config.DATA_DIR.resolve()
    if landing == data or not landing.is_relative_to(data):
        raise RuntimeError(f"Refusing to wipe {landing}: LANDING_DIR must be inside {data}")
    if landing.exists():
        shutil.rmtree(landing)
    landing.mkdir(parents=True)


def write_landing(n_customers: int, seed: int) -> dict:
    """Wipe LANDING_DIR and write all landing files; returns numbers for the summary."""
    if n_customers < 1 or 3 * n_customers >= ORPHAN_ACCOUNT_BASE:
        raise ValueError(f"--customers must be between 1 and {ORPHAN_ACCOUNT_BASE // 3 - 1}")
    _reset_landing_dir()
    landing = config.LANDING_DIR
    files: dict[str, int] = {}

    bank = make_bank(n_customers, _rng(seed, STREAM_BANK))
    patterns = make_patterns(bank, _rng(seed, STREAM_PATTERNS))

    customers = customers_frame(bank, _rng(seed, STREAM_CUSTOMER_DIRT))
    customers.to_json(landing / "customers.jsonl", orient="records", lines=True, force_ascii=False)
    files["customers.jsonl"] = len(customers)
    accounts = accounts_frame(bank)
    accounts.to_csv(landing / "accounts.csv", index=False)
    files["accounts.csv"] = len(accounts)
    patterns.ground_truth.to_csv(landing / "_ground_truth.csv", index=False)
    files["_ground_truth.csv"] = len(patterns.ground_truth)

    txn_dir = landing / "transactions"
    txn_dir.mkdir()
    paths = [txn_dir / f"part-{p:05d}.csv" for p in range(config.N_TXN_PART_FILES)]
    for path in paths:
        pd.DataFrame(columns=TXN_COLUMNS).to_csv(path, index=False)              # header only
    rows_per_part = np.zeros(config.N_TXN_PART_FILES, dtype=np.int64)
    dirty = Counter({"mangled segment": int((~customers["segment"].isin(SEGMENTS)).sum())})

    next_txn_no = 1
    for m_idx in range(MONTHS.size):
        events = make_month_transactions(bank, patterns, m_idx, _rng(seed, STREAM_MONTH, m_idx))
        df, part, stats = format_transactions(events, next_txn_no, _rng(seed, STREAM_TXN_DIRT, m_idx))
        next_txn_no += events["ts"].size
        for p, chunk in df.groupby(part):
            chunk.to_csv(paths[p], mode="a", header=False, index=False)
        rows_per_part += np.bincount(part, minlength=config.N_TXN_PART_FILES)
        dirty.update(stats)
        print(f"  {MONTHS[m_idx]}: {len(df):>9,} transaction rows")

    for path, count in zip(paths, rows_per_part):
        files[f"transactions/{path.name}"] = int(count)
    return {"files": files, "patterns": patterns.counts, "dirty": dirty}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic GenericBank landing data.")
    parser.add_argument("--customers", type=int, default=config.N_CUSTOMERS, help="number of customers")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    print(f"Generating GenericBank landing data: {args.customers:,} customers, seed {args.seed} "
          f"-> {config.LANDING_DIR}")
    summary = write_landing(args.customers, args.seed)

    txn_total = sum(v for k, v in summary["files"].items() if k.startswith("transactions/"))
    print("\nRows per file:")
    for name, count in summary["files"].items():
        print(f"  {name:<32}{count:>12,}")
    print(f"  {'transactions total':<32}{txn_total:>12,}")
    print("\nInjected patterns:")
    for name, count in summary["patterns"].items():
        print(f"  {name:<52}{count:>8,}")
    print("\nDirty data:")
    for name, count in summary["dirty"].items():
        print(f"  {name:<52}{count:>8,}")
    print(f"\nDone in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
