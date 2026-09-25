"""Central configuration: paths, data-generation settings and business rules."""
from datetime import date
from decimal import Decimal
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/genericbank/config.py -> two levels up
DATA_DIR = PROJECT_ROOT / "data"
LANDING_DIR = DATA_DIR / "landing"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
QUARANTINE_DIR = DATA_DIR / "quarantine"

# --- Synthetic data generation ---
SEED = 42                                   # same seed -> identical data on every run
N_CUSTOMERS = 50_000
AVG_TXN_PER_CUSTOMER_PER_MONTH = 8          # ~50k * 12 months * 8 ≈ 4.8M transactions
PERIOD_START = date(2025, 10, 1)
PERIOD_END = date(2026, 9, 30)

# --- AML (structuring) ---
AML_REPORTING_THRESHOLD_EUR = Decimal("15000")   # cash transaction reporting threshold
EUR_PLN_RATE = Decimal("4.25")                   # illustrative rate, parameterised on purpose
AML_THRESHOLD_PLN = AML_REPORTING_THRESHOLD_EUR * EUR_PLN_RATE
AML_NEAR_THRESHOLD_RATIO = Decimal("0.85")       # "just below threshold" = from 85% of it
AML_WINDOW_DAYS = 7
AML_MIN_DEPOSITS = 3

# --- Churn signals ---
CHURN_BASELINE_MONTHS = 3        # baseline = average of the previous 3 months
CHURN_DROP_RATIO = 0.5           # flag when activity falls below 50% of baseline
CHURN_MIN_BASELINE_TXN = 5       # skip customers who were barely active anyway

# --- Generator (synthetic landing data, see generate_data.py) ---
N_TXN_PART_FILES = 10
PREMIER_SHARE = 0.10
SAVINGS_ACCOUNT_SHARE = 0.40
CARD_ACCOUNT_SHARE = 0.50
SALARY_SHARE = 0.85                                  # customers with a monthly salary credit
BIRTH_YEAR_RANGE = (1950, 2005)
RELATIONSHIP_START_RANGE = (date(2005, 1, 1), date(2025, 12, 31))

# Injected patterns (recorded in _ground_truth.csv); customer groups are mutually disjoint
CHURN_CUSTOMER_SHARE = 0.015
CHURN_START_RANGE = (date(2026, 2, 1), date(2026, 8, 1))      # month the drop starts, inclusive
CHURN_DROP_RANGE = (0.70, 1.00)
TEMP_DIP_CUSTOMER_SHARE = 0.02                                 # one bad month, then back to normal
TEMP_DIP_MONTH_RANGE = (date(2026, 1, 1), date(2026, 8, 1))
TEMP_DIP_DROP_RANGE = (0.60, 0.90)
STRUCTURING_CUSTOMER_SHARE = 0.002
STRUCTURING_EPISODES_RANGE = (1, 3)
STRUCTURING_DEPOSITS_RANGE = (AML_MIN_DEPOSITS, 6)            # cash deposits per episode, within AML_WINDOW_DAYS
STRUCTURING_MULTI_ACCOUNT_PROB = 0.4                           # episode split across CURRENT and SAVINGS
NEAR_THRESHOLD_DEPOSIT_RATIO_RANGE = (float(AML_NEAR_THRESHOLD_RATIO), 0.99)   # share of AML_THRESHOLD_PLN
AML_DECOY_PAIR_SHARE = 0.002                                   # 2 near-threshold deposits within the window
AML_DECOY_SPREAD_SHARE = 0.002                                 # 3-4 deposits, never 3 inside one window
AML_DECOY_SPREAD_DEPOSITS_RANGE = (3, 4)
AML_DECOY_SPREAD_DAYS_RANGE = (10, 20)
NEAR_THRESHOLD_NOISE_SHARE = 0.005                             # one innocent near-threshold deposit, not recorded

# Dirty data (per transaction row unless stated otherwise)
DIRTY_DUPLICATE_RATE = 0.005
DIRTY_INJECTED_DUPLICATE_RATE = 0.05     # extra duplicates on injected AML deposits: dedup must precede counting
DIRTY_EMPTY_AMOUNT_RATE = 0.001
DIRTY_ALT_TS_FORMAT_RATE = 0.02          # dd.MM.yyyy HH:mm
DIRTY_COMMA_DECIMAL_RATE = 0.01          # 1234,56
DIRTY_ORPHAN_ACCOUNT_RATE = 0.001        # account_id missing from accounts.csv
DIRTY_CASE_WHITESPACE_RATE = 0.03        # segment (customers) and channel (transactions)

# --- Spark (local) ---
SPARK_DRIVER_MEMORY = "6g"
SHUFFLE_PARTITIONS = 16          # default 200 is tuned for clusters; too many tiny tasks locally

# --- General ---
APP_NAME = "genericbank"
BUSINESS_TIMEZONE = "Europe/Warsaw"   # decides which day/month a transaction belongs to