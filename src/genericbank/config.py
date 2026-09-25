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

# --- Spark (local) ---
SPARK_DRIVER_MEMORY = "6g"
SHUFFLE_PARTITIONS = 16          # default 200 is tuned for clusters; too many tiny tasks locally

# --- General ---
APP_NAME = "genericbank"
BUSINESS_TIMEZONE = "Europe/Warsaw"   # decides which day/month a transaction belongs to