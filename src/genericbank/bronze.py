"""Bronze layer: raw landing files 1:1 as strings, plus ingestion metadata."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from genericbank import config


def _all_strings(*names: str) -> StructType:
    """Build a schema where every column is a nullable string."""
    return StructType([StructField(name, StringType(), True) for name in names])


CUSTOMERS_SCHEMA = _all_strings("customer_id", "segment", "birth_year", "region", "relationship_start")
ACCOUNTS_SCHEMA = _all_strings("account_id", "customer_id", "account_type", "currency", "open_date")
TRANSACTIONS_SCHEMA = _all_strings(
    "transaction_id", "account_id", "txn_ts", "amount", "direction",
    "txn_type", "channel", "merchant_category", "counterparty_country",
)


def _with_source_file(df: DataFrame) -> DataFrame:
    """Keep the path of the file each row came from (file sources only)."""
    return df.select("*", F.col("_metadata.file_path").alias("_source_file"))


def read_customers(spark: SparkSession) -> DataFrame:
    df = spark.read.schema(CUSTOMERS_SCHEMA).json(str(config.LANDING_DIR / "customers.jsonl"))
    return _with_source_file(df)


def read_accounts(spark: SparkSession) -> DataFrame:
    df = spark.read.schema(ACCOUNTS_SCHEMA).option("header", True).csv(str(config.LANDING_DIR / "accounts.csv"))
    return _with_source_file(df)


def read_transactions(spark: SparkSession) -> DataFrame:
    df = spark.read.schema(TRANSACTIONS_SCHEMA).option("header", True).csv(str(config.LANDING_DIR / "transactions"))
    return _with_source_file(df)


def add_ingestion_metadata(df: DataFrame, run_id: str) -> DataFrame:
    """Pure transformation: stamp every row with the load time."""
    return (
        df
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_run_id", F.lit(run_id))
    )