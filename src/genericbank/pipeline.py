"""Pipeline entry point: python -m genericbank.pipeline --stage bronze|silver|all"""
import argparse
import uuid
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from genericbank import bronze, config, quality, silver
from genericbank.spark import get_spark


def _write_delta(df: DataFrame, path: Path, partition_by: str | None = None) -> None:
    """Overwrite a Delta table and print its row count."""
    writer = df.write.format("delta").mode("overwrite").option("overwriteSchema", True)
    if partition_by:
        writer = writer.partitionBy(partition_by)
    writer.save(str(path))
    rows = df.sparkSession.read.format("delta").load(str(path)).count()
    print(f"{path.relative_to(config.DATA_DIR)}: {rows:,} rows")


def _read_delta(spark: SparkSession, path: Path) -> DataFrame:
    return spark.read.format("delta").load(str(path))


def run_bronze(spark: SparkSession, run_id: str) -> None:
    sources = {
        "customers": bronze.read_customers,
        "accounts": bronze.read_accounts,
        "transactions": bronze.read_transactions,
    }
    for name, read in sources.items():
        df = bronze.add_ingestion_metadata(read(spark), run_id=run_id)
        _write_delta(df, config.BRONZE_DIR / name)


def run_silver(spark: SparkSession, run_id: str) -> None:
    customers = silver.clean_customers(_read_delta(spark, config.BRONZE_DIR / "customers"))
    accounts = silver.clean_accounts(_read_delta(spark, config.BRONZE_DIR / "accounts"))
    bronze_txn = _read_delta(spark, config.BRONZE_DIR / "transactions")
    txn = silver.deduplicate_transactions(silver.parse_transactions(bronze_txn))

    flagged = silver.flag_rejects(txn, accounts).cache()      # used twice below: compute once
    valid = flagged.filter(F.col("reject_reason").isNull())
    rejected = flagged.filter(F.col("reject_reason").isNotNull())

    _write_delta(customers, config.SILVER_DIR / "customers")
    _write_delta(accounts, config.SILVER_DIR / "accounts")
    _write_delta(
        silver.enrich_transactions(valid, accounts, customers),
        config.SILVER_DIR / "transactions",
        partition_by="txn_month",
    )
    _write_delta(rejected, config.QUARANTINE_DIR / "transactions")
    flagged.unpersist()

    report = quality.silver_transactions_report(spark, run_id, bronze_txn.count(), flagged)
    report.show(truncate=False)
    report.write.format("delta").mode("append").save(str(config.DATA_DIR / "quality" / "silver_transactions"))
    flagged.unpersist()


def main() -> None:
    parser = argparse.ArgumentParser(description="GenericBank medallion pipeline")
    parser.add_argument("--stage", choices=["bronze", "silver", "all"], default="all")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex
    print(f"run_id = {run_id}")
    spark = get_spark("genericbank-pipeline")
    if args.stage in ("bronze", "all"):
        run_bronze(spark, run_id)
    if args.stage in ("silver", "all"):
        run_silver(spark, run_id)
    spark.stop()


if __name__ == "__main__":
    main()