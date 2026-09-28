"""Pipeline entry point: python -m genericbank.pipeline --stage bronze"""
import argparse
import uuid

from pyspark.sql import SparkSession

from genericbank import bronze, config
from genericbank.spark import get_spark


def run_bronze(spark: SparkSession, run_id: str) -> None:
    sources = {
        "customers": bronze.read_customers,
        "accounts": bronze.read_accounts,
        "transactions": bronze.read_transactions,
    }
    for name, read in sources.items():
        target = str(config.BRONZE_DIR / name)
        df = bronze.add_ingestion_metadata(read(spark), run_id=run_id)
        (
            df.write.format("delta")
            .mode("overwrite")                    # full reload on every run
            .option("overwriteSchema", True)      # allow schema changes between runs
            .save(target)
        )
        rows = spark.read.format("delta").load(target).count()
        print(f"bronze.{name}: {rows:,} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="GenericBank medallion pipeline")
    parser.add_argument("--stage", choices=["bronze", "all"], default="all")
    args = parser.parse_args()

    run_id = uuid.uuid4().hex                     # one id per pipeline run
    print(f"run_id = {run_id}")
    spark = get_spark("genericbank-pipeline")
    if args.stage in ("bronze", "all"):
        run_bronze(spark, run_id)
    spark.stop()


if __name__ == "__main__":
    main()