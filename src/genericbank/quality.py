"""Data quality report and reconciliation for the silver layer."""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def silver_transactions_report(
    spark: SparkSession, run_id: str, bronze_rows: int, flagged: DataFrame
) -> DataFrame:
    """Account for every bronze row: duplicate, rejected (by reason) or loaded to silver."""
    after_dedup = flagged.count()
    by_reason = {
        row["reject_reason"]: row["count"]
        for row in flagged.groupBy("reject_reason").count().collect()   # tiny result: safe to collect
    }
    silver_rows = by_reason.pop(None, 0)                                 # null reason = valid row
    duplicates = bronze_rows - after_dedup

    metrics = [("bronze_rows", bronze_rows), ("duplicates_removed", duplicates)]
    metrics += [(f"rejected_{reason.lower()}", n) for reason, n in sorted(by_reason.items())]
    metrics.append(("silver_rows", silver_rows))

    balance = bronze_rows - duplicates - sum(by_reason.values()) - silver_rows
    if balance != 0:
        raise ValueError(f"Reconciliation failed: {balance} rows unaccounted for")

    return (
        spark.createDataFrame(metrics, "metric string, value long")
        .withColumn("run_id", F.lit(run_id))
        .withColumn("checked_at", F.current_timestamp())
    )