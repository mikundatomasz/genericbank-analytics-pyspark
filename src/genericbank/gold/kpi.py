"""Gold mart: monthly KPIs per customer on a complete customer x month grid."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

KPI_COLUMNS = ["inflow", "outflow", "net_flow", "txn_count", "card_txn_count", "active_days"]


def customer_monthly_kpi(transactions: DataFrame, customers: DataFrame) -> DataFrame:
    monthly = (
        transactions
        .groupBy("customer_id", "txn_month")
        .agg(
            F.sum(F.when(F.col("direction") == "CREDIT", F.col("amount")).otherwise(0)).alias("inflow"),
            F.sum(F.when(F.col("direction") == "DEBIT", F.col("amount")).otherwise(0)).alias("outflow"),
            F.sum("signed_amount").alias("net_flow"),
            F.count("*").alias("txn_count"),
            F.sum(F.when(F.col("txn_type") == "CARD_PAYMENT", 1).otherwise(0)).alias("card_txn_count"),
            F.countDistinct("txn_date").alias("active_days"),
        )
    )

    months = transactions.select("txn_month").distinct()
    grid = (
        customers.select("customer_id", "segment", "relationship_start")
        .crossJoin(months)                                                   # every customer x every month
        .filter(F.col("txn_month") >= F.trunc("relationship_start", "month"))  # not before they joined
        .drop("relationship_start")
    )

    return (
        grid
        .join(monthly, on=["customer_id", "txn_month"], how="left")
        .fillna(0, subset=KPI_COLUMNS)                                       # no transactions -> zeros
        .withColumn(
            "avg_txn_amount",
            F.when(
                F.col("txn_count") > 0,
                F.round((F.col("inflow") + F.col("outflow")) / F.col("txn_count"), 2),
            ),
        )
    )