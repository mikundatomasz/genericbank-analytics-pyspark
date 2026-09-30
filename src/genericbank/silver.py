"""Silver layer: typed, cleaned, deduplicated and enriched data."""
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Column, DataFrame, Window

MONEY = "decimal(18,2)"


def _normalized(name: str) -> Column:
    """Dictionary values: trim whitespace, upper-case (' mobile ' -> 'MOBILE')."""
    return F.upper(F.trim(F.col(name)))

def deduplicate_transactions(df: DataFrame) -> DataFrame:
    """Keep exactly one row per transaction_id (latest ingestion wins, ties broken deterministically)."""
    latest_first = (
        Window
        .partitionBy("transaction_id")                                   # one group per transaction id
        .orderBy(F.col("_ingested_at").desc(), F.col("_source_file"))   # which copy comes first
    )
    return (
        df
        .withColumn("_rn", F.row_number().over(latest_first))           # 1, 2, 3... inside each group
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

def flag_rejects(txn: DataFrame, accounts: DataFrame) -> DataFrame:
    """Add reject_reason to every transaction; null means the row is valid."""
    known_accounts = (
        accounts
        .select(F.trim("account_id").alias("account_id"))
        .distinct()
        .withColumn("_account_known", F.lit(True))
    )
    return (
        txn
        .join(F.broadcast(known_accounts), on="account_id", how="left")    # small table -> broadcast
        .withColumn(
            "reject_reason",
            F.when(F.col("amount").isNull(), "INVALID_AMOUNT")            # first matching rule wins
             .when(F.col("txn_ts").isNull(), "INVALID_TIMESTAMP")
             .when(F.col("_account_known").isNull(), "UNKNOWN_ACCOUNT"),
        )
        .drop("_account_known")
    )

def clean_customers(df: DataFrame) -> DataFrame:
    """Typed customer dimension with normalised segment."""
    return df.select(
        F.trim("customer_id").alias("customer_id"),
        _normalized("segment").alias("segment"),                 # ' Retail' -> 'RETAIL'
        F.col("birth_year").cast("int").alias("birth_year"),
        F.trim("region").alias("region"),
        F.to_date("relationship_start").alias("relationship_start"),
    )


def clean_accounts(df: DataFrame) -> DataFrame:
    """Typed account dimension."""
    return df.select(
        F.trim("account_id").alias("account_id"),
        F.trim("customer_id").alias("customer_id"),
        _normalized("account_type").alias("account_type"),
        _normalized("currency").alias("currency"),
        F.to_date("open_date").alias("open_date"),
    )

def parse_transactions(df: DataFrame) -> DataFrame:
    """Cast raw strings to proper types; keep the raw values for audit."""
    return (
        df
        .withColumnRenamed("txn_ts", "txn_ts_raw")
        .withColumnRenamed("amount", "amount_raw")
        .withColumn("transaction_id", F.trim("transaction_id"))
        .withColumn("account_id", F.trim("account_id"))
        .withColumn(
            "txn_ts",
            F.coalesce(                                                  # first format that parses wins
                F.to_timestamp("txn_ts_raw", "yyyy-MM-dd HH:mm:ss"),
                F.to_timestamp("txn_ts_raw", "dd.MM.yyyy HH:mm"),
            ),
        )
        .withColumn("amount", F.regexp_replace("amount_raw", ",", ".").cast(MONEY))
        .withColumn("direction", _normalized("direction"))
        .withColumn("txn_type", _normalized("txn_type"))
        .withColumn("channel", _normalized("channel"))
        .withColumn("merchant_category", _normalized("merchant_category"))
        .withColumn("counterparty_country", _normalized("counterparty_country"))
    )

def enrich_transactions(txn_valid: DataFrame, accounts: DataFrame, customers: DataFrame) -> DataFrame:
    """Attach customer and account attributes; add derived columns; fix the silver column order."""
    acc = accounts.select("account_id", "customer_id", "account_type")
    cust = customers.select("customer_id", "segment")
    return (
        txn_valid
        .join(F.broadcast(acc), on="account_id", how="left")
        .join(F.broadcast(cust), on="customer_id", how="left")
        .withColumn("txn_date", F.to_date("txn_ts"))
        .withColumn("txn_month", F.trunc("txn_date", "month"))                      # 2026-03-17 -> 2026-03-01
        .withColumn(
            "signed_amount",
            F.when(F.col("direction") == "DEBIT", -F.col("amount")).otherwise(F.col("amount")),
        )
        .select(
            "transaction_id", "customer_id", "account_id", "account_type", "segment",
            "txn_ts", "txn_date", "txn_month",
            "amount", "signed_amount", "direction", "txn_type", "channel",
            "merchant_category", "counterparty_country",
        )
    )