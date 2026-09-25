# AGENTS.md

GenericBank Analytics: a local PySpark + Delta Lake pipeline over synthetic retail-banking data
(customers, accounts, transactions) producing analytics marts: monthly customer KPIs, RFM segments,
churn signals, AML structuring alerts and spend by category. All data is synthetic; GenericBank is fictional.

## Working agreement
- Propose shell commands with a one-line explanation; run them only when the user explicitly asks.
- Edit only the files the task names; propose broader changes before making them.
- When writing Spark code, explain what each transformation does and whether it triggers a shuffle.

## Architecture (medallion)
- `data/landing/`: raw files produced by the synthetic generator.
- **bronze**: raw 1:1 copy, all columns as strings, plus `_ingested_at` and `_source_file`.
- **silver**: typed, cleaned, deduplicated and enriched; rejected rows go to `data/quarantine/` with a `reject_reason`.
- **gold**: business marts, one module per mart in `src/genericbank/gold/`.
- All layers are Delta tables under `data/` (git-ignored).

## Code conventions
- Transformations are pure functions `DataFrame -> DataFrame` with no I/O; reading and writing live in `pipeline.py`.
- Every transformation has a pytest test built from a small hand-written DataFrame.
- Use built-in `pyspark.sql.functions`; a Python UDF needs a written justification
  (UDFs serialise rows to Python and are opaque to the Catalyst optimizer).
- Keep large data in Spark; `collect()` / `toPandas()` only on small gold results.
- Money is `DecimalType(18,2)`.
- Business parameters (AML threshold, EUR/PLN rate, window lengths, data scale, random seed) live in `src/genericbank/config.py`.
- One SparkSession factory: `src/genericbank/spark.py`.
- Code, comments and commit messages in English; Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`).

## Environment
- Java 17 is required (Spark runs on the JVM).
- Dependencies are managed with uv: `uv sync` to install, `uv run <cmd>` to run.