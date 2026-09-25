from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# 1. Build the "recipe" for the Spark session
builder = (
    SparkSession.builder
    .appName("smoke")                       # application name (visible in the Spark UI)
    .master("local[*]")                     # local mode: driver + executors on my laptop, * = use all cores
    .config("spark.driver.memory", "6g")    # how much RAM for the driver
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")                  # enable Delta support in SQL
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")  # table catalog that understands Delta
)

# 2. Pull in the Delta JARs (Java code) and start the session
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# 3. Transformation: DataFrame with numbers 0-9 (LAZY, nothing is computed yet)
df = spark.range(10).withColumnRenamed("id", "n")

# 4. Action: write to Delta (NOW Spark actually does the work)
df.write.format("delta").mode("overwrite").save("/tmp/smoke_delta")

# 5. Read from Delta and display
spark.read.format("delta").load("/tmp/smoke_delta").show()

# 6. Table history: Delta can do this, plain Parquet can't
spark.sql("DESCRIBE HISTORY delta.`/tmp/smoke_delta`").select("version", "operation").show()

print("Spark", spark.version, "OK")
spark.stop()
