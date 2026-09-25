"""Single place to create the SparkSession used by the pipeline, tests and notebooks."""
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from genericbank import config


def get_spark(app_name: str = config.APP_NAME) -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")                                          # local mode, use all CPU cores
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", config.SHUFFLE_PARTITIONS)
        .config("spark.sql.session.timeZone", config.BUSINESS_TIMEZONE)       # fixed business timezone
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()   # adds Delta JARs, reuses an existing session
    spark.sparkContext.setLogLevel("WARN")
    return spark