"""Smoke test: Spark starts, Delta can write, read and show table history."""
from genericbank.spark import get_spark

spark = get_spark("smoke")

df = spark.range(10).withColumnRenamed("id", "n")                      # transformation: lazy
df.write.format("delta").mode("overwrite").save("/tmp/smoke_delta")    # action: Spark actually runs
spark.read.format("delta").load("/tmp/smoke_delta").show()
spark.sql("DESCRIBE HISTORY delta.`/tmp/smoke_delta`").select("version", "operation").show()

print("Spark", spark.version, "OK")
spark.stop()