from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# 1. Budujemy "przepis" na sesję Sparka
builder = (
    SparkSession.builder
    .appName("smoke")                       # nazwa aplikacji (widoczna w Spark UI)
    .master("local[*]")                     # tryb lokalny: driver + executory na moim laptopie, * = użyj wszystkich rdzeni
    .config("spark.driver.memory", "6g")    # ile RAM dla drivera
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")                  # włącz obsługę Delty w SQL
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")  # katalog tabel rozumiejący Deltę
)

# 2. Dociągnij JAR-y Delty (kod w Javie) i uruchom sesję
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# 3. Transformacja: DataFrame z liczbami 0-9 (LAZY, nic się jeszcze nie liczy)
df = spark.range(10).withColumnRenamed("id", "n")

# 4. Akcja: zapis do Delty (TERAZ Spark naprawdę pracuje)
df.write.format("delta").mode("overwrite").save("/tmp/smoke_delta")

# 5. Odczyt z Delty i wyświetlenie
spark.read.format("delta").load("/tmp/smoke_delta").show()

# 6. Historia tabeli: to potrafi Delta, zwykły parquet nie
spark.sql("DESCRIBE HISTORY delta.`/tmp/smoke_delta`").select("version", "operation").show()

print("Spark", spark.version, "OK")
spark.stop()