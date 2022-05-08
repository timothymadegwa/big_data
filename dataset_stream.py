import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType
from pyspark.sql.functions import explode, split
from nltk.tokenize import regexp_tokenize

spark = SparkSession.builder.appName("structuredstream").getOrCreate()

struct_type = StructType().add("clean_text", "string").add("category", "string")

filestream = spark.readStream.csv("sentiment_data/*.csv", schema=struct_type, encoding="utf-8")
wordCount = filestream.groupBy("category").count()
query = wordCount.writeStream.format("console").outputMode("complete").start()
query.awaitTermination()
