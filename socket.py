
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split
from pyspark.ml.feature import StopWordsRemover
import matplotlib.pyplot as plt

spark = SparkSession.builder.master("local[1]").appName("assignment").getOrCreate()


lines = spark.readStream.format('socket').option("host", "127.0.0.1").option("port", "9996").load()
words = lines.select(explode(split(lines.value, " ")).alias("word"))
wordCounts = words.groupBy('word').count().sort('count', ascending=False)

query = wordCounts.writeStream.outputMode('complete').format("console").start()


query.awaitTermination()

