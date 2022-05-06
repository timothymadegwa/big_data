import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import Tokenizer, RegexTokenizer, HashingTF, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.evaluation import ClusteringEvaluator

from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

findspark.init()
sc = pyspark.SparkContext
spark = SparkSession.builder.master("local[1]").appName("assignment").getOrCreate()

# reading the data

data = spark.read.option("header",True).csv("emails/spam_or_not_spam.csv")
data.show(5)
data = data.dropna()
data = data.drop_duplicates()

tokenizer = RegexTokenizer(inputCol="email", outputCol="words", pattern="\\W")
stop_word_remover = StopWordsRemover(inputCol="words", outputCol="stop_words")
count_vec = CountVectorizer(vocabSize=2**7,inputCol="stop_words", outputCol="count_vec")
idf = IDF(inputCol="count_vec", outputCol="features", minDocFreq=10) 
pipeline = Pipeline(stages=[tokenizer,stop_word_remover, count_vec, idf])

eval = ClusteringEvaluator()
pipelineFit = pipeline.fit(data)
train_df = pipelineFit.transform(data)

print(train_df.show(5))

for k in range(2,6):
    kmeans = KMeans(k=k, maxIter=100, seed=555)

    model = kmeans.fit(train_df)
    predictions = model.transform(train_df)

    score = eval.evaluate(predictions)
    print(f"For K={k}",score)
