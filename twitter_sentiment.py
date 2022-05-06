import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover,NGram, CountVectorizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# import sparknlp

from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

findspark.init()
sc = pyspark.SparkContext
spark = SparkSession.builder.master("local[1]").appName("assignment").getOrCreate()

# reading the data

data = spark.read.option("header",True).csv("twitter/Twitter_Data.csv")
data.show(5)
data = data.dropna()
data = data.drop_duplicates()
evaluator = MulticlassClassificationEvaluator(labelCol="label")
print(f"Defaut metric is : {evaluator.getMetricName()}")


tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\W")
stop_word_remover = StopWordsRemover(inputCol="words", outputCol="stop_words")
count_vec = CountVectorizer(inputCol="stop_words", outputCol="count_vec")
idf = IDF(inputCol="count_vec", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer,stop_word_remover, count_vec, idf, label_stringIdx])

(train, test) = data.randomSplit([0.8, 0.2], seed = 555)

pipelineFit = pipeline.fit(train)
train_df = pipelineFit.transform(train)
test_df = pipelineFit.transform(test)
print(train_df.show(10))

# Machine Learning
lr = LogisticRegression(maxIter=100)
print("LR model training started")
lrModel = lr.fit(train_df)
print("training complete")
train_pred = lrModel.transform(train_df)
lr_predictions = lrModel.transform(test_df)

train_roc = evaluator.evaluate(train_pred)
roc = evaluator.evaluate(lr_predictions)

print(f"Train ROC is {train_roc}")
print(f"Test ROC is {roc}")
print(lr_predictions.show(5))
train_pred = train_pred.toPandas()
lr_pandas = lr_predictions.toPandas()
train_acc = accuracy_score(train_pred['label'], train_pred['prediction'])
acc = accuracy_score(lr_pandas['label'],lr_pandas['prediction'])

print(f"train accuract is {train_acc}")
print(f"Test accuracy is {acc}")

# Naive Bayes

tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\W")
stop_word_remover = StopWordsRemover(inputCol="words", outputCol="stop_words")
bi_grams = NGram(n=2, inputCol="stop_words", outputCol="bigrams")
count_vec = CountVectorizer(inputCol="bigrams", outputCol="count_vec")
idf = IDF(inputCol="count_vec", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer,stop_word_remover,bi_grams, count_vec, idf, label_stringIdx])

(train, test) = data.randomSplit([0.8, 0.2], seed = 555)

pipelineFit = pipeline.fit(train)
train_df = pipelineFit.transform(train)
test_df = pipelineFit.transform(test)
print(train_df.show(10))

nb =NaiveBayes(featuresCol= 'features', labelCol='label')
print("Naive Bayes model training started")
nb_model = nb.fit(train_df)
print("training complete")
train_pred = nb_model.transform(train_df)
nb_predictions = nb_model.transform(test_df)

train_roc = evaluator.evaluate(train_pred)
nb_roc = evaluator.evaluate(nb_predictions)
print(f"Train ROC is {train_roc}")
print(f"Test ROC is {nb_roc}")

train_pred = train_pred.toPandas()
nb_pandas = nb_predictions.toPandas()
train_acc = accuracy_score(train_pred['label'], train_pred['prediction'])
acc = accuracy_score(nb_pandas['label'],nb_pandas['prediction'])


print(f"train accuract is {train_acc}")
print(f"Test accuracy is {acc}")
