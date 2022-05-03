import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import Tokenizer,RegexTokenizer, HashingTF, StopWordsRemover,NGram, CountVectorizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sparknlp

from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

findspark.init()
sc = pyspark.SparkContext
spark = SparkSession.builder.master("local[1]").appName("assignment").getOrCreate()

# reading the data

data = spark.read.option("header",True).csv("twitter/Twitter_Data.csv")
data.show(5)

print(data.count())
data = data.dropna()
data = data.drop_duplicates()
print(data.count())


#tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\W")
stop_word_remover = StopWordsRemover(inputCol="words", outputCol="stop_words")
#bi_grams = NGram(n=2, inputCol="stop_words", outputCol="trigrams")
#count_vec = CountVectorizer(vocabSize=2**16, inputCol="trigrams", outputCol="count_vec")
hashtf = HashingTF(numFeatures=2**16, inputCol="stop_words", outputCol="tf")
idf = IDF(inputCol="tf", outputCol="features", minDocFreq=5) #minDocFreq: remove sparse terms
label_stringIdx = StringIndexer(inputCol = "category", outputCol = "label")
pipeline = Pipeline(stages=[tokenizer,stop_word_remover, hashtf, idf, label_stringIdx])

(train, test) = data.randomSplit([0.8, 0.2], seed = 555)

pipelineFit = pipeline.fit(train)
train_df = pipelineFit.transform(train)
test_df = pipelineFit.transform(test)
print(train_df.show(10))

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression(maxIter=100)
print("DT model training started")
lrModel = lr.fit(train_df)
print("training complete")
predictions = lrModel.transform(test_df)

evaluator = MulticlassClassificationEvaluator(labelCol="label")
roc = evaluator.evaluate(predictions)

print(roc)
print(predictions.show(5))
dt_pandas = predictions.toPandas()
acc = accuracy_score(dt_pandas['label'],dt_pandas['prediction'])
print(acc)
'''
rf = RandomForestClassifier(featuresCol= 'features', labelCol='label')
print("Random forest model training started")
rf_model = rf.fit(train_df)
print("training complete")
rf_predictions = rf_model.transform(test_df)
rf_roc = evaluator.evaluate(rf_predictions)
print(rf_roc)
'''
