import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

findspark.init()
sc = pyspark.SparkContext
spark = SparkSession.builder.master("local[1]").appName("assignment").getOrCreate()

data_desc = pd.read_csv('UNSW-NB15_features.csv', encoding = "ISO-8859-1")
cols = data_desc['Name'].to_list()
data_types = data_desc['Type '].to_list()
data_types = list(map(str.lower,data_types))

data = spark.read.option("header",False).csv("UNSW-NB15.csv")


for old,new, dtype in zip(data.columns, cols, data_types):
    data = data.withColumnRenamed(old, new)
    if dtype == "nominal":
        pass
    if dtype == "integer":
        data = data.withColumn(new,data[new].cast('integer'))
    if dtype == "float":
        data = data.withColumn(new,data[new].cast('float'))
    if dtype == "binary":
        data = data.withColumn(new,data[new].cast('integer'))
    if dtype == "timestamp":
        data = data.withColumn(new,data[new].cast('timestamp'))


#dropping columns/features that are not required
data = data.drop('srcip', 'dstip', 'sport','dsport', 'Stime','Ltime', 'Label')

#fixing the attack category rows for fuzzers and reconnaissance
data = data.withColumn('attack_cat', when(data['attack_cat'] == 'Fuzzers ', 'Fuzzers').otherwise(data['attack_cat']))
data = data.withColumn('attack_cat', when(data['attack_cat'] == 'Reconnaissance ', 'Reconnaissance').otherwise(data['attack_cat']))
data = data.na.fill('Normal', subset=['attack_cat'])

#dropping duplicate data
data = data.drop_duplicates()

data = data.dropna(how='any')
print(data.columns)

string_cols = ['proto', 'state', 'service', 'attack_cat']
indexed_cols = [col+'_index' for col in string_cols]
indexer = StringIndexer(inputCols=string_cols, outputCols=indexed_cols)
indexed = indexer.fit(data).transform(data)
string_cols.remove('attack_cat')
indexed_cols.remove('attack_cat_index')
attack_cat_index = indexed.select('attack_cat_index')
indexed = indexed.drop('attack_cat')
encoded_cols = [col+'_encoded' for col in indexed_cols]
encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
encoded = encoder.fit(indexed).transform(indexed)

cols_to_remove = string_cols + indexed_cols
feature_cols = encoded.columns
for col in cols_to_remove:
    feature_cols.remove(col)

feature_cols.remove('attack_cat_index')
assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")
output = assembler.transform(encoded)

# Split the data into training and test sets
(trainingData, testData) = output.randomSplit([0.7, 0.3],seed=555)

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="attack_cat_index", featuresCol="features")
#pipeline = Pipeline(stages=[dt])
print("Decision tree Training started ")
dt_model = dt.fit(trainingData)
print("Training ended")

# Make predictions.
print("testing started")
dt_predictions = dt_model.transform(testData)
print("testing ended")

# Select example rows to display.
dt_predictions.select('prediction', 'features', 'attack_cat_index').show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="attack_cat_index")
dt_accuraccy = evaluator.evaluate(dt_predictions)
print(dt_accuraccy)

evaluator = MulticlassClassificationEvaluator(labelCol="attack_cat_index")
rf = RandomForestClassifier(featuresCol= 'features', labelCol='attack_cat_index')
print("Random forest model training started")
rf_model = rf.fit(trainingData)
print("training complete")
rf_predictions = rf_model.transform(testData)
rf_accuraccy = evaluator.evaluate(rf_predictions)
print(rf_accuraccy)

