import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import col,isnan, when, count
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

findspark.init()
sc = pyspark.SparkContext
spark = SparkSession.builder.master("local[1]").appName("assignment").getOrCreate()

# reading the data
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
data = data.drop('srcip', 'dstip', 'sport','dsport', 'Stime','Ltime')

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
indexed = indexed.drop('attack_cat_index', 'attack_cat')
encoded_cols = [col+'_encoded' for col in indexed_cols]
encoder = OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols)
encoded = encoder.fit(indexed).transform(indexed)

cols_to_remove = string_cols + indexed_cols
feature_cols = encoded.columns
for col in cols_to_remove:
    feature_cols.remove(col)

feature_cols.remove('Label')
assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")
output = assembler.transform(encoded)

# Split the data into training and test sets
(trainingData, testData) = output.randomSplit([0.7, 0.3],seed=555)

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="Label", featuresCol="features")

print("Decision tree Training started ")
dt_model = dt.fit(trainingData)
print("Training ended")

# Make predictions.
print("testing started")
dt_predictions = dt_model.transform(testData)
print("testing ended")

dt_pandas = dt_predictions.toPandas()
dt_acc = accuracy_score(dt_pandas['Label'],dt_pandas['prediction'])
dt_rec = recall_score(dt_pandas['Label'],dt_pandas['prediction'])
dt_prec = precision_score(dt_pandas['Label'],dt_pandas['prediction'])
dt_f1 = f1_score(dt_pandas['Label'],dt_pandas['prediction'])

# Select (prediction, true label) and compute test error
evaluator = BinaryClassificationEvaluator(labelCol="Label")
dt_roc = evaluator.evaluate(dt_predictions)



lr = LogisticRegression(featuresCol = 'features', labelCol = 'Label', maxIter=10)
print("Logistic regression traing started")
lrModel = lr.fit(trainingData)
print("training ended")
lr_predictions = lrModel.transform(testData)

lr_pandas = lr_predictions.toPandas()
lr_acc = accuracy_score(lr_pandas['Label'],lr_pandas['prediction'])
lr_rec = recall_score(lr_pandas['Label'],lr_pandas['prediction'])
lr_prec = precision_score(lr_pandas['Label'],lr_pandas['prediction'])
lr_f1 = f1_score(lr_pandas['Label'],lr_pandas['prediction'])
evaluator = BinaryClassificationEvaluator(labelCol="Label")
lr_roc = evaluator.evaluate(lr_predictions)


rf = RandomForestClassifier(featuresCol= 'features', labelCol='Label')
print("Random forest model training started")
rf_model = rf.fit(trainingData)
print("training complete")
rf_predictions = rf_model.transform(testData)

rf_pandas = rf_predictions.toPandas()
rf_acc = accuracy_score(rf_pandas['Label'],rf_pandas['prediction'])
rf_rec = recall_score(rf_pandas['Label'],rf_pandas['prediction'])
rf_prec = precision_score(rf_pandas['Label'],rf_pandas['prediction'])
rf_f1 = f1_score(rf_pandas['Label'],rf_pandas['prediction'])
rf_roc = evaluator.evaluate(rf_predictions)


print("-------------------------RESULTS-------------------------")
print("Decision tree")
print("---------------------------------------------------------")
print("Accuracy", dt_acc)
print("Recall", dt_rec)
print("Precision", dt_prec)
print("F1 Score", dt_f1)
print("ROC", dt_roc)
print("----------------------------------------------------------")

print("Logistic regression")
print("---------------------------------------------------------")
print("Accuracy", lr_acc)
print("Recall", lr_rec)
print("Precision", lr_prec)
print("F1 score", lr_f1)
print("ROC", lr_roc)
print("----------------------------------------------------------")

print("Random Forest")
print("---------------------------------------------------------")
print("Accuracy", rf_acc)
print("Recall", rf_rec)
print("Precision", rf_prec)
print("F1 score", rf_f1)
print("ROC", rf_roc)
print("----------------------------------------------------------")

'''

lr_predictions = lrModel.transform(testData)

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
'''
'''
treeModel = dt_model.stages[2]
# summary only
print(treeModel)
'''
#https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa