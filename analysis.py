import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,isnan, when, count

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
data = data.drop('srcip', 'dstip', 'sport','dsport', 'Stime','Ltime')

# Viewing select columns
data.select("attack_cat", "Label", "service").show()

#fixing the attack category rows for fuzzers and reconnaissance
data = data.withColumn('attack_cat', when(data['attack_cat'] == 'Fuzzers ', 'Fuzzers').otherwise(data['attack_cat']))
data = data.withColumn('attack_cat', when(data['attack_cat'] == 'Reconnaissance ', 'Reconnaissance').otherwise(data['attack_cat']))
print(data.count())

# checking for the schema
print(data.printSchema())

# distinct vales in the proto column

print(data.select('proto').distinct().count())
data.filter(data['proto'] =='dns').show()

#filling null values in the attack category with Normal
data = data.na.fill('Normal', subset=['attack_cat'])

#dropping duplicate data
data = data.drop_duplicates()
print("number of records after droping duplicates")
#print(data.count())
print("number of rows after dropping null")
data = data.dropna(how='any')
#print(data.count())

print(data.columns)
cols_of_interest = ['sbytes','dbytes', 'sload','dload', 'sloss','dloss']

for col in cols_of_interest:
    data.select(col).describe().show()
    print('--------------------------------------------')
for col in cols_of_interest:
    print('Correlation for ', col, 'is',data.corr(col, 'Label'))

attack_cat_group = data.groupBy('attack_cat').count()
attack_cat_pandas = attack_cat_group.toPandas()
attack_cat_pandas.plot.bar()
plt.title("Attack Category")
plt.yscale('log')
plt.xlabel('category')
plt.ylabel('count of category')
plt.show()

data.groupBy("attack_cat")
service_group =data.groupBy('service').count().orderBy('count', ascending=False)
service_group.show()


data.groupBy('proto').count().orderBy('count', ascending=False).show()


dur = data.select('dur').toPandas()
dur.plot.hist(bins=100)
plt.yscale('log')
plt.title("Histogram of duration")
plt.show()

service_crosstab = data.crosstab('Label','service').sort('Label_service')
service_crosstab.show()
service_crosstab_pandas = service_crosstab.toPandas()
service_crosstab_pandas.plot.bar()
plt.title("Services Grouped by Label")
plt.yscale('log')
plt.xlabel('Label')
plt.ylabel('count of service')
plt.show()

state_crosstab = data.crosstab('Label','state').sort('Label_state')
state_crosstab.show()
state_crosstab_pandas = state_crosstab.toPandas()
state_crosstab_pandas.plot.bar()
plt.title("State Grouped by Label")
plt.yscale('log')
plt.xlabel('Label')
plt.ylabel('count of State')
plt.show()
