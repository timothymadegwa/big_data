import pandas as pd
import findspark
import pyspark
from pyspark.sql import SparkSession

from pyspark.sql.functions import col,isnan, when, count
from wordcloud import WordCloud, STOPWORDS
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
total = data.count()
print(total)

#Positive
positive = data.where('category == 1')
pos_count = positive.count()
print("Positive", pos_count)
print("Percentage", (pos_count/total)*100)

#Negative
negative = data.where('category == -1')
neg_count = negative.count()
print("Negative", neg_count)
print("Percentage", (neg_count/total)*100)

#Positive
neutral = data.where('category == 0')
neu_count = neutral.count()
print("Neutral", neu_count)
print("Percentage", (neu_count/total)*100)

stopwords = set(STOPWORDS)
#common words

#Positive
pd_positive = positive.toPandas()
print(pd_positive.head())
all_positive = " ".join([sentence for sentence in pd_positive['clean_text']]).lower()

wordcloud = WordCloud(width=800, height=500, random_state=555, max_font_size=100, stopwords=stopwords).generate(all_positive)
print(wordcloud)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Negative

pd_negative = negative.toPandas()
print(pd_negative.head())
all_negative = " ".join([sentence for sentence in pd_negative['clean_text']]).lower()

wordcloud = WordCloud(width=800, height=500, random_state=555, max_font_size=100, stopwords=stopwords).generate(all_negative)
print(wordcloud)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

#Neutral

pd_neutral = neutral.toPandas()
print(pd_neutral.head())
all_neutral = " ".join([sentence for sentence in pd_neutral['clean_text']]).lower()

wordcloud = WordCloud(width=800, height=500, random_state=555, max_font_size=100, stopwords=stopwords).generate(all_neutral)
print(wordcloud)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
