from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
"""
This file preprocess tweets data using spark SQL
"""

#to be configured
SparkContext.setSystemProperty('spark.executor.memory', '2g')
sc = SparkContext("local[4]", "App Name")
sqlContext = SQLContext(sc)


slen = udf(lambda s: len(s), IntegerType())
#load twitter data as dataframe, and register it as a table
#df = SqlContext.jsonFile("~/Downloads/ntweets_2015-03-31.22.4.txt")
df = sqlContext.jsonFile("Downloads/twitter_raw")
print "loaded"
df.cache()
#df.registerTempTable("tweets")

#filter fields and tweets:
#Only keep body, actor.favoritesCount, actor.followersCount, actor.friendsCount, actor.verified,
#favoritesCount, retweetCount, verb actor.followersCount AS actor.followersCount, \
    #actor.friendsCount AS actor.friendsCount, actor.verified AS actor.verified,
#sqlContext.sql("SELECT body, actor.favoritesCount AS favoritesCount,  favoritesCount, retweetCount, verb FROM tweets ").save("Downloads/processed.json", 'json')
df.filter("twitter_lang = 'en' ").filter(slen(df.body) > 10).select(df.body, df.favoritesCount, df.retweetCount, df.verb, 
    df["actor.favoritesCount"].alias('actor.favoritesCount'), df["actor.followersCount"].alias('actor.followersCount'), 
    df['actor.friendsCount'].alias('actor.friendsCount'), df['actor.verified'].alias('actor.verified')).save("Downloads/processed_tweets_test_5", "parquet", "append")

#df2 = sqlContext.jsonFile("Downloads/processed_tweets_test")
#df2.printSchema()
