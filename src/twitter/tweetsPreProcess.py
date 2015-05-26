from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import udf
import datetime
import pickle
"""
This file preprocess tweets data using spark SQL
"""

#to be configured
sc = SparkContext()
sqlContext = SQLContext(sc)



def dateStrToCol(date_str):
    begin_date = datetime.datetime.strptime('2013-01-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2015-04-01', '%Y-%m-%d')
    if len(date_str) < 10:
        return 0
    else:
        date = datetime.datetime.strptime(date_str[0:10], '%Y-%m-%d')
        max_idx = (end_date-begin_date).days
        return min([(date-begin_date).days, max_idx])

slen = udf(lambda s: len(s), IntegerType())
dateNum = udf(dateStrToCol, IntegerType())

f = open('twitter_schema.pkl')
schema_read = pickle.load(f)
f.close()
#load twitter data as dataframe, and register it as a table
#df = SqlContext.jsonFile("~/Downloads/ntweets_2015-03-31.22.4.txt")
#df = sqlContext.jsonFile("Downloads/twitter_raw")
df = sqlContext.jsonFile("s3n://cs341-data/twitter_test_500", schema = schema_read)
print "loaded"
#df.cache()
#df.registerTempTable("tweets")

#filter fields and tweets:
#Only keep body, actor.favoritesCount, actor.followersCount, actor.friendsCount, actor.verified,
#favoritesCount, retweetCount, verb actor.followersCount AS actor.followersCount, \
    #actor.friendsCount AS actor.friendsCount, actor.verified AS actor.verified,
#sqlContext.sql("SELECT body, actor.favoritesCount AS favoritesCount,  favoritesCount, retweetCount, verb FROM tweets ").save("Downloads/processed.json", 'json')
df.filter("twitter_lang = 'en' ").filter(slen(df.body) > 10).select(df.body, df.favoritesCount, df.retweetCount, df.verb, 
    df["actor.favoritesCount"].alias('actor.favoritesCount'), df["actor.followersCount"].alias('actor.followersCount'), 
    df['actor.friendsCount'].alias('actor.friendsCount'), df['actor.verified'].alias('actor.verified'), dateNum(df['postedTime']).alias('date'), 
    df['twitter_entities.user_mentions.screen_name'].alias('mentions_screen'), df['twitter_entities.user_mentions.name'].alias('mentions_name')).save("s3n://cs341-data/processed_tweets_test_8", "parquet")

#df['twitter_entities.user_mentions.screen_name'].alias('mentions_screen'), df['twitter_entities.user_mentions.name'].alias('mentions_name')
#df2 = sqlContext.jsonFile("Downloads/processed_tweets_test")
#df2.printSchema()
df2 = sqlContext.parquetFile("s3n://cs341-data/processed_tweets_test_8")
df2.printSchema()
