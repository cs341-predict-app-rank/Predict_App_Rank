from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import IntegerType, StructType
from pyspark.sql.functions import udf
import pickle
"""
This file preprocess tweets data using spark SQL
"""

#to be configured
SparkContext.setSystemProperty('spark.executor.memory', '2g')
sc = SparkContext("local[4]", "App Name")
sqlContext = SQLContext(sc)

#slen = udf(lambda s: len(s), IntegerType())
#load twitter data as dataframe, and register it as a table
#df = SqlContext.jsonFile("~/Downloads/ntweets_2015-03-31.22.4.txt")
df =  sqlContext.jsonFile("Downloads/ntweets_2015-03-31.22.4.txt")

schema = df.schema
f = open('twitter_schema.pkl', 'w')
pickle.dump(schema, f)
f.close()
print "loaded"

f = open('twitter_schema.pkl')
schema_read = pickle.load(f)
f.close()
df =  sqlContext.jsonFile("Downloads/ntweets_2015-03-31.22.4.txt", schema = schema_read)
