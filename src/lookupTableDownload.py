import datetime
import json
import pickle
import mysql.connector as sql
import sys
import os
import time

"""
Usage: python2 matrixBuild.py username password
Username and password should not be contained in code because we ARE OPEN SOURCE!
"""

try: username = sys.argv[1]
except IndexError:
    print "user name is not provided!\nUsage: python2 matrixBuild.py username password"
    sys.exit(1)
try: passwd = sys.argv[2]
except IndexError:
    print "password is not provided!\nUsage: python2 matrixBuild.py username password"
    sys.exit(1)

#setup connection configuration
config = {
    'user': username,
    'password': passwd,
    'host': 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
    'port': '3306',
    'database': 'appannie',
}

#connect to db
cn = sql.connect(**config)
cursor = cn.cursor()

#Download lookup table
lookup_table = {}

#get the category list
query = ("SELECT DISTINCT category FROM Product_category_lookup WHERE market = %s")
cursor.execute(query, (1,))
for category in cursor:
    lookup_table[category[0].encode('ascii')] = {}
#for each category get its number of apps
for category in lookup_table.keys():
    print category
    query = ("SELECT COUNT(*) FROM Product_category_lookup "
            "WHERE category = %s ")
    cursor.execute(query, (category,))
    lookup_table[category]['num_of_rows'] = (cursor.fetchall())[0][0]
    query = ("SELECT id, idx FROM Product_category_lookup "
             "WHERE market = 1 AND category = %s ")
    cursor.execute(query, (category, ))
    for item in cursor:
        lookup_table[category][int(item[0].encode('ascii'))] = item[1]
        time.sleep(0.001)


#save lookup table
f = open('lookup_table.pkl', 'w')
pickle.dump(lookup_table, f)

#clean up
f.close()
cursor.close()
cn.close()
