import numpy as np
import scipy.sparse as sp
import sparseIO
import datetime
import json
import pickle
import mysql.connector as sql
import sys
import os
import time

"""
This script create matrices for the ratings/reviews time series of apps. For each
category one matrix will be created, with each row representing each app
in this category and each column representing each day. The range of dates
is from 2013-01-01 to 2015-04-01. Only U.S apps are considered.

Since the matrices are sparse, scipy csr format is used to store these matrices.
And a dict to map appid to row index is also created for each category.

Usage: python2 matrixBuild.py username password
Username and password should not be contained in code because we ARE OPEN SOURCE!
"""

#set the time range that the matrix represent
begin_date = datetime.datetime.strptime('2013-01-01', '%Y-%m-%d')
end_date = datetime.datetime.strptime('2015-04-01', '%Y-%m-%d')
num_of_days = (end_date-begin_date).days + 1
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
    'host': 'google-app.coqatsb9cruk.us-west-2.rds.amazonaws.com',
    'port': '3306',
    'database': 'appleapps',
}

#connect to db
cn = sql.connect(**config)
cursor = cn.cursor()

#Download review and ratings data
data_table = {}
query = ("SELECT app_id, ratings_graph, reviews_graph FROM Ratings WHERE country = 'us'")
cursor.execute(query)
for data in cursor:
    data_table[data[0]] = data[1:]
    time.sleep(0.001)

cursor.close()
cn.close()
#save data table
f = open('data_table.pkl', 'w')
pickle.dump(data_table, f)
f.close()


f = open('lookup_table.pkl')
lookup_table = pickle.load(f)
f.close()

for category in lookup_table.keys():
    print category
    ratings_mtx = sp.lil_matrix((lookup_table[category]['num_of_rows'], num_of_days))
    reviews_mtx = sp.lil_matrix((lookup_table[category]['num_of_rows'], num_of_days))
    for app_id in lookup_table[category].keys():
        if app_id in data_table:
            try:
                ratings_series = json.loads(data_table[app_id][0].encode('ascii'))
                for date in ratings_series.keys():
                    delta = (datetime.datetime.strptime(date, '%Y-%m-%d') - begin_date).days
                    if delta >= 0 and delta < num_of_days and ratings_series[date]>=0 and ratings_series[date]<=5:
                        ratings_mtx[lookup_table[category][appid], delta] = ratings_series[date]
            except:
                print"cannot parse ", app_id, " for ratings"
                print data_table[app_id][0]
            try:
                reviews_series = json.loads(data_table[app_id][1].encode('ascii'))
                for date in reviews_series.keys():
                    delta = (datetime.datetime.strptime(date, '%Y-%m-%d') - begin_date).days
                    if delta >= 0 and delta < num_of_days and reviews_series[date]>0:
                        reviews_mtx[lookup_table[category][appid], delta] = reviews_series[date]
            except:
                print"cannot parse ", app_id, "for reviews"
                print data_table[app_id][1]
    ratings_mtx = sp.csr_matrix(ratings_mtx)
    reviews_mtx = sp.csr_matrix(reviews_mtx)
    path = '1/' + category + '/'
    sparseIO.csrSave(ratings_mtx, path + 'datamatrix_ratings'+'.npz')
    sparseIO.csrSave(reviews_mtx, path + 'datamatrix_reviews'+'.npz')


