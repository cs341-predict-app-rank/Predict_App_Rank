import numpy as np
import scipy.sparse as sp
import datetime
import json
import mysql.connector as sql
import sys

"""
This script create matrices for the metric time series of apps. For each
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
    'host': 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
    'port': '3306',
    'database': 'appannie',
}

#connect to db
cn = sql.connect(**config)
cursor = cn.cursor()

#get the category list
query = ("SELECT * FROM Categories WHERE market = 1 OR market = 3")
cursor.execute(query)
cate = cursor.fetchall()

#for each category get its app data and transform it to a matrix
query = ("SELECT id, graph FROM Metrics "
    "WHERE metric  = %s and country = 'US' and (device = 'android' or device = 'iphone')"
    " LIMIT 10")
cursor.execute(query, (1,))
data = cursor.fetchall()
# here I use lil matrix instead of csr because csr is not suitable for mutable object
mtx = sp.lil_matrix((len(data), num_of_days))

idx = 0;
id_dict = {}

for i in data:
    id_dict[i[0].encode('ascii')] = idx
    series = json.loads(i[1].encode('ascii'))
    for date in series.keys():
        delta = (datetime.datetime.strptime(date, '%Y-%m-%d') - begin_date).days
        if delta >= 0 and delta < num_of_days:
            mtx[idx, delta] = series[date]
    idx += 1

mtx = sp.csr_matrix(mtx)

#clean up
cursor.close()
cn.close()
