import numpy as np
import scipy.sparse as sp
import sparseIO
import datetime
import json
import mysql.connector as sql
import sys
import os

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
category_list = {}
for market in (1,3):
    query = ("SELECT DISTINCT category FROM Product_category_lookup WHERE market = %s")
    cursor.execute(query, (market,))
    category_list[market] = cursor.fetchall()

#for each market / category get its app data and transform it to a matrix
for market in (1,3):
    for category in category_list[market]:
        for metric in range(1,4):
            print category
            query = ("SELECT COUNT(*) "
                "FROM Product_category_lookup INNER JOIN Metrics "
                "ON Product_category_lookup.id = Metrics.id "
                "WHERE Metrics.market = %s "
                "AND Product_category_lookup.category = %s "
                "AND metric  = %s "
                "AND country = 'US' "
                "AND (device = 'android' or device = 'iphone') ")
            cursor.execute(query, (market,category[0].encode('ascii'),metric))
            num_of_rows = (cursor.fetchall())[0][0]
            print num_of_rows
            mtx = sp.lil_matrix((num_of_rows, num_of_days))
            query = ("SELECT Metrics.id, Metrics.graph "
                "FROM Product_category_lookup INNER JOIN Metrics "
                "ON Product_category_lookup.id = Metrics.id "
                "WHERE Metrics.market = %s "
                "AND Product_category_lookup.category = %s "
                "AND metric  = %s "
                "AND country = 'US' "
                "AND (device = 'android' or device = 'iphone') ")
            cursor.execute(query, (market,category[0].encode('ascii'),metric))

            idx = 0;
            id_dict = {}
            error_log = {}

            for i in cursor:
                id_dict[i[0].encode('ascii')] = idx
                valid_flag = True
                try:
                    series = json.loads(i[1].encode('ascii'))
                except ValueError:
                    valid_flag = False
                    error_log[i[0].encode('ascii')] = ['no data',]

                if valid_flag:
                    for date in series.keys():
                        try:
                            delta = (datetime.datetime.strptime(date, '%Y-%m-%d') - begin_date).days
                            if delta >= 0 and delta < num_of_days:
                                mtx[idx, delta] = series[date]
                        except:
                            error_log[i[0].encode('ascii')] = error_log.get(i[0].encode('ascii'), []) + [date]
                idx += 1
            #save the data
            mtx = sp.csr_matrix(mtx)
            path = market.__str__() + '/' + category[0].encode('ascii') + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            sparseIO.csrSave(mtx, path + 'datamatrix_metric_'+metric.__str__()+'.npz')
            with open(path+'id_dict_metric_' + metric.__str__() +'.json', 'w') as fp:
                json.dump(id_dict, fp)
            with open(path+'error_log_metric_' + metric.__str__() +'.json', 'w') as fp:
                json.dump(error_log, fp)

#clean up
cursor.close()
cn.close()
