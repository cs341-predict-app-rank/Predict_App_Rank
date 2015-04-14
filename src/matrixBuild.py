import numpy as np
import scipy.sparse as sp
import sparseIO
import datetime
import json
import mysql.connector as sql
import sys
import os
import time

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
        query = ("SELECT id FROM Product_category_lookup "
                "WHERE category = %s ")
        cursor.execute(query, (category[0].encode('ascii'),))
        id_dict = {}
        num_of_rows = 0
        for i in cursor:
            id_dict[i[0].encode('ascii')] = num_of_rows
            num_of_rows += 1

        print category
        print num_of_rows

        for metric in range(1,4):
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
            error_log = {}

            for i in cursor:
                if (i[0].encode('ascii') in id_dict):
                    idx = id_dict[i[0].encode('ascii')]
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
                else:
                    error_log[i[0].encode('ascii')] = ['no data',]
            #save the data
            mtx = sp.csr_matrix(mtx)
            path = market.__str__() + '/' + category[0].encode('ascii') + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            sparseIO.csrSave(mtx, path + 'datamatrix_metric_'+metric.__str__()+'.npz')
            with open(path+'error_log_metric_' + metric.__str__() +'.json', 'w') as fp:
                json.dump(error_log, fp)
        """
        path = market.__str__() + '/' + category[0].encode('ascii') + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path+'id_dict' + '.json', 'w') as fp:
            json.dump(id_dict, fp)
        """

#clean up
cursor.close()
cn.close()
