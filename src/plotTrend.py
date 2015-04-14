import sys
import scipy.sparse
from sparseIO import *
from lookUpTable import *
import matplotlib.pyplot as plt
import os

def queryId(cursor, appname):
    query = 'SELECT id FROM Products WHERE name = "%s"' % appname
    cursor.execute(query)
    return [id for (id,) in list(cursor.fetchall())]

def queryCategory(cursor, product_id):
    query = 'SELECT market,category,idx FROM Product_category_lookup WHERE id = "%s"' % product_id
    cursor.execute(query)
    return cursor.fetchall()[0]

def queryDownloadMatrix(market, category, idx, metric):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    market_dir = curr_dir + '/' + str(market)
    category_dir = market_dir + '/' + category
    filename = category_dir + '/datamatrix_metric_' + str(metric) + '.npz'
    try: whole_matrix = csrLoad(filename)
    except IOError: print "File not exist!"
    return whole_matrix[idx, :]

def queryInfo(cursor, appname, metric):
    id_list = queryId(cursor, appname)
    market_category_idx_list = [queryCategory(cursor, id) for id in id_list]
    for (market, category, idx) in market_category_idx_list:
        row = queryDownloadMatrix(market, category, idx, metric)
        plt.plot(row)

if __name__ == '__main__':
    username = sys.argv[1]
    password = sys.argv[2]
    connection, cursor = setup_connection(username, password)
    while(True):
        print "App name: "
        appname = sys.stdin.readline()
        if appname == "":
            close_connection(connection, cursor)
            exit(0)
        else: appname = appname[:-1]
        print "Metric: "
        metric = sys.stdin.readline()
        if metric == "":
            close_connection(connection, cursor)
            exit(0)
        else: metric = int(metric)
        queryInfo(cursor, appname, metric)
