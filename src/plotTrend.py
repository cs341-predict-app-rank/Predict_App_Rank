import sys
import scipy.sparse
from sparseIO import *
from lookUpTable import *
import matplotlib.pyplot as plt
import os

def queryId(cursor, appname):
    query = 'SELECT id FROM Products WHERE name = "%s"' % appname
    print query
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
    except IOError: print "File not exist!"; return None
    return whole_matrix[idx, :]

def queryInfo(cursor, appname, metric):
    id_list = queryId(cursor, appname)
    if not id_list:
        print "Query Failed"
        return 1
    market_category_idx_list = [queryCategory(cursor, id) for id in id_list]
    for (i, (market, category, idx)) in enumerate(market_category_idx_list):
        row = queryDownloadMatrix(market, category, idx, metric)
        if row is not None:
            row = np.array(row.todense()[0,:-6])
            plt.plot(range(row.shape[1]), row[0,:], label = 'data')
            plt.legend(loc = 2, title = 'product id: ' + id_list[i] + '\n' + 'metric: ' + str(metric))
            plt.savefig(appname + '_' + str(i) + '_market_' + str(market) + '_metric_' + str(metric) + '.pdf')

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
        error = queryInfo(cursor, appname, metric)
        if error is not None:
            print "App not found in such metric"
