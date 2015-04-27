import sys
import scipy.sparse
from sparseIO import *
from lookUpTable import *
import matplotlib.pyplot as plt
import os
from os.path import expanduser
home = expanduser("~")

"""
Function: This script draw Daily Download and Rank Data of apps. For each

Usage: python plotHistory.py username password -s 
    -s : show plot and save plot to pdf
    -n : only save plot to pdf
Username and password should not be contained in code because we ARE OPEN SOURCE!

NOTE: you can add '%' to the left or right side of your app name to do partial match.

"""


def queryId(cursor, appname):
    query = 'SELECT id, name FROM Products WHERE UPPER(name) LIKE UPPER("%s")' % appname
    print query
    cursor.execute(query)
    result = cursor.fetchall()
    idx = list()
    name = list()
    for val in result:
        idx.append(val[0])
        name.append(val[1])
    print 'Find indices: ', idx 
    print 'App names:    ', name
    return idx, name

def queryCategory(cursor, product_id):
    query = 'SELECT market,category,idx FROM Product_category_lookup WHERE id = "%s"' % product_id
    cursor.execute(query)
    return cursor.fetchall()[0]

def queryDownloadMatrix(market, category, idx, metric):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    market_dir = curr_dir + '/' + str(market)
    category_dir = market_dir + '/' + category
    filename = category_dir + '/datamatrix_metric_' + str(metric) + '.npz'
    try: 
        whole_matrix = csrLoad(filename)
        print 'Successfully Load', filename
    except IOError: 
        print "File not exist!", filename
        return None
    return whole_matrix[idx, :]

def queryRankMatrix(market, category, idx):
    if market == 1: market_name = '.iphone'
    elif market == 3: market_name = '.android'
    country = '.us'
    directory = 'rank/'
    filename = directory + category + country + market_name + '.npz'
    try: 
        whole_matrix = np.load(filename)['mtx_gros'][()]
        print 'Successfully Load', filename
    except IOError: 
        print "File not exist!", filename
        return None
    return whole_matrix[idx, :]

def queryInfo(cursor, appname, metric, show):
    id_list, name_list = queryId(cursor, appname)
    if not id_list:
        print "Query Failed"
        return 1
    market_category_idx_list = [queryCategory(cursor, id) for id in id_list]
    for (i, (market, category, idx)) in enumerate(market_category_idx_list):
        row = queryDownloadMatrix(market, category, idx, metric)
        tmp_name = appname
        name_idx = 0
        for tmp_id in id_list:
            if tmp_id == idx:
                tmp_name = name_list[name_idx]
                break
            else:
                name_idx = name_idx + 1
        if row is not None:
            row = np.array(row.todense()[0,:-6])
            plt.plot(range(row.shape[1]), row[0,:], label = 'data')
            plt.legend(loc = 2, title = 'product id: ' + id_list[i] + '\n' + 'metric: ' + str(metric)+ '\n Cate:' + category)
            plt.title('Download of App:' + tmp_name )
            plt.savefig(appname + '_' + str(i) + '_market_' + str(market) + '_metric_' + str(metric) + '.pdf')
            plt.clf()
        rank = queryRankMatrix(market, category, idx)
        if rank is not None:
            rank = np.array(rank.todense()[0,:-6])
            rank += 999 * (rank == 0)
            # print rank[0,:]
            plt.plot(range(rank.shape[1]), rank[0,:], label = 'data')
            plt.legend(loc = 2, title = 'product id: ' + id_list[i] + '\n' + 'metric: ' + str(metric)+ '\n Cate:' + category)
            plt.title('Rank of App:' + tmp_name + ' in Cate:' + category)
            if max(rank[0,:]) > 1000:
                plt.axis([1, 900 ,1000, 1])
            else: plt.axis([1, 900 ,max(rank[0,:])+1, 1])
            plt.savefig(appname + '_' + str(i) + '_market_' + str(market) + 'rank' + '.pdf')
            if show == '-s':
                plt.show()
            plt.clf()

if __name__ == '__main__':
    username = sys.argv[1]
    password = sys.argv[2]
    show = sys.argv[3]
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
        error = queryInfo(cursor, appname, metric, show)
        if error is not None:
            print "App not found in such metric"
