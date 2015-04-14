import sys
import scipy.sparse
from sparseIO import *
from lookUpTable import *

def queryId(cursor, appname):
    query = 'SELECT id FROM Products WHERE name = "%s"' % appname
    cursor.execute(query)
    return [a for (a,) in list(cursor.fetchall())]

def queryCategory(cursor, product_id):
    query = 'SELECT category,idx FROM Product_category_lookup WHERE id = "%s"' % product_id
    cursor.execute(query)
    return cursor.fetchall()[0]

def queryInfo(cursor, appname, metric):
    id_list = queryId(cursor, appname)
    category_idx_list = [queryCategory(cursor, id) for id in id_list]
    print id_list, category_idx_list

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
