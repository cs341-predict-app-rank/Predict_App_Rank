import sys
import scipy.sparse
from sparseIO import *

def queryInfo(appname, metric):
    pass

if __name__ == '__main__':
    username = sys.argv[0]
    password = sys.argv[1]
    while(True):
        print "App name: "
        appname = input()
        print "Metric: "
        metric = input()
        queryInfo(appname, metric)
