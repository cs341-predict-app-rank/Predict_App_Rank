import numpy as np
import scipy.sparse as sp
import datetime
import json
import mysql.connector as sql
import sys
import pprint as pp
import zlib
import struct
import time
from glob import glob
from tempfile import TemporaryFile
import datetime

"""
This script create matrices for the Daily Ranking Data of apps. For each
category one matrix will be created, with each row representing each app
in this category and each column representing each day. 

The range of dates is from 2013-01-01 to 2015-04-01. 
Only U.S apps are considered.

Since the matrices are sparse, scipy csr format is used to store these matrices.
And a dict to map appid to row index is also created for each category.
Usage: python2 matrixBuild.py username password
Username and password should not be contained in code because we ARE OPEN SOURCE!

"""

def setup_connection(username, passwd):
    # setup connection configuration
    config = {
        'user': username,
        'password': passwd,
        'host': 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
        'port': '3306',
        'database': 'appannie',
    }
    return config

def get_category(cursor, mrkt_num):
    query = ("SELECT distinct category FROM TopCharts "
        "WHERE country = 'US' and market = %s" % mrkt_num)
    cursor.execute(query)
    cate = cursor.fetchall()
    return cate

def get_data_in_one_category(cursor, mrkt_num, cate_name, limit = ''):
    query = ('SELECT rank_date, data FROM TopCharts '
            'WHERE market = %s'
            ' and category = "%s"'
            ' and country = "US"'
            ' and device = "iphone" %s' % (mrkt_num, cate_name, limit)
        ) 
    cursor.execute(query)
    raw_data = cursor.fetchall()
    return raw_data

def get_idx_in_one_category(cursor, mrkt_num, cate_name):
    query = ("SELECT id, idx FROM Product_category_lookup "
            'WHERE market = %s'
            ' and category = "%s"' % (mrkt_num, cate_name)
        ) 
    cursor.execute(query)
    index = dict()
    for row in cursor:
        time.sleep(0.00005)
        index[str(row[0])] = row[1]
    return index

def get_rank_in_one_day(cate_data, date):
    # rank_all is a metrix. Each raw represents a app:
    #   Three columns represent: 
    #     0 feed type(free=0, paid=1, grossing=2), 
    #     1 rank in this feed type,
    #     2 prduct id 
    # num_free, num_paid, num_gros are total numbers of ranking in each feed
    date_data = cate_data[date][0]
    bolb_data = cate_data[date][1]
    json_data = zlib.decompress(str(bolb_data))
    data = json.loads(json_data)
    rank_all = np.zeros((len(data['list']), 3), dtype=np.int) #feed, rank, product id
    num_free = 0
    num_paid = 0
    num_gros = 0
    for r in range(0,len(data['list'])):
        if data['list'][r]['feed'] == 'free':
            rank_all[r, 0] = 0;
            num_free = num_free + 1
        if data['list'][r]['feed'] == 'paid':
            rank_all[r, 0] = 1;
            num_paid = num_paid + 1
        if data['list'][r]['feed'] == 'grossing':
            rank_all[r, 0] = 2;
            num_gros = num_gros + 1
        rank_all[r, 1] = data['list'][r]['rank'];
        rank_all[r, 2] = data['list'][r]['product_id'];
    return rank_all, num_free, num_paid, num_gros

def get_cate_name(raw):
    new = raw.split(' > ')
    return new[len(new)-1]

if __name__ == '__main__':
    try: username = sys.argv[1]
    except IndexError:
        print "user name is not provided!\nUsage: python2 matrixBuild.py username password"
        sys.exit(1)
    try: passwd = sys.argv[2]
    except IndexError:
        print "password is not provided!\nUsage: python2 matrixBuild.py username password"
        sys.exit(1)

    # connect to db
    config = setup_connection(username, passwd)
    cn = sql.connect(**config)
    cursor = cn.cursor()
   
    # set the time range that the matrix represent
    begin_date = datetime.datetime.strptime('2013-01-01', '%Y-%m-%d')
    end_date = datetime.datetime.strptime('2015-04-01', '%Y-%m-%d')
    num_of_days = (end_date-begin_date).days + 1
    mrkt_num = '1' # set up market choises: 1=IOS or 3=Android
    
    # get the category list
    cate = get_category(cursor, mrkt_num)

    # for each category get its app rank data and transform it to a matrix  
    for i in range(0, len(cate)):
        start = time.time()
        cate_data = get_data_in_one_category(cursor, mrkt_num, cate[i][0])# only inculde iphone in US
        cate_name = get_cate_name(cate[i][0])
        # get index from product_category_lookup
        index = get_idx_in_one_category(cursor, mrkt_num, cate_name)
        print 'Got index.','\t','Run time:', (time.time()-start)
        mtx_gros = sp.lil_matrix((len(index)+1, len(cate_data)), dtype=np.int)
        # mtx_free = sp.lil_matrix(len(cate_data), 1000)
        # mtx_paid = sp.lil_matrix(len(cate_data), 1000)

        # update daily ranking 
        date = -1
        for day in range(0, len(cate_data)):   # for each day in this category
            rank_all, num_free, num_paid, num_gros = get_rank_in_one_day(cate_data, day)
            
            # update each rank
            date = date + 1
            for r in range((num_free+num_paid), len(rank_all)-1):
                app_rank = rank_all[r,1]
                app_id = str(rank_all[r,2])
                if index.has_key(app_id):
                    idx_row = index.get(app_id)
                    # print idx_row, app_rank
                    mtx_gros[idx_row, date] = app_rank
                else:
                    pass
                    # print 'Cannot find index for id =', '\'',app_id,'\''
        print 'Finish Matrix:',cate_name,'\t','Index #:',len(index), '\t', 'Mtx element #:', mtx_gros.nnz,'\t','Day #:',date+1, 'Run time:', (time.time()-start)
        filename = cate_name + '.us.iphone.npz'
        np.savez(filename, mtx_gros = mtx_gros)

    #clean up
    cursor.close()
    cn.close()