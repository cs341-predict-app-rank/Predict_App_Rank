import numpy as np
import scipy.sparse as sp
import datetime
import json
import mysql.connector as sql
import sys
import pprint as pp
import zlib
import struct
from glob import glob
from tempfile import TemporaryFile

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
    query = ("SELECT distinct category FROM TopCharts WHERE market = " + str(mrkt_num))
    cursor.execute(query)
    cate = cursor.fetchall()
    return cate

def get_data_in_one_category(cursor, mrkt_num, cate_name, limit):
    query = ("SELECT rank_date, data FROM TopCharts "
            "WHERE market = " + str(mrkt_num) +
            " and category = '" + str(cate_name) +"'"
            " and country = 'US'"
            " and device = 'iphone' "+ limit
        ) 
    cursor.execute(query)
    raw_data = cursor.fetchall()
    return raw_data

def get_rank_in_one_day(cate_data, date):
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
    mrkt_num = 1 # set up market choises: 1=IOS or 3=Android
    rank_max = 3000 # record Top 1000 app, in each category, and each feed

    # get the category list
    cate = get_category(cursor, mrkt_num)

    # for each category get its app rank data and transform it to a matrix 
    for i in range(0, 9):#len(cate)):
        cate_name = cate[i][0]  
        cate_data = get_data_in_one_category(cursor, mrkt_num, cate_name, "limit 5")# only inculde iphone in US

        # initialize rank matrix for this cate, Top 500
        
        # mtx_free = sp.lil_matrix(len(cate_data), 1000)
        # mtx_paid = sp.lil_matrix(len(cate_data), 1000)
        mtx_gros = np.zeros((rank_max, len(cate_data)), dtype=np.int)
        idx_gros = dict()
        
        # update daily ranking 
        date = -1
        for day in range(0, len(cate_data)):   # for each date in this category
            rank_all, num_free, num_paid, num_gros = get_rank_in_one_day(cate_data, day)
            # rank_all is a metrix. Each raw represents a app:
            #   Three columns represent: 
            #     0 feed type(free=0, paid=1, grossing=2), 
            #     1 rank in this feed type,
            #     2 prduct id 
            # num_free, num_paid, num_gros are total numbers of ranking in each feed
        
            # update each rank
            date = date + 1
            for r in range((num_free+num_paid), len(rank_all)-1):
                idx_row = -1
                if idx_gros.has_key(str(rank_all[r,2])):
                    idx_row = idx_gros.get(str(rank_all[r,2]))
                else:
                    idx_row = len(idx_gros)
                    idx_gros[str(rank_all[r,2])] = idx_row              
                # print idx_row, date, rank_all[r,2]
                mtx_gros[idx_row, date] = rank_all[r,1]
        
        print mtx_gros[0:100,:]
        filename = cate_name + '.us.iphone.npz'
        np.savez(filename, mtx_gros=mtx_gros, idx_gros=idx_gros)

    #clean up
    cursor.close()
    cn.close()