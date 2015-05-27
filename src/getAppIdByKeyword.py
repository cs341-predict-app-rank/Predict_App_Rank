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
import os 
from glob import glob
from tempfile import TemporaryFile
import datetime


"""
This script get App Id and Index By Keyword and market num.

Usage: python2 username password

NOTE: add '%' as a prefix or suffix to keyword to do partial match
e.g python facebook username password
e.g python facebook% username password

Username and password should not be contained in code because we ARE OPEN SOURCE!

"""

def getAppIdByKeyword(cursor, market, keywordList):
	'''
	Input: cursor, market, keywordList
		Example:
		market = 1
		keywordList = ['evernote','facebook','gmail','wechat']
	
	Output:
		A dictionary:
			['keyword'][0] app id List
			['keyword'][1] app idx List
		Example:
			{'facebook': [[u'284882215'], [7]], 
			 'evernote': [[u'281796108'], [0]], 
			 'wechat': [[u'414478124'], [3040]], 
			 'gmail': [[], []]}
	
	NOTE: add '%' as a prefix or suffix to keyword to do partial match
	e.g facebook
	e.g facebook%
	'''
	result = dict()
	for keyword in keywordList:
		query = (
			"select id, idx from Product_category_lookup "
			'where market = %s and id in ('
				'select id from Products '
				"where name like '%s');" % (market, keyword)
		)
		# print query
		cursor.execute(query)
		InfoList = cursor.fetchall()
		idList = list()
		idxList = list()
		for app in InfoList:
			idList.append(app[0])
			idxList.append(app[1])
		result[keyword] = [idList, idxList]
	return result

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

def getAppNameById(cursor, appid):
	query = (
		'select name from Products '
		"where id = %s;" % (appid)
	)
	cursor.execute(query)
	name = cursor.fetchall()
	return name


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
	config = setup_connection(username,passwd)
	cn = sql.connect(**config)
	cursor = cn.cursor()
	
	mrkt_num = '1'
	keywordlist = ['evernote','facebook','gmail','wechat']

	result = getAppIdByKeyword(cursor, mrkt_num, keywordlist)
	
	print result