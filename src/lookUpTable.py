import json
import mysql.connector as sql
import sys


username = sys.argv[1]
password = sys.argv[2]

config = {
    'user': username,
    'password': password,
    'host': 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
    'port': '3306',
    'database': 'appannie',
}

#connect to db
connection = sql.connect(**config)
cursor = connection.cursor()
query = "select data from Products"
cursor.execute(query)
n = cursor.rowcount
i = 0
for row in cursor:
    i += 1
    if i % 10 == 0: print i#, row[0]
    #if i % 100 == 0: print row[0]
    if i == 1000000: break

connection.close()
cursor.close()
