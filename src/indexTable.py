import json
import mysql.connector as sql
import sys
from lookUpTable import *
import os

def load_all():
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    filename = 'id_dict.json'
    indexDict = dict()
    market_list = [1, 3]
    for market in market_list:
        market_dir = curr_dir + '/' + str(market)
        categories = os.listdir(market_dir)
        indexDict[market] = dict()
        for category in categories:
            category_dir = market_dir + '/' + category
            filename_with_path = category_dir + '/' + filename
            indexDict[market][category] = json.load(open(filename_with_path))
    return indexDict

if __name__ == "__main__":
    username = sys.argv[1]
    password = sys.argv[2]
    indexDict = load_all()
    connection_lookup, cursor_lookup = setup_connection(username, password)
    connection_update, cursor_update = setup_connection(username, password)
    query = "SELECT * FROM Product_category_lookup WHERE market = 3"
    cursor_lookup.execute(query)
    insert_frame = '(id, metric, idx)'
    for row in cursor_lookup:
        product_id = row[0]
        market = row[1]
        category = row[2]
        table = indexDict[market][category]
        update_query = 'UPDATE Product_category_lookup SET idx = %d WHERE id = "%s"' % (table[product_id], product_id)
        print update_query
        cursor_update.execute(update_query)
    connection_update.commit()
    close_connection(connection_lookup, cursor_lookup)
    close_connection(connection_update, cursor_update)
