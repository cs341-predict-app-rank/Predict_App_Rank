import json
import mysql.connector as sql
import sys


username = sys.argv[1]
password = sys.argv[2]

def setup_connection(username, password,
                     host = 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
                     port = '3306',
                     database = 'appannie'):
    """
    set up connection to the database.
    Output:
        connection and cursor which respond to query.
    """
    config = {
        'user': username,
        'password': password,
        'host': host,
        'port': port,
        'database': database,
    }
    connection = sql.connect(**config)
    cursor = connection.cursor()
    return connection, cursor

def close_connection(connection, cursor):
    connection.close()
    cursor.close()

def get_market(cursor, product_id):
    query = 'SELECT market FROM Metrics WHERE id = "' + product_id + '" LIMIT 1'
    cursor.execute(query)
    return cursor.fetchall()[0][0]

def insert_row(cursor, table_name, query_frame, query_info):
    query = 'INSERT INTO ' + table_name + ' ' + query_frame + ' VALUES' + query_info + ';'
    print query
    cursor.execute(query)

if __name__ == "__main__":
    connection_product, cursor_product = setup_connection(username, password)
    connection_metric, cursor_metric = setup_connection(username, password)
    connection_new, cursor_new = setup_connection(username, password)
    new_table_name = "Product_category_lookup"
    fields = """id       VARCHAR(255) NOT NULL,
                market   INT          NOT NULL DEFAULT 0,
                category VARCHAR(255) NOT NULL,
                idx      INT          NOT NULL DEFAULT 0,
                PRIMARY KEY (id, market, category, idx)"""
    # create new table
    create_table = "CREATE TABLE " + new_table_name + " (" + fields +")"
    #cursor_new.execute(create_table)
    query = "SELECT * FROM Products LIMIT 1"
    cursor_product.execute(query)
    insert_frame = '(id, market, category)'
    for row in cursor_product:
        product_id = row[0]
        market = get_market(cursor_metric, product_id)
        record = json.loads(row[2])
        category = record[u'product_category']
        insert_info = '("' + product_id + '", ' + str(market) + ', ' + '"' + category + '")'
        insert_row(cursor_new, new_table_name, insert_frame, insert_info)
        #print record

    close_connection(connection_product, cursor_product)
    close_connection(connection_metric, cursor_metric)
    close_connection(connection_new, cursor_new)
