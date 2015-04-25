import datetime
import matplotlib.pyplot as plt
import mysql.connector as sql
import os
import sparseIO

def plotAppWithRow (row_date_list, 
                    market = None, 
                    category = None, 
                    metric = 1, 
                    db_user = None, 
                    db_pswd = None, 
                    matrix_path = './', 
                    output_path = 'plots/'):
    """
    Function: plotAppWithRow
        plot the historical metric of given apps. One plot will be generated for each app, and will indicate
        the app's name and the given time. 
    Input:
        row_date_list: a list of tuples of app row index and date.
        market: the market (1 or 3). If not provided user will have to input it interactively.
        category: the category of the provided app. If not provided user will have to input it interactively.
        metric: the metric to plot. default is 1 (free downloads). 
        db_user: username of the database. If not provided user will have to input it interactively.
        db_pswd: password of the database. If not provided user will have to input it interactively. 
        matrix_path: the directory in which the data matrices are stored, i.e., the super directory of folder 1 and 3. 
        The default is the current directory. 
        output_path: the directory in which the output figures will be saved. The default is plots/market/category where
        market/category is the actual market number/category name. 

    Output:
        The results will be saved to the given directory. 
        The function will return a dict: idx -> name.
    """

    #setup connection configuration and connect to the db
    if (db_user is None):
        db_user = raw_input('username for db: ')
    if (db_pswd is None):
        db_pswd = raw_input('password for db: ')
    config = {
        'user': db_user,
        'password': db_pswd,
        'host': 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
        'port': '3306',
        'database': 'appannie',
    }
    cn = sql.connect(**config)
    cursor = cn.cursor()

    #check the availability of the category and the market
    if (market is None):
        market = raw_input('please specify market: ')
        market = int(market)
    if (category is None):
        category = raw_input('please specify category: ')
    while (True):
        query = ("SELECT DISTINCT category, market FROM Product_category_lookup WHERE category = %s AND market = %s")
        cursor.execute(query, (category, market))
        real_cat_mkt = cursor.fetchall()
        if (len(real_cat_mkt) > 0):
            break
        else:
            category = raw_input('The specified category does not exist. Please specify category again: ')

    #if necessary build the output directory
    output_path = output_path + market.__str__() + '/' + category + '/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #read the data matrix of this category
    data_matrix = sparseIO.csrLoad(matrix_path + market.__str__() + '/' + category + '/' + 
                                   'datamatrix_metric_' + metric.__str__() + '.npz')

    #plot and save results:
    query = ("SELECT Products.name "
             "FROM Product_category_lookup INNER JOIN Products "
             "ON Product_category_lookup.id = Products.id "
             "WHERE Product_category_lookup.idx = %s "
             "AND Product_category_lookup.category = %s")
    name_dict = {}
    num_of_dates = data_matrix.get_shape()[1]
    for item in row_date_list:
        cursor.execute(query, (item[0], category))
        name = cursor.fetchall()
        if (len(name) == 0):
            print 'Warning: index ' + item[0].__str__() + ' cannot be found in the database!'
        else:
            name = name[0][0].encode('ascii')
            name_dict[item[0]] = name
            plt.plot(range(num_of_dates), data_matrix[item[0], :].todense().T, label = 'data')
            plt.axvline(x = item[1], linestyle = '--')
            plt.title(name)
            plt.legend(loc = 2, title = 'metric: ' + metric.__str__()) 
            plt.savefig(output_path + name + '_metric_' + metric.__str__() + '.pdf')
            plt.clf()

    #clean up
    cursor.close()
    cn.close()

    return name_dict













