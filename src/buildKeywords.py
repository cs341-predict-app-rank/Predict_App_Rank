"""
build keyword dictionary for given category
"""
import mysql.connector as sql
import sys
import sparseIO
import pickle

def getSelectedAppIdx(download_file, threshold):
    """
    return a list of row indices in a category with total downloads
    more than given threshold
    """
    total_downloads = sparseIO.csrLoad(download_file).sum(1)
    qualify_list = []
    for i in range(0, total_downloads.size):
        if(total_downloads[i] >= threshold):
            qualify_list.append(i)
    return qualify_list

def getKeywordByIndex(idx, market, category, cursor, excluded):
    """
    return keyword of a given app idx
    the first word in it's name 
    excluding given words
    """
    #get name
    query = ("SELECT Products.name "
         "FROM Product_category_lookup INNER JOIN Products "
         "ON Product_category_lookup.id = Products.id "
         "WHERE Product_category_lookup.idx = %s "
         "AND Product_category_lookup.category = %s" 
         "AND Product_category_lookup.market = %s")
    cursor.execute(query, (idx, category, market))
    name = cursor.fetchall()
    word = ''
    if (len(name) == 0):
        print 'Warning: index ' + item[0].__str__() + ' cannot be found in the database!'
    else:
        try:
            name = name[0][0].encode('ascii').lower().split()
            word = name[0]
            if word in excluded:
                word = ''
        except:
            pass
    return word


if __name__ == '__main__':

    try: username = sys.argv[1]
    except IndexError:
        print "user name is not provided!\nUsage: python2 matrixBuild.py username password"
        sys.exit(1)
    try: passwd = sys.argv[2]
    except IndexError:
        print "password is not provided!\nUsage: python2 matrixBuild.py username password"
        sys.exit(1)

    config = {
        'user': username,
        'password': passwd,
        'host': 'appannie.coqatsb9cruk.us-west-2.rds.amazonaws.com',
        'port': '3306',
        'database': 'appannie',
    }
    cn = sql.connect(**config)
    cursor = cn.cursor()

    market = 1
    excluded_words = set(['a', 'an', 'and', 'at', 'by', 'on', 'in', 'of', 'to', 'be', 'it'])
    category_list = ['Social Networking', 'Photo and Video']

    for category in category_list:
        data_file = '1/'+category + '/datamatrix_metric_1.npz'
        idx_list = getSelectedAppIdx(data_file, 1000)
        print len(idx_list)
        keyword_dict = {}
        for idx in idx_list:
            word = getKeywordByIndex(idx, market, category, cursor, excluded_words)
            if not word == '':
                if word not in keyword_dict:
                    keyword_dict[word] = []
                keyword_dict[word].append(idx)
        f = open(category + '_keywords.pkl', 'w')
        pickle.dump(keyword_dict, f)
        f.close()

    #clean up
    cursor.close()
    cn.close()

