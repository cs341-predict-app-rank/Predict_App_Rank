import matplotlib.pyplot as plt
import numpy as np
import os
import sparseIO

def plotDistribution(market, 
                     category, 
                     metric, 
                     top_percentile = 60,
                     matrix_path = './'):
    #read the data matrix of this category
    data_matrix = sparseIO.csrLoad(matrix_path + market.__str__() + '/' + category + '/' + 
                                   'datamatrix_metric_' + metric.__str__() + '.npz')
    total = data_matrix.sum(1)
    data = total[total > 0]
    num_of_apps = data.size
    data=data.reshape((data.size, 1))
    cumsum = np.cumsum(np.sort(data, axis = 0)[::-1]) / np.sum(data)
    cumsum=cumsum.T
    threshold_pos = cumsum[cumsum < (top_percentile/100.)].size

    threshold = (cumsum[threshold_pos] - cumsum[threshold_pos-1]) * np.sum(data)
    max_power = np.ceil(np.log10(np.max(data)))

    n, bins, pathces = plt.hist((data), bins = np.logspace(0.1, max_power, 100), log = True)
    #n, bins, pathces = plt.hist((data), bins = 1000, log = True)
    bin_centers = np.sqrt (bins[:-1] * bins[1:])
    i = 0
    for p in pathces:
        if bin_centers[i] >= threshold:
            plt.setp(p, 'facecolor', 'r')
        i += 1
    plt.setp(pathces[-1],'label', top_percentile.__str__() + '%'+' category downloads')

    plt.axvline(x = threshold, linestyle = '--', color = 'g')
    plt.title('History Downloads Distribution for Category: '+category)
    plt.xlabel('Downloads (in log scale)')
    plt.ylabel('Amount of Apps (in log scale)')
    plt.legend(title = 'Total Apps: ' + num_of_apps.__str__()+'\n' + top_percentile.__str__() +'%' +' Downloads: ' + (threshold_pos+1).__str__())

    plt.gca().set_xscale("log")
    plt.show()

    plt.clf()

def plotCategoryDownload(market, 
                         category_importance_list, 
                         metric, 
                         matrix_path = './'):
    num_of_days = 821
    num_of_weeks = num_of_days / 7
    data = np.zeros((len(category_importance_list), num_of_weeks))
    i = 0
    for category, is_importtant in category_importance_list:
        download_matrix = sparseIO.csrLoad(matrix_path + market.__str__() + '/' + category + '/' + 
                                       'datamatrix_metric_' + metric.__str__() + '.npz')
        category_sum = download_matrix.sum(0)
        for t in range(num_of_weeks):
            data[i, t] = category_sum[0,(7*t):(7*t + 7)].sum(1)
        if (is_importtant):
            plt.plot(data[i, :], label = category, linewidth = 1.5)
        else:
            plt.plot(data[i, :], color = '0.75', linewidth = 0.6)
        i += 1
    print data.shape
    plt.legend()
    plt.xlim(xmax = 113)
    plt.xlabel('Week')
    plt.ylabel('Total Category Downloads')
    plt.title('Weekly Total Downloads for each Category')
    plt.show()

    plt.clf()

def plotCategoryThreshold(market, 
                          category_importance_list, 
                          metric, 
                          top_percentile = 60,
                          matrix_path = './'):
    num_of_days = 821
    num_of_weeks = num_of_days / 7
    data = np.zeros((len(category_importance_list), num_of_weeks))
    i = 0
    for category, is_importtant in category_importance_list:
        download_matrix = sparseIO.csrLoad(matrix_path + market.__str__() + '/' + category + '/' + 
                                          'datamatrix_metric_' + metric.__str__() + '.npz')
        for t in range(num_of_weeks):
            week_data = download_matrix[:, (7*t):(7*t + 7)].sum(1)
            week_data = week_data[week_data > 0]
            week_data = week_data.T
            cumsum = np.cumsum(np.sort(week_data, axis = 0)[::-1]) / np.sum(week_data)
            data[i, t] = cumsum[cumsum < (top_percentile/100.)].size + 1
        if (is_importtant):
            plt.plot(data[i, :], label = category, linewidth = 1.5)
        else:
            plt.plot(data[i, :], color = '0.75', linewidth = 0.6)

        i+=1
    plt.legend()
    plt.xlim(xmax = 113)
    plt.xlabel('Week')
    plt.ylabel('Amount of Apps ')
    plt.title('Amount of Apps that constitute '+top_percentile.__str__()+'%' + ' category downloads')
    plt.show()

    plt.clf()



if __name__ == '__main__':
    category_list = os.listdir('1/')[1:]
    #Remove games which affects the scale of the plot
    category_list.remove('Games')
    for category in category_list:
        plotDistribution(1, category, 1)

    
    c_i_1 = zip(category_list, [(i == 'Social Networking' or i == 'Entertainment' or i == 'Books')for i in category_list])
    c_i_2 = zip(category_list, [(i == 'Social Networking' or i == 'Business' or i == 'Medical')for i in category_list])
    
    plotCategoryDownload(1, c_i_1, 1)
    plotCategoryThreshold(1, c_i_2, 1)



    

