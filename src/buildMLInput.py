#!/usr/bin/python2
import sparseIO
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt

############################################################################
# Hyper parameters:                                                        #
#   inputFile: filename to the input npz file                              #
#   predictTimeWindow: number of weeks used to generate label              #
#   featureTimeWindow: number of weeks used to generate feature            #
#   slidingWindowSize: number of weeks in a sliding window                 #
#   outOfSigmaSuccess: how many sigmas out counts as success               #
#   successThreshold: win from category have to win how many windows.      #
#   garbageThreshold: total download threshold for candidate               #
#   testPortion: the portion of data treating as test                      #
# Note: successThreshold <= predictTimeWindow - slidingWindowsize + 1      #
############################################################################
WEEK = 7 # don't change this, you know why :-)
inputFile = './Productivity/datamatrix_metric_1.npz'
predictTimeWindow = 10
featureTimeWindow = 10
slidingWindowSize = 4
outOfSigmaSuccess = 2
successThreshold = 5
garbageThreshold = featureTimeWindow * WEEK # a download a day, keep doctors away.
testPortion = 0.2

def rawDataMatrix(inputFile):
    """
    Function: rawDataMatrix
        Read raw data from file.
    Input:
        inputFile: string contains filename.
    Output:
        a csr matrix contains raw data.
    """
    return sparseIO.csrLoad(inputFile)

def compressMatrix(rawData, windowSize = slidingWindowSize * WEEK, skipDay = WEEK):
    """
    Function: compressMatrix
        Generate sliding window data matrix.
    Input:
        rawData: a matrix/sparse matrix contains raw data.
        skipDay: how many days to skip, 1 week as default.
        windowSize: the size of a sliding window,
                slidingWindowSize weeks as default.
    Output:
        transform raw data into sliding windows.
    """
    dataNum = rawData.shape[0]
    windowNum = (rawData.shape[1] - windowSize) // skipDay
    slidingWindowMatrix = np.zeros((dataNum, windowNum))
    for i in range(windowNum):
        slidingWindowMatrix[:,i] = rawData[:,(WEEK * i):(WEEK * i + windowSize)].sum(1).T
    return sps.csr_matrix(slidingWindowMatrix)

def swipeOutInactiveApp(downloadMatrix, predictMatrix, leastDownload = garbageThreshold):
    """
    Function: swipeOutInactiveApp
        Cleaning Apps with too small download amount in feature matrix.
    Input:
        downloadMatrix: cleaning is done from here, type must be scipy.sparse.*
        predictMatrix: also apply the cleaning on prediction, type scipy.sparse.*
        leastDownload: the least download within downloadMatrix to be consider.
    Output:
        cleaned downloadMatrix and predictMatrix
    """
    totalDownload = np.array(downloadMatrix.sum(1)).ravel()
    candidate = totalDownload > leastDownload
    return downloadMatrix[candidate,:].toarray(), predictMatrix[candidate,:].toarray()

def generateAccumulateLabelByCol(dataMatrix, numberSigma = outOfSigmaSuccess):
    """
    Function: generateAccumulateLabelByCol
        Generate out-of-sigma-success label for each column.
    Input:
        dataMatrix: data to generate label.
        numberSigma: how much sigma we want to label it as success.
    Output:
        matrix as same shape with dataMatrix, elements are 0 or 1.
    """
    # expectation
    mean = dataMatrix.mean(0)
    sq = dataMatrix.copy()
    sq.data **= 2
    # expectation of square
    sqmean = sq.mean(0)
    # standard deviation
    std = np.sqrt(sqmean - np.square(mean))
    return dataMatrix > mean + std * numberSigma

def standardize(featureMatrix):
    """
    Function: standardize
        Standardize feature matrix. Transform each col to mu = 0 sigma = 1.
    Input:
        featureMatrix: numpy 2d array, each row is a data, each column is a feature.
    Output:
        standardized feature matrix.
    """
    standardizedMatrix = featureMatrix.copy()
    standardizedMatrix -= standardizedMatrix.mean(0)
    standardizedMatrix /= standardizedMatrix.std(0)
    print standardizedMatrix.mean(0)
    print standardizedMatrix.std(0)
    return standardizedMatrix

def singlePredictTime():
    return

if __name__ == '__main__':
    rawData = rawDataMatrix(inputFile)
    # cleaning the missing data
    rawData = rawData[:,:-6]
    transformed = compressMatrix(rawData)
    label = generateAccumulateLabelByCol(transformed[:,0:2])
    print standardize(transformed[:,0:2].toarray())
