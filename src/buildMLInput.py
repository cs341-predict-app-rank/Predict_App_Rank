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
# Note: successThreshold <= predictTimeWindow - slidingWindowsize + 1      #
############################################################################
inputFile = './Productivity/datamatrix_metric_1.npz'
predictTimeWindow = 10
featureTimeWindow = 10
slidingWindowSize = 4
outOfSigmaSuccess = 2
successThreshold = 5

WEEK = 7

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
        slidingWindowMatrix[:,i] =\
                rawData[:,(WEEK * i):(WEEK * i + windowSize)].sum(1).T
    return sps.csr_matrix(slidingWindowMatrix)

def generateAccumulateLabelByCol(dataMatrix, numberSigma = outOfSigmaSuccess):
    """
    Function: generateAccumulateLabelByCol
        Generate out-of-sigma-success label for each column
    Input:

    """
    return

if __name__ == '__main__':
    rawData = rawDataMatrix(inputFile)
    # cleaning the missing data
    rawData = rawData[:,:-6]
    transformed = compressMatrix(rawData)
