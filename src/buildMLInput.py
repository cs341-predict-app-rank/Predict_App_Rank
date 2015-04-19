#!/usr/bin/python2
import sparseIO
import scipy.sparse as sps
import numpy as np
import matplotlib.pyplot as plt
import random

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
outOfSigmaSuccess = 1
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
        a csr matrix contains raw data with missing date cleaned.
    """
    # cleaning the missing date
    return sparseIO.csrLoad(inputFile)[:,:-6]

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
    for i in xrange(windowNum):
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
        dataMatrix: data to generate label, type numpy array.
        numberSigma: how many sigma we want to label it as success.
    Output:
        matrix as same shape with dataMatrix, elements are 0 or 1.
    """
    # expectation
    mean = dataMatrix.mean(0)
    # standard deviation
    std = dataMatrix.std(0)
    return dataMatrix > (mean + std * numberSigma)

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
    return standardizedMatrix

def sample(dataSet, testPortion = testPortion):
    """
    Function: sample
        split whole dataset into training set and test set.
    Input:
        dataSet: a list or tuple of data to be sampled.
        testPortion: portion to be sampled as test set.
    Output:
        training set and test set, as the same form with dataSet.
    """
    dataNum = dataSet[0].shape[0]
    sample = range(dataNum)
    random.shuffle(sample)
    testNum = int(dataNum * testPortion)
    trainNum = dataNum - testNum
    train = [None] * len(dataSet)
    test = [None] * len(dataSet)
    for i in xrange(len(dataSet)):
        train[i] = dataSet[i][0:trainNum]
        test[i] = dataSet[i][trainNum:dataNum]
    return train, test

def singlePredictTime(totalDataMatrix, predictTimeStamp,
        windowSize = slidingWindowSize,
        featureSize = featureTimeWindow - slidingWindowSize + 1,
        predictSize = predictTimeWindow - slidingWindowSize + 1,
        success = successThreshold):
    """
    Function: singlePredictTime
        Give a predict time, generate standardized features and labels.
    Input:
        totalDataMatrix: the complete data matrix.
        predictTimeStamp: time to start prediction.
        windowSize: size of sliding window.
        featureSize: feature dimension.
        predictSize: prediction dimension to generate label.
        success: win at least these sliding windows to label as successful.
    Output:
        feature matrix, accumulate label and sliding window label.
    Application note: In this function, swipeOutInactiveApp(...) and
    generateAccumulateLabelByCol(...) are called with default parameters.
    """
    featureEndTime = predictTimeStamp - windowSize + 1
    featureStartTime = featureEndTime - featureSize
    featureTotal = totalDataMatrix[:,featureStartTime:featureEndTime]
    predictTotal = totalDataMatrix[:,predictTimeStamp:predictTimeStamp + predictSize]
    featureMatrix, predictMatrix = swipeOutInactiveApp(featureTotal, predictTotal)
    accumulateLabel = generateAccumulateLabelByCol(predictMatrix.sum(1))
    eachWindowLabel = generateAccumulateLabelByCol(predictMatrix)
    slidingWindowLabel = (eachWindowLabel.sum(1) >= success)
    return standardize(featureMatrix), accumulateLabel[:,None], slidingWindowLabel[:,None]

def generateFeatureMatrixAndLabel(totalDataMatrix,
        windowSize = slidingWindowSize,
        featureWindow = featureTimeWindow,
        predictWindow = predictTimeWindow,
        success = successThreshold):
    """
    Function: generateFeatureMatrixAndLabel
        generate feature matrix and label from total matrix, both training set and testing set.
    Input:
        totalDataMatrix: the complete data matrix.
        windowSize: size of sliding window.
        featureWindow: size of feature window.
        predictWindow: size of prediction window.
        success: win at least these sliding windows to label as successful.
    Output:
        (train, test), each one contains (matrix, accumulate label, sliding window label).
    Application note: singlePredictTime(...) is called, where some
    default-parameter-function-call involve. Also, sample(...) is called
    by default parameter.
    """
    featureSize = featureWindow - windowSize + 1
    predictSize = predictWindow - windowSize + 1
    train = [np.zeros((0, featureSize)), np.zeros((0,1)), np.zeros((0,1))]
    test = [np.zeros((0, featureSize)), np.zeros((0,1)), np.zeros((0,1))]
    for predictTime in xrange(featureWindow, totalDataMatrix.shape[1] - predictSize):
        dataSet = singlePredictTime(totalDataMatrix, predictTime,
                windowSize, featureSize, predictSize, success)
        singleTrain, singleTest = sample(dataSet)
        for i in range(len(train)):
            train[i] = np.vstack((train[i],singleTrain[i]))
            test[i] = np.vstack((test[i],singleTest[i]))
    return train, test

def buildMatrix(filename = inputFile):
    """
    Function: buildMatrix
        Wrapper for the whole procedure.
    Input:
        filename: input category file name.
    Output:
        (train, test), each one contains (matrix, accumulate label, sliding window label).
    Application note: This is a wrapper function therefore have many
    default-parameter-function-call.
    """
    rawData = rawDataMatrix(inputFile)
    transformed = compressMatrix(rawData)
    return generateFeatureMatrixAndLabel(transformed)

def plotDownloadInflation(filename = inputFile):
    """
    Function: plotDownloadInflation
        Wrapper for plot download inflation
    """
    rawData = rawDataMatrix(inputFile)
    transformed = compressMatrix(rawData)
    plt.subplot(1,2,1)
    plt.plot(rawData.sum(0).T)
    plt.subplot(1,2,2)
    plt.plot(transformed.sum(0).T)
    plt.show()

if __name__ == '__main__':
    train, test = buildMatrix()
