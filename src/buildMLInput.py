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
#   top: top k success                                                     #
# Note: successThreshold <= predictTimeWindow - slidingWindowsize + 1      #
############################################################################
WEEK = 7 # don't change this, you know why :-)
EPSILON = 0.000001
inputFile = './1/Productivity/datamatrix_metric_1.npz'
predictTimeWindow = 10
featureTimeWindow = 10
slidingWindowSize = 4
outOfSigmaSuccess = 1
successThreshold = 5
garbageThreshold = featureTimeWindow * WEEK # a download a day, keep doctors away.
testPortion = 0.2
top = 100

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

def compressMatrix(rawData, windowSize=None, skipDay=WEEK):
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
    if windowSize is None: windowSize = slidingWindowSize * WEEK
    dataNum = rawData.shape[0]
    windowNum = (rawData.shape[1] - windowSize) // skipDay
    slidingWindowMatrix = np.zeros((dataNum, windowNum))
    for i in xrange(windowNum):
        slidingWindowMatrix[:,i] = rawData[:,(WEEK * i):(WEEK * i + windowSize)].sum(1).T
    return sps.csr_matrix(slidingWindowMatrix)

def swipeOutInactiveApp(downloadMatrix, predictMatrix = None, leastDownload=None):
    """
    Function: swipeOutInactiveApp
        Cleaning Apps with too small download amount in feature matrix.
    Input:
        downloadMatrix: cleaning is done from here, type must be scipy.sparse.*
        predictMatrix: also apply the cleaning on prediction, type scipy.sparse.*
        leastDownload: the least download within downloadMatrix to be consider.
    Output:
        cleaned downloadMatrix, predictMatrix and active index
    """
    if leastDownload is None: leastDownload = garbageThreshold
    totalDownload = np.array(downloadMatrix.sum(1)).ravel()
    candidate = np.where(totalDownload > leastDownload)[0]
    if predictMatrix is not None:
        return downloadMatrix[candidate,:].toarray(), predictMatrix[candidate,:].toarray(), candidate[:,None]
    else:
        return downloadMatrix[candidate,:].toarray(), None, candidate[:,None]

def generateAccumulateLabelByCol(dataMatrix, numberSigma=None):
    """
    Function: generateAccumulateLabelByCol
        Generate out-of-sigma-success label for each column.
    Input:
        dataMatrix: data to generate label, type numpy array.
        numberSigma: how many sigma we want to label it as success.
    Output:
        matrix as same shape with dataMatrix, elements are 0 or 1.
    """
    if numberSigma is None: numberSigma = outOfSigmaSuccess
    # expectation
    mean = dataMatrix.mean(0)
    # standard deviation
    std = dataMatrix.std(0)
    return dataMatrix > (mean + std * numberSigma)

def generateTopkLabelByCol(dataMatrix, k=None):
    """
    Function: generateTopkLabelByCol
        Generate top-k label for each column.
    Input:
        dataMatrix: data to generate label, type numpy array.
        k: top rank.
    Output:
        matrix as same shape with dataMatrix, elements are 0 or 1.
    """
    if k is None: k = top
    label = np.zeros(dataMatrix.shape)
    topkIndex = np.argpartition(dataMatrix, dataMatrix.shape[0] - k, axis=0)[-k:,:]
    for colIdx in xrange(topkIndex.shape[1]):
        label[topkIndex[:,colIdx], colIdx] = 1
    return label

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

def normalize(featureMatrix):
    """
    Function: normalize
        normalize feature matrix by col.
    Input:
        featureMatrix: each row is a data, each column is a feature.
    Output:
        normalized feature matrix.
    """
    return featureMatrix / featureMatrix.sum(0)

def sample(dataSet, seed, portionOfTestSet=None):
    """
    Function: sample
        split whole dataset into training set and test set.
    Input:
        dataSet: a list or tuple of data to be sampled.
        portionOfTestSet: portion to be sampled as test set.
        seed: random number seed.
    Output:
        training set and test set, as the same form with dataSet.
    """
    if portionOfTestSet is None: portionOfTestSet = testPortion
    dataNum = dataSet[0].shape[0]
    sample = range(dataNum)
    random.seed(seed)
    random.shuffle(sample)
    testNum = int(dataNum * portionOfTestSet)
    trainNum = dataNum - testNum
    train = [None] * len(dataSet)
    test = [None] * len(dataSet)
    for i in xrange(len(dataSet)):
        train[i] = dataSet[i][sample[0:trainNum]]
        test[i] = dataSet[i][sample[trainNum:dataNum]]
    return train, test

def singlePredictTime(totalDataMatrix, predictTimeStamp, windowSize=None,
        featureSize=None, predictSize=None, success=None, numberOfSigma=None):
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
        numberOfSigma: outOfSigmaSuccess
    Output:
        feature matrix, accumulate label, sliding window label, prediction window and corresponding index.
    Application note: In this function, swipeOutInactiveApp(...) is called with default parameters.
    """
    if windowSize is None: windowSize = slidingWindowSize
    if featureSize is None: featureSize = featureTimeWindow - slidingWindowSize + 1
    if predictSize is None: predictSize = predictTimeWindow - slidingWindowSize + 1
    if success is None: success = successThreshold
    if numberOfSigma is None: numberOfSigma = outOfSigmaSuccess
    featureEndTime = predictTimeStamp - windowSize + 1
    featureStartTime = featureEndTime - featureSize
    featureTotal = totalDataMatrix[:,featureStartTime:featureEndTime]
    predictTotal = totalDataMatrix[:,predictTimeStamp:predictTimeStamp + predictSize]
    featureMatrix, predictMatrix, remainingIndex = swipeOutInactiveApp(featureTotal, predictTotal)
    accumulateLabel = generateAccumulateLabelByCol(predictMatrix.sum(1), numberOfSigma)
    eachWindowLabel = generateAccumulateLabelByCol(predictMatrix, numberOfSigma)
    slidingWindowLabel = (eachWindowLabel.sum(1) >= success)
    predictTimeCol = np.ones(remainingIndex.shape[0], dtype='int')[:,None]
    return (standardize(featureMatrix),
            accumulateLabel[:,None],
            slidingWindowLabel[:,None],
            standardize(predictMatrix),
            np.hstack((remainingIndex, predictTimeCol)))

def singlePredictTimeNew(totalDataMatrix, predictTimeStamp, standardizeMethod=None,
        windowSize=None, featureSize=None, predictSize=None,
        failTime=2, successTime=2, topk=None):
    """
    Function: singlePredictTimeNew
        Give a predict time, generate features and labels.
    Input:
        totalDataMatrix: the complete data matrix.
        predictTimeStamp: time to start prediction.
        standardizeMethod: method for standardization.
        windowSize: size of sliding window.
        featureSize: feature dimension.
        predictSize: prediction dimension to generate label.
        failTime: Fail as least first several windows to be considered as incremental.
        successTime: Fail as least last several windows to be considered as incremental.
        topk: Top k to be considered as good.
    Output:
        feature matrix, label, prediction window, and corresponding index.
    Application note: In this function, swipeOutInactiveApp(...) is called with default parameters.
    """
    if windowSize is None: windowSize = slidingWindowSize
    if featureSize is None: featureSize = featureTimeWindow - slidingWindowSize + 1
    if predictSize is None: predictSize = predictTimeWindow - slidingWindowSize + 1
    if topk is None: topk = top
    featureEndTime = predictTimeStamp - windowSize + 1
    featureStartTime = featureEndTime - featureSize
    featureTotal = totalDataMatrix[:,featureStartTime:featureEndTime]
    predictTotal = totalDataMatrix[:,predictTimeStamp:predictTimeStamp + predictSize]
    featureMatrix, predictMatrix, remainingIndex = swipeOutInactiveApp(featureTotal, predictTotal)
    eachWindowLabel = generateTopkLabelByCol(predictMatrix, topk)
    consistentLabel = eachWindowLabel.sum(1) >= (predictSize - EPSILON)
    firstFailLabel = eachWindowLabel[:,:failTime].sum(1) < EPSILON
    lastSuccessLabel = eachWindowLabel[:,-successTime:].sum(1) > (successTime - EPSILON)
    incrementalLabel = np.logical_and(firstFailLabel, lastSuccessLabel)
    accumulateLabel = np.logical_or(incrementalLabel, consistentLabel)
    #accumulateLabel = incrementalLabel
    predictTimeCol = np.ones(remainingIndex.shape[0], dtype='int')[:,None]
    if standardizeMethod is not None:
        return (standardizeMethod(featureMatrix),
                accumulateLabel[:,None],
                standardizeMethod(predictMatrix),
                np.hstack((remainingIndex, predictTimeCol)))
    else:
        return (featureMatrix,
                accumulateLabel[:,None],
                predictMatrix,
                np.hstack((remainingIndex, predictTimeCol)))

def generateFeatureMatrixAndLabel(totalDataMatrix, singleMethod=singlePredictTimeNew,
        windowSize=None, featureWindow=None, predictWindow=None, success=None, normalizeFlag=False):
    """
    Function: generateFeatureMatrixAndLabel
        generate feature matrix and label from total matrix, both training set and testing set.
    Input:
        totalDataMatrix: the complete data matrix.
        singleMethod: single time method.
        windowSize: size of sliding window.
        featureWindow: size of feature window.
        predictWindow: size of prediction window.
        success: win at least these sliding windows to label as successful.
        normalizeFlag: whether normalize or not.
    Output:
        (train, test), each one contains (matrix, accumulate label, sliding window label, predict window data, corresponding index).
    Application note: singlePredictTime(...) is called by default parameter, where some
    default-parameter-function-call involve. Also, sample(...) is called
    by default parameter.
    """
    if windowSize is None: windowSize = slidingWindowSize
    if featureWindow is None: featureWindow = featureTimeWindow
    if predictWindow is None: predictWindow = predictTimeWindow
    if success is None: success = successThreshold
    featureSize = featureWindow - windowSize + 1
    predictSize = predictWindow - windowSize + 1
    train = []
    test = []
    for predictTime in xrange(featureWindow, totalDataMatrix.shape[1] - predictSize):
        if normalizeFlag:
            dataSet = singleMethod(totalDataMatrix, predictTime)
        else:
            dataSet = singleMethod(totalDataMatrix, predictTime, normalize)
        singleTrain, singleTest = sample(dataSet, predictTime)
        for i in xrange(len(singleTrain)):
            if i >= len(train):
                train.append(singleTrain[i].copy())
                test.append(singleTest[i].copy())
            else:
                train[i] = np.vstack((train[i],singleTrain[i]))
                test[i] = np.vstack((test[i],singleTest[i]))
    return train, test

def buildMatrix(filename=None,normalizeFlag=False):
    """
    Function: buildMatrix
        Wrapper for the whole procedure.
    Input:
        filename: input category file name.
        normalizeFlag: whether normalize or not.
    Output:
        (train, test), each one contains (matrix, accumulate label, sliding window label).
    Application note: This is a wrapper function therefore have many
    default-parameter-function-call.
    """
    if filename is None: filename = inputFile
    rawData = rawDataMatrix(inputFile)
    transformed = compressMatrix(rawData)
    return generateFeatureMatrixAndLabel(transformed, normalizeFlag=normalizeFlag)

def plotDownloadInflation(filename=None):
    """
    Function: plotDownloadInflation
        Wrapper for plot download inflation
    """
    if filename is None: filename = inputFile
    rawData = rawDataMatrix(filename)
    transformed = compressMatrix(rawData)
    ax = plt.subplot(1,2,1)
    plt.title('Raw download')
    plt.plot(rawData.sum(0).T)
    plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    ax = plt.subplot(1,2,2)
    plt.title('Sliding window smoothened')
    plt.plot(transformed.sum(0).T)
    plt.ticklabel_format(style = 'sci', axis = 'y', scilimits = (0,0))
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect((x1-x0)/(y1-y0))
    plt.show()
    return

def plotDownloadDistribution(filename=None):
    """
    Function: plotDownloadDistribution
        Wrapper for plot download distribution.
    """
    if filename is None: filename = inputFile
    rawData = rawDataMatrix(filename)
    dataPerApp = np.array(rawData.sum(1)).ravel()
    binMax = int(np.log10(dataPerApp.max()) + 0.5)
    plt.hist(dataPerApp, bins = 10**np.linspace(0,binMax,100))
    plt.gca().set_xscale('log')
    return

def plotPerWindowDistribution(compressedMatrix, windowIndex):
    dataPerApp = compressedMatrix[:,windowIndex].toarray().ravel()
    binMax = int(np.log10(dataPerApp.max()) + 0.5)
    plt.hist(dataPerApp, bins = 10**np.linspace(0,binMax,100))
    plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')

def randomPlot(filename=None):
    if filename is None: filename = inputFile
    rawData = rawDataMatrix(filename)
    rowNum = random.randint(1, rawData.shape[0]) - 1
    while rawData[rowNum].sum() < 100000:
        rowNum = random.randint(1, rawData.shape[0]) - 1
    plt.plot(rawData[rowNum].toarray().ravel())
    plt.show()

if __name__ == '__main__':
    trainNormalized, testNormalized = buildMatrix(normalizeFlag=True)
    train, test = buildMatrix(normalizeFlag=False)
    pass
