import buildMLInput as bml
import os
from os.path import isdir
import numpy as np


def allCategory(normailzeFlag=False):
    listFolder = ["1/" + folder for folder in os.listdir('1') if isdir("1/" + folder)]
    for i, folder in enumerate(listFolder):
        if folder  == '1/Games': continue
        print folder
        train, test = bml.buildMatrix(filename=folder+"/datamatrix_metric_1.npz", normalizeFlag=normailzeFlag)
        if i == 0:
            totalTrain = train[:]
            totalTest = test[:]
            totalTrain.append([train[0].shape[0]])
            totalTest.append([test[0].shape[0]])
        else:
            for i in xrange(len(train)):
                totalTrain[i] = np.vstack((totalTrain[i], train[i]))
            totalTrain[-1].append(train[0].shape[0])
            for i in xrange(len(test)):
                totalTest[i] = np.vstack((totalTest[i], test[i]))
            totalTest[-1].append(test[0].shape[0])
    return totalTrain, totalTest

if __name__ == '__main__':
    # trainNormalized, testNormalized = bml.buildMatrix(normalizeFlag=True)
    # train, test = bml.buildMatrix(normalizeFlag=False)
    # raw = bml.rawDataMatrix(bml.inputFile)
    # compressed = bml.compressMatrix(raw)
    train, test = allCategory()
