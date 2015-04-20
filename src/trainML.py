#!user/bin/python2

import os
import sys
import time
import buildMLInput as bml
from sklearn.linear_model import *

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

# default parameter values:
def setBuildMLInputParas(bml):
	"""
	Function: update parameters for ML train and test matrix building
	Input: module 'buildMLInput', import from 'buildMLInput.py'
	Output: bml module with new parameter values

	"""
	bml.WEEK = 7 # don't change this, you know why :-)
	bml.inputFile = './1/Productivity/datamatrix_metric_1.npz'
	bml.predictTimeWindow = 10
	bml.featureTimeWindow = 10
	bml.slidingWindowSize = 4
	bml.outOfSigmaSuccess = 1
	bml.successThreshold = 5
	bml.garbageThreshold = bml.featureTimeWindow * bml.WEEK # a download a day, keep doctors away.
	bml.testPortion = 0.2
	return bml

def printMatrixInfo(train, test):
	"""
	Function: print basic information about train and test matrices
	Input: train matrix, test matrix, generated from 'buildMLInput.py'
	Output: print to screen
	"""
	print '\n==== Train Matrix Info ===='
	print 'Train Feature Matrix:   ',len(train[0][:,0]),',',len(train[0][0,:])
	print 'Train Accumulate Label: ',len(train[1][:,0]),',',len(train[1][0,:])
	print 'Train slidingWin Label: ',len(train[2][:,0]),',',len(train[2][0,:])
	same1 = 0.0
	same0 = 0.0
	for idx in range(0,len(train[1][:,0])):
		if train[1][idx][0] == 1 and train[1][idx][0] == train[2][idx][0]:
			same1 = same1 + 1
		if train[1][idx][0] == 0 and train[1][idx][0] == train[2][idx][0]:
			same0 = same0 + 1
	print 'Positive Accumulate Label #:', sum(train[1][:,0])
	print 'Positive SlidingWin Label #:', sum(train[2][:,0])
	print 'Both Postive Label        #:', same1, '(',same1/float(sum(train[1][:,0])), '%)'
	print '\n==== Test Matrix Info ===='
	print 'Test Feature Matrix: ',len(test[0][:,0]),',',len(test[0][0,:])
	print 'Test Accumulate Label: ',len(test[1][:,0]),',',len(test[1][0,:])
	print 'Test slidingWin Label: ',len(test[2][:,0]),',',len(test[2][0,:])
	same1 = 0.0
	same0 = 0.0
	for idx in range(0,len(test[1][:,0])):
		if test[1][idx][0] == 1 and test[1][idx][0] == test[2][idx][0]:
			same1 = same1 + 1
		if test[1][idx][0] == 0 and test[1][idx][0] == test[2][idx][0]:
			same0 = same0 + 1
	print 'Positive Accumulate Label #:', sum(test[1][:,0])
	print 'Positive SlidingWin Label #:', sum(test[2][:,0])
	print 'Both Postive Label        #:', same1, '(',same1/float(sum(test[1][:,0])), '%)'

def trainLinearSGD(X,Y,model,penalty,regularization_strength):
	"""
	"""
	clf = SGDClassifier(loss = model, penalty = penalty, alpha = regularization_strength)
	clf.fit(X,Y)
	return clf

def printPerformance(modelName, prediction, target, confidence):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	num = float(len(target))
	for i in range(0,len(target)):
		if prediction[i] == 1.0:
			if prediction[i] == target[i][0]:
				TP = TP + 1
			else:
				FP = FP + 1
		if prediction[i] == 0.0:
			if prediction[i] == target[i][0]:
				TN = TN + 1
			else:
				FN = FN + 1
	print '====', modelName, '===='
	print 'Overall accuracy:', (TP+TN)/num*100,'%'
	print 'True positive:   ', TP, '\t\t', TP/float(TP+FN)*100,'%'
	print 'True negative:   ', TN, '\t', TN/float(TN+FP)*100,'%'
	print 'False positive:  ', FP
	print 'False negative:  ', FN
	# for i in range(0,10):
	# 	print prediction[i], target[i][0], confidence[i]
	# return 0


if __name__ == '__main__':
	timeStart = time.time()
	bml = setBuildMLInputParas(bml)
	train, test = bml.buildMatrix()
	printMatrixInfo(train, test)

	trainFeature = train[0]
	trainTargetAcc = train[1]
	trainTargetSld = train[2]

	modelLog = trainLinearSGD(trainFeature, trainTargetAcc, 'log', 'l2', 0.1)
	
	testFeature = test[0]
	testTargetAcc = test[1]
	testTargetSld = test[2]

	prediction = modelLog.predict(testFeature)
	confidence = modelLog.decision_function(testFeature)
	
	printPerformance('SGD Logistic Train',modelLog.predict(trainFeature), trainTargetAcc, confidence)
	printPerformance('SGD Logistic Test',prediction, testTargetAcc, confidence)

	
	runTime = time.time() - timeStart
	print '\nRuntime: ', runTime