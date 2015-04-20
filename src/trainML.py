#!user/bin/python2

import os
import sys
import time
import buildMLInput as bml
import numpy as np
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *

############################################################################
#	Function:
# 		Implement different learing algorithms to predict good apps.
# 	Input:
# 		Raw matrix npz files, created by 'matrixBuild.py'
# 		Data locates at Prediction_App_Trend/src/1/*
# 			NOTE: Must set parameters in setInputParameters(bml)
# 				e.g. bml.inputFile = filename to the input npz file   
# 	Output:
# 		Print prediction accuracy to screen.
# 
############################################################################

# default parameter values:
def setInputParameters(bml):
	"""
	Function: 
		update parameters for ML train and test matrix building
	Input: 
		module 'buildMLInput', import from 'buildMLInput.py'
	Output: 
		bml module with new parameter values
	Parameter Definition:                                                        
	  inputFile: filename to the input npz file                              
	  predictTimeWindow: number of weeks used to generate label              
	  featureTimeWindow: number of weeks used to generate feature            
	  slidingWindowSize: number of weeks in a sliding window                 
	  outOfSigmaSuccess: how many sigmas out counts as success               
	  successThreshold: win from category have to win how many windows.      
	  garbageThreshold: total download threshold for candidate               
	  testPortion: the portion of data treating as test                      
	Note: successThreshold <= predictTimeWindow - slidingWindowsize + 1      

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
	Function: 
		print basic information about train and test matrices
	Input: 
		train matrix, test matrix, generated with parameters in setInputParameters(bml):
	Output: 
		print to screen
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

def useLogSGD(label, loss, penalty, regularization_strength, X1, Y1, X2, Y2):
	"""
	Function: use Stochastic gradient decent logistic regression to predict
	Input:
		label: just a name to recognize input data from output
		loss: str, 'log'
		penalty: str, 'l2'
		regularization_strength: float, 0.1
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
	"""
	model = SGDClassifier(loss = loss, penalty = penalty, alpha = regularization_strength)
	model.fit(X1,Y1)
	label = label + '.'+ str(loss) + '.' + str(penalty) + '.' + str(regularization_strength) 
	TPR, TNR, ACC = getAccuracy('LogisticSGD.Train.'+label, model.predict(X1), Y1)
	TPR, TNR, ACC = getAccuracy('LogisticSGD.Test.'+label, model.predict(X2), Y2)
	return 0

def useSVM(label, kernel, degree, penalty, X1, Y1, X2, Y2):
	"""
	Function: use SVM to predict
	Input:
		label: just a name to recognize input data from output
		kernel: 'poly'
		degree: 3
		penalty: 1
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
	"""
	model = SVC(kernel = kernel, degree = degree, C = penalty)
	model.fit(X1,Y1)
	label = label + '.'+ str(kernel) + '.' + str(degree) + '.' + str(penalty)
	TPR, TNR, ACC = getAccuracy('SVM.Train.'+label, model.predict(X1), Y1)
	TPR, TNR, ACC = getAccuracy('SVM.Test.'+label, model.predict(X2), Y2)
	return 0

def usekernelkNN(label, kernel, k, threshold, X1, Y1, X2, Y2):
	"""
	Function: use SVM to predict
	Input:
		label: just a name to recognize input data from output
		kernel: 'inv', to compute prediction from distances
		k: number of neighbors
		threshold: use to determine if the prediction is positive or not
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
	"""
	tree = KDTree(X1, leaf_size = 5)
	prediction = np.zeros(len(X2))
	for i in range(0, len(X2)):
		dist, idx = tree.query(X2[i], k = k)
		# print dist, idx 
		prediction[i] = kernelkNN(kernel, threshold, dist[0], idx[0], Y1)
	label = label + '.'+ str(kernel) + '.' + str(k) + '.' + str(threshold)
	getAccuracy('kNN.'+label, prediction, Y2)
	return 0

def kernelkNN(kernel, threshold, dist, idx, Y1):
	'''
	Function: implement differnt kernel method to compute prediction from distances
	Input:
		kernel: kernel Name, 'inv'
		threshold: use to determine if the prediction is positive or not
		dist: distance array, from 'usekernelkNN()'
		idx: index array, from 'usekernelkNN()'
		Y1: label array for known neighbors
	Output:
		prediction value
	'''
	if kernel == 'inv': # inverse
		numerator = 0
		denominator = 0
		for i in range(0,len(dist)):
			# print dist[i], Y1[idx[i]]
			dist[i] = dist[i] + 0.0001 # avoid dist == 0
			numerator = numerator + Y1[idx[i]]/dist[i]
			denominator = denominator + 1/dist[i]
		prediction = numerator/denominator
		if prediction > threshold:
			return 1.0
		else:
			return 0.0
	else:
		print 'Cannot find kNN kernel:', kernel
		return 0

def getAccuracy(modelName, prediction, target):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	num = float(len(target))
	for i in range(0,len(target)):
		if prediction[i] == 1.0:
			if prediction[i] == target[i]:
				TP = TP + 1
			else:
				FP = FP + 1
		if prediction[i] == 0.0:
			if prediction[i] == target[i]:
				TN = TN + 1
			else:
				FN = FN + 1
	TPR = TP/float(TP+FN)*100
	TNR = TN/float(TN+FP)*100
	PPV = TP/float(TP+FP)*100
	ACC = (TP+TN)/num*100
	print '\n<'+modelName+'>'
	print 'True pos:', TP, ' False neg:', FN, '\t Sensitivity(TPR):', TPR,'%'
	print 'True neg:', TN, ' False pos:', FP, '\t Precision(PPV):  ', PPV,'%'
	print 'Overall accuracy:', ACC,'%'
	# for i in range(0,10):
	# 	print prediction[i], target[i], confidence[i]
	# return 0
	return TPR, TNR, ACC


if __name__ == '__main__':
	timeStart = time.time()
	bml = setInputParameters(bml)
	train, test = bml.buildMatrix()
	# printMatrixInfo(train, test)

	trainFeature = train[0]	# feature matrix
	trainTargetAcc = train[1][:,0]	# accumulated label
	trainTargetSld = train[2][:,0]	# sliding window label
	testFeature = test[0]	# feature matrix
	testTargetAcc = test[1][:,0]	# accumulated label
	testTargetSld = test[2][:,0]	# sliding window label

	useLogSGD('Acc', 'log', 'l2', 0.1, trainFeature, trainTargetAcc, testFeature, testTargetAcc)
	useLogSGD('Sld', 'log', 'l2', 0.1, trainFeature, trainTargetSld, testFeature, testTargetSld)
	# useSVM('Acc','poly', 3, 1, trainFeature, trainTargetAcc, testFeature, testTargetAcc)
	# useSVM('Sld','poly', 3, 1, trainFeature, trainTargetSld, testFeature, testTargetSld)
	usekernelkNN('Acc', 'inv', 25, 0.55, trainFeature, trainTargetAcc, testFeature, testTargetAcc)
	usekernelkNN('Sld', 'inv', 25, 0.55, trainFeature, trainTargetSld, testFeature, testTargetSld)
	# confidence = modelLog.decision_function(testFeature)
	
	runTime = time.time() - timeStart
	print '\nRuntime: ', runTime