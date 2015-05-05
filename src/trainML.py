#!user/bin/python2

import os
import sys
import time
import buildMLInput as bml
import numpy as np
import random
import matplotlib.pyplot as plt
from plotApp import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.externals import joblib

############################################################################
#	Function:															   #
# 		Implement different learing algorithms to predict good apps.       #
# 	Input:																   #
# 		Raw matrix npz files, created by 'matrixBuild.py'				   #
# 		Data locates at Prediction_App_Trend/src/1/*					   #
# 			NOTE: Must set parameters in setInputParameters(bml)		   #
# 				e.g. bml.inputFile = filename to the input npz file   	   #
# 	Output:																   #
# 		Print prediction accuracy to screen.                               #
# 																		   #
############################################################################

# default parameter values:
ModelFileDir = 'models/' # save existing model in this dir
CategoryName = 'Social Networking'
FeatureMatrixMultiplier = 10000
LabelPercent = 0.6

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
	bml.EPSILON = 0.000001
	bml.inputFile = './1/'+ CategoryName +'/datamatrix_metric_1.npz'
	bml.predictTimeWindow = 12
	bml.featureTimeWindow = 12
	bml.slidingWindowSize = 4
	bml.outOfSigmaSuccess = 1
	bml.successThreshold = 5
	bml.garbageThreshold = bml.featureTimeWindow * bml.WEEK # a download a day, keep doctors away.
	bml.testPortion = 0.2
	bml.top = 60
	bml.percent = LabelPercent
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
	print 'Train Label: ',len(train[1][:,0]),',',len(train[1][0,:])
	print 'Train Baseline Label: ',len(train[4][:,0]),',',len(train[4][0,:])
	same1 = 0.0
	same0 = 0.0
	for idx in range(0,len(train[1][:,0])):
		if train[1][idx][0] == 1 and train[1][idx][0] == train[4][idx][0]:
			same1 = same1 + 1
		if train[1][idx][0] == 0 and train[1][idx][0] == train[4][idx][0]:
			same0 = same0 + 1
	print 'Positive Label          #:', sum(train[1][:,0])
	print 'Positive Baseline Label #:', sum(train[4][:,0])
	print 'Both Postive Label      #:', same1, '(',same1/float(sum(train[1][:,0])), ')'
	print '\n==== Test Matrix Info ===='
	print 'Test Feature Matrix: ',len(test[0][:,0]),',',len(test[0][0,:])
	print 'Test Label: ',len(test[1][:,0]),',',len(test[1][0,:])
	print 'Test Baseline Label: ',len(test[4][:,0]),',',len(test[4][0,:])
	same1 = 0.0
	same0 = 0.0
	for idx in range(0,len(test[1][:,0])):
		if test[1][idx][0] == 1 and test[1][idx][0] == test[4][idx][0]:
			same1 = same1 + 1
		if test[1][idx][0] == 0 and test[1][idx][0] == test[4][idx][0]:
			same0 = same0 + 1
	print 'Positive Label          #:', sum(test[1][:,0])
	print 'Positive Baseline Label #:', sum(test[4][:,0])
	print 'Both Postive Label      #:', same1, '(',same1/float(sum(test[1][:,0])), ')'

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
		return prediction for test
	"""
	model = SGDClassifier(loss = loss, penalty = penalty, alpha = regularization_strength)
	model.fit(X1,Y1)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	label = label + '.'+ str(loss) + '.' + str(penalty) + '.' + str(regularization_strength) 
	getAccuracy('LogisticSGD.Train.'+label, prediction1, Y1)
	getAccuracy('LogisticSGD.Test.'+label, prediction2, Y2)
	return prediction2

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
		return prediction for test
	"""
	label = label + '.'+ str(kernel) + '.' + str(degree) + '.' + str(penalty)
	modelFileName = ModelFileDir + 'svm.'+label+'.pkl'
	try:
		model = joblib.load(modelFileName)
	except:
		model = SVC(kernel = kernel, degree = degree, C = penalty)
		print '\nStart training SVM ...'
		model.fit(X1,Y1)
		joblib.dump(model, modelFileName)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	getAccuracy('SVM.Train.'+label, prediction1, Y1)
	getAccuracy('SVM.Test.'+label, prediction2, Y2)
	return prediction2

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
		return prediction for test
	"""
	tree = KDTree(X1, leaf_size = 5)
	prediction = np.zeros(len(X2))
	for i in range(0, len(X2)):
		dist, idx = tree.query(X2[i], k = k)
		# print dist, idx 
		prediction[i] = kernelkNN(kernel, threshold, dist[0], idx[0], Y1)
	label = label + '.'+ str(kernel) + '.' + str(k) + '.' + str(threshold)
	getAccuracy('kNN.'+label, prediction, Y2)
	return prediction

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
	'''
	Function: compute and print confusion matrix
	Input:
		modelName: str, to distinguish print contents
		prediction: array, prediction label array
		target: array, target label array
	Output:
		print confusion matrix
		return: true positive #, fale negative #, false positive #, true negative #
	'''
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
	ACC = (TP/float(TP+FP+0.01) + TN/float(TN+FN+0.01))/2
	print '\n<'+modelName+'>'
	print 'True pos:', TP, ' False neg:', FN,'\t Precision:', (TP+0.01)/float(TP+FN+0.01)*100,'%'
	print 'False pos:', FP,' True neg:', TN,'\t Precision:', (TN+0.01)/float(FP+TN+0.01)*100,'%'
	print 'Precision:', TP/float(TP+FP+0.01), '\t', TN/float(FN+TN+0.01)
	print 'TP/P:', TP/float(TP+FP+0.01), 'TN/N:', TN/float(TN+FN+0.01)
	print 'Overall accuracy:', ACC,'%'
	# for i in range(0,10):
	# 	print prediction[i], target[i], confidence[i]
	# return 0
	return TP, FN, FP, TN

def plotResultOnFeature(name, feature, real):
	'''
	Function: plot feature 
	Input:
		modelName: str, to distinguish the plot
		feature: array, 
		array: array,
	Output:
		plot
	'''
	print [feature, real]
	timeline1 = range(1, len(feature)+1)
	timeline2 = range(len(feature)+1, len(feature)+len(real)+1)
	tmp = range(len(feature),len(feature)+2)
	between = [feature[len(feature)-1], real[0]]
	print between
	line1 = plt.plot(timeline1, feature, '-b', label= 'Known past',linewidth=2)
	line2 = plt.plot(timeline2, real, '-r', label='Real future',linewidth=2)
	line2 = plt.plot(tmp, between, '-.b', linewidth=2)
	plt.title('Download performance of an App in one category')
	legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
	plt.xlabel('Week number')
	plt.ylabel('Feature')
	plt.grid(color='k', linestyle='-.', linewidth=1)
	# plt.axis([1,14, -3, 20])
	ax = plt.gca()
	ax.set_autoscale_on(False)
	# plt.show(block=True)
	plt.savefig( name + '.pdf')
	plt.clf()
	return 0

def plotResultOnDownload(name, 
						idx, 
						market = None, 
	                    category = None, 
	                    metric = 1, 
	                    db_user = None, 
	                    db_pswd = None, 
	                    matrix_path = './', 
	                    output_path = 'plots/'):
	'''
	Function: plot download
	Input:
		name: str, to distinguish the plot
		idx: A list. Each element of the list is a tuple corresponding to a data point, that is an app at a specific date. First element of each
        tuple is the apps row index, and the second element is the date number. 
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
		plot
	'''
	name = plotAppWithRow(idx, market, category, metric, db_user, db_pswd, matrix_path, output_path)
	# print name
	return name

def compareWithBaseline(modelName, prediction, baseline):
	getAccuracy(modelName, prediction, baseline)
	return 0

def addFiniteDiff(feature):
	rowLen = feature.shape[0]
	colLen = feature.shape[1]
	newfeature = np.zeros((rowLen, colLen * 2 - 1))
	for i in range(0, rowLen - 1):
		newfeature[i, 0:colLen - 1] = feature[i, 0:colLen - 1]
		for j in range(colLen, colLen * 2 - 2):
			newfeature[i, j] = feature[i, j - colLen + 1] - feature[i, j - colLen]
	return newfeature

if __name__ == '__main__':
	timeStart = time.time()
	
	# set parameters
	bml = setInputParameters(bml)
	
	# build matrix
	train, test = bml.buildMatrix()
	
	
	# denote input for ML algorithms
	trainFeature = addFiniteDiff(train[0])*FeatureMatrixMultiplier			# feature matrix
	# trainFeature = train[0]
	trainTarget = (train[1][:, 0]).astype(int)	# label
	trainReal = train[2]			# real feature metrix in prediction window
	trainIdx = train[3]				# index for plotAppWithRow(), need to run .tolist() before input to plotAppWithRow()
	trainBaselineTarget = (train[4][:, 0]).astype(int)	# label for baseline model
	
	testFeature = addFiniteDiff(test[0])*FeatureMatrixMultiplier			# feature matrix
	# testFeature = test[0]
	testTarget = (test[1][:, 0]).astype(int)		# label
	testReal = addFiniteDiff(test[2])*FeatureMatrixMultiplier				# real feature metrix in prediction window
	testIdx = test[3]				# index for plotAppWithRow(), need to run .tolist() before input to plotAppWithRow()
	testBaselineTarget = (test[4][:, 0]).astype(int)		# label for baseline model	
	
	# printMatrixInfo(train, test)

	# run prediction models
	getAccuracy('baseline',testBaselineTarget,testTarget)
	# prediction = useLogSGD('', 'log', 'l2', 0.1, trainFeature, trainTarget, testFeature, testTarget)
	# prediction = useSVM('Fin10000','poly', 4, 0.1, trainFeature, trainTarget, testFeature, testTarget)
	prediction = usekernelkNN('1000', 'inv', 25, 0.55, trainFeature, trainTarget, testFeature, testTarget)
	num = 0
	sampleIdx = range(0, len(prediction))
	random.shuffle(sampleIdx)
	# for i in sampleIdx:
	# 	if prediction[i] == 0 and testTarget[i] == 0:
	# 		plotResultOnFeature(str(i),testFeature[i],testReal[i])
	# 		num =  num +1
	# 		# try:
	# 		# 	names = plotResultOnDownload(str(i),[testIdx[i].tolist()], 1, CategoryName, 1, 'safe3', 'cs341')
	# 		# 	appName = names[testIdx[i][0]]
	# 		# 	print appName
	# 		# 	num =  num +1 
	# 		# except: pass
	# 	if num > 10:
	# 		break
	runTime = time.time() - timeStart
	print '\nRuntime: ', runTime






