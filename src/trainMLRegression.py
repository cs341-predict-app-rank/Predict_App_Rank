#!user/bin/python2

import os
import sys
import time
import math
import buildMLInput as bml
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from plotApp import *
from pylab import * 
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
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
# set labelling parameters
ModelFileDir = 'models/' # save existing model in this dir
CategoryName = 'Utilities'
# CategoryName = 'Social Networking'
LabelPercent = 0.6
predictTimeWindow = 12
featureTimeWindow = 12
lastWindowLength = 8
windowStepLength = 3
NegLabelConst = -1
FeatureMatrixMultiplier = 1
bucketNum = 300
ReviewFeature = True
RatingFeature = True
print CategoryName

def setParameters(bml, CategoryName, LabelPercent, predictTimeWindow, featureTimeWindow, windowStepLength):
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
	bml.categoryDir = './1/'+ CategoryName +'/'
	bml.inputFile = bml.categoryDir + 'datamatrix_metric_1.npz'
	bml.ratingFile = bml.categoryDir + 'datamatrix_ratings.npz'
	bml.reviewFile = bml.categoryDir + 'datamatrix_reviews.npz'
	bml.predictTimeWindow = predictTimeWindow
	bml.featureTimeWindow = featureTimeWindow
	bml.windowStepLength = windowStepLength
	bml.slidingWindowSize = 4
	bml.outOfSigmaSuccess = 1
	bml.successThreshold = 5
	bml.garbageThreshold = bml.featureTimeWindow * bml.WEEK # a download a day, keep doctors away.
	bml.testPortion = 0.2
	bml.top = 60
	bml.percent = LabelPercent
	bml.bucketNum = bucketNum
	bml.lastWindow = lastWindowLength
	print 'bucketNum:', bml.bucketNum
	print 'lastWindowLength:', bml.lastWindow
	print 'predictTimeWindow:', bml.predictTimeWindow
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
	print '\n==== Test Matrix Info ===='
	print 'Test Feature Matrix: ',len(test[0][:,0]),',',len(test[0][0,:])
	print 'Test Label: ',len(test[1][:,0]),',',len(test[1][0,:])

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

def useRandomForestRegression(label, X1, Y1, X2, Y2, n_estimators=25, 
			max_depth = None, min_samples_split=1, verbose=False):
	"""
	Function: use ForestRegression to predict
	Input:
		label: just a name to recognize input data from output
		n_estimators: int, number of trees 
		X1, Y1: train
		X2, Y2: test
		verbose: print train test accuracy
	Output:
		print accuracy to screen
		return prediction for test
	"""
	label = label + '.ForestRegression.' + str(n_estimators)
	model = RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth, 
		min_samples_split = min_samples_split, n_jobs=32)
	model.fit(X1,Y1)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	if verbose:
		getMSRE('ForestRegression.Train.'+label, prediction1, Y1)
		getMSRE('ForestRegression.Test.'+label, prediction2, Y2)
	return [prediction1, prediction2]

def useAdaBoost(label, X1, Y1, X2, Y2, n_estimators=100):
	"""
	Function: use AdaBoost to predict
	Input:
		label: just a name to recognize input data from output
		n_estimators: int, The maximum number of estimators at which boosting is terminated
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
		return prediction for test
	"""
	label = label + '.AdaBoost.' + str(n_estimators)
	model = AdaBoostClassifier(n_estimators=n_estimators)
	print '\nStart training AdaBoost ...'
	model.fit(X1,Y1)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	getAccuracy('AdaBoost.Train.'+label, prediction1, Y1)
	getAccuracy('AdaBoost.Test.'+label, prediction2, Y2)
	return prediction2

def useGBR(label, X1, Y1, X2, Y2, n_estimators=100, 
			max_depth = 3, min_samples_split=1, verbose=False):
	"""
	Function: use GradientBoostingRegressor to predict
	Input:
		label: just a name to recognize input data from output
		n_estimators: int, The maximum number of estimators at which boosting is terminated
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
		return prediction for test
	"""
	label = label + '.GradientBoostingRegressor.' + str(n_estimators)
	model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth = max_depth, 
		min_samples_split = min_samples_split)
	model.fit(X1,Y1)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	if verbose:
		getMSRE('GradientBoostingRegressor.Train.'+label, prediction1, Y1)
		getMSRE('GradientBoostingRegressor.Test.'+label, prediction2, Y2)
	return prediction2

def getAccuracy(modelName, prediction, target, verbose=True):
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
	Precision = TP/float(TP+FP+0.01)
	Recall = TP/float(TP+FN+0.01)
	F1 = 2*(Precision*Recall)/(Precision+Recall+0.0001)
	if verbose:
		print '\n<'+modelName+'>'
		print 'True pos:', TP, ' False neg:', FN #,'\t Precision:', (TP+0.01)/float(TP+FN+0.01)*100,'%'
		print 'False pos:', FP,' True neg:', TN #,'\t Precision:', (TN+0.01)/float(FP+TN+0.01)*100,'%'
		print 'F1:', F1
	return F1, Precision, Recall

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

def addLinearFitFeatures(feature, orgFeatureNum = 9):
	'''
	Function: add linear fitting charactors and corresponding prediction points as new features
	Input:
		feature: orginal feature matrix
		orgFeatureNum: original download feature numbers  
	Output:
		newfeature: new feature matrix
	'''
	rowLen = feature.shape[0]
	colLen = feature.shape[1]
	newfeature = np.zeros((rowLen, colLen + 8))
	for i in range(0, rowLen):
		newfeature[i, 0:colLen] = feature[i, 0:colLen]
		
		# calculate linear fit on all features
		x = range(0, orgFeatureNum)
		y = feature[i, 0:orgFeatureNum].tolist()
		fit = polyfit(x, y, 1)
		newfeature[i, colLen] = fit[0]
		newfeature[i, colLen + 1] = fit[1]

		# calculate linear fit on last 1/3 features
		x = range(int(orgFeatureNum*2/3), orgFeatureNum)
		y = feature[i, int(orgFeatureNum*2/3):orgFeatureNum].tolist()
		fit = polyfit(x, y, 1)
		fit_fn = poly1d(fit)
		newfeature[i, colLen + 2] = fit[0]
		newfeature[i, colLen + 3] = fit[1]
		x = [orgFeatureNum, orgFeatureNum+1, orgFeatureNum+2, orgFeatureNum+3]
		y = fit_fn(x)
		newfeature[i, colLen + 4:colLen + 8] = y
	return newfeature

def balanceData(train, posRate=0.5, noiseRate=0.01, growRate=1):	# data, postive date rate
	'''
	Function: balance training samples
	Input:
		train: original samples
		posRate: the rate of positive sample among all samples
		noiseRate: the amplitude of noise added to new smaples
		growRate: the rate of new total sample number over original sample number
	Output:
		newTrain: new train data with balanced samples
	'''
	orgNum = len(train[0])
	posNum = sum(train[1][:, 0])
	negNum = orgNum - posNum
	
	totalNum = int(posNum / posRate * growRate)
	newPosNum = int(totalNum * posRate)
	newNegNum = totalNum - newPosNum

	# initialize new train list
	newTrain = list()	
	for matrixIdx in range(len(train)):
		newTrain.append(np.zeros((totalNum, len(train[matrixIdx][0, :]))))	# init new matrix
	# create random smapling index
	sampleIdx = range(orgNum)
	rd.shuffle(sampleIdx)
	newIdx = 0
	posIdx = 0
	negIdx = 0
	idx = 0
	while newIdx < totalNum:	
		if idx >= orgNum:
			idx = 0
		if (train[1][sampleIdx[idx]]).astype(int) == 1 and posIdx < newPosNum:	# if it's a positive sample
			for column in range(len(train[0][0,:])):	# copy feature and add noise to the new sample 
				newTrain[0][newIdx, column] = train[0][sampleIdx[idx], column] * (1+rd.uniform(-1,1)*noiseRate)
			for matrixIdx in range(1,len(train)):		# copy other data as the same to the new sample
				newTrain[matrixIdx][newIdx, :] = train[matrixIdx][sampleIdx[idx], :]
			# print 'p', posIdx, newIdx, newTrain[0][newIdx]
			newIdx = newIdx + 1
			posIdx = posIdx + 1
		if (train[1][sampleIdx[idx]]).astype(int) == 0 and negIdx < newNegNum:	# if it's a negtive sample
			for matrixIdx in range(0,len(train)):		# copy all data as the same to the new sample
				newTrain[matrixIdx][newIdx, :] = train[matrixIdx][sampleIdx[idx], :]
			# print 'n', negIdx, newIdx, newTrain[0][newIdx]
			newIdx = newIdx + 1
			negIdx = negIdx + 1
		idx = idx + 1
	# print orgNum, posNum
	# print len(newTrain[0]), sum(newTrain[1][:, 0])
	return newTrain


def plotMultipleResults(prediction, 
						testTarget, 
						testBaselineTarget,
						testIdx,
						CategoryName,
						limit):
	'''
	Function: plot multiple downloads figures to files
	Input:
		prediction: array, prediction results, 0 or 1
		testTarget: array, targets, 0 or 1
		testBaselineTarget: array, prediction made by baseline method, 0 or 1
		testIdx: list, contains all index relationship between those arrays and original data matrix
		CategoryName: str, current category name
	Output:
		pdf files in /plot/1/CategoryName/
	Note: 
		user need to change this line to control the output contents:
		if prediction[i] == 0 and testTarget[i] == 0 and testBaselineTarget[i] == 1:
	'''
	num = 0
	sampleIdx = range(0, len(prediction))
	rd.shuffle(sampleIdx)
	for i in sampleIdx:
		if prediction[i] == 1 and testTarget[i] == 1 and testBaselineTarget[i] == 0:
			# plotResultOnFeature(str(i),testFeature[i],testReal[i])
			try:
				names = plotResultOnDownload(str(i),[testIdx[i].tolist()], 1, CategoryName, 1, 'safe3', 'cs341')
				appName = names[testIdx[i][0]]
				print appName
				num =  num +1 
			except: pass
		if num > limit:
			break
	return 0

def getMSRE(modelName, prediction, target, verbose=True):
	mse = mean_squared_error(target, prediction)
	if verbose:
		print '\n<'+modelName+'>'
		print 'MSRE:', math.sqrt(mse)
	return math.sqrt(mse)

def useLinearRegression(label, X1, Y1, X2, Y2):
	"""
	Function: use LinearRegression to predict
	Input:
		label: just a name to recognize input data from output
		n_estimators: int, The maximum number of estimators at which boosting is terminated
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
		return prediction for test
	"""
	label = label + '.LinearRegression.'
	model = linear_model.LinearRegression()
	model.fit(X1,Y1)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	getMSRE('LinearRegression.Train.'+label, prediction1, Y1)
	getMSRE('LinearRegression.Test.'+label, prediction2, Y2)
	return prediction2

def useSVR(label, X1, Y1, X2, Y2):
	"""
	Function: use SVR to predict
	Input:
		label: just a name to recognize input data from output
		X1, Y1: train
		X2, Y2: test
	Output:
		print accuracy to screen
		return prediction for test
	"""
	label = label + '.SVR.'
	model = SVR(kernel='rbf', degree= 3, C=1.0, epsilon=0.2)
	model.fit(X1,Y1)
	prediction1 = model.predict(X1)
	prediction2 = model.predict(X2)
	getMSRE('LinearRegression.Train.'+label, prediction1, Y1)
	getMSRE('LinearRegression.Test.'+label, prediction2, Y2)
	return prediction2

def integrateNegLabel(target, label):
	for i in range(len(target)):
		if target[i] < 0:
			target[i] = target[i]
	return target

def getPosMSRE(modelName, prediction, target, verbose=True):
	posTar = list()
	posPre = list()
	negTar = list()
	negPre = list()
	for i in range(len(prediction)):
		if target[i] > 0:
			posTar.append(target[i])
			posPre.append(prediction[i])
		if target[i] < 0:
			negTar.append(target[i])
			negPre.append(prediction[i])
	mseP = mean_squared_error(posTar, posPre)
	mseN = mean_squared_error(negTar, negPre)
	if verbose:
		print '\nOverall  Sample Number:', len(target)
		print 'Positive Sample Number:', len(posTar)
		print 'Negative Sample Number:', len(negTar)
		print '<'+modelName+'>', 'MSRE of Positive Targets:', math.sqrt(mseP)
		print '<'+modelName+'>', 'MSRE of Negative Targets:', math.sqrt(mseN)

def plotTargetHist(title, target):
	plt.hist(target, bins=bucketNum/2, range=[-150, 150])
	plt.title(title)
	plt.xlabel("Rank Bucket Change")
	plt.ylabel("Number of Samples")
	plt.show()

def getInputForML(bml, FeatureMatrixMultiplier, Review = True, Rating = True, \
	LinearFit = True, LoadDir = None):
	if LoadDir != None:
		# load existing matrix
		data = joblib.load(loadDir)
		train = data[0]
		test = data[1]

		trainFeature = train[0]
		trainTarget = (train[1][:, 0]).astype(int)		# label
		trainReal = train[2]								# real feature metrix in prediction window
		trainIdx = train[3]								# index for plotAppWithRow(), need to run .tolist() before input to plotAppWithRow()
		# trainBaselineTarget = (newtrain[4][:, 0]).astype(int)# label for baseline model
		testFeature = test[0]
		testTarget = (test[1][:, 0]).astype(int)			# label
		testReal = test[2]									# real feature metrix in prediction window
		testIdx = test[3]									# index for plotAppWithRow(), need to run .tolist() before input to plotAppWithRow()
		# testBaselineTarget = (test[4][:, 0]).astype(int)	# label for baseline model
		print 'Finished labelling'
		return train, test, trainFeature, trainTarget, trainReal,\
				testFeature, testTarget, testReal
	else:
		# build new matrix
		train, test = bml.buildMatrix()

		orgtrain = train
		orgtest = test

		if LinearFit:
			print 'Start adding Linear Fit points'
			train[0] = addLinearFitFeatures(train[0])
			test[0] = addLinearFitFeatures(test[0])
			print 'Finished adding linear fit feature'

		if Review:
			print 'Start adding Review Feature'
			rawReviewMat = bml.rawDataMatrix(bml.reviewFile)
			reviewMatrix = bml.compressMatrix(bml.buildReviewMatrix(rawReviewMat))
			bml.pruneMatrix(train, reviewMatrix)
			bml.pruneMatrix(test, reviewMatrix)
			print 'Finished adding Review feature'
		
		if Rating:
			print 'Start adding Rating Feature'
			rawReviewMat = bml.rawDataMatrix(bml.ratingFile)
			reviewMatrix = bml.compressMatrix(bml.buildReviewMatrix(rawReviewMat))
			bml.pruneMatrix(train, reviewMatrix)
			bml.pruneMatrix(test, reviewMatrix)
			print 'Finished adding Rating feature'

		train[0] = train[0]*FeatureMatrixMultiplier
		train[2] = train[2]*FeatureMatrixMultiplier
		test[0] = test[0]*FeatureMatrixMultiplier
		test[2] = test[2]*FeatureMatrixMultiplier
		# balance samples
		# newtrain = balanceData(train, posRate=0.0096)
		
		# create input for ML algorithms
		trainFeature = train[0]
		trainTarget = (train[1][:, 0]).astype(int)		# label
		trainReal = train[2]								# real feature metrix in prediction window
		trainIdx = train[3]								# index for plotAppWithRow(), need to run .tolist() before input to plotAppWithRow()
		# trainBaselineTarget = (newtrain[4][:, 0]).astype(int)# label for baseline model
		testFeature = test[0]
		testTarget = (test[1][:, 0]).astype(int)			# label
		testReal = test[2]									# real feature metrix in prediction window
		testIdx = test[3]									# index for plotAppWithRow(), need to run .tolist() before input to plotAppWithRow()
		# testBaselineTarget = (test[4][:, 0]).astype(int)	# label for baseline model
		print 'Finished labelling'
		return train, test, trainFeature, trainTarget, trainReal,\
				testFeature, testTarget, testReal

if __name__ == '__main__':
	timeStart = time.time()
	rd.seed(time.time())
	bml = setParameters(bml, CategoryName, LabelPercent, predictTimeWindow, featureTimeWindow, windowStepLength)
	
	# get input data for ML algorithms
	train, test, \
	trainFeature, trainTarget, trainReal, \
	testFeature, testTarget, testReal\
	= getInputForML(bml, FeatureMatrixMultiplier, Review = ReviewFeature, Rating = RatingFeature)


	# unify negtive label
	# trainTarget = integrateNegLabel(trainTarget, NegLabelConst)
	# testTarget = integrateNegLabel(testTarget, NegLabelConst)
	
	# print data details
	printMatrixInfo(train, test)

	# run prediction models
	prediction_rfr = useRandomForestRegression(CategoryName, trainFeature, trainTarget, testFeature, testTarget, \
                        n_estimators=200, max_depth = 8, verbose=True)
	prediction_gbr = useGBR(CategoryName, trainFeature, trainTarget, testFeature, testTarget, \
                        n_estimators=200, max_depth = 3, verbose=True)
	# prediction_svr = useSVR(CategoryName, trainFeature, trainTarget, testFeature, testTarget)
	prediction_lin = useLinearRegression(CategoryName, trainFeature, trainTarget, testFeature, testTarget)
	baseline = testTarget*0

	getMSRE('baseline',baseline,testTarget)
	getPosMSRE('RFR',prediction_rfr[1],testTarget)
	# getPosMSRE('SVR',prediction_svr,testTarget)
	getPosMSRE('LinearRegression',prediction_lin,testTarget)
	getPosMSRE('baseline',baseline,testTarget)
	
	# plotMultipleResults(prediction, testTarget, testBaselineTarget, testIdx, CategoryName, 50)

	runTime = time.time() - timeStart
	print '\nRuntime: ', runTime


	SaveDir = CategoryName + '-' + predictTimeWindow + '-' +bucketNum + '.pkl'
	joblib.dump([train, test], SaveDir)




