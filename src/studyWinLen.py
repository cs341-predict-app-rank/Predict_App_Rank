#!user/bin/python2
from trainML import *
import time
import random

def plotStepLength(result):
	length = len(result[0])
	plt.plot(result[0], result[1], '-r', linewidth = 2,label='Baseline F1')
	plt.plot(result[0], result[2], '-.y', linewidth = 2,label='Baseline Precision')
	plt.plot(result[0], result[3], '-.k', linewidth = 2,label='Baseline Recall')
	plt.plot(result[0], result[4], '-g', linewidth = 2,label='RamdomForest F1')
	plt.plot(result[0], result[5], '-.b', linewidth = 2,label='RamdomForest Precision')
	plt.plot(result[0], result[6], '-.c', linewidth = 2,label='RamdomForest Recall')
	plt.axis([result[0][0], result[0][-1], 0.3, 1])
	plt.grid(color='k', linestyle='-.', linewidth=1)
	plt.title('Fix predict window 12, feature 12, change step length = 1,...,48')
	plt.xlabel('Step length')
	plt.ylabel('Scores')
	plt.legend(loc=3)
	plt.show()
	return 0

def plotPredictLength(result):
	length = len(result[0])
	plt.plot(result[0], result[1], '-r', linewidth = 2, label='Baseline F1')
	plt.plot(result[0], result[2], '-.y', linewidth = 2,label='Baseline Precision')
	plt.plot(result[0], result[3], '-.k', linewidth = 2,label='Baseline Recall')
	plt.plot(result[0], result[4], '-g', linewidth = 2,label='RamdomForest F1')
	plt.plot(result[0], result[5], '-.b', linewidth = 2,label='RamdomForest Precision')
	plt.plot(result[0], result[6], '-.c', linewidth = 2,label='RamdomForest Recall')
	plt.axis([9, 48, 0.3, 1])
	plt.grid(color='k', linestyle='-.', linewidth=1)
	plt.title('Fix step length = 1, change predict length = 9,...,48')
	plt.xlabel('Predict length')
	plt.ylabel('Scores')
	plt.legend(loc=3)
	plt.show()
	return 0

def plotFeatureLength(result):
	length = len(result[0])
	plt.plot(result[0], result[1], '-r', linewidth = 2, label='Baseline F1')
	plt.plot(result[0], result[2], '-.y', linewidth = 2,label='Baseline Precision')
	plt.plot(result[0], result[3], '-.k', linewidth = 2,label='Baseline Recall')
	plt.plot(result[0], result[4], '-g', linewidth = 2,label='RamdomForest F1')
	plt.plot(result[0], result[5], '-.b', linewidth = 2,label='RamdomForest Precision')
	plt.plot(result[0], result[6], '-.c', linewidth = 2,label='RamdomForest Recall')
	plt.axis([9, 36, 0, 1])
	plt.grid(color='k', linestyle='-.', linewidth=1)
	plt.title('Fix predict length = 36, change feature length = 9,...,36')
	plt.xlabel('feature length')
	plt.ylabel('Scores')
	plt.legend(loc=3)
	plt.show()
	return 0

if __name__ == '__main__':
	timeStart = time.time()
	random.seed(time.time())
	
	# set labelling parameters
	ModelFileDir = 'models/' # save existing model in this dir
	CategoryName = 'Social Networking'
	LabelPercent = 0.6
	FeatureMatrixMultiplier = 10000
	predictTimeWindow = 36
	featureTimeWindow = 12
	windowStepLength = 1
	
	variableLength = range(9,36,3)	
	baselineF1 = list()
	baselinePrecision = list()
	baselineRecall = list()
	rfF1 = list()
	rfPrecision = list()
	rfRecall = list()

	for length in variableLength:
		bml = setParameters(bml, CategoryName, LabelPercent, predictTimeWindow, featureTimeWindow, length)
		print '\n', bml.windowStepLength
		# get input data for ML algorithms
		train, test, trainFeature, trainTarget, trainReal, trainIdx, trainBaselineTarget,\
		testFeature, testTarget, testReal, testIdx, testBaselineTarget\
		= getInputForML(bml, FeatureMatrixMultiplier)

		# print data details
		# printMatrixInfo(train, test)
		f1, pr, re = getAccuracy('baseline',testBaselineTarget,testTarget)
		baselineF1.append(f1)
		baselinePrecision.append(pr)
		baselineRecall.append(re)
		
		meanf1 = 0
		meanpr = 0
		meanre = 0
		for i in range(1,7):
			prediction = useRandomForest(CategoryName, trainFeature, trainTarget, testFeature, testTarget, n_estimators=30)
			f1, pr, re = getAccuracy('random forest',prediction,testTarget)
			meanf1 = (meanf1*(i-1) + f1)/float(i)
			meanpr = (meanpr*(i-1) + pr)/float(i)
			meanre = (meanre*(i-1) + re)/float(i)
		rfF1.append(meanf1)
		rfPrecision.append(meanpr)
		rfRecall.append(meanre)
	# plotMultipleResults(prediction, testTarget, testBaselineTarget, testIdx, CategoryName, 50)
	result = [variableLength, baselineF1, baselinePrecision, baselineRecall, rfF1, rfPrecision, rfRecall]
	runTime = time.time() - timeStart
	print '\nRuntime: ', runTime