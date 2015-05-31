from pybrain.datasets				import ClassificationDataSet
from pybrain.utilities				import percentError
from pybrain.tools.shortcuts		import buildNetwork
from pybrain.supervised.trainers 	import BackpropTrainer
from pybrain.structure.modules   	import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal


def useNueralNetwork():
	means = [(-1,0),(2,4),(3,1)]
	cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
	alldata = ClassificationDataSet(2, 1, nb_classes=3)
	for n in xrange(400):
		for klass in range(3):
			input = multivariate_normal(means[klass],cov[klass])
			alldata.addSample(input, [klass])

	# Randomly split the dataset into 75% training and 25% test data sets. 
	tstdata_temp, trndata_temp = alldata.splitWithProportion( 0.25 )

	# Encode classes with one output neuron per class
	tstdata = ClassificationDataSet(2, 1, nb_classes=3)
	for n in xrange(0, tstdata_temp.getLength()):
			tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )
	trndata = ClassificationDataSet(2, 1, nb_classes=3)
	for n in xrange(0, trndata_temp.getLength()):
			trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
	trndata._convertToOneOfMany()
	tstdata._convertToOneOfMany()

	# Test our dataset by printing a little information about it
	print "Number of training patterns: ", len(trndata)
	print "Input and output dimensions: ", trndata.indim, trndata.outdim
	print "First sample (input, target, class):"
	print trndata['input'][0], trndata['target'][0], trndata['class'][0]

	# Build a feed-forward network with 5 hidden units.
	fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

	# Set up a trainer that basically takes the network and training dataset as input. For 
	# a list of trainers, see trainers. We are using a BackpropTrainer for this.
	trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

	# Now generate a square grid of data points and put it into a dataset, which we can then classify 
	# to obtain a nice contour field for visualization. Therefore the target values for this data set can be ignored.
	ticks = arange(-3.,6.,0.2)
	X, Y = meshgrid(ticks, ticks)
	# need column vectors in dataset, not arrays
	griddata = ClassificationDataSet(2,1, nb_classes=3)
	for i in xrange(X.size):
			griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
	griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

	# Start the training iterations.
	for i in range(5):
		# Train the network for some epochs. Usually you would set something like 5 here
		trainer.trainEpochs( 1 )

		# Evaluate the network on the training and test data. 
		trnresult = percentError( trainer.testOnClassData(),trndata['class'])
		tstresult = percentError( trainer.testOnClassData(dataset=tstdata ), tstdata['class'])

		print "epoch: %4d" % trainer.totalepochs, \
			"  train error: %5.2f%%" % trnresult, \
			"  test error: %5.2f%%" % tstresult

		# Run our grid data through the FNN, get the most likely class and shape it into a square array again.
		out = fnn.activateOnDataset(griddata)
		out = out.argmax(axis=1)  # the highest output activation gives the class
		out = out.reshape(X.shape)

		# Now plot the test data and the underlying grid as a filled contour.
		figure(1)
		ioff()  # interactive graphics off
		clf()   # clear the plot
		hold(True) # overplot on
		for c in [0,1,2]:
			here, _ = where(tstdata['class']==c)
			plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
		if out.max()!=out.min():  # safety check against flat field
			contourf(X, Y, out)   # plot the contour
		ion()   # interactive graphics on
		draw()  # update the plot

	ioff()
	show()