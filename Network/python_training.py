import os
import sys
import caffe
from pylab import *
from caffe import layers as L
from caffe import params as P
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob

test_iter = 8
batch_size_train = 30
batch_size_test = 25 #test_iter*batch_size = nb of test images
max_iter = 30000 #Number of iterations for the training
test_interval = 200 #interval between two tests

if sys.argv[1]=='cpu':
	caffe.set_mode_cpu()
elif sys.argv[1]=='gpu':
	caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')

# Clearing the snap directory
directory='./snap'
os.chdir(directory)
files=glob.glob('*')
for filename in files:
    os.remove(filename)
os.chdir('..') # coming back to Network directory

# Clearing previous training figures
pathfigs='./figures'
os.chdir(pathfigs)
files=glob.glob('*')
for filename in files:
    os.remove(filename)
os.chdir('..') # coming back to Network directory

# Printing outputs size and weights size for the current training
# each output is (batch size, feature dim, spatial dim)
a = [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print(a)
# just print the weight sizes (not biases)
b = [(k, v[0].data.shape) for k, v in solver.net.params.items()]
print(b)
solver.net.forward()
solver.test_nets[0].forward()

# Ploting the first images of the train and val datasets (mosaique des images si batch size > 1, ici selection de 1 dans data)
fig = figure()
imshow(solver.net.blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(240, 352),cmap='gray')
fig.savefig(pathfigs+'/image1_trainset.png')
fig.clear()
imshow(solver.test_nets[0].blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(240,352),cmap='gray')
fig.savefig(pathfigs+'/image1_valset.png')

# Training
train_loss = zeros(max_iter)
train_loss_manual = zeros(max_iter)
# the main solver loop
for it in range(max_iter):
	solver.step(1)  # SGD by Caffe
	# store the train loss
	train_loss[it] = solver.net.blobs['loss'].data
	label_loss = solver.net.blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_train,4)
	pred_loss = solver.net.blobs['fc8'].data[:]
	train_loss_manual_vect = ((label_loss[:,0]-pred_loss[:,0])*(label_loss[:,0]-pred_loss[:,0])+(label_loss[:,1]-pred_loss[:,1])*(label_loss[:,1]-pred_loss[:,1])+(label_loss[:,2]-pred_loss[:,2])*(label_loss[:,2]-pred_loss[:,2])+(label_loss[:,3]-pred_loss[:,3])*(label_loss[:,3]-pred_loss[:,3]))
	train_loss_manual[it] = 0.5*np.mean(train_loss_manual_vect)
	print(train_loss[it]-train_loss_manual[it])
	# start the forward pass at conv1 to avoid loading new data
	solver.test_nets[0].forward(start='conv1')
	# run a full test every so often (Caffe can also do this for us and write to a log, but we show here how to do it directly in Python, where more complicated things are easier.)
	if it % test_interval == 0:
		print 'Iteration', it, 'testing...'
		solver.test_nets[0].forward()
		labels = solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_test,4)
		pred = solver.test_nets[0].blobs['fc8'].data[:]
		for test_it in range(test_iter-1):
			solver.test_nets[0].forward()
			labels = np.concatenate((labels,solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_test,4)))
			pred = np.concatenate((pred,solver.test_nets[0].blobs['fc8'].data[:]))
		# Plotting conv1 weights layer at test interval
		fig.clear()
		imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
		fig.savefig(pathfigs+'/conv1_'+str(it)+'.png')
		# Plotting output of pool5 layer at test interval
		fig.clear()
		imshow(solver.test_nets[0].blobs['pool5'].data[:,0].reshape(batch_size_test,7*10),cmap='gray')
		fig.savefig(pathfigs+'/pool5_'+str(it)+'.png')
		# Plotting position and predicted position at test interval
		fig.clear()
		scatter(labels[:,0],labels[:,1],s=25,c='g',marker='+')
		scatter(labels[:,2],labels[:,3],s=25,c='r',marker='+')
		scatter(pred[:,0],pred[:,1],s=25,c='b',marker='+')
		scatter(pred[:,2],pred[:,3],s=25,c='m',marker='+')
		quiver(labels[:,0],labels[:,1],labels[:,2]-labels[:,0],labels[:,3]-labels[:,1],color='g')
		quiver(pred[:,0],pred[:,1],pred[:,2]-pred[:,0],pred[:,3]-pred[:,1],color='r')
		fig.savefig(pathfigs+'/labels_view_'+str(it)+'.png')
		# Plotting prediction error for the position X/Y
		fig.clear()
		scatter((pred[:,0]-labels[:,0]),(pred[:,1]-labels[:,1]),s=25,c='g')
		fig.savefig(pathfigs+'/error_xy_'+str(it)+'.png')
		# Plotting angle error (histogram)
		fig.clear()
		hist(np.arctan(pred[:,3]-pred[:,1],pred[:,2]-pred[:,0])-np.arctan(labels[:,3]-labels[:,1],labels[:,2]-labels[:,0]))
		fig.savefig(pathfigs+'/error_angle_'+str(it)+'.png')
		if (it>=50):
			# Plotting loss 
			fig.clear()
			plt.plot(arange(it)[50:it], train_loss[50:it])
			plt.xlabel('iteration')
			plt.ylabel('train loss')
			fig.savefig(pathfigs+'/loss_'+str(it)+'.png')

## Plotting loss 
#figure(6)
#_, ax1 = subplots()
#ax1.plot(arange(max_iter), train_loss)
#ax1.set_xlabel('iteration')
#ax1.set_ylabel('train loss')

## Plotting output and found label along iterations
#for i in range(8):
    #figure(figsize=(2, 2))
    #imshow(solver.test_nets[0].blobs['images'].data[i, 0], cmap='gray')
    #figure(figsize=(4, 2))
    #imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    #xlabel('iteration')
    #ylabel('label')

##for i in range(8):
    ##figure(figsize=(2, 2))
    ##imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    ##figure(figsize=(10, 2))
    ##imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    ##xlabel('iteration')
    ##ylabel('label')

#subplot(1,3,1)
# plt.imshow()
# plt.subplot(1,3,2)
# plt.imshow()
