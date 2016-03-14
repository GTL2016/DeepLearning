import os
import sys
import caffe
from pylab import *
from caffe import layers as L
from caffe import params as P
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

if sys.argv[1]=='cpu':
	caffe.set_mode_cpu()
elif sys.argv[1]=='gpu':
	caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')
# each output is (batch size, feature dim, spatial dim)
a = [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print(a)
# just print the weight sizes (not biases)
b = [(k, v[0].data.shape) for k, v in solver.net.params.items()]
print(b)
solver.net.forward()
solver.test_nets[0].forward()
# we use a little trick to tile the first image (mosaique des images si batch size > 1)
figure(1)
imshow(solver.net.blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(240, 352),cmap='gray')
figure(2)
imshow(solver.test_nets[0].blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(240,352),cmap='gray')

# 1 Iteration to display conv 1 layer
solver.step(1)
figure(3)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
figure(4)
imshow(solver.test_nets[0].blobs['pool5'].data[:,0].reshape(5,7*10),cmap='gray')

# Complete training
max_iter = 100
test_interval = 25
# losses will also be stored in the log
#test_acc = zeros(int(np.ceil(max_iter / test_interval)))
train_loss = zeros(max_iter)
output = zeros((max_iter, 5, 4)) # second argument is number of classes (4) and first is the number of examples to display/select (here we choose 8 first images)
# the main solver loop
for it in range(max_iter):
	solver.step(1)  # SGD by Caffe
	# store the train loss
	train_loss[it] = solver.net.blobs['loss'].data
	# store the output on the first test batch
	# (start the forward pass at conv1 to avoid loading new data)
	solver.test_nets[0].forward(start='conv1')
	output[it] = solver.test_nets[0].blobs['fc8'].data[:5]
	# run a full test every so often (Caffe can also do this for us and write to a log, but we show here how to do it directly in Python, where more complicated things are easier.)
	if it % test_interval == 0:
		print 'Iteration', it, 'testing...'
		correct = 0
		for test_it in range(100):
			solver.test_nets[0].forward()

# Display conv1 layer after max_iter iterations:
figure(5)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
figure(6)
imshow(solver.test_nets[0].blobs['pool5'].data[:,0].reshape(5,7*10),cmap='gray')

# Plotting position and predicted position
figure(7)
labels_in = solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(5,4)
labels_out = solver.test_nets[0].blobs['fc8'].data[:]
scatter(labels_in[:,0],labels_in[:,1],s=25,c='g',marker='+')
scatter(labels_in[:,2],labels_in[:,3],s=25,c='r',marker='+')
scatter(labels_out[:,0],labels_out[:,1],s=25,c='b',marker='+')
scatter(labels_out[:,2],labels_out[:,3],s=25,c='m',marker='+')
quiver(labels_in[:,0],labels_in[:,1],labels_in[:,2]-labels_in[:,0],labels_in[:,3]-labels_in[:,1],color='g')
quiver(labels_out[:,0],labels_out[:,1],labels_out[:,2]-labels_out[:,0],labels_out[:,3]-labels_out[:,1],color='r')
# Show all figures
show()

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
