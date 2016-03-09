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
show()
print solver.net.blobs['labels'].data[:1]
figure(2)
imshow(solver.test_nets[0].blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(240,352),cmap='gray')
show()
print solver.test_nets[0].blobs['labels'].data[:1]

# plotting conv1 layer before iterations begin
solver.step(1)
figure(3)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
show()

# 1 Iteration to display conv 1 layer
solver.step(1)
figure(4)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
show()

# # 10 iterations to display conv 1
#for i in range(10):
	#solver.step(1)
#figure(5)
#imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
#show()

max_iter = 200
test_interval = 25
## losses will also be stored in the log
test_acc = zeros(int(np.ceil(max_iter / test_interval)))
train_loss = zeros(max_iter)
output = zeros((max_iter, 8, 4)) # second argument is number of classes (4) and first is the number of examples to display/select (here we choose 8 first images)
# the main solver loop
for it in range(max_iter):
	solver.step(1)  # SGD by Caffe
	
	# store the train loss
	train_loss[it] = solver.net.blobs['loss'].data
	
	# store the output on the first test batch
	## (start the forward pass at conv1 to avoid loading new data)
	solver.test_nets[0].forward(start='conv1')
	output[it] = solver.test_nets[0].blobs['fc8'].data[:8]
	## run a full test every so often
    ## (Caffe can also do this for us and write to a log, but we show here
    ##  how to do it directly in Python, where more complicated things are easier.)
    if it%test_interval == 0:
		print 'Iteration', it, 'testing...'
		correct = 0
		for test_it in range(100):
			solver.test_nets[0].forward()
			correct += sum(solver.test_nets[0].blobs['fc8'].data.argmax(1)== solver.test_nets[0].blobs['labels'].data
		test_acc[it // test_interval] = correct / 1e4

# Display conv1 layer after max_iter iterations:
solver.step(1)
figure(5)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
show()

# Plotting loss and accuracy
_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')

# Plotting output and found label along iterations
#for i in range(8):
    #figure(figsize=(2, 2))
    #imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    #figure(figsize=(10, 2))
    #imshow(output[:50, i].T, interpolation='nearest', cmap='gray')
    #xlabel('iteration')
    #ylabel('label')

#for i in range(8):
    #figure(figsize=(2, 2))
    #imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    #figure(figsize=(10, 2))
    #imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    #xlabel('iteration')
    #ylabel('label')
