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


if sys.argv[1]=='cpu':
	caffe.set_mode_cpu()
elif sys.argv[1]=='gpu':
	caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')

os.chdir('../Database')
f=open('scale.txt',"r")
lines = f.readlines()
for line in lines:
	s = line
scale = float(s)
os.chdir('../Network_rescale')

# Clearing the snap directory
directory='./snap'
os.chdir(directory)
files=glob.glob('*')
for filename in files:
    os.remove(filename)

os.chdir('..')

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
imshow(solver.test_nets[0].blobs['pool5'].data[:,0].reshape(23,7*10),cmap='gray')

# Complete training
max_iter = 31000
test_interval = 1000
test_iter = 101
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
		for test_it in range(test_iter):
			solver.test_nets[0].forward()

print solver.test_nets[0].blobs['fc8'].data[:].shape

# Display conv1 layer after max_iter iterations:
figure(5)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
figure(6)
imshow(solver.test_nets[0].blobs['pool5'].data[:,0].reshape(5,7*10),cmap='gray')

solver.test_nets[0].forward()
labels = solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(5,4)
pred = solver.test_nets[0].blobs['fc8'].data[:]
for test_it in range(test_iter-1):
	solver.test_nets[0].forward()
	labels = np.concatenate((labels,solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(5,4)))
	pred = np.concatenate((pred,solver.test_nets[0].blobs['fc8'].data[:]))
# Taking into account the scaling factor
labels = labels/scale
pred = pred/scale

# Plotting position and predicted position
figure(7)
scatter(labels[:,0],labels[:,1],s=25,c='g',marker='+')
scatter(labels[:,2],labels[:,3],s=25,c='r',marker='+')
scatter(pred[:,0],pred[:,1],s=25,c='b',marker='+')
scatter(pred[:,2],pred[:,3],s=25,c='m',marker='+')
quiver(labels[:,0],labels[:,1],labels[:,2]-labels[:,0],labels[:,3]-labels[:,1],color='g')
quiver(pred[:,0],pred[:,1],pred[:,2]-pred[:,0],pred[:,3]-pred[:,1],color='r')

# Plotting prediction error for the position X/Y
figure(8)
scatter((pred[:,0]-labels[:,0]),(pred[:,1]-labels[:,1]),s=25,c='g')

# Plotting angle error (histogram)
figure(9)
hist(np.arctan(pred[:,3]-pred[:,1],pred[:,2]-pred[:,0])-np.arctan(labels[:,3]-labels[:,1],labels[:,2]-labels[:,0]))

# Plotting loss 
figure(10)
plt.plot(arange(max_iter-50), train_loss[50:])
plt.xlabel('iteration')
plt.ylabel('train loss')

# Show all figures
show()

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
