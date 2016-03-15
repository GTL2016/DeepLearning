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

# scaling factor
scale = 0.000000194325685545

# Snapshot to test
net = caffe.Net('train_val.prototxt', './snap/_iter_900.caffemodel', caffe.TEST)


net.forward()
labels = net.blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(5,4)
pred = net.blobs['fc8'].data[:]
for test_it in range(4):
	net.forward()
	labels = np.concatenate((labels,net.blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(5,4)))
	pred = np.concatenate((pred,net.blobs['fc8'].data[:]))
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

# Show all figures
show()
