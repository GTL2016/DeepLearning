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

# conv1 layer before iterations begin
solver.step(1)
figure(3)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
show()

# Iterations to display conv 1 layer
solver.step(1)
figure(4)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
show()

for i in range(10):
	solver.step(1)
figure(5)
imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(12,8, 11, 11).transpose(0, 2, 1, 3).reshape(12*11, 8*11),cmap='gray')
show()
