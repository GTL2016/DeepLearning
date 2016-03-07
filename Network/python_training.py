import os
import sys
import caffe
from pylab import *
from caffe import layers as L
from caffe import params as P
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


caffe.set_mode_cpu()
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
imshow(solver.net.blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(210, 352),cmap='gray')
show()
print solver.net.blobs['labels'].data[:1]
figure(2)
imshow(solver.test_nets[0].blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(210,352),cmap='gray')
show()
print solver.test_nets[0].blobs['labels'].data[:1]
solver.step(1)
#imshow(solver.net.params['conv1'][0].diff[:, 0].reshape(4, 5, 5, 5).transpose(0, 2, 1, 3).reshape(4*5, 5*5), cmap='gray')
