import os
import sys
import caffe
from pylab import *
from caffe import layers as L
from caffe import params as P
import numpy as np

train_val_path = 'train_val.prototxt'
caffemodel_path = './snap/_iter_1000.caffemodel'
weights_path = './weights'


if sys.argv[1]=='cpu':
	caffe.set_mode_cpu()
elif sys.argv[1]=='gpu':
	caffe.set_mode_gpu()

# Load snapshot
net = caffe.Net(train_val_path, caffemodel_path, caffe.TEST)
net.forward()
labels = net.blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_test,4)
pred = net.blobs['fc8'].data[:]
for test_it in range(test_iter):
	net.forward()
	labels = np.concatenate((labels,net.blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_test,4)))
	pred = np.concatenate((pred,net.blobs['fc8'].data[:]))

np.save(weights_path+'/conv1.npy',solver.net.params['conv1'][0].data)
np.save(weights_path+'/conv2.npy',solver.net.params['conv2'][0].data)
np.save(weights_path+'/conv3.npy',solver.net.params['conv3'][0].data)
np.save(weights_path+'/conv4.npy',solver.net.params['conv4'][0].data)
np.save(weights_path+'/conv5.npy',solver.net.params['conv5'][0].data)
