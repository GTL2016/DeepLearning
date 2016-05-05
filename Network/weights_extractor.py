import os
import sys
import caffe
from pylab import *
from caffe import layers as L
from caffe import params as P
import numpy as np

train_val_path = '../../../classif/snap/caffenet_train9/train_val.prototxt'
caffemodel_path = '../../../classif/snap/caffenet_train9/snap_iter_250000.caffemodel'
weights_path = './weights'


if sys.argv[1]=='cpu':
	caffe.set_mode_cpu()
elif sys.argv[1]=='gpu':
	caffe.set_mode_gpu()

# Load snapshot
net = caffe.Net(train_val_path, caffemodel_path, caffe.TEST)
net.forward()

np.save(weights_path+'/conv1.npy',net.params['conv1'][0].data)
np.save(weights_path+'/conv2.npy',net.params['conv2'][0].data)
np.save(weights_path+'/conv3.npy',net.params['conv3'][0].data)
np.save(weights_path+'/conv4.npy',net.params['conv4'][0].data)
np.save(weights_path+'/conv5.npy',net.params['conv5'][0].data)

np.save(weights_path+'/conv1_b.npy',net.params['conv1'][1].data)
np.save(weights_path+'/conv2_b.npy',net.params['conv2'][1].data)
np.save(weights_path+'/conv3_b.npy',net.params['conv3'][1].data)
np.save(weights_path+'/conv4_b.npy',net.params['conv4'][1].data)
np.save(weights_path+'/conv5_b.npy',net.params['conv5'][1].data)
