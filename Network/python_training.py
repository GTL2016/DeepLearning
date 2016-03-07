import os
import sys
import caffe
import pylab
import matplotlib
from caffe import layers as L
from caffe import params as P


caffe.set_mode_cpu()
solver = caffe.SGDSolver('solver.prototxt')
[(k, v.data.shape) for k, v in solver.net.blobs.items()]

