import os
import sys
import caffe
import matplotlib
from caffe import layers as L
from caffe import params as P

#cat train_val.prototxt
#cat solver.prototxt

caffe.set_mode_cpu()
solver = caffe.SGDSolver('solver.prototxt')

