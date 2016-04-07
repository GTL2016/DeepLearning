import os
import sys
import caffe
from pylab import *
from caffe import layers as L
from caffe import params as P
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import glob
import shutil

test_iter = 8
batch_size_test = 25 #test_iter*batch_size = nb of test images
batch_size_train = 30
max_iter = 200000 #Number of iterations for the training
test_interval = 200 #interval between two tests
stepsize = 50000 #interval between each learning rate decrease

if sys.argv[1]=='cpu':
	caffe.set_mode_cpu()
elif sys.argv[1]=='gpu':
	caffe.set_mode_gpu()
solver = caffe.SGDSolver('solver.prototxt')

scale = 0;
os.chdir('../Database')
f=open('scale.txt',"r")
lines = f.readlines()
for line in lines:
	scale = float(line)
f.close()
os.chdir('../Network')
print('Scale = '+str(scale))

# Clearing the snap directory
directory='./snap'
os.chdir(directory)
files=glob.glob('*')
for filename in files:
	os.remove(filename)
os.chdir('..') # coming back to Network directory

# Clearing previous training figures
pathfigs='./figures'
os.chdir(pathfigs)
files=glob.glob('*.png')
for filename in files:
	os.remove(filename)
dirs=glob.glob('*')
for dire in dirs:
	shutil.rmtree(dire)
os.chdir('..') # coming back to Network directory
os.makedirs(pathfigs+'/label_views')

def vis_square(data, padsize=1, padval=0):
	data1 = data - data.min()
	data1 /= data1.max()
	# force the number of filters to be square
	n = int(np.ceil(np.sqrt(data1.shape[0])))
	padding = ((0, n ** 2 - data1.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data1.ndim - 3)
	data1 = np.pad(data1, padding, mode='constant', constant_values=(padval, padval))

	# tile the filters into an image
	data1 = data1.reshape((n, n) + data1.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data1.ndim + 1)))
	data1 = data1.reshape((n * data1.shape[1], n * data1.shape[3]) + data1.shape[4:])
	return data1


def test_and_plot( it ):
	"Create the result plots"
	print 'Iteration', it, 'testing...'
	solver.test_nets[0].forward()
	pathiter = pathfigs+'/iter_'+str(it)
	os.makedirs(pathiter)
	labels = solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_test,4)
	pred = solver.test_nets[0].blobs['fc8'].data[:]
	test_loss[it/test_interval] = solver.test_nets[0].blobs['loss'].data
	for test_it in range(test_iter-1):
		solver.test_nets[0].forward()
		labels = np.concatenate((labels,solver.test_nets[0].blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_test,4)))
		pred = np.concatenate((pred,solver.test_nets[0].blobs['fc8'].data[:]))
		test_loss[it/test_interval] = test_loss[it/test_interval]+solver.test_nets[0].blobs['loss'].data
	test_loss[it/test_interval] = test_loss[it/test_interval]/(test_it*scale*scale)
	# Plotting conv1 weights layer at test interval
	fig.clear()
	filters = solver.net.params['conv1'][0].data
	imshow(vis_square(filters.transpose(0, 2, 3, 1)),cmap='gray')
	fig.savefig(pathiter+'/conv1_'+str(it)+'.png')
	# Plotting output of Conv 1
	fig.clear()
	imshow(vis_square(solver.net.blobs['conv1'].data[0],padval=1),cmap='gray')
	fig.savefig(pathiter+'/conv1_out_'+str(it)+'.png')
	# Plotting output of pool1 layer at test interval
	fig.clear()
	#imshow(solver.test_nets[0].blobs['pool1'].data[:,0].reshape(batch_size_test,29*43),cmap='gray')
	imshow(vis_square(solver.test_nets[0].blobs['pool1'].data[0],padval=1),cmap='gray')
	fig.savefig(pathiter+'/pool1_'+str(it)+'.png')
	# Plotting output of norm1 layer at test interval
	fig.clear()
	#imshow(solver.test_nets[0].blobs['norm1'].data[:,0].reshape(batch_size_test,29*43),cmap='gray')
	imshow(vis_square(solver.test_nets[0].blobs['norm1'].data[0],padval=1),cmap='gray')
	fig.savefig(pathiter+'/norm1_'+str(it)+'.png')
	# Plotting output of pool2 layer at test interval
	fig.clear()
	#imshow(solver.test_nets[0].blobs['pool2'].data[:,0].reshape(batch_size_test,14*21),cmap='gray')
	imshow(vis_square(solver.test_nets[0].blobs['pool2'].data[0],padval=1),cmap='gray')
	fig.savefig(pathiter+'/pool2_'+str(it)+'.png')
	# Plotting output of norm2 layer at test interval
	fig.clear()
	#imshow(solver.test_nets[0].blobs['norm2'].data[:,0].reshape(batch_size_test,14*21),cmap='gray')
	imshow(vis_square(solver.test_nets[0].blobs['norm2'].data[0],padval=1),cmap='gray')
	fig.savefig(pathiter+'/norm2_'+str(it)+'.png')
	# Plotting output of pool5 layer at test interval
	fig.clear()
	#imshow(solver.test_nets[0].blobs['pool5'].data[:,0].reshape(batch_size_test,7*10),cmap='gray')
	imshow(vis_square(solver.test_nets[0].blobs['pool5'].data[0],padval=1),cmap='gray')
	fig.savefig(pathiter+'/pool5_'+str(it)+'.png')
	# Plotting position and predicted position at test interval
	fig.clear()
	scatter(labels[:,0],labels[:,1],s=25,c='g',marker='+')
	scatter(labels[:,2],labels[:,3],s=25,c='r',marker='+')
	scatter(pred[:,0],pred[:,1],s=25,c='b',marker='+')
	scatter(pred[:,2],pred[:,3],s=25,c='m',marker='+')
	quiver(labels[:,0],labels[:,1],labels[:,2]-labels[:,0],labels[:,3]-labels[:,1],color='g')
	quiver(pred[:,0],pred[:,1],pred[:,2]-pred[:,0],pred[:,3]-pred[:,1],color='r')
	fig.savefig(pathiter+'/labels_view_'+str(it)+'.png')
	fig.savefig(pathfigs+'/label_views'+'/labels_view_'+str(it)+'.png')
	# Plotting prediction error for the position X/Y
	fig.clear()
	scatter((pred[:,0]-labels[:,0]),(pred[:,1]-labels[:,1]),s=25,c='g')
	fig.savefig(pathiter+'/error_xy_'+str(it)+'.png')
	# Plotting angle error (histogram)
	fig.clear()
	hist(np.arctan(pred[:,3]-pred[:,1],pred[:,2]-pred[:,0])-np.arctan(labels[:,3]-labels[:,1],labels[:,2]-labels[:,0]))
	fig.savefig(pathiter+'/error_angle_'+str(it)+'.png')
	if (it>=2*test_interval):
		# Plotting loss 
		fig.clear()
		plt.plot(arange(it)[test_interval:it], train_loss[test_interval:it], marker='.', linestyle='None', color='b')
		plt.xlabel('iteration')
		plt.ylabel('train loss')
		fig.savefig(pathiter+'/loss_'+str(it)+'.png')
		# Plotting test loss and train loss average
		fig.clear()
		axis = arange(2*test_interval,it+1,test_interval)
		plt.plot(axis, test_loss[2:1+it/test_interval], 'g')
		plt.plot(axis, train_loss_average[2:1+it/test_interval], 'b')
		plt.ylabel('test loss (green), train loss (blue)')
		fig.savefig(pathiter+'/test_loss_'+str(it)+'.png')
		#~ if (it>stepsize):
			#~ # Plotting test loss and train loss average on the current step
			#~ fig.clear()
			#~ axis = arange(floor(it/stepsize)*stepsize,it+1,test_interval)
			#~ plt.plot(axis, test_loss[floor(it/stepsize)*stepsize+1:1+it/test_interval], 'g')
			#~ plt.plot(axis, train_loss_average[floor(it/stepsize)*stepsize+1:1+it/test_interval], 'b')
			#~ plt.ylabel('test loss (green), train loss (blue)')
			#~ fig.savefig(pathiter+'/test_loss_step'+str(it)+'.png')
	# Tests to visualize histograms of conv and fc param weights
	# Conv1
	fig.clear()
	hist(solver.net.params['conv1'][0].data.flatten())
	fig.savefig(pathiter+'/conv1_hist_'+str(it)+'.png')
	# Conv2
	fig.clear()
	hist(solver.net.params['conv2'][0].data.flatten())
	fig.savefig(pathiter+'/conv2_hist_'+str(it)+'.png')
	# Conv3
	fig.clear()
	hist(solver.net.params['conv3'][0].data.flatten())
	fig.savefig(pathiter+'/conv3_hist_'+str(it)+'.png')
	# Conv4
	fig.clear()
	hist(solver.net.params['conv4'][0].data.flatten())
	fig.savefig(pathiter+'/conv4_hist_'+str(it)+'.png')
	# Conv5
	fig.clear()
	hist(solver.net.params['conv5'][0].data.flatten())
	fig.savefig(pathiter+'/conv5_hist_'+str(it)+'.png')
	# fc6
	fig.clear()
	hist(solver.net.params['fc6'][0].data.flatten())
	fig.savefig(pathiter+'/fc6_hist_'+str(it)+'.png')
	# fc7
	fig.clear()
	hist(solver.net.params['fc7'][0].data.flatten())
	fig.savefig(pathiter+'/fc7_hist_'+str(it)+'.png')
	# fc8
	fig.clear()
	hist(solver.net.params['fc8'][0].data.flatten())
	fig.savefig(pathiter+'/fc8_hist_'+str(it)+'.png')
	# fc8 bias
	fig.clear()
	hist(solver.net.params['fc8'][1].data.flatten())
	fig.savefig(pathiter+'/fc8_bias_hist_'+str(it)+'.png')
	# Norm1 out
	fig.clear()
	hist(solver.test_nets[0].blobs['norm1'].data[:,:].flatten())
	fig.savefig(pathiter+'/norm1_out_hist_'+str(it)+'.png')
	# Norm2 out
	fig.clear()
	hist(solver.test_nets[0].blobs['norm2'].data[:,:].flatten())
	fig.savefig(pathiter+'/norm2_out_hist_'+str(it)+'.png')
	# Pool5 out
	fig.clear()
	hist(solver.test_nets[0].blobs['pool5'].data[:,:].flatten())
	fig.savefig(pathiter+'/pool5_out_hist_'+str(it)+'.png')
	# fc8 out
	fig.clear()
	hist(solver.test_nets[0].blobs['fc8'].data[:,:].flatten())
	fig.savefig(pathiter+'/fc8_out_hist_'+str(it)+'.png')
	# fc6 out
	fig.clear()
	hist(solver.test_nets[0].blobs['fc6'].data[:,:].flatten())
	fig.savefig(pathiter+'/fc6_out_hist_'+str(it)+'.png')
	return




# Printing outputs size and weights size for the current training
# each output is (batch size, feature dim, spatial dim)
a = [(k, v.data.shape) for k, v in solver.net.blobs.items()]
print(a)
# just print the weight sizes (not biases)
b = [(k, v[0].data.shape) for k, v in solver.net.params.items()]
print(b)



solver.test_nets[0].forward()
# Plotting the initialisation
fig = figure()
test_loss = zeros(1+floor(max_iter/test_interval))
train_loss_average = zeros(1+floor(max_iter/test_interval))
test_and_plot(0)



solver.net.forward()

# Ploting the first images of the train and val datasets (mosaique des images si batch size > 1, ici selection de 1 dans data)
fig.clear()
#print(shape(solver.net.blobs['images'].data[:1, 0].transpose(1, 0, 2)))
#im_1 = np.zeros((240,352,3))
#im_1[:,:,0] = solver.net.blobs['images'].data[:1, 0].transpose(1, 0, 2).reshape(im_height, im_width)
#im_1[:,:,1] = solver.net.blobs['images'].data[:1, 1].transpose(1, 0, 2).reshape(im_height, im_width)
#im_1[:,:,2] = solver.net.blobs['images'].data[:1, 2].transpose(1, 0, 2).reshape(im_height, im_width)
#imshow(im_1, norm=Normalize())

shape = solver.net.blobs['images'].data[:1, 0].transpose(1, 2, 0).shape
imshow(solver.net.blobs['images'].data[:1, 0].transpose(1, 2, 0).reshape(shape[0],shape[1]),cmap='gray')
fig.savefig(pathfigs+'/image1_trainset.png')
fig.clear()
imshow(solver.test_nets[0].blobs['images'].data[:1, 0].transpose(1, 2, 0).reshape(shape[0],shape[1]),cmap='gray')
fig.savefig(pathfigs+'/image1_valset.png')

# Training
train_loss = zeros(max_iter)
#train_loss_manual = zeros(max_iter)
# the main solver loop
for it in range(1,max_iter):
	solver.step(1)  # SGD by Caffe
	# store the train loss
	train_loss[it] = solver.net.blobs['loss'].data/(scale*scale)
	label_loss = solver.net.blobs['labels'].data[:].transpose(0, 2, 1, 3).reshape(batch_size_train,4)
	pred_loss = solver.net.blobs['fc8'].data[:]
	#train_loss_manual_vect = ((label_loss[:,0]-pred_loss[:,0])*(label_loss[:,0]-pred_loss[:,0])+(label_loss[:,1]-pred_loss[:,1])*(label_loss[:,1]-pred_loss[:,1])+(label_loss[:,2]-pred_loss[:,2])*(label_loss[:,2]-pred_loss[:,2])+(label_loss[:,3]-pred_loss[:,3])*(label_loss[:,3]-pred_loss[:,3]))
	#train_loss_manual[it] = 0.5*np.mean(train_loss_manual_vect)
	#print(train_loss[it]-train_loss_manual[it])
	# start the forward pass at conv1 to avoid loading new data
	solver.test_nets[0].forward(start='conv1')
	# run a full test every so often (Caffe can also do this for us and write to a log, but we show here how to do it directly in Python, where more complicated things are easier.)
	if it % test_interval == 0:
		train_loss_average[it/test_interval] = np.sum(train_loss[it-test_interval:it])/test_interval
		test_and_plot(it)
		fig.clear()
		scatter(label_loss[:,0],label_loss[:,1],s=25,c='g',marker='+')
		scatter(label_loss[:,2],label_loss[:,3],s=25,c='r',marker='+')
		scatter(pred_loss[:,0],pred_loss[:,1],s=25,c='b',marker='+')
		scatter(pred_loss[:,2],pred_loss[:,3],s=25,c='m',marker='+')
		quiver(label_loss[:,0],label_loss[:,1],label_loss[:,2]-label_loss[:,0],label_loss[:,3]-label_loss[:,1],color='g')
		quiver(pred_loss[:,0],pred_loss[:,1],pred_loss[:,2]-pred_loss[:,0],pred_loss[:,3]-pred_loss[:,1],color='r')
		pathiter = pathfigs+'/iter_'+str(it)
		fig.savefig(pathiter+'/labels_view_train_'+str(it)+'.png')
		fig.savefig(pathfigs+'/label_views'+'/labels_view_train_'+str(it)+'.png')

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

