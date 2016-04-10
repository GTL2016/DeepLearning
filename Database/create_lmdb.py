import lmdb
import re, fileinput, math
import numpy as np
import caffe
import sys
from PIL import Image
from skimage.transform import resize
import skimage.io
import shutil
import os

interp_order = 1

if ((len(sys.argv)==2 or len(sys.argv)==4)):
	# Command line to check created files:
	# python -mlmdb stat --env=./Downloads/caffe-master/data/liris-accede/train_score_lmdb/
	# python -mlmdb stat --env=./Downloads/caffe-master/data/liris-accede/val_score_lmdb/

	data = sys.argv[1]+'.txt'
	command = 'shuf '+data+' > '+sys.argv[1]+'2.txt'
	os.system(command)
	data = sys.argv[1]+'2.txt'
	print(data)
	lmdb_data_name = sys.argv[1]+'_data_lmdb'
	lmdb_label_name = sys.argv[1]+'_label_lmdb'
	
	if (os.path.exists(lmdb_label_name)):
		shutil.rmtree(lmdb_label_name)
	if (os.path.exists(lmdb_data_name)):
		shutil.rmtree(lmdb_data_name)
	
	Inputs = []
	Label1 = []
	Label2 = []
	Label3 = []
	Label4 = []

	finput = fileinput.input(data);
	for line in finput:
		entries = re.split(' ', line.strip())
		Inputs.append(entries[0])
		Label1.append(entries[1])
		Label2.append(entries[2])
		Label3.append(entries[3])
		Label4.append(entries[4])
	finput.close()

	print('Writing labels')

	# Size of buffer: 1000 elements to reduce memory consumption
	for idx in range(int(math.ceil(len(Label1)/1000.0))):
		in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))
		with in_db_label.begin(write=True) as in_txn:
			for label_idx, label_ in enumerate(Label1[(1000*idx):(1000*(idx+1))]):
				im_dat = caffe.io.array_to_datum(np.array([label_,Label2[label_idx],Label3[label_idx],Label4[label_idx]]).astype(float).reshape(4,1,1))
				in_txn.put('{:0>10d}'.format(1000*idx + label_idx), im_dat.SerializeToString())

				string_ = str(1000*idx+label_idx+1) + ' / ' + str(len(Label1))
				sys.stdout.write("\r%s" % string_)
				sys.stdout.flush()
		in_db_label.close()
	print('')

	print('Writing image data')

	for idx in range(int(math.ceil(len(Inputs)/1000.0))):
		in_db_data = lmdb.open(lmdb_data_name, map_size=int(1e12))
		with in_db_data.begin(write=True) as in_txn:
			for in_idx, in_ in enumerate(Inputs[(1000*idx):(1000*(idx+1))]):
				im = caffe.io.load_image(in_)
				
				if len(sys.argv)==4:
					im_min, im_max = im.min(), im.max()
					if im_max > im_min:
						# skimage is fast but only understands {1,3} channel images
						# in [0, 1].
						im_std = (im - im_min) / (im_max - im_min)
						resized_std = resize(im_std, (float(sys.argv[2]),float(sys.argv[3])), order=interp_order)
						resized_im = resized_std * (im_max - im_min) + im_min
						resized_im.astype(np.float32)
						im = resized_im
					else:
						ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
						dtype=np.float32)
						ret.fill(im_min)
						im = ret;
						
				im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
				in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())

				string_ = str(1000*idx+in_idx+1) + ' / ' + str(len(Inputs))
				sys.stdout.write("\r%s" % string_)
				sys.stdout.flush()
		in_db_data.close()
	print('')
else:
	print('Incorrect number of parameter or wrong train/val mode');
	print('Usage: First argument=<train or val mode> Second Argument(optional)= <New height> Third Argument(optional)=<New width>');
