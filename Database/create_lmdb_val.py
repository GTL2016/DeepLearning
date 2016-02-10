import lmdb
import re, fileinput, math
import numpy as np
import caffe
import sys

# Command line to check created files:
# python -mlmdb stat --env=./Downloads/caffe-master/data/liris-accede/val_score_lmdb/

data = '/home/yr/Documents/Project/val.txt'
lmdb_data_name = 'val_data_lmdb'
lmdb_label1_name = 'val_score_lmdb_1'
lmdb_label2_name = 'val_score_lmdb_2'
lmdb_label3_name = 'val_score_lmdb_3'

Inputs = []
Label1 = []
Label2 = []
Label3 = []

finput = fileinput.input(data);
for line in finput:
	entries = re.split(' ', line.strip())
	Inputs.append(entries[0])
	Label1.append(entries[1])
	Label2.append(entries[2])
	Label3.append(entries[3])
finput.close()

print('Writing label 1')

# Size of buffer: 1000 elements to reduce memory consumption
for idx in range(int(math.ceil(len(Label1)/1000.0))):
	in_db_label = lmdb.open(lmdb_label1_name, map_size=int(1e12))
	with in_db_label.begin(write=True) as in_txn:
		for label_idx, label_ in enumerate(Label1[(1000*idx):(1000*(idx+1))]):
			im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(1,1,1))
			in_txn.put('{:0>10d}'.format(1000*idx + label_idx), im_dat.SerializeToString())

			string_ = str(1000*idx+label_idx+1) + ' / ' + str(len(Label1))
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
	in_db_label.close()
print('')

print('Writing label 2')

# Size of buffer: 1000 elements to reduce memory consumption
for idx in range(int(math.ceil(len(Label2)/1000.0))):
	in_db_label = lmdb.open(lmdb_label2_name, map_size=int(1e12))
	with in_db_label.begin(write=True) as in_txn:
		for label_idx, label_ in enumerate(Label2[(1000*idx):(1000*(idx+1))]):
			im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(1,1,1))
			in_txn.put('{:0>10d}'.format(1000*idx + label_idx), im_dat.SerializeToString())

			string_ = str(1000*idx+label_idx+1) + ' / ' + str(len(Label2))
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
	in_db_label.close()
print('')

print('Writing label 3')

# Size of buffer: 1000 elements to reduce memory consumption
for idx in range(int(math.ceil(len(Label3)/1000.0))):
	in_db_label = lmdb.open(lmdb_label3_name, map_size=int(1e12))
	with in_db_label.begin(write=True) as in_txn:
		for label_idx, label_ in enumerate(Label3[(1000*idx):(1000*(idx+1))]):
			im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(1,1,1))
			in_txn.put('{:0>10d}'.format(1000*idx + label_idx), im_dat.SerializeToString())

			string_ = str(1000*idx+label_idx+1) + ' / ' + str(len(Label3))
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
			im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
			in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())

			string_ = str(1000*idx+in_idx+1) + ' / ' + str(len(Inputs))
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
	in_db_data.close()
print('')
