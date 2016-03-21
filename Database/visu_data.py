import numpy as np
import math
import sys
import os
from pylab import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


file_name = "noname";
if sys.argv[1]=='train':
	file_name = "train.txt"
elif sys.argv[1]=='val':
	file_name = "val.txt"
elif sys.argv[1]=='test':
	file_name = "test.txt"

if file_name !="noname":
	f=open(file_name,"r")
	l = f.readlines()
	L=[s.strip().split(' ') for s in l]
	
	nb_lines = 0;
	for s in L:
		nb_lines = nb_lines+1;
	
	
	i = 0
	labels = np.zeros((nb_lines,4))
	for s in L:
		lab = np.array([[float(s[1]), float(s[2]), float(s[3]), float(s[4])]])
		labels[i,:]=lab
		i=i+1

	print(labels.shape)

	# Plotting position and predicted position
	figure(1)
	scatter(labels[:,0],labels[:,1],s=25,c='g',marker='+')
	scatter(labels[:,2],labels[:,3],s=25,c='r',marker='+')

	# Show all figures
	show()
else:
	print('Please state which dataset you want to plot (train, val or test)')
