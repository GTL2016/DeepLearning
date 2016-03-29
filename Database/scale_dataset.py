import numpy as np
import math
import sys
import os
from pylab import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


f=open('scale.txt',"r")
lines = f.readlines()
scale = 0
for line in lines:
	scale = float(line)
f.close()


f=open('reference_point.txt',"r")
line = f.readline()
ref_x = float(line)
line = f.readline()
ref_y = float(line)
f.close()

filenames=["train","val","test"]

for name in filenames:
	t = open(name+"_rescaled.txt","w")
	if (os.stat(name+"_rescaled.txt").st_size != 0):
		os.remove(name+"_rescaled.txt")
		t=open(name+"_rescaled.txt","w")
	
	data=open(name+".txt","r")
	l = data.readlines()
	L=[s.strip().split(' ') for s in l if s[0]!='%']
	for s in L:
		t.write(s[0] + " " +str((float(s[1])-ref_x)*scale)+" "+str((float(s[2])-ref_y)*scale)+" "+str((float(s[3])-ref_x)*scale)+" "+str((float(s[4])-ref_y)*scale)+"\n")
	t.close()

