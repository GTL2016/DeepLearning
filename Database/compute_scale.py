import math
import sys
import os

# This script computes the scaling factor of the data contained in the 'filenames files'
# This factor is computed so that the rescaled values are distributed in an interval of size 'label_range'

filenames=["train","val","test"]
label_range = 10

t=open("scale.txt","w")
if (os.stat("scale.txt").st_size != 0):
	os.remove('scale.txt')
	t=open("scale.txt","w")
labelx_min = float('inf')
labelx_max = -float('inf')
labely_min = float('inf')
labely_max = -float('inf')
for f in filenames:
	data=open(f+".txt","r")
	l = data.readlines()
	L=[s.strip().split(' ') for s in l if s[0]!='%']
	for s in L:
		labelx_max = max(labelx_max, float(s[1]), float(s[3]))
		labelx_min = min(labelx_min, float(s[1]), float(s[3]))
		labely_max = max(labely_max, float(s[2]), float(s[4]))
		labely_min = min(labely_min, float(s[2]), float(s[4]))
scale = label_range/(max(labelx_max-labelx_min,labely_max-labely_min))
print "Scaling factor = "+str(scale)
t.write(str(scale))
t.close()
