import math
import sys
import os

filenames=["train","val","test"]

t=open("mean.txt","w")
if (os.stat("mean.txt").st_size != 0):
	os.remove('mean.txt')
	t=open("mean.txt","w")
labelx_tot = 0
labely_tot = 0
i = 0
for f in filenames:
	data=open(f+".txt","r")
	l = data.readlines()
	L=[s.strip().split(' ') for s in l if s[0]!='%']
	for s in L:
		i = i+2
		labelx_tot = labelx_tot + float(s[1]), float(s[3])
		labely_tot = labely_tot + float(s[2]), float(s[4])
mean_x = labelx_tot/i
mean_y = labely_tot/i
print "Mean x = "+str(mean_x)
print "Mean y = "+str(mean_y)
t.write(str(mean_x)+"\n")
t.write(str(mean_y))
t.close
