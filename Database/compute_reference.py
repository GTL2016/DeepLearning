import math
import sys
import os

filenames=["train","val","test"]

t=open("reference_point.txt","w")
if (os.stat("reference_point.txt").st_size != 0):
	os.remove('reference_point.txt')
	t=open("reference_point.txt","w")



## If we want the reference to best the mean point
#labelx_tot = 0
#labely_tot = 0
#i = 0
#for f in filenames:
	#data=open(f+".txt","r")
	#l = data.readlines()
	#L=[s.strip().split(' ') for s in l if s[0]!='%']
	#for s in L:
		#i = i+2
		#labelx_tot = labelx_tot + float(s[1])+ float(s[3])
		#labely_tot = labely_tot + float(s[2])+ float(s[4])
	#data.close()
#mean_x = labelx_tot/i
#mean_y = labely_tot/i
#print "Mean x = "+str(mean_x)
#print "Mean y = "+str(mean_y)
#t.write(str(mean_x)+"\n")
#t.write(str(mean_y))
#t.close()

# If we want the reference to best the min point (to have positive labels)
labelx_min = float('inf')
labely_min = float('inf')
for f in filenames:
	data=open(f+".txt","r")
	l = data.readlines()
	L=[s.strip().split(' ') for s in l if s[0]!='%']
	for s in L:
		labelx_min = min(labelx_min,float(s[1]),float(s[3]))
		labely_min = min(labely_min,float(s[2]),float(s[4]))
	data.close()
print "Min x = "+str(labelx_min)
print "Min y = "+str(labely_min)
t.write(str(labelx_min)+"\n")
t.write(str(labely_min))
t.close()

