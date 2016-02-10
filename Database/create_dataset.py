import numpy as np
import math
import sys
import os

pathtoimages = "nopath";
if sys.argv[1]=='supelec':
	pathtoimages = "/data/cei/VBags/"
elif sys.argv[1]=='gtl':
	pathtoimages = "/mnt/tale/"


if pathtoimages !="nopath":
	train_date=["150505"]
	val_date=["150522"]

	file_name=["train.txt","val.txt"]

	ref_gps=[49.103938,6.217547]#centre lac
	ref_utm=[296920,5442739]

	print("writing training document")
	t=open("train.txt","w")
	if (os.stat("train.txt").st_size != 0):
		os.remove('train.txt')
		t=open("train.txt","w")
	for date in train_date:
		print date
		path=pathtoimages+date

		f=open(path+"/image_auxilliary.csv","r")
		l = f.readlines()
		L=[s.strip().split(',') for s in l if s[0]!='%']

		
		for s in L:
			if (float(s[5])<1.8)&(float(s[5])>1.5)&(float(s[6])<0.21)&(float(s[4])-float(s[5])>math.pi/3)&(float(s[4])-float(s[5])<(math.pi/3)+(math.pi/180)):
				nb=int(float(s[1]))
				if nb/1000<10:
					tag="0"+str(nb/1000)
				else:
					tag=str(nb/1000)
				if nb%1000<10:
					index="00"+str(nb%1000)
				else:
					if nb%1000<100:
						index="0"+str(nb%1000)
					else:
						index=str(nb%1000)  
				
				t.write(path+"/00"+tag+"/0"+index+".jpg "+s[2]+" "+s[3]+" "+s[4]+"\n")
		f.close()
		print "fin "+date
	t.close()

	print("writing validation document")
	t=open("val.txt","w")
	if (os.stat("val.txt").st_size != 0):
		os.remove('val.txt')
		t=open("val.txt","w")
	for date in val_date:
		print date
		path=pathtoimages+date

		f=open(path+"/image_auxilliary.csv","r")
		l = f.readlines()
		L=[s.strip().split(',') for s in l if s[0]!='%']

		
		for s in L:
			if (float(s[5])<1.8)&(float(s[5])>1.5)&(float(s[6])<0.21)&(float(s[4])-float(s[5])>math.pi/3)&(float(s[4])-float(s[5])<(math.pi/3)+(math.pi/180)):
				nb=int(float(s[1]))
				if nb/1000<10:
					tag="0"+str(nb/1000)
				else:
					tag=str(nb/1000)
				if nb%1000<10:
					index="00"+str(nb%1000)
				else:
					if nb%1000<100:
						index="0"+str(nb%1000)
					else:
						index=str(nb%1000)  
				
				t.write(path+"/00"+tag+"/0"+index+".jpg "+s[2]+" "+s[3]+" "+s[4]+"\n")
		f.close()
		print "fin "+date
	t.close()
else:
	print('Please state which machine you are using (gtl or supelec)')	
