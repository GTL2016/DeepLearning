import numpy as np
import math
import sys
import os
import random
import matplotlib.pyplot as plt


#param
estmax=350 #m
nordmax=600 #max

step=2.5 #m
angl_pos=36 # 10 deg. zone
ref_gps=[49.101254,6.215672]#ref bas gauche
ref_utm=[296771.703361087,5442445.66151712]

def isValid(s):
	#checks whether the image is good enough to be used in database
	#todo : isvalid for other files with less classes
	return (float(s[6])<0.20)&(int(float(s[15]))==0)&(abs(float(s[5]))<2)

def getLabel(s):
	pos=[0,0,0]
	x=float(s[2])-float(ref_utm[0])
	y=float(s[3])-float(ref_utm[1])
	pos[0]=int(round(x/step))
	pos[1]=int(round(y/step))
	orient=math.fmod((float(s[4])-float(s[5])),(2*math.pi))
	pos[2]=int(round(orient*(angl_pos-1)/(2*math.pi)))
	
	return str(pos[0])+str(pos[1])+str(pos[2])

def load_classes():
	f=open("best_classes.txt","r")
	l = f.readlines()
	L=[s.strip().split(',') for s in l]
	classes=[]
	for s in L:
		classes+=s
	return classes

def reindex(classes):
	index={}
	ind=0
	for c in classes:
		index[c]=str(ind)
		ind+=1
	return index	

def writeToFile(s,label, b):
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
	angle = float(s[4])-float(s[5])
	proj_x = float(s[2])+math.cos(angle)*10;
	proj_y = float(s[3])+math.sin(angle)*10;
	b.write("/mnt/tale/"+date+"/00"+tag+"/0"+index+".jpg "+s[2]+" "+s[3]+" "+str(proj_x)+" "+str(proj_y)+"\n")


pathtoimages = "nopath";
if sys.argv[1]=='supelec':
	pathtoimages = "/data/cei/VBags/"
elif sys.argv[1]=='gtl':
	pathtoimages = "/mnt/tale/"


if pathtoimages !="nopath":
    
	dates = [s for s in os.listdir(pathtoimages) if os.path.isdir(pathtoimages+s)]
	dates.remove('150414')#bogus dataset
	dates.remove('131209')#bogus dataset
	dates.remove('151214')
	random.shuffle(dates)
	
	
	#creating file handles and line count
	file_name=["train","val","test"]
	best_classes=load_classes()
	N=306
	classes=random.sample(best_classes,N)
	ind=reindex(classes)
	lcount={} #counter for databases
	ccount=[0 for s in range(N)] #counter for classes
	files={} #database file file handles
	
	for bdd in file_name:
		files[bdd]=open("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt","w")
		#remove existing files
		if (os.stat("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt").st_size != 0):
			os.remove("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt")
			files[bdd]=open("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt","w")
		lcount[bdd]=0
	
	lmax={}
	lmax["train"]=300000
	lmax["val"]=10000
	lmax["test"]=5000
	
	for date in dates:
		
		print date
		path=pathtoimages+date

		f=open(path+"/image_auxilliary.csv","r")
		l = f.readlines()
		L=[s.strip().split(',') for s in l if s[0]!='%']
		if len(L[0])<15:
			f.close()
			continue

		for s in L:
			if not isValid(s): continue
			
			base = random.choice(file_name)
			current_file=files[base]
			label = getLabel(s)
			if label in classes:
				writeToFile(s,ind[label],current_file)
				lcount[base]+=1
				ccount[int(ind[label])]+=1

			if lcount[base]>=lmax[base]:
				files[base].close()
				file_name.remove(base)

			if (lcount["train"]>=lmax["train"])&(lcount["val"]>=lmax["val"])&(lcount["test"]>=lmax["test"]):
				break

		f.close()
		print "fin "+date
		
		if (lcount["train"]>=lmax["train"])&(lcount["val"]>=lmax["val"])&(lcount["test"]>=lmax["test"]):
			break

	plt.plot(range(N), ccount, '+')
	plt.axis([0, N, 0, max(ccount)])
	plt.savefig("classcount.png")
	print "My work here is done."
else:
	print('Please state which machine you are using (gtl or supelec)')
    
    

    

