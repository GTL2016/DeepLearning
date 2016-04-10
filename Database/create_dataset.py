import numpy as np
import math
import sys
import os
import random
import matplotlib.pyplot as plt

step=2.5 #m
angl_pos=36 # 10 deg. zone
ref_gps=[49.101254,6.215672]#ref bas gauche
ref_utm=[296771.703361087,5442445.66151712]

def isValid(s):
	#checks whether the image is good enough to be used in database
	#todo : isvalid for other files with less classes
	return (float(s[6])<0.20)&(int(float(s[15]))==0)&(abs(float(s[5]))<2)

def getDicretePos(s):
	pos=[0,0,0]
	x=float(s[2])-float(ref_utm[0])
	y=float(s[3])-float(ref_utm[1])
	pos[0]=int(round(x/step))
	pos[1]=int(round(y/step))
	orient=(float(s[4])-float(s[5]))%(2*math.pi)
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
	#angle = float(s[4])-float(s[5])
	angle = math.fmod((float(s[4])-float(s[5])),(2*math.pi))
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
	
	# Number of different classes in the datasets
	number_of_classes = 200
	# Max number of instances for each class
	max_in_class={}
	max_in_class["train"]=3
	max_in_class["val"]=math.floor(max_in_class["train"]/3)
	max_in_class["test"]=math.floor(max_in_class["train"]/3)
	# Size of each dataset accordingly
	lmax={}
	lmax["train"]=number_of_classes*max_in_class["train"]
	lmax["val"]=number_of_classes*max_in_class["val"]
	lmax["test"]=number_of_classes*max_in_class["test"]
	print "Considering "+str(number_of_classes)+" different classes in the train/val/test datasets"
	print "Train dataset: "+str(lmax["train"])+" images. "+str(max_in_class["train"])+" images per class"
	print "Val dataset: "+str(lmax["val"])+" images. "+str(max_in_class["val"])+" images per class"
	print "Test dataset: "+str(lmax["test"])+" images. "+str(max_in_class["test"])+" images per class"
	
	#creating file handles and line count
	file_name=["train","val","test"]
	best_classes=load_classes()
	classes=random.sample(best_classes,number_of_classes)
	print(classes[:])
	ind=reindex(classes)
	lcount={} # counter for databases
	ccount = {} # counter for classes
	files={} # database file file handles
	
	for bdd in file_name:
		files[bdd]=open("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt","w")
		#remove existing files
		if (os.stat("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt").st_size != 0):
			os.remove("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt")
			files[bdd]=open("/home/gpu_user/local/pfe/regression/DeepLearning/Database/"+bdd+".txt","w")
		lcount[bdd]=0
		ccount[bdd]=[0 for s in range(number_of_classes)] 
	
	# While the datasets are not full
	while (lcount["train"]<lmax["train"])or(lcount["val"]<lmax["val"])or(lcount["test"]<lmax["test"]):
	#for date in dates:
		date = random.choice(dates)
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
			pos = getDicretePos(s)
			# Only select classes among the best ones selected
			if (pos in classes) and (ccount[base][int(ind[pos])]<max_in_class[base]) :
				writeToFile(s,ind[pos],current_file)
				lcount[base]+=1
				ccount[base][int(ind[pos])]+=1
			# Closing current file if the maximum number of images is reached
			if lcount[base]>=lmax[base]:
				files[base].close()
				file_name.remove(base)
			# Stopping everything if databases filled or overfilled
			if (lcount["train"]>=lmax["train"])&(lcount["val"]>=lmax["val"])&(lcount["test"]>=lmax["test"]):
				break
		f.close()
		print "fin "+date
		print "Train :"+str(lcount["train"])+" images, Val :"+str(lcount["val"])+" images, Test :"+str(lcount["test"])
		if (lcount["train"]>=lmax["train"])&(lcount["val"]>=lmax["val"])&(lcount["test"]>=lmax["test"]):
			break
	#for bdd in file_name:
		## Plotting classes representation for the 3 datasets
		#plt.plot(range(number_of_classes), ccount[bdd], '+')
		#plt.axis([0, number_of_classes, 0, max(ccount[bdd])*3])
		#plt.savefig("classcount_"+bdd+".png")
	print "My work here is done."
else:
	print('Please state which machine you are using (gtl or supelec)')
    
    

    

