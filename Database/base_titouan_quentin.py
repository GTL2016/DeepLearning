import numpy as np
import math
import sys
import os
import random
import matplotlib.pyplot as plt

def isValid(s):
	#checks whether the image is good enough to be used in database
	#todo : isvalid for other files with less classes
	
	return (float(s[6])<0.20)&(int(float(s[15]))==0)&(float(s[5])>0)

def getLabel(s):
	ref_gps=[49.103938,6.217547]#centre lac
	ref_utm=[296920,5442739]

	x=float(s[2])-float(ref_utm[0])
	y=float(s[3])-float(ref_utm[1])
	theta=math.atan2(y,x)+math.pi
	label=(int(theta/(2*math.pi/1000)))
	#heading = int(float(s[4])*5*(1+math.pi)/math.pi)
	heading=int((float(s[4])+math.pi)*9/(2*math.pi))
	
	return str(label)+str(heading)

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
	b.write("tale/"+date+"/00"+tag+"/0"+index+".jpg "+str(label)+"\n")

pathtoimages = "nopath";
if sys.argv[1]=='supelec':
	pathtoimages = "/data/cei/VBags/"
elif sys.argv[1]=='gtl':
	pathtoimages = "/mnt/tale/"


if pathtoimages !="nopath":
    
	#remove existing files, uncomment if needed
	#todo : check that the file exists before removal
	#os.remove('train.txt')
	#os.remove('val.txt')
	#os.remove('test.txt')

	dates = [s for s in os.listdir(pathtoimages) if os.path.isdir(pathtoimages+s)]
	dates.remove('150414')#bogus dataset
	dates.remove('131209')#bogus dataset
	dates.remove('151214')
	random.shuffle(dates)
	
	
	#creating file handles and line count
	file_name=["train","val","test"]
	
	classes=random.sample(range(10000),500)
	lcount={} #counter for databases
	ccount=[0 for s in range(10000)] #counter for classes
	files={} #database file file handles
	
	for bdd in file_name:
		files[bdd]=open(bdd+".txt","w")
		lcount[bdd]=0
	
	for c in classes:
		ccount[c]=0

	lmax={}
	lmax["train"]=150000
	lmax["val"]=5000
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
			if int(label) in classes:
				writeToFile(s,label,current_file)
				lcount[base]+=1
				ccount[int(label)]+=1

			if lcount[base]>=lmax[base]:
				files[base].close()
				file_name.remove(base)

			if (lcount["train"]>=lmax["train"])&(lcount["val"]>=lmax["val"])&(lcount["test"]>=lmax["test"]):
				break

		f.close()
		print "fin "+date
		
		if (lcount["train"]>=lmax["train"])&(lcount["val"]>=lmax["val"])&(lcount["test"]>=lmax["test"]):
			break
	
	plt.plot(range(10000), ccount, '-')
	plt.axis([0, 10000, 0, 10000])
	plt.savefig("classcount.png")
	
	print "My work here is done."
else:
	print('Please state which machine you are using (gtl or supelec)')
    
    

    
