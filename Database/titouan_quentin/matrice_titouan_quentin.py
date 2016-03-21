import numpy as np
import math
import sys
import os
import random
import matplotlib.pyplot as plt


#param
estmax=350 #m
nordmax=600 #max
ratio=0.5
step=2.5 #m
angl_pos=36 # 10 deg. zone
ref_gps=[49.101254,6.215672]#ref bas gauche
ref_utm=[296771.703361087,5442445.66151712]
pathfigs='/home/gpu_user/local/pfe/regression/DeepLearning/Database/titouan_quentin'
n_total=0



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
	
	return pos

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
	
	#create matrix
	occ_mat=np.zeros((round(estmax/step),round(nordmax/step),angl_pos))
	
	
	
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
			pos = getLabel(s)
			occ_mat[pos[0],pos[1],pos[2]]+=1
			n_total+=1
		f.close()
		print "fin "+date
		
	occ_space=np.zeros((round(estmax/step),round(nordmax/step)))
	best_class_ever=occ_mat.max()
	#occ_mat/=occ_mat.max()
	classes=[]
	b=open("best_classes.txt","w")
	for i in range(occ_mat.shape[0]):
		for j in range(occ_mat.shape[1]):
			for k in range(occ_mat.shape[2]):
				if occ_mat[i,j,k]>ratio*best_class_ever:
					classes+=[str(i)+str(j)+str(k)]
					b.write(str(i)+str(j)+str(k)+"\n")
				else:
					occ_mat[i,j,k]=0
					
		
	
	n_kept=sum(sum(sum(occ_mat)))
	occ_mat/=occ_mat.max()
	fig = plt.figure()
	for i in range(angl_pos):
		occ_space+=occ_mat[:,:,i]
		fig.clear()
		plt.imshow(occ_mat[:,:,i])
		fig.savefig(pathfigs+'/occ'+str(i)+'.png')
		
	occ_space/=occ_space.max()
	
	
	fig.clear()
	plt.imshow(occ_space)
	fig.savefig('occ.png')

	print "ratio : "+str(ratio)+" ; n_kept : "+str(n_kept)+" ; nb of classes : "+str(len(classes)); 
	print "My work here is done."
else:
	print('Please state which machine you are using (gtl or supelec)')
    
    

    


