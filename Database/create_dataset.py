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
	train_date=["140625"]
	val_date=["140625"]

	file_name=["train.txt","val.txt"]

	ref_gps=[49.103938,6.217547]#centre lac
	ref_utm=[296920,5442739]
	
	print("writing dataset documents")
	
	t1=open("train.txt","w")
	if (os.stat("train.txt").st_size != 0):
		os.remove('train.txt')
		t1=open("train.txt","w")
	
	t2=open("val.txt","w")
	if (os.stat("val.txt").st_size != 0):
		os.remove('val.txt')
		t2=open("val.txt","w")
	
	index_sample = 0
	number_test = 0
	number_train = 0
	for date in train_date:
		print date
		path=pathtoimages+date

		f=open(path+"/image_auxilliary.csv","r")
		l = f.readlines()
		L=[s.strip().split(',') for s in l if s[0]!='%']
		
		#parcours des lignes du csv
		for s in L:
			# s[5] correspond au pan, on veut 1.5 < pan < 1.8 (cote exterieur du lac, plus eviter les images d initilisation de la camera)
			# s[6] correspond au tilt on veut tilt < 0.21 (ne pas regarder le ciel)
			# s[4]-s[5] correspond a theta-pan (l angle de vision)
			# s[2] correspond a x, superieur a 296921 partie sud du lac evitee pour probleme d'initialisation et de retour a la base
			if (float(s[5])<1.8)&(float(s[5])>1.5)&(float(s[6])<0.21)&(float(s[4])-float(s[5])>math.pi/3)&(float(s[4])-float(s[5])<(math.pi/3)+(15*math.pi/180))&(float(s[2])>296921):
				index_sample = index_sample + 1
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
				
				if (index_sample % 5 == 0):
					number_test = number_test +1
					# ecriture du dataset test
					t2.write(path+"/00"+tag+"/0"+index+".jpg "+s[2]+" "+s[3]+" "+str(proj_x)+" "+str(proj_y)+"\n")
				else:
					number_train = number_train +1
					# ecriture du dataset train
					t1.write(path+"/00"+tag+"/0"+index+".jpg "+s[2]+" "+s[3]+" "+str(proj_x)+" "+str(proj_y)+"\n")
		f.close()
		print "fin "+date
	t1.close()
	t2.close()
	print "Total number of inputs = "+str(index_sample)
	print "Train dataset = "+str(number_train)
	print "Test dataset = "+str(number_test)
	
else:
	print('Please state which machine you are using (gtl or supelec)')	
