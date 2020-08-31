import sys
import time
import numpy as np
import cv2
import glob
from math import sqrt
argvs=sys.argv
argc=len(argvs)
filelist=sorted(glob.glob(argvs[1]+"*.ppm"))

#cv_imgs=[]
counter=0
for img in filelist:
    n=cv2.imread(img)
    nb,ng,nr=cv2.split(n)
    print(n.shape)
    #n00=n[::2,::2]
    #n11=n[1::2,1::2]
    #n01=n[::2,1::2]
    #n10=n[1::2,::2]
    #print(n00.shape)

    av00=np.average(nr)
    av01=np.average(nb)
    av10=np.average(ng)
    #av11=np.average(n11)
    avdats=np.array([[counter,av00,av01,av10]])
    #cv_imgs.append(n)
    if counter==0:
        avs=avdats
        counter+=1
    else:
        avs=np.append(avs,avdats,axis=0)
        counter+=1
np.savetxt(argvs[1]+".dat",avs)
