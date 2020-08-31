import sys
import time
import numpy as np
import cv2
import glob
from math import sqrt
argvs=sys.argv
argc=len(argvs)
filelist=sorted(glob.glob(argvs[1]+"*.raw"))
#cv_imgs=[]
width=640
height=480
counter=0
for img in filelist:
    fd=open(img,'rb')
    print("openfile",img)
    f=np.fromfile(fd,dtype=np.uint8,count=height*width*3)
    f1=f[::3]
    f2=f[1::3]
    f3=f[2::3]
    n=f3.reshape((height,width))
    fd.close()
    #n=cv2.imread(img)
    #print(n.shape)
    #n00=n[::2,::2]
    #n11=n[1::2,1::2]
    #n01=n[::2,1::2]
    #n10=n[1::2,::2]
    #print(n00.shape)

    av00=np.average(f1)
    av01=np.average(f2)
    av10=np.average(f3)
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
