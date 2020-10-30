import sys
import time
argvs=sys.argv
argc=len(argvs)
import numpy as np
import scipy as sp
from scipy import signal
from skimage import data,img_as_float
from skimage.feature import blob_log
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
import cv2
import glob
from math import sqrt
num=0
num2=100
fn=3
filelist=sorted(glob.glob("*.raww"))
#covk=np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])
covk=np.full((fn,fn),1/(fn*fn))
covy=np.array([[1/9,1/9,1/9],[0,0,0],[-1/9,-1/9,-1/9]])
covx=np.array([[1/9,0,-1/9],[1/9,0,-1/9],[1/9,0,-1/9]])
covs=np.array([[1,2,3],[4,0,5],[6,7,8]])
#filelist=filelist[num:num2]
#print(filelist)
cv_imgs=[]
maximg=np.zeros((1024,1024),dtype='u2')
minimg=np.full((1024,1024),4098,dtype='u2')
counter=0
for fnum in filelist:
    f = open(fnum,mode='rb')
    img0=np.fromfile(f, dtype='u2',sep='').reshape(1024,1024)
    maximg=np.maximum(img0,maximg)
    minimg=np.minimum(img0,minimg)
#maxval=np.max(maximg)
#minval=np.min(minimg)
partimg=maximg-minimg
maxp=np.max(maximg)
minp=np.min(minimg)
#print(minp,maxp)
#print("out")
frmnum=0
imgh=np.zeros((4,512,512),dtype='u2')
imgc=np.zeros((4,510,510))
imgdx=np.zeros((8,510,510))
iflg=np.zeros((4,510,510),dtype='uint8')
for fnum in filelist:
    iflg=np.zeros((4,510,510),dtype='uint8')
    f = open(fnum,mode='rb')
    img0=np.fromfile(f, dtype='u2',sep='').reshape(1024,1024)
    img1=img0-minimg
    for i in range(2):
        for j in range(2):
            num=i*2+j
            imgh[num,:,:]=img1[i::2,j::2]
            imgc[num,:,:]=signal.convolve2d(covk,imgh[num,:,:],'valid')
            imgdx[num*2,:,:]=signal.convolve2d(covx,imgh[num,:,:],'valid')
            imgdx[num*2+1,:,:]=signal.convolve2d(covy,imgh[num,:,:],'valid')
            tmp=img_as_float(imgc[num,:,:])
            coord=peak_local_max(tmp, min_distance=1*fn)
            #print(len(coord))
            for dist in coord:
                #print(dist)
                if imgc[num,dist[0],dist[1]]>0:
                    iflg[num,dist[0],dist[1]]=1
                    #print (num,frmnum,dist[0],dist[1],imgc[num,dist[0],dist[1]],imgdx[num*2,dist[0],dist[1]],imgdx[num*2+1,dist[0],dist[1]])
    tmp2=np.zeros((512,512))
    tmp2[1:511,1:511]=iflg[0,:,:]
    tmp3=signal.convolve2d(covs,tmp2,'valid')
    for i in range(1,4):
        #print (i,len(list(zip(*np.where(iflg[0,:,:]==1)))))
        for ind in list(zip(*np.where(iflg[i,:,:]==1))):
            a=iflg[0,ind[0],ind[1]]
            b=tmp3[ind[0],ind[1]]
            if a==0 and b==0 :
                iflg[i,ind[0],ind[1]]=0
            elif b==1 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0]+1,ind[1]+1]=1
                imgc[i,ind[0]+1,ind[1]+1]=imgc[i,ind[0],ind[1]]
            elif b==2 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0],ind[1]+1]=1
                imgc[i,ind[0],ind[1]+1]=imgc[i,ind[0],ind[1]]
            elif b==3 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0]-1,ind[1]+1]=1
                imgc[i,ind[0]-1,ind[1]+1]=imgc[i,ind[0],ind[1]]
            elif b==4 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0]+1,ind[1]]=1
                imgc[i,ind[0]+1,ind[1]]=imgc[i,ind[0],ind[1]]
            elif b==5 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0]-1,ind[1]]=1
                imgc[i,ind[0]-1,ind[1]]=imgc[i,ind[0],ind[1]]
            elif b==6 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0]+1,ind[1]-1]=1
                imgc[i,ind[0]+1,ind[1]-1]=imgc[i,ind[0],ind[1]]
            elif b==7 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0],ind[1]-1]=1
                imgc[i,ind[0],ind[1]-1]=imgc[i,ind[0],ind[1]]
            elif b==8 :
                iflg[i,ind[0],ind[1]]=0
                iflg[i,ind[0]-1,ind[1]-1]=1
                imgc[i,ind[0]-1,ind[1]-1]=imgc[i,ind[0],ind[1]]
            
                #print(i,ind,tmp3[ind[0],ind[1]],iflg[0,ind[0],ind[1]])
    img8=imgc/maxp*255*iflg
    for ind in list(zip(*np.where(iflg[0,:,:]==1))):
        if img8[1,ind[0],ind[1]]>0 and img8[2,ind[0],ind[1]]>0 and img8[3,ind[0],ind[1]]>0 :
            print(frmnum,ind[0],ind[1],img8[0,ind[0],ind[1]],img8[1,ind[0],ind[1]],img8[2,ind[0],ind[1]],img8[3,ind[0],ind[1]])
    img9a0=(img8[0,:,:]+img8[3,:,:])/2
    img45a=(img8[1,:,:]+img8[2,:,:])/2
    atf90=np.where(img9a0>4,1,0)
    atf45=np.where(img45a>4,1,0)
    img9s0=(img8[0,:,:]-img8[3,:,:])*atf90*atf45
    img45s=(img8[1,:,:]-img8[2,:,:])*atf90*atf45
    img8i=img8.astype('uint8')
    img9a0i=img9a0.astype('uint8')
    img9s0i=img9s0.astype('uint8')
    img45ai=img45a.astype('uint8')
    img45si=img45s.astype('uint8')
    colorimg=cv2.merge((img9a0i,img9s0i,img45si))
    #colorimg=cv2.cvtColor(colorimg,cv2.COLOR_BGR2HSV_FULL)
    cv2.imshow("a",colorimg)   
    cv2.waitKey(1)
    frmnum=frmnum+1 
    #iflg=int(0)
#print(np.max(maximg),np.max(minimg),np.max(partimg))
#print(np.min(maximg),np.min(minimg),np.min(partimg))
maximg1=(maximg)/np.max(maximg)*255
minimg1=(minimg)/np.max(minimg)*255
partimg1=(partimg)/(np.max(partimg))*255
maximg1=maximg1.astype('uint8')
minimg1=minimg1.astype('uint8')
partimg1=partimg1.astype('uint8')
part0=partimg1[::2,::2]
part1=partimg1[1::2,::2]
part2=partimg1[::2,1::2]
partc=cv2.merge((part0,part1,part2))
partc=cv2.cvtColor(partc,cv2.COLOR_BGR2HSV)
#minimg0=minimg[::2,::2]
#print(maxval,minval,np.max(maximg),np.min(maximg))
#cv2.imshow("max",maximg1[::2,::2])
cv2.imshow("min",minimg1[::2,::2])
cv2.imshow("max-min",partimg1[::2,::2])
#cv2.imshow("color",partc)
cv2.waitKey(0)