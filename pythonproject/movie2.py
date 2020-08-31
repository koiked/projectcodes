import cv2
import numpy as np
import math
skip=1
fnum=0
mxskip=100
minskip=-100
stt=1600
edf=1640
yt=0.4
yb=0.7
cam=cv2.VideoCapture("2018-10-10 10-10.MOV")
fourcc=cv2.VideoWriter_fourcc(*'XVID')
totf=cam.get(cv2.CAP_PROP_FRAME_COUNT)
print(cam.get(cv2.CAP_PROP_FPS))
print(totf)
cam.set(cv2.CAP_PROP_POS_FRAMES,stt)
ret,oframe=cam.read()
oh,ow,dep=oframe.shape
frame=oframe[int(oh*yt):int(oh*yb),:,:].copy()
h2,w2,dp2=frame.shape
print(h2,w2)
out=cv2.VideoWriter('testout.avi',fourcc,30.0,(w2,h2))
bf,gf,rf0=cv2.split(frame)
ar=np.zeros(bf.shape,dtype=float)
ab=np.zeros(bf.shape,dtype=float)
ag=np.zeros(bf.shape,dtype=float)
#nr=np.zeros(bf.shape,dtype=float)
#nb=np.zeros(bf.shape,dtype=float)
#ng=np.zeros(bf.shape,dtype=float)
#xr=np.zeros(bf.shape,dtype=float)
#xb=np.zeros(bf.shape,dtype=float)
#xg=np.zeros(bf.shape,dtype=float)
sdg=np.zeros(bf.shape,dtype=float)
sdr=np.zeros(bf.shape,dtype=float)
sdb=np.zeros(bf.shape,dtype=float)
 #dx,dy=cv2.phaseCorrelate(gray12,gray11)
height,width=rf0.shape
print(height,width)
dxmin=0
dxmax=0
dymin=0
dymax=0
for fnum in range(stt,int(edf)):
    cam.set(cv2.CAP_PROP_POS_FRAMES,fnum)
    ret,oframe=cam.read()
    oh,ow,dep=oframe.shape
    frame=oframe[int(oh*yt):int(oh*yb),:,:].copy()  
    if ret :
        bf,gf,rf=cv2.split(frame)
    d,v=cv2.phaseCorrelate(rf0.astype(np.float32),rf.astype(np.float32))
    dx,dy=d
    #print(dx,dy)
    if dxmin>dx:
       dxmin=dx
    if dxmax<dx:
        dxmax=dx
    if dymin>dy:
        dymin=dy
    if dymax<dy:
        dymax=dy
    M=np.float32([[1,0,-dx],[0,1,-dy]])
    rf=cv2.warpAffine(rf,M,(width,height))
    bf=cv2.warpAffine(bf,M,(width,height))
    gf=cv2.warpAffine(gf,M,(width,height))
    #if fnum == stt:
    #    xr=rf
    #    nr=rf
    #    xb=bf
    #    nb=bf
    #    xg=gf
    #    ng=gf
    #else :
    #    xr=np.where(rf>xr,rf,xr)
    #    xb=np.where(bf>xb,bf,xb)
    #    xg=np.where(gf>xg,gf,xg)
    #    nr=np.where(rf<nr,rf,nr)
    #    nb=np.where(bf<nb,bf,nb)
    #    ng=np.where(gf<ng,gf,ng)
    ar=ar+rf/(edf-stt)
    sdr+=np.square(rf)/(edf-stt)
    ab=ab+bf/(edf-stt)
    sdb+=np.square(bf)/(edf-stt)
    ag=ag+gf/(edf-stt)
    sdg+=np.square(gf)/(edf-stt)
    if fnum%20 == 0: 
        print(fnum)  
print(dxmin,dxmax,dymin,dymax)
sdr-=np.square(ar)
sdb-=np.square(ab)
sdg-=np.square(ag)
sdr=np.where(sdr>0,np.sqrt(sdr),1)
sdb=np.where(sdb>0,np.sqrt(sdb),1)
sdg=np.where(sdg>0,np.sqrt(sdg),1)
hab=cv2.resize(ab.astype(np.uint8),(int(width/4),int(height/4)))
har=cv2.resize(ar.astype(np.uint8),(int(width/4),int(height/4)))
hag=cv2.resize(ag.astype(np.uint8),(int(width/4),int(height/4)))
cv2.imshow("average-blue",hab ) 
cv2.imshow("average-red",har)   
cv2.imshow("average-green",hag)
fnum=stt
while(cam.isOpened()):
    cam.set(cv2.CAP_PROP_POS_FRAMES,fnum)
    ret,oframe=cam.read()
    
    if ret :
        oh,ow,dep=oframe.shape
        frame=oframe[int(oh*yt):int(oh*yb),:,:].copy()  
        bf,gf,rf=cv2.split(frame)
        color2=cv2.merge((bf,gf,rf))
    d,v=cv2.phaseCorrelate(ar.astype(np.float32),rf.astype(np.float32))
    dx,dy=d
    M=np.float32([[1,0,-dx],[0,1,-dy]])
    rf=cv2.warpAffine(rf,M,(width,height))
    bf=cv2.warpAffine(bf,M,(width,height))
    gf=cv2.warpAffine(gf,M,(width,height))
    #bfd=np.where(ab/xb>nb/ab,(xb-bf).astype(np.uint8),(bf-nb).astype(np.uint8))
    #rfd=np.where(ar/xr>nr/ar,(xr-rf).astype(np.uint8),(rf-nr).astype(np.uint8))
    #gfd=np.where(ag/xg>ng/ag,(xg-gf).astype(np.uint8),(gf-ng).astype(np.uint8))
    rfd=np.abs(rf-ar).astype(np.uint8)  
    bfd=np.abs(bf-ab).astype(np.uint8)
    gfd=np.abs(gf-ag).astype(np.uint8)
    rfm=np.where(rfd>4*sdr,255,0).astype(np.uint8)
    bfm=np.where(bfd>4*sdb,255,0).astype(np.uint8)
    gfm=np.where(gfd>4*sdg,255,0).astype(np.uint8)
    rf=cv2.bitwise_and(rf,rf,mask=rfm)
    bf=cv2.bitwise_and(bf,bf,mask=bfm)
    gf=cv2.bitwise_and(gf,gf,mask=gfm)
    co3=cv2.merge((bf,gf,rf))
    cv2.putText(color2,"frame="+str(fnum)+",skip="+str(skip),(100,100),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0))
    hcolor2=cv2.resize(color2,(int(width/2),int(height/2)))
    hrf=cv2.resize(rf,(int(width/2),int(height/2)))
    hbf=cv2.resize(bf,(int(width/2),int(height/2)))
    hgf=cv2.resize(gf,(int(width/2),int(height/2)))
    hcolor3=cv2.merge((hbf,hgf,hrf))
    cv2.imshow('frame',hcolor2)
    cv2.imshow('red',hrf)
    cv2.imshow('blue',hbf)
    cv2.imshow('green',hgf)
    cv2.imshow('clopc',hcolor3)
    cv2.imwrite("test.png",hcolor3)
    bf2=bf.copy()
    out.write(co3)
    cmd=cv2.waitKey(25)
    if cmd&0xFF == ord('q'):
        break
    if cmd&0xFF == ord('f'):
        if skip<mxskip:
            skip+=1
    if cmd&0xFF == ord('r'):
        if skip>minskip:
            skip-=1
    if cmd&0xFF == ord('p'):
        skip=0
    if cmd&0xFF == ord('e'):
        if fnum<totf:
            fnum+=1
    if cmd&0xFF == ord('d'):
        if fnum>0:
            fnum-=1    
    if skip>0:
        if totf-skip>fnum :
            fnum+=skip
        else :
            fnum=0
    else:
        if fnum+skip>1:
            fnum+=skip
        else:
            fnum=totf-1
   
cam.release()
out.release()
cv2.destroyAllWindows()