import cv2
import numpy as np
import math
cam=cv2.VideoCapture("1116.1.MOV")
skip=1
totf=cam.get(cv2.CAP_PROP_FRAME_COUNT)
fnum=0
mxskip=100
print(cam.get(cv2.CAP_PROP_FPS))
minskip=-100
methods = [ cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
while(cam.isOpened()):
    cam.set(cv2.CAP_PROP_POS_FRAMES,fnum)
    ret,frame=cam.read()
    if ret :
        bf,gf,rf=cv2.split(frame)
        color2=cv2.merge((bf,gf,rf))
        ag=np.average(gf)
        ab=np.average(bf)
        ar=np.average(rf)
    height,width=gf.shape
    if fnum ==0 :
        bf2=bf.copy()
    div=16
    #gf2=gf*ab/ag
    #3gf3=gf*ar/ag
    bf-=gf.astype(np.uint8)
    rf-=gf.astype(np.uint8)
    hdiv=int(div/2)
    wins=4
    for j in range(1,int(height/hdiv)-1):
        for i in range(1,int(width/hdiv-1)):
            orgx=i*hdiv  
            orgy=j*hdiv
            orgx1=int(orgx+div/2)
            orgy1=int(orgy+div/2)
            gray112=np.uint8(rf[orgy:orgy+div,orgx:orgx+div])
            if orgy-wins>0 and orgy+div+wins<height and orgx-wins>0 and orgx+div+wins<width:
                gray13=np.uint8(bf[orgy-wins:orgy+div+wins,orgx-wins:orgx+div+wins])
                res=cv2.matchTemplate(gray13,gray112, cv2.TM_CCOEFF_NORMED)
                min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
                distx=orgx1-wins+2*int(max_loc[0]-1)
                disty=orgy1-wins+2*int(max_loc[1]-1)
                distxo=orgx1-wins+2*int(max_loc[0]-1)
                distyo=orgy1-wins+2*int(max_loc[1]-1)
                rgbc=(100,0,int(max_val*255))
                rgbco=(100,0,int((1-min_val)*255))
                if max_val>0.9 and np.linalg.norm(max_loc)>math.sqrt(12):
                    cv2.arrowedLine(gf,(orgx1,orgy1),(distxo,distyo),rgbco,thickness=2,line_type=4)
    cv2.putText(color2,"frame="+str(fnum)+",skip="+str(skip),(100,100),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0))
    hcolor2=cv2.resize(color2,(int(width/4),int(height/4)))
    hrf=cv2.resize(rf,(int(width/4),int(height/4)))
    hbf=cv2.resize(bf,(int(width/4),int(height/4)))
    hgf=cv2.resize(gf,(int(width/4),int(height/4)))
    cv2.imshow('frame',hcolor2)
    cv2.imshow('red',hrf)
    cv2.imshow('blue',hbf)
    cv2.imshow('green',hgf)
    bf2=bf.copy()
    
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
cv2.destroyAllWindows()