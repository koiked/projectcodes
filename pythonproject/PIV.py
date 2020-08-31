import cv2
import numpy as np
import matplotlib as plt
import sys
import time
import glob
argvs=sys.argv
argc=len(argvs)
num=int(argvs[2])
num2=int(argvs[3])
filelist=sorted(glob.glob(argvs[1]+"*.ppm"))
div=int(argvs[4])
wins=int(argvs[5])
#filelist=sorted(glob.glob(argvs[1]))
filelist=filelist[num2:num]
cv_imgs=[]
counter=0
for img in filelist:
    n=cv2.imread(img)
    nb,ng,nr=cv2.split(n)
    cv_imgs.append(n)
    if counter==0:
        maskb=nb
        maskr=nr
        counter+=1
    else:
        cb1=cv2.compare(maskb,nb,cv2.CMP_LT)
        cr1=cv2.compare(maskr,nr,cv2.CMP_LT)
        cb2=cv2.bitwise_not(cb1)
        cr2=cv2.bitwise_not(cr1)
        mb1=cv2.bitwise_and(maskb,cb1)
        mr1=cv2.bitwise_and(maskr,cr1)
        mb2=cv2.bitwise_and(nb,cb2)
        mr2=cv2.bitwise_and(nr,cr2)
        maskb=cv2.bitwise_or(mb1,mb2)
        maskr=cv2.bitwise_or(mr1,mr2)
        counter+=1
for cvimg in cv_imgs:
    bcv,gcv,rcv=cv2.split(cvimg)
    height,width=gcv.shape
    bcv=cv2.subtract(bcv,maskb)
    rcv=cv2.subtract(rcv,maskr)
    height,width=rcv.shape
    #div=32
    hdiv=int(div/2)
    #wins=4  
    methods = [ cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
    vect=np.array([[0,0,0,0,0,0]])
    #print(vect.shape)
    for j in range(1,int(height/hdiv)-1):
        for i in range(1,int(width/hdiv-1)):
            orgx=i*hdiv  
            orgy=j*hdiv
            orgx1=int(orgx+div/2)
            orgy1=int(orgy+div/2)
            gray11=np.float32(bcv[orgy:orgy+div,orgx:orgx+div])
            gray112=np.uint8(bcv[orgy:orgy+div,orgx:orgx+div])
            orgave=np.average(gray112)
            gray12=np.float32(rcv[orgy:orgy+div,orgx:orgx+div])
            if orgy-wins>0 and orgy+div+wins<height and orgx-wins>0 and orgx+div+wins<width:
                gray13=np.uint8(rcv[orgy-wins:orgy+div+wins,orgx-wins:orgx+div+wins])
                res=cv2.matchTemplate(gray13,gray112,cv2.TM_CCOEFF_NORMED)
                min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
                vecd=np.array([[orgx1,orgy1,max_loc[0],max_loc[1],max_val,orgave]])
                vect=np.append(vect,vecd,axis=0)
                distx=orgx1-wins+int(max_loc[0])
                disty=orgy1-wins+int(max_loc[1])
                distxo=orgx1-wins+int(max_loc[0])
                distyo=orgy1-wins+int(max_loc[1])
                rgbc=(100,int(orgave*4),int(max_val*255))
                rgbco=(100,0,int((1-min_val)*255))
                cv2.arrowedLine(cvimg,(orgx1,orgy1),(distx,disty),rgbc,thickness=2,line_type=4)
                          
    cv2.imshow("vectors",cvimg)
    #print(vect.shape)
    np.savetxt(argvs[1]+"{0:03d}".format(counter)+"vec.dat",vect)
    k=cv2.waitKey(1)
    counter +=1
    if k==27:
        break
#cap.release()
cv2.destroyAllWindows()