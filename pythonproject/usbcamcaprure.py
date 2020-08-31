
import cv2
#import openpiv.tools
#import openpiv.process
#import openpiv.scaling
#import openpiv.validation
#import openpiv.filters
import numpy as np
import matplotlib as plt
import sys
import time
cap=cv2.VideoCapture(0)#fps=cap.get(cv2.CAP_PROP_FPS)
#gain=cap.get(cv2.CAP_PROP_GAIN)
#gain=cap.get(cv2.CAP_PROP_FPS)
#print(fps,gain)
#cap.set(cv2.CAP_PROP_FPS,60)
#cap.set(cv2.CAP_PROP_GAIN,-20)
#cap=cv2.VideoCapture(0)
#fps=cap.get(cv2.CAP_PROP_FPS)
#gain=cap.get(cv2.CAP_PROP_GAIN)
#print(fps,gain)
cascade=cv2.CascadeClassifier(r"C:\Users\kohik\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_upperbody.xml")
count=0
while True:
    ret,frame=cap.read()
    bf,gf,rf=cv2.split(frame)#print(ret)as
    if count==0:
        gray2=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        gray2=gray.copy()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bf=cv2.cvtColor(bf,cv2.COLOR_GRAY2BGR)
    rf=cv2.cvtColor(rf,cv2.COLOR_GRAY2BGR)
    gf=cv2.cvtColor(gf,cv2.COLOR_GRAY2BGR)

    #stat=time.time()
    #cv2.imwrite("1st.bmp",gray)
    #cv2.imwrite("2nd.bmp",gray2)
    #cv2.imwrite("3rd.bmp",bf)
    #cv2.imwrite("4th.bmp",rf)
    #gray01=openpiv.tools.imread('1st.bmp')
    #gray02=openpiv.tools.imread('2nd.bmp')
    #bf0=openpiv.tools.imread('3rd.bmp')
    #rf0=openpiv.tools.imread('4th.bmp')
    gray01=gray.copy()
    gray02=gray2.copy()
    #bf0=bf.copy()
    #rf0=rf.copy()
    #gray01=bf.copy()
    #gray02=rf.copy()
    #if count%2==0:
    face=cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3,minSize=(30,30))
    height,width=gray01.shape
    #time2=time.time()-stat
    div=32
    hdiv=int(div/2)
    wins=4  
    methods = [ cv2.TM_CCOEFF_NORMED,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF_NORMED]
      
    for j in range(1,int(height/hdiv)-1):
        for i in range(1,int(width/hdiv-1)):
            orgx=i*hdiv  
            orgy=j*hdiv
            orgx1=int(orgx+div/2)
            orgy1=int(orgy+div/2)
            gray11=np.float32(gray01[orgy:orgy+div,orgx:orgx+div])
            gray112=np.uint8(gray01[orgy:orgy+div,orgx:orgx+div])
            #avrg=gray110.mean()
            #print(avrg)
            gray12=np.float32(gray02[orgy:orgy+div,orgx:orgx+div])
            if orgy-wins>0 and orgy+div+wins<height and orgx-wins>0 and orgx+div+wins<width:
                gray13=np.uint8(gray02[orgy-wins:orgy+div+wins,orgx-wins:orgx+div+wins])
                count2=0
                for meth in methods:
                    res=cv2.matchTemplate(gray13,gray112,meth)
                    min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(res)
                    distx=orgx1-wins+int(max_loc[0])
                    disty=orgy1-wins+int(max_loc[1])
                    distxo=orgx1-wins+int(max_loc[0])
                    distyo=orgy1-wins+int(max_loc[1])
                    rgbc=(100,0,int(max_val*255))
                    rgbco=(100,0,int((1-min_val)*255))
                    if count2==0 and max_val>0.5:
                        cv2.arrowedLine(rf,(orgx1,orgy1),(distx,disty),rgbc,thickness=2,line_type=4)
                    if count2==1 and max_val>0.5:
                        cv2.arrowedLine(bf,(orgx1,orgy1),(distx,disty),rgbc,thickness=2,line_type=4)
                    if count2==2 and min_val<0.5:
                        cv2.arrowedLine(gf,(orgx1,orgy1),(distxo,distyo),rgbco,thickness=2,line_type=4)
                    count2+=1


            #gray11=np.zeros((div+2*wins,div+2*wins),dtype=float)
            #gray11[np.where(gray11<1)]=avrg
            #gray11[wins:wins+div,wins:wins+div]=gray110
            #gray11=np.float32(gray11)
            #dx,dy=cv2.phaseCorrelate(gray12,gray11)
            #idx=int(dx[0])

            #idy=int(dx[1])
            #idx=0
            #idy=0
            #print(dx,dy)
            
            #distx=int(orgx1+idx)
            #disty=int(orgy1+idy)  
            #cv2.imshow("org",cv2.resize(np.uint8(gray11),(256,256)))
            #cv2.imshow("dist",cv2.resize(np.uint8(gray12),(256,256))) 
            #hsv=(dy,1,1)
            #rgbc=(100,0,int(dy*255))
            #rgbc=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            #if dy > 0.2:
            #    cv2.arrowedLine(frame,(orgx1,orgy1),(distx,disty),rgbc,thickness=2,line_type=4)
            #cv2.waitKey(0)
    
    #u,v,sig2noise =openpiv.process.extended_search_area_piv(gray01.astype(np.int32),gray02.astype(np.int32),window_size=32,overlap=16,dt=0.02,search_area_size=32,sig2noise_method='peak2peak')
    #u2,v2,sig2noise2 =openpiv.process.extended_search_area_piv(rf0.astype(np.int32),bf0.astype(np.int32),window_size=32,overlap=16,dt=0.02,search_area_size=32,sig2noise_method='peak2peak')
    #x,y=openpiv.process.get_coordinates(image_size=gray01.shape,window_size=32,overlap=16)
    #time3=time.time()-stat-time2
    #x2,y2=openpiv.process.get_coordinates(image_size=rf0.shape,window_size=32,overlap=16)
    #u,v,mask =openpiv.validation.sig2noise_val(u,v,sig2noise,threshold=1.3)
    #u2,v2,mask2 =openpiv.validation.sig2noise_val(u2,v2,sig2noise2,threshold=1.3)
    #u,v =openpiv.filters.replace_outliers(u,v,method='localmean',max_iter=10,kernel_size=2)
    #u2,v2 =openpiv.filters.replace_outliers(u2,v2,method='localmean',max_iter=10,kernel_size=2)
    #x,y,u,v=openpiv.scaling.uniform(x,y,u,v,scaling_factor=1.0)
    #x2,y2,u2,v2=openpiv.scaling.uniform(x2,y2,u2,v2,scaling_factor=1.0)
    #time4=time.time()-stat-time2-time3
    #plt.quiver(x,y,u,v)
    #plt.quiver(x2,y2,u2,v2)
    #fps=cap.get(cv2.CAP_PROP_POS_MSEC)
    #gain=cap.get(cv2.CAP_PROP_FPS)
    #expo=cap.get(cv2.CAP_PROP_POS_FRAMES)
    #print(fps,gain,expo)
    #edges=cv2.Canny(bf,70,200)
    #time5=time.time()-stat-time2-time3-time4
    for(x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,200),3)
    #cv2.putText(frame,tirme2,(0,0),cv2.FONT_ITALIC,0.6,(255,255,255))
    #print(time2,time3,time4,time5)
    #cv2.imshow("camera",edges)
    cv2.imshow("facedetect",frame)
    cv2.imshow("red",rf)
    cv2.imshow("blue",bf)
    cv2.imshow("green",gf)
    k=cv2.waitKey(1)
    count +=1
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()