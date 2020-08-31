import cv2
import numpy as np
import matplotlib.pyplot as plt
def display(img,cmap=None):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap=cmap)
#im0 = cv2.imread('/home/koh/media/subtract.png')
im0 = cv2.imread('/home/koh/media/originalcropped.png')
#im0 = cv2.imread('/home/koh/media/edgedetection.png')
im =cv2.bitwise_not(im0)#invert image if background was white otherwise use below
#im=im0.copy()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
vim = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
result = cv2.Canny(vim,30, 150)
result0=result.copy()
result=cv2.dilate(result,kernel3,iterations=3)
th, bw = cv2.threshold(vim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #threshold procedure
morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
dist2=cv2.bitwise_and(vim,morph)
dist3=cv2.cvtColor(dist2,cv2.COLOR_GRAY2BGR)
morph2=cv2.bitwise_and(morph,cv2.bitwise_not(result))
sure_bg = cv2.dilate(morph2,kernel,iterations=3)
dist = cv2.distanceTransform(morph2, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
ret, sure_fg = cv2.threshold(dist,0.1*dist.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers1 = cv2.watershed(dist3,markers)
contor2=dist3.copy()
contor2[markers1 == -1] = [255,0,0]
data=[[0,0,0,0,0,0,0,0,0,0,0,0,0]]
for i in range(markers1.max()):
    if i>0 :
       cont0=morph2.copy()
       cont0[markers1!=i]=0
       cont0 = cv2.dilate(cont0,kernel2,iterations=3)
       #cv2.imshow('cont0',cont0)
       #cv2.waitKey(10)
       contours0, hierarchy = cv2.findContours(cont0,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       if len(contours0)>0:
           area = cv2.contourArea(contours0[0])
           if area>16 and area<10000:
            #x,y,w,h=cv2.boundingRect(contours0[0])
            #elp=cv2.fitEllipse(contours0[0])
            rect=cv2.minAreaRect(contours0[0])
            box=cv2.boxPoints(rect)
            box = np.int0(box)
            #ellipse = cv2.fitEllipse(contours0[0])
            #c,r=cv2.minEnclosingCircle(contours0[0])
            data0=[i,len(contours0),box[0][0],box[0][1],box[1][0],box[1][1],box[2][0],box[2][1],box[3][0],box[3][1],int(area),int(np.linalg.norm(box[1]-box[0],ord=2)),int(np.linalg.norm(box[2]-box[1],ord=2))]
            #print(data0)
            cv2.drawContours(contor2,[box],0,(0,0,255),2)
            data=np.append(data,[data0],axis=0)
            #cv2.ellipse(contor2,ellipse,(0,255,0),2)
            #cv2.rectangle(contor2, (x, y), (x+w, y+h), (0, 255, 255), 2)
            #cv2.circle(contor2, (int(c[0]), int(c[1])), int(r),(255, 0, 0), 3)
        
#    x, y, w, h = cv2.boundingRect(contours[i])
#    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
#    cv2.circle(contor2, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
#    

print(data.shape)
np.savetxt("output.dat",np.int0(data),fmt='%d')
display(contor2)
display(markers1)
#display(dist3)
fig= plt.figure(figsize=(10,8))
plt.hist(np.sqrt(data[:,10])*50/95,bins=25,range=(0,50),normed=True)
#plt.hist(data[:,11]*50/95,bins=25,range=(0,50),normed=True)
#plt.hist(data[:,12]*50/95,bins=25,range=(0,50),normed=True)

#display(dist2)
#display(result0)
#display(sure_fg)
display(morph2)
plt.show()
