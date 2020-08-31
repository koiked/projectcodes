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
im =cv2.bitwise_not(im0)#invert image
gap = 1
borderSize = 20
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
#kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
#kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
#                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
#distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)#print(im0.shape)
#hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)#convert rgb>HSV (if you use color image it will be change)
vim = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
result = cv2.Canny(vim, 0, 255)
#cv2.imshow('Canny',result)
result2 =cv2.Laplacian(vim,cv2.CV_8U,15)
#cv2.imshow("Laplacian",result2)
#th, result3 = cv2.threshold(result2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
th, bw = cv2.threshold(vim, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #threshold procedure
morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
dist2=cv2.bitwise_and(vim,morph)
cv2.imshow('org and morph', dist2)
morph2=cv2.bitwise_and(morph,cv2.bitwise_not(result))
cv2.imshow('canny and morph', morph2)
#print(result2.dtype,result2.shape)

#blob detection
params = cv2.SimpleBlobDetector_Params()#blob parameters<from here>
params.minThreshold = 55;#threshold
params.maxThreshold = 100;
params.filterByArea = True# Filter by Area.
params.minArea = 200
params.filterByCircularity = True# Filter by Circularity
params.minCircularity = 0.0
params.filterByConvexity = True# Filter by Convexity
params.minConvexity = 0.0
params.filterByInertia = True# Filter by Inertia
params.minInertiaRatio = 0.0

#detector = cv2.SimpleBlobDetector_create(params)
#keypoints2 = detector.detect(im0) #blob detection here
#writing result of detection in image above
#with_keypoints= cv2.drawKeypoints(im0, keypoints2, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#simple contour
#contours0, hierarchy = cv2.findContours(bw2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#contimg=im0.copy()
#print(len(contours0))
#for i in range(len(contours0)):
#    if(hierarchy[0][i][3]==-1):
#        x, y, w, h = cv2.boundingRect(contours0[i])
#        cv2.rectangle(contimg, (x, y), (x+w, y+h), (255*i/len(contours0), 255, 255), 2)
#        cv2.drawContours(contimg,contours0[i],-1,(0,255,0),3)

#th2, bw2 = cv2.threshold(dist2, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #threshold procedure
#kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#morph2 = cv2.morphologyEx(bw2, cv2.MORPH_CLOSE, kernel2)

#morphorlogy detection <from here>
#cv2.drawContours(contimg,contours0,-1,(0,255,0),3)
#cv2.imshow('findcont', img3)
sure_bg = cv2.dilate(morph2,kernel,iterations=3)
dist = cv2.distanceTransform(morph2, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)
dist2 = cv2.distanceTransform(morph, cv2.DIST_L1, cv2.DIST_MASK_PRECISE)

ret, sure_fg = cv2.threshold(dist,0.1*dist.max(),255,0)
#display(dist,cmap='gray')
#print(dist.max(),dist.dtype)
cv2.imshow('dist', sure_fg)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
markers[unknown==255] = 0
markers = cv2.watershed(im,markers)
im0[markers == -1] = [255,0,0]
print(markers.shape, markers.dtype,markers.max())
display(dist)
display(dist2)
plt.show()
#dist2=cv2.distanceTransform(morph2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
#distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
#                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
#distborder2 = cv2.copyMakeBorder(dist2, borderSize, borderSize, borderSize, borderSize, 
#                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
#nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)
#nxcor2 = cv2.matchTemplate(distborder2, distTempl, cv2.TM_CCOEFF_NORMED)
#mn, mx, _, _ = cv2.minMaxLoc(nxcor)
#cv2.imshow("diff",cv2.bitwise_xor(nxcor,nxcor2))
#th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
#peaks8u = cv2.convertScaleAbs(peaks)
#contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
#peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
contor2=im0.copy()
#cv2.drawContours(contor2, contours, -1, (0, 0, 255), 2)
#for i in range(len(contours)):
#    x, y, w, h = cv2.boundingRect(contours[i])
#    _, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
#    cv2.circle(contor2, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
#    cv2.rectangle(contor2, (x, y), (x+w, y+h), (0, 255, 255), 2)

#cv2.imshow('contor', contor2)
#cv2.imshow('bw',cv2.bitwise_xor(morph,bw))
#cv2.imshow('threshold', morph)
cv2.imshow('watershed', im0)

#cv2.imshow('simplecont', contimg)
#cv2.imshow('dist2', im_with_keypoints2)

cv2.waitKey(0)
