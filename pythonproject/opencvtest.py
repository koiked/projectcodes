import cv2
import numpy as np
import dlib
#import skimage
#from skimage.viewer import ImageViewer
#from skimage import data
cap=cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_predict=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    maskim=np.zeros(frame.shape,dtype=np.uint8)
    wid,hei=gray.shape[:2]
    faces = cascade.detectMultiScale(gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
    if len(faces) > 0:
        x, y, w, h = faces[0, :]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face=dlib.rectangle(x,y,x+w,y+h)
        face_parts=face_predict(gray,face).parts()
        l_eye=np.array([[face_parts[42].x,face_parts[42].y],[face_parts[43].x,face_parts[43].y],[face_parts[44].x,face_parts[44].y],[face_parts[45].x,face_parts[45].y],[face_parts[46].x,face_parts[46].y],[face_parts[47].x,face_parts[47].y]])
        r_eye=np.array([[face_parts[36].x,face_parts[36].y],[face_parts[37].x,face_parts[37].y],[face_parts[38].x,face_parts[38].y],[face_parts[39].x,face_parts[39].y],[face_parts[40].x,face_parts[40].y],[face_parts[41].x,face_parts[41].y]])
        cv2.fillConvexPoly(maskim,l_eye,color=(255,255,255))
        cv2.fillConvexPoly(maskim,points=r_eye,color=(255,255,255))
        maskedeye=cv2.bitwise_and(frame,maskim)
        #maskedeye=cv2.resize(maskedeye,(hei*2,wid*2))
        cv2.imshow("mask",maskedeye)
        for i in face_parts:
            cv2.circle(frame, (i.x,i.y),1,(0,0,255),-1)
    #frame=cv2.resize(frame,(hei*2,wid*2))
    cv2.imshow("facedetect",frame)
    #cv2.imshow("gray",gray)
    k=cv2.waitKey(1)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()