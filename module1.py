
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import time
#import keras
#import tensorflow as tf


camera = cv2.VideoCapture(0)
def cam():
    while True:
        #time.sleep(10)
        return_value,image = camera.read()
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('image',gray)
        time.sleep(5)
        #if cv2.waitKey(1)& 0xFF == ord('s'):
        cv2.imwrite('test.jpg',image)
        #break

        time.sleep(5)
        txtfiles = [] 
        for file in glob.glob("*.jpg"):
            txtfiles.append(file)
    
        for ix in txtfiles:
            img = cv2.imread(ix,cv2.IMREAD_COLOR)
            imgtest1 = img.copy()
            imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)   
            facecascade = cv2.CascadeClassifier('D:\Kinshuk Gupta\Documents\Hack\haarcascade_frontalface_default.xml')    
            eye_cascade = cv2.CascadeClassifier('D:\Kinshuk Gupta\Documents\Hack\haarcascade_eye.xml')
   
            faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=5) 
            print("Total number of Faces found",len(faces))
    
            if(len(faces)>1):
                print("Make sure there is no one else around you")
                time.sleep(5)

            for (x, y, w, h) in faces:
                face_detect = cv2.rectangle(imgtest, (x, y), (x+w, y+h), (255, 0, 255), 2)
                roi_gray = imgtest[y:y+h, x:x+w]
                roi_color = imgtest[y:y+h, x:x+w]        
                plt.imshow(face_detect)
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    eye_detect = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,255),2)
                    plt.imshow(eye_detect)

        if cv2.waitKey(1)& 0xFF == ord('s'):
            camera.release()
            cv2.destroyAllWindows()














cam()