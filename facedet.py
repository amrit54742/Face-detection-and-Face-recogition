import cv2 
import pickle
import numpy as np
#img = cv.imread('photos/dog2.jpg')
#cv.imshow('Dog',img)
#cv.waitKey(0)
face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

#blank=np.zeros((500,500),dtype='uint8')
recognizer = cv2.face.LBPHFaceRecognizer_create()


recognizer.read("trainer.yml")
labels = {"person_name": 1}

#with open("labels.pickle",'rb') as f:
 #   og_labels = pickle.load(f)
  #  labels = {v:k for k,v in og_labels.items()}



with open("labels.pickle", "rb") as f:
    unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
    og_labels = unpickler.load()
    labels = {v:k for k,v in og_labels.items()}








capture = cv2.VideoCapture(0)
while True:
    istrue, frame = capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=2)
        #print(x,y,w,h)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        
        # recognize ? deep learned model keras tensorflow pytorch scikit learn

        
        id_, conf =recognizer.predict(roi_gray)
        if conf>=50 and conf<= 85:
            print(id_)
            print(labels[id_])
        
        img_item="my_img.png"
        cv2.imwrite(img_item,roi_color)

            
        color=(255,0,0)
        stroke=2
        end_cord_x=x+w
        end_cord_y=y+h
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=2)

        
    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv2.destroyAllWindows() 

cv2.waitKey(0)