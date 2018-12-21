from imutils import face_utils
import dlib
import cv2
import numpy as np

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)

while True:
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #shape[36] is left point of left eye
        #shape[45] is right point of right eye
        ab,cb=shape[36]
        ab2,cb2=shape[45]        
        cv2.line(image,(ab,cb),(ab2,cb2),(255,0,0),5)        
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)        
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)        
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    cv2.imshow("Output", image)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()