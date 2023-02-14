import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils

import cv2
image = cv2.imread('Assets\images\kid.jpg')
classifier =cv2. CascadeClassifier('Assets\Haarcascadefiles\haarcascade_eye_tree_eyeglasses.xml')
boxes = classifier.detectMultiScale(image)
for box in boxes:
 x, y, width, height = box
 x2, y2 = x + width, y + height
 cv2.rectangle(image, (x, y), (x2, y2), (0,255,0), 5)
cv2.imshow('Glasses detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()