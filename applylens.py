import cv2
import numpy as np
import dlib

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('Assets\images\sample images\girl.png')
glasses = cv2.imread("Assets\images\sunglasses\glass.png", cv2.IMREAD_UNCHANGED)
glasses = cv2.cvtColor(glasses, cv2.COLOR_BGRA2RGBA)

face_cascade = cv2.CascadeClassifier('Assets\Haarcascadefiles\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('Assets\Haarcascadefiles\haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('Assets\Haarcascadefiles\haarcascade_mcs_nose.xml')
# plt.imshow(img)

eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
noses = nose_cascade.detectMultiScale(img, 1.3, 5)
(x, y, w, h) = eyes[0]
glasses = cv2.resize(glasses, (w, h))
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
for i in range(glasses.shape[0]):
    for j in range(glasses.shape[1]):
     if (glasses[i, j, 3] > 0):
            img[y+i, x+j, :] = glasses[i, j, :-1]

    for (nx, ny, nw, nh) in noses:
        cv2.rectangle(img, (nx, ny), (nx+nw, ny+nh), (0, 255, 0), 2)

    # cv2.imshow("Img",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(img)
    plt.show()
