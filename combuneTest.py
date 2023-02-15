import cv2
import numpy as np
import dlib
from imutils import face_utils

# Load the cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(
    'Assets\Haarcascadefiles\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(
    'Assets\Haarcascadefiles\haarcascade_eye.xml')

# Load the dlib model for facial landmarks detection
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the image and the sunglasses
image = cv2.imread('Assets\images\sample images\sample4.jpg')
glass_img = cv2.imread('Assets\images\sunglasses\greenGlass.png')

# Detect faces in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Iterate over each face and apply the sunglasses

if len(faces) > 0:
    for (x, y, w, h) in faces:
        # Create a region of interest for the face and the eyes

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Detect facial landmarks
        rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        landmarks = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        # Calculate the width of the sunglasses
        glasses_width = 2.16 * abs(landmarks[16][0] - landmarks[1][0])

        # Resize the sunglasses to the width of the face
        scaling_factor = glasses_width / glass_img.shape[1]
        overlay_glasses = cv2.resize(
            glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Calculate the position of the sunglasses
        x = landmarks[1][0]
        y = landmarks[24][1]

        # Adjust the position of the sunglasses
        x -= 0.26 * overlay_glasses.shape[1]
        y += 0.85 * overlay_glasses.shape[0]

        # Overlay the sunglasses on the face
        h, w, _ = overlay_glasses.shape
        overlay_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(overlay_mask, cv2.convexHull(
            landmarks[[0, 1, 16], :]), 255, 8, 0)
        overlay_mask = cv2.erode(overlay_mask, np.ones(
            (5, 5), np.uint8), iterations=1)
        overlay_mask = cv2.dilate(
            overlay_mask, np.ones((5, 5), np.uint8), iterations=1)
        mask_inv = cv2.bitwise_not(overlay_mask)
        roi = roi_color[y:y+h, x:x+w]
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(
            overlay_glasses, overlay_glasses, mask=overlay_mask)
        dst = cv2.add(bg, fg)
        roi_color[y:y+h, x:x+w] = dst

     # Show the result
    cv2.imshow('Sunglasses', image)
    cv2.destroyAllWindows()
else:
    print("No faces detected.")
