import cv2
import numpy as np
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt 
#%matplotlib inline

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('Assets\Haarcascadefiles\haarcascade_frontalface_default.xml')

# Load the dlib model for facial landmarks detection
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to detect facial landmarks
def detect_landmarks(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # Otherwise, find the facial landmarks for each face
    face = faces[0]
    x, y, w, h = face
    rect = dlib.rectangle(left=x, top=y, right=x + w, bottom=y + h)
    landmarks = predictor(gray, rect)
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    
    return landmarks

# Function to apply virtual makeup
def virtual_makeup(image, color, alpha):
    # Detect the facial landmarks
    landmarks = detect_landmarks(image)
    
    # If no landmarks are detected, return the original image
    if landmarks is None:
        return image
    
    # Otherwise, apply virtual makeup to the landmarks
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, color, -1)
    
    # Blend the original image with the virtual makeup
    result = cv2.addWeighted(image, alpha, np.zeros_like(image), 1 - alpha, 0)
    
    return result

# Load an image
image = cv2.imread('Assets\images\sample images\sample18.jpg')

# Apply virtual makeup with blue color and 0.7 alpha
result = virtual_makeup(image, (255, 0, 0), 0.7)

# Show the result
cv2.imshow('Virtual Makeup', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
