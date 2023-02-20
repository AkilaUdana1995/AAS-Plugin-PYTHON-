import cv2
import dlib
import numpy as np

# Load the face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image and resize it
img = cv2.imread("Assets\images\sample images\sample4.jpg")
img = cv2.resize(img, (500, 500))

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the faces in the image
faces = detector(gray)

# Loop through each face and find the landmarks
for face in faces:
    landmarks = predictor(gray, face)
    
    # Extract the coordinates of the left and right eye
    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)
    
    # Calculate the distance between the eyes
    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    
    # Load the sunglasses image and resize it
    sunglasses = cv2.imread("Assets\images\sunglasses\glass.png", cv2.IMREAD_UNCHANGED)
    sunglasses = cv2.resize(sunglasses, (int(eye_distance * 3), int(eye_distance * 1.5)))

    # Rotate the sunglasses
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
    M = cv2.getRotationMatrix2D((sunglasses.shape[1] / 2, sunglasses.shape[0] / 2), angle, 1)
    sunglasses = cv2.warpAffine(sunglasses, M, (sunglasses.shape[1], sunglasses.shape[0]))

    # Calculate the position of the top-left corner of the sunglasses
    x = int(left_eye[0] - eye_distance)
    y = int(left_eye[1] - eye_distance / 2)

    # Create a mask for the sunglasses and the region of interest in the image
    sunglasses_mask = sunglasses[:, :, 3] / 255.0
    img_mask = 1.0 - sunglasses_mask
    sunglasses = sunglasses[:, :, 0:3]
    roi = img[y:y+sunglasses.shape[0], x:x+sunglasses.shape[1]]

    # Blend the sunglasses and the region of interest in the image
    blended_roi = cv2.addWeighted(roi, img_mask, sunglasses, sunglasses_mask, 0)
    img[y:y+sunglasses.shape[0], x:x+sunglasses.shape[1]] = blended_roi

# Display the final image
cv2.imshow("Virtual Makeup", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
