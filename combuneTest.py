import cv2
import numpy as np
import dlib
from imutils import face_utils
import matplotlib.pyplot as plt
# %matplotlib inline

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(
    'Assets\Haarcascadefiles\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'Assets\Haarcascadefiles\haarcascade_eye.xml')

# read both the images of the face and the glasses
# image = cv2.imread('Assets\images\sample images\Spongebob.png')
image = cv2.imread('Assets\images\sample images\sample4.jpg')


# glass_img = cv2.imread('Assets\images\sunglasses\greenGlass.png')
glass_img = cv2.imread('Assets\images\sunglasses\glass.png')

# convert image into gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    # detect the faces in gray scale image
centers = []
faces = face_cascade.detectMultiScale(gray, 1.3, 5)


if len(faces) > 0:
    # iterating over the face detected
    for (x, y, w, h) in faces:

        # create two Regions of Interest.
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Store the coordinates of eyes in the image to the 'center' array
        for (ex, ey, ew, eh) in eyes:
            centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))

    if len(centers) > 0:
        # change the given value of 2.15 according to the size of the detected face
        glasses_width = 2.16 * abs(centers[1][0] - centers[0][0])
        overlay_img = np.ones(image.shape, np.uint8) * 255
        h, w = glass_img.shape[:2]
        # we can change the glass size on adjusting below (  >>>>  scaling_factor =1.25*glasses_width / w)
        scaling_factor = glasses_width / w

        overlay_glasses = cv2.resize(
            glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]

        # The x and y variables below depend upon the size of the detected face.
        x -= 0.26 * overlay_glasses.shape[1]
        y += 0.85 * overlay_glasses.shape[0]

        # Slice the height, width of the overlay image.
        h, w = overlay_glasses.shape[:2]

        # Overlay the glasses on the image
        overlay_img[int(y):int(y + h), int(x):int(x + w)] = overlay_glasses

        # Create a mask and generate it's inverse.
        gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        temp = cv2.bitwise_and(image, image, mask=mask)

        temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        final_img = cv2.add(temp, temp2)

# Show the original image and the final image with the glasses overlay at the same time
 # Apply virtual makeup with blue color and 0.7 alpha
        result = virtual_makeup(image, (255, 0, 0), 0.7)
        combined_img = np.hstack((image, final_img))
        cv2.imshow(
            'Original Image (Left) and Glasses Overlay (Right)', combined_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
else:
    print("No faces detected.")
