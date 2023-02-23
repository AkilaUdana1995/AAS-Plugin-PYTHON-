import cv2
import numpy as np
#this import is used to file load dynamically
import tkinter as tk
from tkinter import filedialog
# Create a Tkinter window
root = tk.Tk()
root.withdraw()

# haarcade classifiers for detect face and eyes
face_cascade = cv2.CascadeClassifier('Assets\Haarcascadefiles\haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('Assets\Haarcascadefiles\haarcascade_eye.xml')

# error handelling
if face_cascade.empty():
    # print("Unable to load the face cascade classifier xml file")
    raise IOError('Unable to load the face cascade classifier xml file')

if eye_cascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file')


# read both the images of the face and the glasses(by hard coding)
#image = cv2.imread('Assets\images\sample images\Spongebob.png')
#image = cv2.imread('Assets\images\sample images\sample18.jpg')

# Load the selected image file
print("Press 'o' to select an image again.")
print("Press 'x' to continue with the selected image.")

image =None
while image is None:
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
    else:
        raise ValueError('No image selected')
    cv2.imshow('Press "O" to reselect, Press "X" to continue', image)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('x'):
        break
    else:
        image = None

cv2.destroyAllWindows()




#glass_img = cv2.imread('Assets\images\sunglasses\greenGlass.png')
glass_img = cv2.imread('Assets\images\sunglasses\glassDefault.png')


# convert image into gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    scaling_factor =glasses_width / w

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
    combined_img = np.hstack((image, final_img))
    cv2.imshow('Original Image (Left) and Glasses Overlay (Right)', combined_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
 print("No faces detected.")
