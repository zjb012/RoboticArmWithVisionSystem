import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)

# define the list of boundaries (lower and upper bounds in RGB) for red, blue and yellow
boundaries = [([17, 15, 100], [50, 56, 200]), ([86, 31, 4], [220, 88, 50]), ([25, 146, 190], [62, 174, 250])]

# lower = [17, 15, 100]
# upper = [50, 56, 200]
# lower = np.array(lower, dtype = "uint8")
# upper = np.array(upper, dtype = "uint8")

# loop over the boundaries
while True:
    for _ in range(100):

        ret, image = cap.read()
        # create NumPy arrays from the boundaries
        lowerRed = np.array([17, 15, 100], dtype = "uint8")
        upperRed = np.array([50, 56, 200], dtype = "uint8")

        lowerBlue = np.array([86, 31, 4], dtype="uint8")
        upperBlue = np.array([220, 88, 50], dtype="uint8")

        # Problem with yellow and green masks overlapping
        lowerYellow = np.array([22, 93, 0], dtype="uint8")
        upperYellow = np.array([45, 255, 255], dtype="uint8")

        lowerGreen = np.array([65,60,60], dtype="uint8")
        upperGreen = np.array([80,255,255], dtype="uint8")

        # find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(image, lowerRed, upperRed)  # red
        mask += cv2.inRange(image, lowerBlue, upperBlue)  # blue
        mask += cv2.inRange(image, lowerYellow, upperYellow)  # yellow
        mask += cv2.inRange(image, lowerYellow, upperYellow)  # green

        # Inverts the mask, does not display the color
        # mask = cv2.bitwise_not(mask)

        # Applies the mask to the image
        output = cv2.bitwise_and(image, image, mask = mask)

        # shows both images
        cv2.imshow("images", np.hstack([image, output]))

        # show the image
        # cv2.imshow("image", output)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sys.exit(1)



