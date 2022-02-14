import numpy as np
import cv2
import sys


def colorDetectionHSV():
    cap = cv2.VideoCapture(0)
    while True:
        for _ in range(100):

            ret, image = cap.read()

            # Colors in HSV
            # Hue - Pure Color - 8 bits 0-180, find Hue value and divide by 2 to get values on scale
            # Saturation - Purity/Intensity - 0-255
            # Value - Relative Lightness or darkness - 0-255
            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Red
            lowerRed = np.array([0, 150, 0], dtype="uint8")
            upperRed = np.array([10, 255, 255], dtype="uint8")

            # Blue
            lowerBlue = np.array([100, 150, 0], dtype="uint8")
            upperBlue = np.array([125, 255, 255], dtype="uint8")

            # Yellow
            lowerYellow = np.array([20, 150, 0], dtype="uint8")
            upperYellow = np.array([30, 255, 255], dtype="uint8")

            # Green
            lowerGreen = np.array([36, 150, 0], dtype="uint8")
            upperGreen = np.array([86, 255, 255], dtype="uint8")

            # find the colors within the specified boundaries and apply the mask
            maskR = cv2.inRange(image_HSV, lowerRed, upperRed)  # red
            maskB = cv2.inRange(image_HSV, lowerBlue, upperBlue)  # blue
            maskY = cv2.inRange(image_HSV, lowerYellow, upperYellow)  # yellow
            maskG = cv2.inRange(image_HSV, lowerGreen, upperGreen)  # green

            # Inverts the mask, does not display the color
            # mask = cv2.bitwise_not(mask)

            # Applies the masks to the image
            outputR = cv2.bitwise_and(image, image, mask=maskR)
            outputB = cv2.bitwise_and(image, image, mask=maskB)
            outputY = cv2.bitwise_and(image, image, mask=maskY)
            outputG = cv2.bitwise_and(image, image, mask=maskG)

            # shows both images
            row1 = np.concatenate((outputR, outputB), axis=1)
            row2 = np.concatenate((outputY, outputG), axis=1)
            imageStack = np.concatenate((row1, row2), axis=0)
            cv2.imshow("images", imageStack)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(1)


def colorDetectionRGB():
    cap = cv2.VideoCapture(0)
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
            mask += cv2.inRange(image, lowerGreen, upperGreen)  # green

            # Inverts the mask, does not display the color
            mask = cv2.bitwise_not(mask)

            # Applies the mask to the image
            output = cv2.bitwise_and(image, image, mask=mask)

            # shows both images
            cv2.imshow("images", np.hstack([image, output]))

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(1)


def main():
    colorDetectionHSV()
    # colorDetectionRGB()


if __name__ == "__main__":
    sys.exit(main())



