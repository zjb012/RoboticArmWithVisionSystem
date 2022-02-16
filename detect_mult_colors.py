import numpy as np
import cv2
import sys
import os

# Colors in HSV
# Hue - Pure Color - 8 bits 0-180, find Hue value and divide by 2 to get values on scale
# Saturation - Purity/Intensity - 0-255
# Value - Relative Lightness or darkness - 0-255


# Red
lowerRed = np.array([160, 150, 0], dtype="uint8")
upperRed = np.array([180, 255, 255], dtype="uint8")

# Blue
lowerBlue = np.array([100, 150, 0], dtype="uint8")
upperBlue = np.array([125, 255, 255], dtype="uint8")

# Yellow
lowerYellow = np.array([12, 150, 100], dtype="uint8")
upperYellow = np.array([35, 255, 255], dtype="uint8")

# Green
lowerGreen = np.array([36, 150, 0], dtype="uint8")
upperGreen = np.array([86, 255, 255], dtype="uint8")


def colorDetectionHSV():
    cap = cv2.VideoCapture(1)
    while True:
        for _ in range(100):

            ret, image = cap.read()


            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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


def cyclePictures():
    for filename in os.listdir("test_pics"):
        for _ in range(100):

            image = cv2.imread(os.path.join("test_pics", filename))
            image = cv2.resize(image, (300,300), interpolation=cv2.INTER_LINEAR)

            # Colors in HSV
            # Hue - Pure Color - 8 bits 0-180, find Hue value and divide by 2 to get values on scale
            # Saturation - Purity/Intensity - 0-255
            # Value - Relative Lightness or darkness - 0-255
            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
            row1 = np.concatenate((outputR, outputB, image), axis=1)
            row2 = np.concatenate((outputY, outputG, image), axis=1)
            imageStack = np.concatenate((row1, row2), axis=0)
            cv2.imshow("images", imageStack)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(1)


def viewPicture(filename):
    #ret, image = cap.read()
    image = cv2.imread(os.path.join("test_pics", filename))
    image = cv2.resize(image, (300,300), interpolation=cv2.INTER_LINEAR)

    # Colors in HSV
    # Hue - Pure Color - 8 bits 0-180, find Hue value and divide by 2 to get values on scale
    # Saturation - Purity/Intensity - 0-255
    # Value - Relative Lightness or darkness - 0-255
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

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
    row1 = np.concatenate((outputR, outputB, image), axis=1)
    row2 = np.concatenate((outputY, outputG, image), axis=1)
    imageStack = np.concatenate((row1, row2), axis=0)
    cv2.imshow("images", imageStack)
    cv2.waitKey(0);

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(1)

def main():
    colorDetectionHSV()
    # viewPicture("dobot_11.jpg")


if __name__ == "__main__":
    sys.exit(main())



