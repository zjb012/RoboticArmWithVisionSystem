import numpy as np
import cv2
import sys
import os

"""
detect_multi_colors.py

Developed to detect tokens used in pick/place operation with a dobot Magician robotic arm
This program detects Red, Blue, Yellow, and Green tokens and creates a bounding box around them in order to
sort with the dobot Magician

Developed by Zack Bowen with assistance from Quinn Geist and Zheng Zeng
Robotic Arm with Vision System, Penn State Behrend ECE
"""


# Colors in HSV
# Hue - Pure Color - 8 bits 0-180, find Hue value and divide by 2 to get values on scale
# Saturation - Purity/Intensity - 0-255
# Value - Relative Lightness or darkness - 0-255

# Red
lowerRed1 = np.array([160, 0, 0], dtype="uint8")
upperRed1 = np.array([180, 255, 255], dtype="uint8")
lowerRed2 = np.array([0, 0, 0], dtype="uint8")
upperRed2 = np.array([10, 255, 255], dtype="uint8")

# Blue
lowerBlue = np.array([100, 80, 0], dtype="uint8")
upperBlue = np.array([125, 255, 255], dtype="uint8")

# Yellow
lowerYellow = np.array([25, 0, 0], dtype="uint8")
upperYellow = np.array([40, 255, 255], dtype="uint8")

# Green
lowerGreen = np.array([45, 125, 125], dtype="uint8")
upperGreen = np.array([80, 255, 255], dtype="uint8")


def colorDetectionHSV():
    """
    Function used to detect live video images in HSV format
    """
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        for _ in range(100):
            ret, image = cap.read()
            #image = zoom(image, 1, 1)

            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #cv2.imshow("i",image)

            # find the colors within the specified boundaries and apply the mask
            maskR1 = cv2.inRange(image_HSV, lowerRed1, upperRed1)  # red
            maskR2 = cv2.inRange(image_HSV, lowerRed2, upperRed2)  # red
            maskR = cv2.bitwise_or(maskR1, maskR2)
            maskB = cv2.inRange(image_HSV, lowerBlue, upperBlue)  # blue
            maskY = cv2.inRange(image_HSV, lowerYellow, upperYellow)  # yellow
            maskG = cv2.inRange(image_HSV, lowerGreen, upperGreen)  # green

            # Inverts the mask, does not display the color
            # mask = cv2.bitwise_not(mask)

            # Applies the masks to the image
            outputR = cv2.bitwise_and(image, image, mask=maskR)
            #outputB = cv2.bitwise_and(image, image, mask=maskB)
            #outputY = cv2.bitwise_and(image, image, mask=maskY)
            #outputG = cv2.bitwise_and(image, image, mask=maskG)

            # shows both images
            #row1 = np.concatenate((outputR, outputB), axis=1)
            #row2 = np.concatenate((outputY, outputG), axis=1)
            #imageStack = np.concatenate((row1, row2), axis=0)
            #cv2.imshow("images", outputR)

            drawBoundingBox(outputR)
            #drawBoundingBox(outputB)
            #drawBoundingBox(outputY)
            #drawBoundingBox(outputG)

            #drawBoundingBox(imageStack)
            #sys.exit(1)
            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(1)


def cyclePictures():
    """
    Function used for testing purposes
    Displays all images in test_pics folder
    """
    for filename in os.listdir("test_pics"):
        for _ in range(100):

            image = cv2.imread(os.path.join("test_pics", filename))
            image = cv2.resize(image, (300,300), interpolation=cv2.INTER_LINEAR)
            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # find the colors within the specified boundaries and apply the mask
            maskR = cv2.inRange(image_HSV, lowerRed1, upperRed1)  # red
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
            #cv2.imshow("images", imageStack)

            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(1)


def viewPicture(filename):
    """
    Used in testing to view pictures saved on local machine
    Input: filename - name of file to view
    """
    image = cv2.imread(os.path.join("test_pics", filename))
    image = zoom(image, 1, 1)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # find the colors within the specified boundaries and apply the mask
    maskR = cv2.inRange(image_HSV, lowerRed1, upperRed1)  # red
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
    cv2.imwrite("outputR.jpg", outputR)

    outputImages = [outputR, outputB, outputY, outputG]

    # shows both images
    row1 = np.concatenate((outputR, outputB, image), axis=1)
    row2 = np.concatenate((outputY, outputG, image), axis=1)
    imageStack = np.concatenate((row1, row2), axis=0)
    imageStack = cv2.resize(imageStack, (1500, 900), interpolation=cv2.INTER_LINEAR)

    drawBoundingBox(outputR)
    drawBoundingBox(outputB)
    drawBoundingBox(outputY)
    drawBoundingBox(outputG)

    #cv2.imshow("images", imageStack)
    #cv2.waitKey(0)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(1)


def drawBoundingBox(img):
    """
    Creates a bounding box around the given image
    Input: img - Image opened in CV2
    Returns the position of the bounding box as a tuple of x,y,w,h
    """


    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray.jpg", gray)

    #  _____________________________________________________________________
    # Eventually want to update based on what the image looks like (change number of iterations)
    # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((5, 5), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    # gray = cv2.erode(gray, kernel, iterations=5)
    #  _____________________________________________________________________


    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_DEFAULT)
    #cv2.imshow("Smoothed", smoothed)
    #cv2.imwrite("smoothed.jpg", smoothed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # threshold
    thresh = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)[1]
    #cv2.imshow("Threshold", thresh)
    #cv2.imwrite("thresh.jpg", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # finds all contours and appends them to array contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #print("x,y,w,h:", x, y, w, h)
        if w>60 or h>60 or w<10 or h<10:
            contours = ()
    #print(contours)

    # show thresh and result
    #cv2.imshow("bounding_box", thresh)
    #cv2.waitKey(1)
    return contours

    #cv2.destroyAllWindows()


def zoom(img, hZoom, wZoom):
    """
    Provides the functionality to zoom in on an image, this can zoom in so that only the blocks on the conveyor are
    visible
    Input: img - image to zoom
    Input: hZoom - how much height to zoom
    Input: wZoom - how much width to zoom
    """
    h, w, c = img.shape

    center = (h / 2, w / 2)
    new_h = center[0] / hZoom
    new_w = center[1] / wZoom
    
    return img[int(center[0]-new_h):int(center[0]+new_h), int(center[1]-new_w):int(center[1]+new_w)]


def main():
    colorDetectionHSV()
    # blobDetection()
    # viewPicture("dobot_9.jpg")
    # zoom(cv2.imread(os.path.join("test_pics", "dobot_1.jpg")), 2, 1)


if __name__ == "__main__":
    sys.exit(main())


# Ideas to get correct bounding box:
    # Remove glare (yellow blends with white)
    # Zoom image to just the conveyor
