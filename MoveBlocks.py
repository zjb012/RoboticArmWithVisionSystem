

#Setup
import threading
import DobotDllType as dType
import numpy as np
import cv2
import sys
import os


import time

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}


#Load Dll and get the CDLL object
api = dType.load()

#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
print("Connect status:",CON_STR[state])


#Color Detection Setup

# Colors in HSV
# Hue - Pure Color - 8 bits 0-180, find Hue value and divide by 2 to get values on scale
# Saturation - Purity/Intensity - 0-255
# Value - Relative Lightness or darkness - 0-255

# Red
lowerRed = np.array([160, 0, 0], dtype="uint8")
upperRed = np.array([180, 255, 255], dtype="uint8")

# Blue
lowerBlue = np.array([100, 0, 0], dtype="uint8")        #old
#lowerBlue = np.array([100, 50, 50], dtype="uint8")
upperBlue = np.array([125, 255, 255], dtype="uint8")     #Old
#upperBlue = np.array([125, 250, 250], dtype="uint8")

# Yellow
lowerYellow = np.array([25, 0, 0], dtype="uint8")
upperYellow = np.array([40, 255, 255], dtype="uint8")

# Green
lowerGreen = np.array([45, 0, 0], dtype="uint8")
upperGreen = np.array([80, 255, 255], dtype="uint8")

#Color Detection Functions

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
    kernel = np.ones((5, 5), np.uint8)

    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    #gray = cv2.erode(gray, kernel, iterations=5)
    #  _____________________________________________________________________


    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_DEFAULT)
    #cv2.imshow("Smoothed", smoothed)
    #cv2.imwrite("smoothed.jpg", smoothed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # threshold
    #thresh = cv2.threshold(smoothed, 128, 255, cv2.THRESH_BINARY)[1]
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
        #cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        #print("x,y,w,h:", x, y, w, h)

    # show thresh and result
    #cv2.imshow("bounding_box", result)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return contours


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

cap = cv2.VideoCapture(0)


#Camera Calibration


#Control loop
if (state == dType.DobotConnect.DobotConnect_NoError):
    print("Here")
    dType.SetInfraredSensor(api, 1 ,2, 1)
    while(1):
        #Move conveyor until block in front
        #If block do nothing
        STEP_PER_CRICLE = 360.0 / 1.8 * 10.0 * 16.0
        MM_PER_CRICLE = 3.1415926535898 * 36.0
        while dType.GetInfraredSensor(api, 2)[0] == 0:
            vel = float((-50)) * STEP_PER_CRICLE / MM_PER_CRICLE
            dType.SetEMotorEx(api, 0, 1, int(vel), 1)
        vel = float(0) * STEP_PER_CRICLE / MM_PER_CRICLE
        dType.SetEMotorEx(api, 0, 0, int(vel), 1)


		#Get block location in image
        ret, image = cap.read()
        #image = zoom(image, 1, 4)
        #cv2.imshow("im",image)
        #image = zoom(image, 1, 1)
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
        # row1 = np.concatenate((outputR, outputB), axis=1)
        # row2 = np.concatenate((outputY, outputG), axis=1)
        # imageStack = np.concatenate((row1, row2), axis=0)
        # cv2.imshow("images", imageStack)

        RedContours = drawBoundingBox(outputR)
        BlueContours = drawBoundingBox(outputB)
        YellowContours = drawBoundingBox(outputY)
        GreenContours = drawBoundingBox(outputG)
        # print(RedContours)
        # print(BlueContours)
        # print(YellowContours)
        # print(GreenContours)


        if RedContours:
           x, y, w, h = cv2.boundingRect(RedContours[0])
           #cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
           print("Red: x,y,w,h:", x, y, w, h)

        if BlueContours:
           x, y, w, h = cv2.boundingRect(BlueContours[0])
           #cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
           print("Blue: x,y,w,h:", x, y, w, h)

        if YellowContours:
           x, y, w, h = cv2.boundingRect(YellowContours[0])
           #cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
           print("Yellow: x,y,w,h:", x, y, w, h)

        if GreenContours:
           x, y, w, h = cv2.boundingRect(GreenContours[0])
           #cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
           print("Green: x,y,w,h:", x, y, w, h)

        #drawBoundingBox(imageStack)



		#Translate Location

		#Move Block
        #time.sleep(2)

