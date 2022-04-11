import threading
import DobotDllType as dType
import numpy as np
import cv2
import sys
import os
import detect_mult_colors as dmc
import DobotControl as Dobot
import time

# Conveyor Constants
STEP_PER_CIRCLE = 360.0 / 1.8 * 10.0 * 16.0
MM_PER_CIRCLE = 3.1415926535898 * 36.0
vel = float((-50)) * STEP_PER_CIRCLE / MM_PER_CIRCLE

# Get dType
api = dType.load()

# Block Sorting Locations
redSortLoc = [1, 1, 1]
blueSortLoc = [2, 2, 2]
greenSortLoc = [3, 3, 3]
yellowSortLoc = [4, 4, 4]


def calcMidpoint(coordinate):
    # x = coordinate[0] | y = coordinate[1] | w = coordinate[2] | h = coordinate[3]
    # mp_x = x + w/2 | mp_y = y + h/2

    return [coordinate[0]+coordinate[2]/2, coordinate[1] + coordinate[3]/2]


def moveBlock(mp, sortLoc):
    # Translate Location

    # Move to Starting Location

    # Pick up Block

    # Move to Ending Location

    # Release Block

    # Sleep for 2 seconds
    time.sleep(2)


def main():
    # Setup
    CON_STR = Dobot.CON_STR

    # Load Dll and get the CDLL object

    # Connect Dobot
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:", CON_STR[state])

    # Camera Capture
    cap = cv2.VideoCapture(1)

    # Camera Calibration

    # Control loop
    if state == dType.DobotConnect.DobotConnect_NoError:
        print("Here")
        dType.SetInfraredSensor(api, 1, 2, 1)
        ret, image = cap.read()
        while True:
            # Move conveyor until block in front
            # If block do nothing

            while dType.GetInfraredSensor(api, 2)[0] == 0:
                Dobot.conveyorOn(api, int(vel))
            Dobot.conveyorOff(api)

            # Get block location in image
            time.sleep(1)
            ret, image = cap.read()

            # image = zoom(image, 1, 4)
            cv2.imshow("im",image)
            cv2.waitKey(0)
            # image = zoom(image, 1, 1)
            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # find the colors within the specified boundaries and apply the mask
            maskR = cv2.inRange(image_HSV, dmc.lowerRed, dmc.upperRed)  # red
            maskB = cv2.inRange(image_HSV, dmc.lowerBlue, dmc.upperBlue)  # blue
            maskY = cv2.inRange(image_HSV, dmc.lowerYellow, dmc.upperYellow)  # yellow
            maskG = cv2.inRange(image_HSV, dmc.lowerGreen, dmc.upperGreen)  # green

            # Applies the masks to the image
            outputR = cv2.bitwise_and(image, image, mask=maskR)
            outputB = cv2.bitwise_and(image, image, mask=maskB)
            outputY = cv2.bitwise_and(image, image, mask=maskY)
            outputG = cv2.bitwise_and(image, image, mask=maskG)

            # Sort Contours based on color
            RedContours = dmc.drawBoundingBox(outputR)
            BlueContours = dmc.drawBoundingBox(outputB)
            YellowContours = dmc.drawBoundingBox(outputY)
            GreenContours = dmc.drawBoundingBox(outputG)

            if RedContours:
                # x, y, w, h = cv2.boundingRect(RedContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #print("Red: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(RedContours[0]))
                print("Red MP: ", mp[0], " ", mp[1])
                moveBlock(mp, redSortLoc)
                #cv2.imshow("R", outputR)
                #cv2.waitKey(0)

            elif BlueContours:
                #x, y, w, h = cv2.boundingRect(BlueContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #print("Blue: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(BlueContours[0]))
                print("Blue MP: ", mp[0], " ", mp[1])
                moveBlock(mp, blueSortLoc)
                #cv2.imshow("B", outputB)
                #cv2.waitKey(0)

            elif YellowContours:
                #x, y, w, h = cv2.boundingRect(YellowContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #print("Yellow: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(YellowContours[0]))
                print("Yellow MP: ", mp[0], " ", mp[1])
                moveBlock(mp, yellowSortLoc)
                #cv2.imshow("Y", outputY)
                #cv2.waitKey(0)

            elif GreenContours:
                #x, y, w, h = cv2.boundingRect(GreenContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #print("Green: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(GreenContours[0]))
                print("Green MP: ", mp[0], " ", mp[1])
                moveBlock(mp, greenSortLoc)
                #cv2.imshow("G", outputG)
                #cv2.waitKey(0)





if __name__ == "__main__":
    sys.exit(main())
