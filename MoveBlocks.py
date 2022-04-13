import threading
import DobotDllType as dType
import numpy as np
import cv2
import sys
import os
import detect_mult_colors as dmc
import DobotControl as Dobot
import TransformationMatrix as tm
import time

# Conveyor Constants
STEP_PER_CIRCLE = 360.0 / 1.8 * 10.0 * 16.0
MM_PER_CIRCLE = 3.1415926535898 * 36.0
vel = float((-50)) * STEP_PER_CIRCLE / MM_PER_CIRCLE

# Get dType
api = dType.load()

motion = 1
# Block Sorting Locations
redSortLoc = [50, 300, -40]
blueSortLoc = [0, 300, -40]
yellowSortLoc = [-50, 300, -40]
greenSortLoc = [-100, 300, -40]



# H=30


def calcMidpoint(coordinate):
    # x = coordinate[0] | y = coordinate[1] | w = coordinate[2] | h = coordinate[3]
    # mp_x = x + w/2 | mp_y = y + h/2

    return [coordinate[0] + coordinate[2] / 2, coordinate[1] + coordinate[3] / 2]


def moveBlock(transform_x, transform_y, mp, sortLoc):
    # Translate Location
    # Move to Starting Location
    x, y = transform_x.dot(np.array([mp[0], mp[1]]).reshape(2, 1)) + transform_y
    print(x, " ", y)

    dType.SetPTPCmdEx(api, motion, 270, 130, 80, 0, 1)
    dType.SetQueuedCmdClear(api)

    dType.SetPTPCmdEx(api, motion, x, y, 50, 0, 1)
    dType.SetQueuedCmdClear(api)

    dType.SetPTPCmdEx(api, motion, x, y, 15, 0, 1)
    dType.SetQueuedCmdClear(api)

    # Pick up Block
    Dobot.suctionON(api)

    dType.SetPTPCmdEx(api, motion, x, y, 80, 0, 1)
    dType.SetQueuedCmdClear(api)

    dType.SetPTPCmdEx(api, motion, 270, 130, 80, 0, 1)
    dType.SetQueuedCmdClear(api)

    # Move to Ending Location

    # dType.SetPTPCmdEx(api, motion, 0,  300,  80, 0, 1)
    # dType.SetQueuedCmdClear(api)
    dType.SetPTPCmdEx(api, motion, sortLoc[0], sortLoc[1], 80, 0, 1)
    dType.SetQueuedCmdClear(api)

    dType.SetPTPCmdEx(api, motion, sortLoc[0], sortLoc[1], sortLoc[2], 0, 1)
    dType.SetQueuedCmdClear(api)
    sortLoc[2] = sortLoc[2] + 25;

    # Release Block
    Dobot.suctionOFF(api)

    dType.SetPTPCmdEx(api, motion, sortLoc[0], sortLoc[1], 80, 0, 1)
    dType.SetQueuedCmdClear(api)

    # Sleep for 2 seconds
    # time.sleep(2)


def main():
    # Setup
    CON_STR = Dobot.CON_STR

    # Load Dll and get the CDLL object

    # Connect Dobot
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:", CON_STR[state])

    # Camera Capture
    # cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    ret, image = cap.read()
    # import matplotlib.pyplot as plt
    # plt.rcParams['figure.figsize'] = (20,30)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RBG))
    # plt.plot([200], [200], "ro")

    # cv2.imshow("i", image)
    # cv2.waitKey(0)

    # Control loop
    if state == dType.DobotConnect.DobotConnect_NoError:

        # Clean Command Queued
        dType.SetQueuedCmdClear(api)

        # Home location x, y, z, r
        # dType.SetHOMEParams(api, 260, 0, 50, 0, isQueued=1)
        # dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)

        dType.SetPTPJointParams(api, 40, 40, 40, 40, 40, 40, 40, 40, isQueued=1)

        # dType.SetPTPCommonParams(api, 100, 100, isQueued=1)

        # Async Home
        dType.SetHOMECmdEx(api, temp=0, isQueued=1)
        dType.SetQueuedCmdClear(api)
        # time.sleep(15)
        # print("Time")
        # Camera Calibration
        transform_x, transform_y, status = tm.getTransformationFromCamToArm(api, cap)
        print("Calibration Successful")
        time.sleep(5)

        dType.SetPTPCmdEx(api, 2, 230, 130, 80, 0, 1)
        # time.sleep(5)
        dType.SetQueuedCmdClear(api)
        dType.SetPTPCmdEx(api, 2, 0, 300, 80, 0, 1)
        # time.sleep(5)
        dType.SetQueuedCmdClear(api)
        # dType.SetHOMECmdEx(api, temp=0, isQueued=1)

        dType.SetInfraredSensor(api, 1, 2, 1)
        while True:
            # Move conveyor until block in front
            # If block do nothing
            while dType.GetInfraredSensor(api, 2)[0] == 0:
                Dobot.conveyorOn(api, int(vel))

            Dobot.conveyorOff(api)

            # Get block location in image
            time.sleep(1)
            print("Image Taken")
            ret, image = cap.read()
            ret, image = cap.read()
            ret, image = cap.read()
            ret, image = cap.read()
            ret, image = cap.read()
            while image.max() < 100:
                ret, image = cap.read()

            # image = zoom(image, 1, 4)
            # cv2.imshow("im",image)
            # cv2.waitKey(0)
            # image = zoom(image, 1, 1)

            image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # find the colors within the specified boundaries and apply the mask
            maskR1 = cv2.inRange(image_HSV, dmc.lowerRed1, dmc.upperRed1)  # red
            maskR2 = cv2.inRange(image_HSV, dmc.lowerRed2, dmc.upperRed2)  # red
            maskR = cv2.bitwise_or(maskR1, maskR2)
            maskB = cv2.inRange(image_HSV, dmc.lowerBlue, dmc.upperBlue)  # blue
            maskY = cv2.inRange(image_HSV, dmc.lowerYellow, dmc.upperYellow)  # yellow
            maskG = cv2.inRange(image_HSV, dmc.lowerGreen, dmc.upperGreen)  # green

            # Applies the masks to the image

            outputR = cv2.bitwise_and(image, image, mask=maskR)
            outputB = cv2.bitwise_and(image, image, mask=maskB)
            outputY = cv2.bitwise_and(image, image, mask=maskY)
            outputG = cv2.bitwise_and(image, image, mask=maskG)

            row1 = np.concatenate((outputR, outputB, image), axis=1)
            row2 = np.concatenate((outputY, outputG, image), axis=1)
            imageStack = np.concatenate((row1, row2), axis=0)
            imageStack = cv2.resize(imageStack, (1500, 900), interpolation=cv2.INTER_LINEAR)
            # cv2.imshow("imst", imageStack)

            # Sort Contours based on color
            RedContours = dmc.drawBoundingBox(outputR)
            BlueContours = dmc.drawBoundingBox(outputB)
            YellowContours = dmc.drawBoundingBox(outputY)
            GreenContours = dmc.drawBoundingBox(outputG)

            if RedContours:
                x, y, w, h = cv2.boundingRect(RedContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print("Red: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(RedContours[0]))
                print("Red")
                moveBlock(transform_x, transform_y, mp, redSortLoc)
                # cv2.imshow("R", outputR)
                # cv2.waitKey(0)

            elif BlueContours:
                # x, y, w, h = cv2.boundingRect(BlueContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print("Blue: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(BlueContours[0]))
                print("Blue")
                moveBlock(transform_x, transform_y, mp, blueSortLoc)
                # cv2.imshow("B", outputB)
                # cv2.waitKey(0)

            elif YellowContours:
                # x, y, w, h = cv2.boundingRect(YellowContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print("Yellow: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(YellowContours[0]))
                print("Yellow")
                moveBlock(transform_x, transform_y, mp, yellowSortLoc)
                # cv2.imshow("Y", outputY)
                # cv2.waitKey(0)

            elif GreenContours:
                # x, y, w, h = cv2.boundingRect(GreenContours[0])
                # cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # print("Green: x,y,w,h:", x, y, w, h)
                mp = calcMidpoint(cv2.boundingRect(GreenContours[0]))
                print("Green")
                moveBlock(transform_x, transform_y, mp, greenSortLoc)
                # cv2.imshow("G", outputG)
                # cv2.waitKey(0)


if __name__ == "__main__":
    sys.exit(main())