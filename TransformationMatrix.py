import cv2 as cv
#import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
import DobotDllType as dType

# input: Dobot api variable
# output: a (2x2), b (2x1), status (for finding the checkboardcorners)
# use a.dot(np.array([xInImg,yInImg]).reshape(2,1))+b to find the (x,y) coordiantes to the arm
def getTransformationFromCamToArm(api, vid):
    # physical width/length between corners in mm
    width = 137.5 / 5
    length = 137.5 / 5

    horizontalAxis = 5
    verticalAxis = 4

    def getCalibrationBoardCorners():
        #vid = cv.VideoCapture(1)

        # found camera intrinsic parameters
        distCoeffs = np.array([-0.33607246, 0.18490096, -0.03402017, -0.00373651, -0.10872384], np.float32)
        cameraMatrix = np.array([[665.44238746, 0., 320.], [0., 660.30501976, 240.], [0., 0., 1.]], np.float32)

        cornersInImg2D = np.zeros((horizontalAxis * verticalAxis, 1, 2), np.float32)

        patternDim = (horizontalAxis, verticalAxis)

        key = ''
        while key != 'q':

            ret, frame = vid.read()

            while (ret == False):
                ret, frame = vid.read()

            while (frame.max() < 100):
                ret, frame = vid.read()

            frame = cv.undistort(frame, cameraMatrix, distCoeffs, None)

            frameInGray = np.dot(frame[..., :3], [0.299, 0.587, 0.114])

            # del frame

            foundAllCorners, tmp = cv.findChessboardCorners(np.array(frameInGray, np.uint8), patternDim)
            del frameInGray

            if not (foundAllCorners):
                print("Corners not found, please reposition the calibration board.")
                key = input("(input q to quit, any other key to continue)")
                status = False
            else:
                cornersInImg2D = np.array(tmp, np.float32)
                status = True
                break

        cornersInImg2D = np.squeeze(cornersInImg2D, -2)
        # vid.release()

        return cornersInImg2D, status

    def getCornersWithArm(xOffset, yOffset):
        widthSpan = np.arange(0, width * horizontalAxis, width)
        lengthSpan = np.arange(0, length * verticalAxis, length)
        cornersToBoard3D = np.zeros((horizontalAxis * verticalAxis, 2), np.float32)
        cornersToBoard3D[:, 0] = -np.matmul(np.expand_dims(lengthSpan, -1), np.ones((1, horizontalAxis))).reshape(
            -1) + xOffset
        cornersToBoard3D[:, 1] = -np.matmul(np.ones((verticalAxis, 1)), np.expand_dims(widthSpan, 0)).reshape(
            -1) + yOffset

        return cornersToBoard3D

    def requestArmPosition(api):
        input("Move the arm so that the end effector points at the designated corner, enter any key when this is done.")

        x, y = dType.GetPose(api)[0:2]

        return x, y

    cornersInImg2D, status = getCalibrationBoardCorners()

    (x, y) = requestArmPosition(api)


    cornersToBoard3D = getCornersWithArm(x, y)

    A, _ = cv.estimateAffine2D(cornersInImg2D, cornersToBoard3D)

    a = A[:, :-1]
    b = A[:, -1].reshape((2, 1))

    return a, b, status
