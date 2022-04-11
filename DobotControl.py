import threading
import sys
import DobotDllType as dType
import time

# Turn Conveyor On
STEP_PER_CIRCLE = 360.0 / 1.8 * 10.0 * 16.0
MM_PER_CIRCLE = 3.1415926535898 * 36.0
vel = float(50) * STEP_PER_CIRCLE / MM_PER_CIRCLE


CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}


def conveyorOn(api, velo):
    dType.SetEMotorEx(api, 0, 1, int(velo), 1)


def conveyorOff(api):
    dType.SetEMotorEx(api, 0, 1, 0, 1)


def suctionON(api):
    dType.SetEndEffectorSuctionCup(api, 1, 1, isQueued=1)


def suctionOFF(api):
    dType.SetEndEffectorSuctionCup(api, 1, 0, isQueued=1)


def movetoCoordinate(api, x, y, z):
    return dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, x, y, z, 0, isQueued=1)[0]


def main():
    # Load Dll and get the CDLL object
    api = dType.load()


    # Connect Dobot
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:", CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):

        # Clean Command Queued
        dType.SetQueuedCmdClear(api)

        # Home location x, y, z, r
        dType.SetHOMEParams(api, 260, 0, 50, 0, isQueued=1)
        # dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)

        dType.SetPTPJointParams(api, 20, 20, 20, 20, 20, 20, 20, 20, isQueued=1)

        dType.SetPTPCommonParams(api, 100, 100, isQueued=1)

        # Async Home
        dType.SetHOMECmd(api, temp=0, isQueued=1)

        # Async PTP Motion
        # for i in range(0, 5):
        #    if i % 2 == 0:
        #        offset = 50
        #    else:
        #        offset = -50
        #    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 200 + offset, offset, offset, offset, isQueued = 1)[0]
        lastIndex = movetoCoordinate(api, 275, 0, 100)


        # Start to Execute Command Queue
        dType.SetQueuedCmdStartExec(api)

        # Wait for Executing Last Command
        while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
            dType.dSleep(100)

        # Stop to Execute Command Queued
        dType.SetQueuedCmdStopExec(api)

    # Disconnect Dobot
    dType.DisconnectDobot(api)


if __name__ == "__main__":
    sys.exit(main())