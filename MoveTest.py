import threading
import DobotDllType as dType
import time
CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}


#Load Dll and get the CDLL object
api = dType.load()

#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0]
dType.SetHOMECmdEx(api, temp=0, isQueued=1)
dType.SetQueuedCmdClear(api)
current_pose = dType.GetPose(api)
dType.SetPTPCmdEx(api, 2, 166,  142,  50, current_pose[3], 1)
dType.SetQueuedCmdClear(api)

dType.SetPTPCmdEx(api, 2, 166,  142,  10, current_pose[3], 1)
dType.SetQueuedCmdClear(api)
time.sleep(10)

