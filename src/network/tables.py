import ntcore
import math
from config import ConfigPaths, WorbotsConfig
from detection import PoseDetection
from wpimath.geometry import *
from typing import Optional

class WorbotsTables:
    config: WorbotsConfig
    ntInstance = None

    # Output Publishers
    fpsPublisher = None
    dataPublisher = None

    # Config Subscribers
    cameraIdSubscriber: ntcore.IntegerSubscriber

    def __init__(self, configPaths: ConfigPaths):
        self.config = WorbotsConfig(configPaths)
        self.ntInstance = ntcore.NetworkTableInstance.getDefault()
        if self.config.SIM_MODE:
            self.ntInstance.setServer("127.0.0.1")
            self.ntInstance.startClient4(f"VisionModule{self.config.MODULE_ID}")
        else:
            self.ntInstance.setServerTeam(self.config.TEAM_NUMBER)
            self.ntInstance.startClient4(f"VisionModule{self.config.MODULE_ID}")

        self.moduleTable = self.ntInstance.getTable(f"/module{self.config.MODULE_ID}")
        table = self.moduleTable.getSubTable("output")
        configTable = self.moduleTable.getSubTable("config")
        self.dataPublisher = table.getDoubleArrayTopic("data").publish(ntcore.PubSubOptions())
        self.fpsPublisher = table.getDoubleTopic("fps").publish(ntcore.PubSubOptions())
        configTable.getBooleanTopic("liveCalib").publish().set(False)
        self.calibListener = configTable.getBooleanTopic("liveCalib").subscribe(False)

    def sendPoseDetection(self, poseDetection: Optional[PoseDetection], timestamp: float):
        if poseDetection is None:
            return
        dataArray = [0]
        if poseDetection.err1 and poseDetection.pose1 is not None:
            dataArray[0] = 1
            dataArray.append(poseDetection.err1)
            dataArray.append(poseDetection.pose1.x)
            dataArray.append(poseDetection.pose1.y)
            dataArray.append(poseDetection.pose1.z)
            dataArray.append(poseDetection.pose1.rw)
            dataArray.append(poseDetection.pose1.rx)
            dataArray.append(poseDetection.pose1.ry)
            dataArray.append(poseDetection.pose1.rz)
        else:
            return
        if poseDetection.err2 and poseDetection.pose2 is not None:
            dataArray[0] = 2
            dataArray.append(poseDetection.err2)
            dataArray.append(poseDetection.pose2.x)
            dataArray.append(poseDetection.pose2.y)
            dataArray.append(poseDetection.pose2.z)
            dataArray.append(poseDetection.pose2.rw)
            dataArray.append(poseDetection.pose2.rx)
            dataArray.append(poseDetection.pose2.ry)
            dataArray.append(poseDetection.pose2.rz)
        for tag_id in poseDetection.tag_ids:
            dataArray.append(float(tag_id))
        self.dataPublisher.set(dataArray, int(math.floor(timestamp * 1000000)))
    
    def sendPose3d(self, pose: Pose3d):
        self.dataPublisher.set(self.getArrayFromPose3d(pose))

    def sendFps(self, fps):
        self.fpsPublisher.set(fps)

    def sendConfig(self):
        configTable = ntcore.NetworkTableInstance.getDefault().getTable(f"/module{self.config.MODULE_ID}/config")
        self.cameraIdSubscriber = configTable.getDoubleTopic("cameraId").subscribe(self.config.CAMERA_ID, ntcore.PubSubOptions())

    def getArrayFromPose3d(self, pose: Pose3d) -> any:
        outArray = []
        outArray.append(pose.X())
        outArray.append(pose.Y())
        outArray.append(pose.Z())
        outArray.append(pose.rotation().getQuaternion().W())
        outArray.append(pose.rotation().getQuaternion().X())
        outArray.append(pose.rotation().getQuaternion().Y())
        outArray.append(pose.rotation().getQuaternion().Z())
        return outArray