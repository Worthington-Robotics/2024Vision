import numpy as np
import cv2
from typing import List, Optional, Union
import math
from config import WorbotsConfig
from wpimath.geometry import *
from robotpy_apriltag import *


class PoseCalculator:
    config: WorbotsConfig
    aprilTagLayout = AprilTagFieldLayout("2024-crescendo.json")
    tagPoseArray = np.array([])
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(1))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(2))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(3))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(4))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(5))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(6))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(7))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(8))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(9))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(10))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(11))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(12))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(13))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(14))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(15))
    tagPoseArray = np.append(tagPoseArray, aprilTagLayout.getTagPose(16))

    def __new__(self):
        return super(PoseCalculator, self).__new__(self)

    def __init__(self):
        pass

    def openCvtoWpi(self, tvec, rvec) -> Pose3d:
        return Pose3d(
            Translation3d(tvec[2][0], -tvec[0][0], -tvec[1][0]),
            Rotation3d(
                np.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
                math.sqrt(
                    math.pow(rvec[0][0], 2) + math.pow(rvec[1][0], 2) + math.pow(rvec[2][0], 2))
            ))

    def wpiTranslationToOpenCV(self, translation: Translation3d) -> List[float]:
        return [-translation.Y(), -translation.Z(), translation.X()]

    def getPose3dFromTagID(self, id: int) -> Optional[Pose3d]:
        try:
            return self.tagPoseArray[id-1][0]
        except:
            print("Not a valid id")

    def getPosefromTag(self, id, tvec, rvec) -> Pose3d:
        camToTag = np.array([[0, 0, 0, tvec[0][0]],
                            [0, 0, 0, tvec[1][0]],
                            [0, 0, 0, tvec[2][0]],
                            [0, 0, 0, 1]], dtype=float)
        camToTag[:3, :3], _ = cv2.Rodrigues(rvec)
        tagToWorld = self.getTMatrixFromID(id)
        camToRobot = self.getCameraToRobotMatrix()
        finalPose = np.matmul(
            np.matmul(tagToWorld, np.linalg.inv(camToTag)), camToRobot)
        return self.pose3dFromMatrix(finalPose)

    def internal(self, id) -> np.array:
        try:
            pose = self.aprilTagLayout.getTagPose(id)
            rotation = pose.rotation()
            tvec = np.array([[pose.x], [pose.y], [pose.z]])
            rvec = np.array([[rotation.x], [rotation.y], [rotation.z]])

            return self.rvecTvecToMatrix(rvec, tvec)
        except:
            print("Field layout had a lil trouble")

    def rvecTvecToMatrix(self, rvec, tvec) -> np.array:
        matrix = np.array([[0, 0, 0, tvec[0][0]],
                           [0, 0, 0, tvec[1][0]],
                           [0, 0, 0, tvec[2][0]],
                           [0, 0, 0, 1]], dtype=float)
        matrix[:3, :3], _ = cv2.Rodrigues(rvec)
        returnMatrix = np.array(matrix)
        return returnMatrix

    def getCameraToRobotMatrix(self) -> np.array:
        tvec = np.array([[self.config.CAM_TO_ROBOT_X], [
                        self.config.CAM_TO_ROBOT_Y], [self.config.CAM_TO_ROBOT_Z]])
        rvec = np.array([[np.deg2rad(self.config.CAM_TO_ROBOT_ROLL)], [np.deg2rad(
            self.config.CAM_TO_ROBOT_PITCH)], [np.deg2rad(self.config.CAM_TO_ROBOT_YAW)]])
        return self.rvecTvecToMatrix(rvec, tvec)
