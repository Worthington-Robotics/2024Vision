import cv2
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from config import ConfigPaths, WorbotsConfig
from wpimath.geometry import *
from detection import Detection, PoseDetection
from .poseCalculator import PoseCalculator, Pose3d
import os


class WorbotsVision:
    worConfig: WorbotsConfig
    poseCalc = PoseCalculator()

    apriltagDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector()

    def __init__(self, configPaths: ConfigPaths):
        self.worConfig = WorbotsConfig(configPaths)
        self.mtx, self.dist = self.worConfig.getCameraIntrinsicsFromJSON()
        self.tag_size = self.worConfig.TAG_SIZE_METERS
        self.obj_1 = [-self.tag_size / 2, self.tag_size / 2, 0.0]
        self.obj_2 = [self.tag_size / 2, self.tag_size / 2, 0.0]
        self.obj_3 = [self.tag_size / 2, -self.tag_size / 2, 0.0]
        self.obj_4 = [-self.tag_size / 2, -self.tag_size / 2, 0.0]
        self.obj_all = self.obj_1 + self.obj_2 + self.obj_3 + self.obj_4
        self.objPoints = np.array(self.obj_all).reshape(4, 3)

        # self.detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detectorParams.minMarkerPerimeterRate = 0.03
        # self.detectorParams.maxMarkerPerimeterRate = 0.0
        self.detectorParams.minDistanceToBorder = 3
        self.detector.setDetectorParameters(self.detectorParams)
        self.detector.setDictionary(self.apriltagDict)

    def processFrame(
        self, frame: Optional[cv2.Mat]
    ) -> Tuple[Optional[cv2.Mat], Optional[PoseDetection]]:
        """Processes a vision frame to find and draw tags"""

        if frame is None:
            return None, None

        # Make the frame grayscale if it isn't already
        if self.worConfig.MAKE_BW:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        imgPoints = []
        objPoints = []
        tag_ids = []

        (corners, ids, _) = self.detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))
            if len(ids) > 0:
                for index, id in enumerate(ids):
                    # Skip the tag if it is ignored
                    if id in self.worConfig.IGNORED_TAGS:
                        continue

                    pose = self.poseCalc.getPose3dFromTagID(id)
                    if pose is None:
                        return (None, None)
                    corner0 = pose + Transform3d(
                        (Translation3d(0, self.tag_size / 2.0, -self.tag_size / 2.0)),
                        Rotation3d(),
                    )
                    corner1 = pose + Transform3d(
                        (Translation3d(0, -self.tag_size / 2.0, -self.tag_size / 2.0)),
                        Rotation3d(),
                    )
                    corner2 = pose + Transform3d(
                        (Translation3d(0, -self.tag_size / 2.0, self.tag_size / 2.0)),
                        Rotation3d(),
                    )
                    corner3 = pose + Transform3d(
                        (Translation3d(0, self.tag_size / 2.0, self.tag_size / 2.0)),
                        Rotation3d(),
                    )
                    objPoints += [
                        self.poseCalc.wpiTranslationToOpenCV(corner0.translation()),
                        self.poseCalc.wpiTranslationToOpenCV(corner1.translation()),
                        self.poseCalc.wpiTranslationToOpenCV(corner2.translation()),
                        self.poseCalc.wpiTranslationToOpenCV(corner3.translation()),
                    ]
                    imgPoints += [
                        [corners[index][0][0][0], corners[index][0][0][1]],
                        [corners[index][0][1][0], corners[index][0][1][1]],
                        [corners[index][0][2][0], corners[index][0][2][1]],
                        [corners[index][0][3][0], corners[index][0][3][1]],
                    ]
                    tag_ids.append(id)
            index = 0
            if len(ids) == 1:
                _, rvec, tvec, errors = cv2.solvePnPGeneric(
                    self.objPoints,
                    np.array(imgPoints),
                    self.mtx,
                    self.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                field_to_tag_pose = self.poseCalc.getPose3dFromTagID(ids[0])
                camera_to_tag_pose_0 = self.poseCalc.openCvtoWpi(tvec[0], rvec[0])
                camera_to_tag_pose_1 = self.poseCalc.openCvtoWpi(tvec[1], rvec[1])
                camera_to_tag_0 = Transform3d(
                    camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation()
                )
                camera_to_tag_1 = Transform3d(
                    camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation()
                )
                field_to_camera_0 = field_to_tag_pose.transformBy(
                    camera_to_tag_0.inverse()
                )
                field_to_camera_1 = field_to_tag_pose.transformBy(
                    camera_to_tag_1.inverse()
                )
                field_to_camera_pose_0 = Pose3d(
                    field_to_camera_0.translation(), field_to_camera_0.rotation()
                )
                field_to_camera_pose_1 = Pose3d(
                    field_to_camera_1.translation(), field_to_camera_1.rotation()
                )
                return frame, PoseDetection(
                    field_to_camera_pose_0,
                    errors[0][0],
                    field_to_camera_pose_1,
                    errors[1][0],
                    tag_ids,
                )
            if len(ids) > 1:
                _, rvec, tvec, errors = cv2.solvePnPGeneric(
                    np.array(objPoints),
                    np.array(imgPoints),
                    self.mtx,
                    self.dist,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                camera_to_field_pose = self.poseCalc.openCvtoWpi(tvec[0], rvec[0])
                camera_to_field = Transform3d(
                    camera_to_field_pose.translation(), camera_to_field_pose.rotation()
                )
                field_to_camera = camera_to_field.inverse()
                field_to_camera_pose = Pose3d(
                    field_to_camera.translation(), field_to_camera.rotation()
                )
                return frame, PoseDetection(
                    field_to_camera_pose, errors[0][0], None, None, tag_ids
                )
        else:
            return frame, None
