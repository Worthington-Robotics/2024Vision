import cv2
from typing import Any, List, Tuple, Union
import numpy as np
from config import ConfigPaths, WorbotsConfig
from wpimath.geometry import *
from detection import Detection, PoseDetection
from .poseCalculator import PoseCalculator, Pose3d
import os

class WorbotsVision:
    worConfig: WorbotsConfig
    axis_len = 0.1
    poseCalc = PoseCalculator()
    
    axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
    apriltagDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detectorParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector()

    def __init__(self, configPaths: ConfigPaths):
        self.worConfig = WorbotsConfig(configPaths)
        self.mtx, self.dist = self.worConfig.getCameraIntrinsicsFromJSON()
        self.tag_size = self.worConfig.TAG_SIZE_METERS
        self.obj_1 = [-self.tag_size/2, self.tag_size/2, 0.0]
        self.obj_2 = [self.tag_size/2, self.tag_size/2, 0.0]
        self.obj_3 = [self.tag_size/2, -self.tag_size/2, 0.0]
        self.obj_4 = [-self.tag_size/2, -self.tag_size/2, 0.0]
        self.obj_all = self.obj_1 + self.obj_2 + self.obj_3 + self.obj_4
        self.objPoints = np.array(self.obj_all).reshape(4,3)

        # self.detectorParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # self.detectorParams.maxMarkerPerimeterRate = 3.5
        self.detectorParams.minDistanceToBorder = 10
        self.detector.setDetectorParameters(self.detectorParams)
        self.detector.setDictionary(self.apriltagDict)
    
    def mainPnP(self):
        while True:
            returnArray = np.array([], dtype=Detection)
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
            detectorParams = cv2.aruco.DetectorParameters()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(image=gray, dictionary=dictionary, parameters=detectorParams)

            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    ret, rvec, tvec = cv2.solvePnP(self.objPoints, corners[i], self.mtx, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    # print(f"Translation: {tvec[0]},{tvec[1]},{tvec[2]}, Rotation: {rvec[0]},{rvec[1]},{rvec[2]}")
                    detection = Detection(ids[i], tvec, rvec)
                    returnArray = np.append(returnArray, detection)
                    frame = cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, self.axis_len)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))
            print(returnArray.size)
            cv2.imshow("out", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def mainPnPSingleFrame(self) -> any:
        returnArray = np.array([], Detection)
        ret, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        detectorParams = cv2.aruco.DetectorParameters()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image=gray, dictionary=dictionary, parameters=detectorParams)
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                ret, rvec, tvec = cv2.solvePnP(self.objPoints, corners[i], self.mtx, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # print(f"Translation: {tvec[0]},{tvec[1]},{tvec[2]}, Rotation: {rvec[0]},{rvec[1]},{rvec[2]}")
                detection = Detection(ids[i], tvec, rvec)
                returnArray = np.append(returnArray, detection)
                frame = cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec, tvec, self.axis_len)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))
        return frame, returnArray
 
    def processFrame(self, frame: Union[Any, None]) -> Tuple[Union[Any, None], Union[PoseDetection, None]]:
        if frame is None:
            return None, None
        
        if self.worConfig.MAKE_BW:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        imgPoints = []
        objPoints = []
        tag_ids = []

        (corners, ids, _) = self.detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 0, 255))
            if len(ids) > 0:
                index = 0
                for id in ids:
                    pose = self.poseCalc.getPose3dFromTagID(id)
                    if pose is None:
                        return (None, None)
                    corner0 = pose + Transform3d((Translation3d(0, self.tag_size/2.0, -self.tag_size/2.0)), Rotation3d())
                    corner1 = pose + Transform3d((Translation3d(0, -self.tag_size/2.0, -self.tag_size/2.0)), Rotation3d())
                    corner2 = pose + Transform3d((Translation3d(0, -self.tag_size/2.0, self.tag_size/2.0)), Rotation3d())
                    corner3 = pose + Transform3d((Translation3d(0, self.tag_size/2.0, self.tag_size/2.0)), Rotation3d())
                    objPoints += [
                        self.poseCalc.wpiTranslationToOpenCV(corner0.translation()),
                        self.poseCalc.wpiTranslationToOpenCV(corner1.translation()),
                        self.poseCalc.wpiTranslationToOpenCV(corner2.translation()),
                        self.poseCalc.wpiTranslationToOpenCV(corner3.translation())
                    ]
                    imgPoints += [
                        [corners[index][0][0][0], corners[index][0][0][1]],
                        [corners[index][0][1][0], corners[index][0][1][1]],
                        [corners[index][0][2][0], corners[index][0][2][1]],
                        [corners[index][0][3][0], corners[index][0][3][1]]
                    ]
                    tag_ids.append(id)
                    index +=1
            index = 0
            if len(ids)==1:
                _, rvec, tvec, errors = cv2.solvePnPGeneric(self.objPoints, np.array(imgPoints), self.mtx, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                field_to_tag_pose = self.poseCalc.getPose3dFromTagID(ids[0])
                camera_to_tag_pose_0 = self.poseCalc.openCvtoWpi(tvec[0], rvec[0])
                camera_to_tag_pose_1 = self.poseCalc.openCvtoWpi(tvec[1], rvec[1])
                camera_to_tag_0 = Transform3d(camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation())
                camera_to_tag_1 = Transform3d(camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation())
                field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
                field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())
                field_to_camera_pose_0 = Pose3d(field_to_camera_0.translation(), field_to_camera_0.rotation())
                field_to_camera_pose_1 = Pose3d(field_to_camera_1.translation(), field_to_camera_1.rotation())
                return frame, PoseDetection(field_to_camera_pose_0, errors[0][0], field_to_camera_pose_1, errors[1][0], tag_ids)
            if len(ids)>1:
                _, rvec, tvec, errors = cv2.solvePnPGeneric(np.array(objPoints), np.array(imgPoints), self.mtx, self.dist, flags=cv2.SOLVEPNP_SQPNP)
                camera_to_field_pose = self.poseCalc.openCvtoWpi(tvec[0], rvec[0])
                camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
                field_to_camera = camera_to_field.inverse()
                field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())
                return frame, PoseDetection(field_to_camera_pose, errors[0][0], None, None, tag_ids)
        else:
            return frame, None


    def checkCalib(self):
        mtx, dist, rvecs, tvecs = self.worConfig.getCameraIntrinsicsFromJSON()
        while True:
            ret, frame = self.cap.read()
            cv2.undistort(frame, mtx, dist, None)

            cv2.imshow("out", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
