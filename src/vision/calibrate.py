import os
from typing import List, Tuple, Union
import cv2

import numpy as np
from config import ConfigPaths, WorbotsConfig

from vision.camera import ThreadCamera


def calibrateCameraImages(folderName, configPaths: ConfigPaths):
    worConfig = WorbotsConfig(configPaths)

    images = os.listdir(folderName)
    print(len(images))

    (detector, board) = createDetectorAndBoard()

    allCharucoCorners: List[np.ndarray] = []
    allCharucoIds: List[np.ndarray] = []

    for frame in images:
        img = cv2.imread(os.path.join(folderName, frame))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img", gray)
        cv2.waitKey(1)

        # Detect corners as well as markers
        ret = findCharucos(gray, detector)
        if ret is None:
            continue
        allCharucoCorners.append(ret[0])
        allCharucoIds.append(ret[1])

    if len(allCharucoCorners) > 0:
        saveCalibration(worConfig, allCharucoCorners,
                        allCharucoIds, board, gray)
    else:
        print("No Charuco corners were detected for calibration.")


def calibrateCamLive(configPaths: ConfigPaths):
    worConfig = WorbotsConfig(configPaths)

    (detector, board) = createDetectorAndBoard()

    allCharucoCorners: List[np.ndarray] = []
    allCharucoIds: List[np.ndarray] = []

    # Use the camera
    cam = ThreadCamera(configPaths)
    while True:
        frame = cam.getRawFrame()

        if frame is None:
            continue

        frame = frame.frame

        ret = findCharucos(frame, detector)
        if ret is not None:
            allCharucoCorners.append(ret[0])
            allCharucoIds.append(ret[1])

            print(len(allCharucoCorners))

            if len(allCharucoCorners) > 50:
                saveCalibration(worConfig, allCharucoCorners,
                                allCharucoIds, board, frame)
                break

            cv2.imwrite(f"./camera_images/{len(allCharucoCorners)}.jpeg", frame)
        cv2.imshow("out", frame)
        cv2.waitKey(1)

def createDetectorAndBoard() -> Tuple[cv2.aruco.CharucoDetector, cv2.aruco.CharucoBoard]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((11, 8), 0.024, 0.019, dictionary)
    charucoParams = cv2.aruco.CharucoParameters()
    detectorParams = cv2.aruco.DetectorParameters()
    return (cv2.aruco.CharucoDetector(board, charucoParams, detectorParams), board)


def findCharucos(frame: cv2.Mat, detector: cv2.aruco.CharucoDetector) -> Union[Tuple[List[np.ndarray], List[np.ndarray]], None]:
    # Detect corners as well as markers
    (charucoCorners, charucoIds, markerCorners,
     markerIds) = detector.detectBoard(frame)

    if charucoCorners is not None and charucoIds is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            frame, charucoCorners, charucoIds, (0, 0, 255))
        if len(charucoCorners) > 10:
            if len(charucoCorners) == len(charucoIds):
                return (charucoCorners, charucoIds)
    return None


def saveCalibration(worConfig: WorbotsConfig, allCharucoCorners: List[np.ndarray], allCharucoIds: List[np.ndarray], board: cv2.aruco.CharucoBoard, frame: cv2.Mat):
    print(frame.shape)
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        allCharucoCorners, allCharucoIds, board, frame.shape[::-1], None, None
    )
    worConfig.saveCameraIntrinsics(mtx, dist, rvecs, tvecs)
    # Print out the reprojection error in pixels. Should be between 0.1 and 1.0
    print(ret)
