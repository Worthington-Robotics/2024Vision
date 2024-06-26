import math
import os
from pathlib import Path

# This environment variable is set here before we import cv2 because it only applies to the
# current process and OpenCV needs to read it. Without this flag, MSMF backend cameras take over a minute
# to initialize
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cProfile
from queue import Empty
import queue
from threading import Thread, Event
import sys
import time
from typing import Any, List, Optional
import cv2
from wpimath.geometry import *
from cscore import CameraServer
from utils import MovingAverage
from vision import WorbotsVision
from network import WorbotsTables
from config import ConfigPaths, WorbotsConfig
from vision.calibrate import calibrateCamLive, calibrateCameraImages
from vision.camera import WorbotsCamera
from detection import PoseDetection
from argparse import ArgumentParser

def main(configPaths: ConfigPaths):
    config = WorbotsConfig(configPaths)
    prof = cProfile.Profile()
    if config.PROFILE:
        prof.enable()

    camera = None
    output = None

    try:
        output = Output(configPaths)
        vision = WorbotsVision(configPaths)
        tables = WorbotsTables(configPaths)

        camera = WorbotsCamera(configPaths, tables)
        
        # Used so that the printed FPS is only updated every couple of frames so it doesnt look
        # so jittery
        averageFps = MovingAverage(80)
        lastFrameTime = time.time()
        while True:
            # Try to get a frame from the camera
            frame = camera.getFrame()

            # Process the frame
            if frame is None:
                continue

            timestamp = frame.timestamp
            poseDetection = None
            if config.PROCESS_VIDEO:
                frame, poseDetection = vision.processFrame(frame.frame)
            else:
                frame = None

            elapsed = time.time() - lastFrameTime
            lastFrameTime = time.time()
            if elapsed != 0.0:
                fps = 1 / elapsed
                averageFps.add(fps)

            if config.PRINT_FPS:
                sys.stdout.write(f"\rFPS: {averageFps.average()}")
                sys.stdout.flush()

            # Send processed data to the output
            output.sendData(OutputData(DetectionData(poseDetection, timestamp), frame, averageFps.average()))

            if config.RUN_ONCE:
                break
    except Exception as e:
        print(e)

    if config.PROFILE:
        prof.disable()
        prof.dump_stats("prof/prof")

    print("Stopping!")
    if camera is not None:
        camera.stop()
    if output is not None:
        output.stop()

class DetectionData:
    poseData: Optional[PoseDetection]
    timestamp: float

    def __init__(self, poseData, timestamp):
        self.poseData = poseData
        self.timestamp = timestamp

class OutputData:
    detection: Optional[DetectionData]
    frame: Optional[Any]
    fps: Optional[float]

    def __init__(self, detection, frame, fps):
        self.detection = detection
        self.frame = frame
        self.fps = fps

class Output:
    thread: Thread
    inQueue = queue.Queue()
    stop: Event
    lastSnapshotTime = [0.0]

    def __init__(self, configPaths: ConfigPaths):
        self.removeOldSnapshots()

        self.stop = Event()
        self.thread = Thread(target=runOutput, args=(self.stop, configPaths, self.inQueue, self.lastSnapshotTime), name="Vision Output", daemon=True)
        self.thread.start()

    def removeOldSnapshots(self):
        """Removes old snapshot images from the snapshots folder"""
        folder = "./snapshots"
        files = os.listdir(folder)
        files.sort()
        maxCount = 150
        if len(files) > maxCount:
            toRemove = files[:maxCount]
            for file in toRemove:
                os.remove(folder + "/" + file)

    def sendData(self, data: OutputData):
        # If they are all none, there is no need to put anything in the queue
        if data.detection is None and data.frame is None and data.fps is None:
            return
        self.inQueue.put(data)

    def stop(self):
        self.stop.set()
        self.thread.join()

def runOutput(stop: Event, configPaths: ConfigPaths, inQueue: queue.Queue, lastSnapshotTime: List[float]):
    config = WorbotsConfig(configPaths)
    network = WorbotsTables(configPaths)
    CameraServer.enableLogging()
    csOutput = CameraServer.putVideo("Module"+str(config.MODULE_ID), config.RES_W, config.RES_H)
    print(f"Optimized used?: {cv2.useOptimized()}")
    network.sendConfig()
    
    while not stop.is_set():
        try:
            data: OutputData = inQueue.get()
        except Empty:
            data = None
            continue

        # Pose detection
        if config.SEND_POSE_DATA and data.detection is not None:
            network.sendPoseDetection(data.detection.poseData, data.detection.timestamp)

        # Camera frames
        if data.frame is not None:
            csOutput.putFrame(data.frame)

            # Record snapshots
            if config.SNAPSHOT_INTERVAL != -1:
                currentTime = time.time()
                if currentTime - lastSnapshotTime[0] > config.SNAPSHOT_INTERVAL:
                    lastSnapshotTime[0] = currentTime
                    try:
                        path = Path("./snapshots")
                        path.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(path.joinpath(f"{time.time_ns()}.jpg")), data.frame)
                    except Exception as e:
                        print(f"Failed to record snapshot due to error:\n{e}")

            # Show image on screen
            if config.SHOW_IMAGE:
                cv2.imshow('image', data.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        # FPS
        if data.fps is not None:
            network.sendFps(data.fps)

        if config.RUN_ONCE:
            break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config-path", default="config.json", help="Path to the config file. Defaults to ./config.json")
    parser.add_argument("-C", "--calibration-path", default="calibration.json", help="Path to the camera calibration file. Defaults to ./calibration.json")
    parser.add_argument("--calibrate", default="false", help="Whether to run in calibration mode. Defaults to false")
    parser.add_argument("--calibrate-folder", default="false", help="Whether to run in calibration mode from the camera_images folder. Defaults to false")
    args = parser.parse_args()
    print(f"Config path: {args.config_path} Calibration path: {args.calibration_path}")
    print(f"CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}")
    configPaths = ConfigPaths(args.config_path, args.calibration_path)
    calibrate = args.calibrate == "true"
    calibrateFolder = args.calibrate_folder == "true"
    if calibrateFolder:
        calibrateCameraImages("./camera_images", configPaths)
    elif calibrate:
        calibrateCamLive(configPaths)
    else:
        main(configPaths)
