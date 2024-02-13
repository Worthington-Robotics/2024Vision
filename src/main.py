import cProfile
from queue import Empty
import queue
from threading import Thread, Event
import sys
import time
from typing import Any, Optional
import cv2
from wpimath.geometry import *
from cscore import CameraServer
from utils import MovingAverage
from vision import WorbotsVision
from network import WorbotsTables
from config import ConfigPaths, WorbotsConfig
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

    def __init__(self, configPaths: ConfigPaths):
        self.stop = Event()
        self.thread = Thread(target=runOutput, args=(self.stop, configPaths, self.inQueue,), name="Vision Output", daemon=True)
        self.thread.start()

    def sendData(self, data: OutputData):
        # If they are all none, there is no need to put anything in the queue
        if data.detection is None and data.frame is None and data.fps is None:
            return
        self.inQueue.put(data)

    def stop(self):
        self.stop.set()
        self.thread.join()

def runOutput(stop: Event, configPaths: ConfigPaths, inQueue: queue.Queue):
    config = WorbotsConfig(configPaths)
    network = WorbotsTables(configPaths)
    CameraServer.enableLogging()
    output = CameraServer.putVideo("Module"+str(config.MODULE_ID), config.RES_W, config.RES_H)
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
            output.putFrame(data.frame)
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
    args = parser.parse_args()
    print(f"Config path: {args.config_path} Calibration path: {args.calibration_path}")
    print(f"CUDA: {cv2.cuda.getCudaEnabledDeviceCount()}")
    paths = ConfigPaths(args.config_path, args.calibration_path)
    main(paths)
