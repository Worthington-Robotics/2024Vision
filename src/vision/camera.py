import math
from queue import Queue, Empty, Full
import time
from typing import Any, Optional
import cv2
from cProfile import Profile
from threading import Thread, Event
import subprocess

from config import ConfigPaths, WorbotsConfig
from network.tables import WorbotsTables


class TimedFrame:
    """A frame along with the time it was captured"""

    frame: cv2.Mat
    timestamp: float

    def __init__(self, frame: cv2.Mat, timestamp: float):
        self.frame = frame
        self.timestamp = timestamp


class WorbotsCamera:
    thread: Thread
    queue = Queue(1)
    stop: Event

    def __init__(self, configPaths: ConfigPaths, tables: WorbotsTables):
        self.stop = Event()
        self.thread = Thread(
            target=runCameraThread,
            args=(
                self.stop,
                configPaths,
                tables,
                self.queue,
            ),
            name="Camera Thread",
            daemon=True,
        )
        self.thread.start()

    def getFrame(self) -> Optional[TimedFrame]:
        try:
            return self.queue.get()
        except Empty:
            return None

    def stop(self):
        self.stop.set()
        self.thread.join()


def runCameraThread(
    stop: Event, configPaths: ConfigPaths, tables: WorbotsTables, out: Queue
):
    prof = Profile()
    config = WorbotsConfig(configPaths)
    if config.PROFILE:
        prof.enable()
    cam = ThreadCamera(configPaths)

    while not stop.is_set():
        frame = cam.getRawFrame()
        if frame is not None:
            # Try to put a frame in the processing queue for the vision. If it is full,
            # then that means the camera is running faster than the vision and we have to throw the frame away.
            try:
                out.put_nowait(frame)
            except Full:
                # print("Drop")
                pass

        cam.checkConfig(tables)

        if config.RUN_ONCE:
            break

    if config.PROFILE:
        prof.disable()
        prof.dump_stats("prof/cam_prof")


class ThreadCamera:
    worConfig: WorbotsConfig
    cap: cv2.VideoCapture
    # Used for offsetting timestamps from the camera
    startTime: float

    def __init__(self, configPaths: ConfigPaths):
        self.worConfig = WorbotsConfig(configPaths)
        self.initializeCapture()

    def initializeCapture(self):
        if self.worConfig.USE_GSTREAMER:
            print("Initializing camera with GStreamer...")
            cmd = ""
            # Base v4l2 command
            cmd += f"gst-launch-1.0 -v v4l2src device=/dev/video{self.worConfig.CAMERA_ID} always-copy=false"
            # JPEG video
            cmd += f" ! image/jpeg, width={self.worConfig.RES_W}, height={self.worConfig.RES_H}, format=MJPG, framerate={self.worConfig.CAM_FPS}/1"
            # Choose decoder based on presence of GPU
            if self.worConfig.USE_GPU:
                cmd += f" ! jpegparse ! nvv4l2decoder ! nvvidconv"
            else:
                cmd += f" ! jpegdec ! videoconvert"
            # Finalize by converting to I420 and setting the appsink to pipe the data into OpenCV
            cmd += f" ! video/x-raw,format=I420 ! appsink max-buffers=1 drop=1"
            print("GStreamer command: " + cmd)

            self.cap = cv2.VideoCapture(cmd, cv2.CAP_GSTREAMER)
            self.startTime = time.time()

            # Configure the camera using v4l2-ctl
            subprocess.run(["v4l2-ctl", "-c", f"exposure_auto=1"])
            subprocess.run(
                ["v4l2-ctl", "-c", f"exposure_absolute={self.worConfig.CAM_EXPOSURE}"]
            )
            subprocess.run(
                ["v4l2-ctl", "-c", f"brightness={self.worConfig.CAM_BRIGHTNESS}"]
            )
            subprocess.run(
                ["v4l2-ctl", "-c", f"contrast={self.worConfig.CAM_CONTRAST}"]
            )
        else:
            print("Initializing camera with default backend...")
            # Use either MSMF or DSHOW on Windows. They both have benefits and drawbacks.
            self.cap = cv2.VideoCapture(self.worConfig.CAMERA_ID, cv2.CAP_MSMF)
            self.startTime = time.time()
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.worConfig.RES_H)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.worConfig.RES_W)
            # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            # self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1.0)
            self.cap.set(cv2.CAP_PROP_FPS, self.worConfig.CAM_FPS)
            # Makes exposure manually controlled
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            # Windows uses a lookup table for exposure with 14 different values. Here we are choosing
            # the closest value in the table by converting the exposure to seconds, then doing log_2 of it,
            # then truncating it to an integer
            exposure = int(math.log2(self.worConfig.CAM_EXPOSURE * 100 / 1e6))
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.worConfig.CAM_BRIGHTNESS)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.worConfig.CAM_CONTRAST)

        if self.cap.isOpened():
            print(f"Initialized camera with {self.cap.getBackendName()} backend")
            print(f"Camera running at {self.cap.get(cv2.CAP_PROP_FPS)} fps")
        else:
            print("Failed to initialize camera")

    # Reconnects the camera
    def reconnect(self):
        self.cap.release()
        self.initializeCapture()

    # Checks for remote config changes; should be run periodically
    def checkConfig(self, tables: WorbotsTables):
        return
        changes = tables.exposureSubscriber.readQueue()
        # Get the latest config change, modify the config, and reinitialize
        # the camera
        if len(changes) > 0:
            last = changes[len(changes) - 1]
            self.worConfig.CAM_EXPOSURE = last.value
            self.reconnect()

    def getRawFrame(self) -> Optional[TimedFrame]:
        ret, frame = self.cap.read()
        if self.worConfig.USE_EXACT_TIMESTAMPS:
            timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            timestamp += self.startTime
        else:
            timestamp = time.time()
        if ret:
            if self.worConfig.USE_GSTREAMER:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            return TimedFrame(frame, timestamp)
        else:
            print("Camera disconnected!")
            self.reconnect()
            return None
