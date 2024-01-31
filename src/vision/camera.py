from queue import Queue, Empty, Full
import time
from typing import Any, Optional
import cv2
from cProfile import Profile
from threading import Thread, Event

from config import ConfigPaths, WorbotsConfig

# A frame along with the time it was captured
class TimedFrame:
    frame: Any
    timestamp: float

    def __init__(self, frame: Any, timestamp: float):
        self.frame = frame
        self.timestamp = timestamp

class WorbotsCamera:
    thread: Thread
    queue = Queue(1)
    stop: Event
    
    def __init__(self, configPaths: ConfigPaths):
        self.stop = Event()
        self.thread = Thread(target=runCameraThread, args=(self.stop, configPaths, self.queue,), name="Camera Thread", daemon=True)
        self.thread.start()

    def getFrame(self) -> Optional[TimedFrame]:
        try:
            return self.queue.get()
        except Empty:
            return None
        
    def stop(self):
        self.stop.set()
        self.thread.join()

def runCameraThread(stop: Event, configPaths: ConfigPaths, out: Queue):
    prof = Profile()
    config = WorbotsConfig(configPaths)
    if config.PROFILE:
        prof.enable()
    cam = ThreadCamera(configPaths)

    while not stop.is_set():
        frame = cam.getRawFrame()
        if frame is not None:
            try:
                out.put_nowait(frame)
            except Full:
                print("Drop")
                pass

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
        if self.worConfig.USE_GSTREAMER:
            print("Initializing camera with GStreamer...")
            cmd = ""
            if self.worConfig.USE_GPU:
                cmd = f"gst-launch-1.0 -v v4l2src device=/dev/video{self.worConfig.CAMERA_ID} ! image/jpeg, width={self.worConfig.RES_W}, height={self.worConfig.RES_H}, format=MJPG, framerate={self.worConfig.CAM_FPS}/1 ! jpegparse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=I420 ! appsink max-buffers=1 drop=1"
            else:
                cmd = f"gst-launch-1.0 -v v4l2src device=/dev/video{self.worConfig.CAMERA_ID} always-copy=false extra_controls=\"c,exposure_auto=false,exposure_absolute=1,gain=1,sharpness=0,brightness=0\" ! image/jpeg, width={self.worConfig.RES_W}, height={self.worConfig.RES_H}, format=MJPG, framerate={self.worConfig.CAM_FPS}/1 ! jpegdec ! videoconvert ! video/x-raw,format=I420 ! appsink max-buffers=1 drop=1"
            print("GStreamer command: " + cmd)

            self.cap = cv2.VideoCapture(cmd, cv2.CAP_GSTREAMER)
            self.startTime = time.time()
        else:
            print("Initializing camera with default backend...")
            self.cap = cv2.VideoCapture(0)
            self.startTime = time.time()
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.worConfig.RES_H)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.worConfig.RES_W)
            # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            # self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1.0)
            self.cap.set(cv2.CAP_PROP_FPS, self.worConfig.CAM_FPS)

        if self.cap.isOpened():
            print(f"Initialized camera with {self.cap.getBackendName()} backend")
            print(f"Camera running at {self.cap.get(cv2.CAP_PROP_FPS)} fps")
        else:
            print("Failed to initialize camera")

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
            return None
