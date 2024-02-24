import json
import numpy as np
from typing import Optional, Dict
   
class ConfigPaths:
    path: Optional[str] = "config.json"
    calibPath: Optional[str] = "calibration.json"

    def __init__(self, path, calibPath):
        self.path = path
        self.calibPath = calibPath

class WorbotsConfig:
    paths: ConfigPaths

    CAMERA_ID = 0
    TEAM_NUMBER = 4145
    MODULE_ID = None
    SIM_MODE = False
    RES_W = 1280
    RES_H = 720
    CAM_FPS = 60
    TAG_SIZE_METERS = 0.1524
    USE_GSTREAMER = False
    RUN_ONCE = False
    PROCESS_VIDEO = True
    SEND_POSE_DATA = True
    SHOW_IMAGE = False
    PRINT_FPS = True
    PROFILE = False
    # Whether to convert the video to black and white.
    # Should be disabled for grayscale cameras
    MAKE_BW = False
    USE_GPU = False
    # Whether to use exact frame timestamps by querying the camera
    USE_EXACT_TIMESTAMPS = True
    CAM_EXPOSURE = 1
    IGNORED_TAGS = []

    def __new__(cls, paths: ConfigPaths):
        with open(paths.path, "r") as read_file:
            data: Dict = json.load(read_file)

            val = data.get("CameraId")
            if val is not None:
                cls.CAMERA_ID = val
            val = data.get("TeamNumber")
            if val is not None:
                cls.TEAM_NUMBER = val
            val = data.get("ModuleId")
            if val is not None:
                cls.MODULE_ID = val
            val = data.get("SimMode")
            if val is not None:
                cls.SIM_MODE = val
            val = data.get("ResolutionW")
            if val is not None:
                cls.RES_W = val
            val = data.get("ResolutionH")
            if val is not None:
                cls.RES_H = val
            val = data.get("CameraFPS")
            if val is not None:
                cls.CAM_FPS = val
            val = data.get("TagSizeinMeters")
            if val is not None:
                cls.TAG_SIZE_METERS = val
            val = data.get("UseGStreamer")
            if val is not None:
                cls.USE_GSTREAMER = val
            val = data.get("RunOnce")
            if val is not None:
                cls.RUN_ONCE = val
            val = data.get("ProcessVideo")
            if val is not None:
                cls.PROCESS_VIDEO = val
            val = data.get("SendPoseData")
            if val is not None:
                cls.SEND_POSE_DATA = val
            val = data.get("ShowImage")
            if val is not None:
                cls.SHOW_IMAGE = val
            val = data.get("PrintFPS")
            if val is not None:
                cls.PRINT_FPS = val
            val = data.get("Profile")
            if val is not None:
                cls.PROFILE = val
            val = data.get("MakeBW")
            if val is not None:
                cls.MAKE_BW = val
            val = data.get("UseGPU")
            if val is not None:
                cls.USE_GPU = val
            val = data.get("UseExactTimestamps")
            if val is not None:
                cls.USE_EXACT_TIMESTAMPS = val
            val = data.get("CamExposure")
            if val is not None:
                cls.CAM_EXPOSURE = val
            val = data.get("IgnoredTags")
            if val is not None:
                cls.IGNORED_TAGS = val
        return super(WorbotsConfig, cls).__new__(cls)

    def __init__(self, paths: ConfigPaths):
        self.paths = paths

    def getKey(self, key) -> any:
        return WorbotsConfig.data[key]

    def saveCameraIntrinsics(self, cameraMatrix, cameraDist, rvecs, tvecs):
        intrinsics = {
            "cameraMatrix": cameraMatrix.tolist(),
            "cameraDist": cameraDist.tolist()
        }

        with open(self.paths.calibPath, "w") as f:
            json.dump(intrinsics, f)

    def getCameraIntrinsicsFromJSON(self):
        try:
            with open(self.paths.calibPath, "r") as f:
                data = json.load(f)

            cameraMatrix = np.array(data["cameraMatrix"])
            cameraDist = np.array(data["cameraDist"])

            return cameraMatrix, cameraDist

        except FileNotFoundError:
            print("Calibration file 'calibration.json' not found.")
            return None, None, None, None
        except (json.JSONDecodeError, KeyError):
            print("Error reading camera intrinsics from 'calibration.json'.")
            return None, None, None, None
