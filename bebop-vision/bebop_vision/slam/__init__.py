"""Live observe-only visual localization (RGB-D POC).

Public surface (stable for tests/tools):

    load_calibration(role, config_dir=None, rig_path=None)
    RigCalibration, CameraModel
    DepthRegistration
    Pose, TrackResult, TrackerBackend
    CuVSLAMTracker, FakeTracker
    LiveSlamConfig, LiveSlam
    SlamStatusServer
    SlamConfigError, CalibrationError

The pose is ``world_from_camera_optical`` (x right, y down, z forward).
No wheel odometry and no IMU are used.  ``LiveSlam`` only consumes frames
from an existing rig slot; it never opens a camera.
"""

from .backend import CuVSLAMTracker, Pose, TrackerBackend, TrackResult
from .calibration import (FRAME_ID, CalibrationError, CameraModel,
                          RigCalibration, SlamConfigError, load_calibration)
from .depth import DepthRegistration
from .fake import FakeTracker
from .live import (IMU_REJECT_REASON, POSE_FRAME, LiveSlam, LiveSlamConfig,
                   json_safe)
from .map import (DepthGeometryConfig, DepthGeometryMap,
                  undistort_pixels)
from .server import DEFAULT_PORT, SlamStatusServer
from .structure import STRUCTURE_FRAME_ID, StructureConfig, StructureMap

__all__ = [
    "CameraModel",
    "CalibrationError",
    "CuVSLAMTracker",
    "DEFAULT_PORT",
    "DepthGeometryConfig",
    "DepthGeometryMap",
    "DepthRegistration",
    "FRAME_ID",
    "FakeTracker",
    "IMU_REJECT_REASON",
    "LiveSlam",
    "LiveSlamConfig",
    "POSE_FRAME",
    "Pose",
    "RigCalibration",
    "SlamConfigError",
    "SlamStatusServer",
    "StructureConfig",
    "StructureMap",
    "STRUCTURE_FRAME_ID",
    "TrackResult",
    "TrackerBackend",
    "json_safe",
    "load_calibration",
    "undistort_pixels",
]
