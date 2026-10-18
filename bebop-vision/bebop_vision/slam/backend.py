"""Tracker backends: a thin cuVSLAM adapter plus a deterministic fake.

The rest of the live pipeline only knows ``TrackerBackend.track`` and
``TrackResult``; cuVSLAM is imported lazily so the package, HTTP service,
and tests run on a workstation without the wheel installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .calibration import FRAME_ID, RigCalibration, SlamConfigError


@dataclass(frozen=True)
class Pose:
    """A rigid pose in a named frame (xyzw quaternion)."""

    t_ns: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    frame_id: str = FRAME_ID

    def as_dict(self) -> dict:
        return {
            "t_ns": int(self.t_ns),
            "translation": [float(v) for v in self.translation],
            "rotation": [float(v) for v in self.rotation],
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class TrackResult:
    """One tracker step.

    ``pose`` is the continuous odometry pose.  ``corrected_pose`` is the
    optional SLAM/map-corrected pose and is kept separate on purpose: it may
    jump when the map is corrected, while odometry stays continuous.
    """

    pose: Pose | None
    corrected_pose: Pose | None = None
    tracking_state: str = "lost"
    error: str | None = None


@runtime_checkable
class TrackerBackend(Protocol):
    def track(self, timestamp_ns: int, rgb: np.ndarray,
              depth_mm: np.ndarray) -> TrackResult: ...

    def close(self) -> None: ...


def _vec(value, size: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != size or not np.all(np.isfinite(arr)):
        return None
    return arr


def _state_name(raw) -> str | None:
    if raw is None:
        return None
    name = getattr(raw, "name", None) or str(raw)
    name = str(name).split(".")[-1].strip().lower()
    if "lost" in name or "fail" in name or "not_track" in name:
        return "lost"
    if "track" in name or name in ("ok", "success"):
        return "tracking"
    return name


def _pose_from(value, t_ns: int) -> Pose | None:
    """Accept a PoseEstimate (world_from_rig.pose) or a bare Pose."""
    if value is None:
        return None
    wfr = getattr(value, "world_from_rig", None)
    if wfr is not None:
        value = getattr(wfr, "pose", wfr)
    tr = _vec(getattr(value, "translation", None), 3)
    rot = _vec(getattr(value, "rotation", None), 4)
    if tr is None or rot is None:
        return None
    norm = float(np.linalg.norm(rot))
    if norm <= 1e-9:
        return None
    rot = rot / norm
    return Pose(int(t_ns), tuple(float(v) for v in tr),
                tuple(float(v) for v in rot), FRAME_ID)


class CuVSLAMTracker:
    """NVIDIA cuVSLAM RGB-D odometry (optionally with SLAM correction)."""

    def __init__(self, calibration: RigCalibration, *, enable_slam: bool = False,
                 use_gpu: bool = True, map_path: str | None = None):
        try:
            import cuvslam as vslam
        except ImportError as exc:  # pragma: no cover - wheel is robot-only
            raise SlamConfigError(
                "live SLAM needs NVIDIA cuVSLAM (python module 'cuvslam'); "
                "install the cuVSLAM wheel into the bebop-vision venv") from exc

        self.vslam = vslam
        self.calibration = calibration
        self.enable_slam = bool(enable_slam)
        cal = calibration.color

        cam = vslam.Camera()
        cam.focal = (cal.fx, cal.fy)
        cam.principal = (cal.cx, cal.cy)
        cam.size = (cal.width, cal.height)
        # RGB-D odometry supports the calibrated polynomial model on Thor
        # v17.  Do not silently downgrade to pinhole: DepthRegistration
        # still projects with the polynomial coefficients, so a pinhole
        # tracker would be fed pixels it does not model.  Fail loudly.
        try:
            cam.distortion = vslam.Distortion(
                vslam.Distortion.Model.Polynomial,
                list(cal.distortion))
        except Exception as exc:
            raise SlamConfigError(
                "cuVSLAM rejected the calibrated polynomial distortion; "
                "refusing to silently fall back to pinhole (registration "
                f"would disagree): {type(exc).__name__}: {exc}") from exc

        rig = vslam.Rig()
        rig.cameras = [cam]
        rig.imus = []

        rgbd = vslam.Tracker.OdometryRGBDSettings()
        # We pass aligned uint16 millimetres.
        rgbd.depth_scale_factor = 1000.0
        rgbd.depth_camera_id = 0
        rgbd.enable_depth_stereo_tracking = False

        odom = vslam.Tracker.OdometryConfig(
            odometry_mode=vslam.Tracker.OdometryMode.RGBD,
            rgbd_settings=rgbd,
            use_gpu=bool(use_gpu),
            async_sba=True,
            enable_final_landmarks_export=False,
            enable_landmarks_export=False,
            enable_observations_export=False,
        )

        slam = None
        if self.enable_slam:
            slam = vslam.Tracker.SlamConfig()
            slam.use_gpu = bool(use_gpu)
            slam.sync_mode = False
            slam.map_cache_path = str(map_path) if map_path else ""
            slam.planar_constraints = False

        self._tracker = vslam.Tracker(rig, odom, slam)
        self.name = "cuvslam-rgbd" + ("-slam" if self.enable_slam else "")

    def track(self, timestamp_ns: int, rgb: np.ndarray,
              depth_mm: np.ndarray) -> TrackResult:
        try:
            pe, slam_pose = self._tracker.track(
                int(timestamp_ns),
                images=[np.ascontiguousarray(rgb)],
                depths=[np.ascontiguousarray(depth_mm)],
            )
        except Exception as exc:  # tracker faults must not kill the service
            return TrackResult(None, None, "error",
                               f"cuvslam track failed: {type(exc).__name__}: {exc}")

        state = _state_name(getattr(pe, "tracking_state", None))
        pose = _pose_from(pe, timestamp_ns)
        if pose is None or state == "lost":
            return TrackResult(None, None, "lost", None)
        corrected = None
        if self.enable_slam and slam_pose is not None:
            corrected = _pose_from(slam_pose, timestamp_ns)
        return TrackResult(pose, corrected, "tracking", None)

    def close(self) -> None:
        # The Python Tracker owns no explicit close in the 17.x bindings.
        self._tracker = None
