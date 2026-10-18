"""Per-camera calibration for live visual odometry.

Source of truth is the same pair the recorder uses:

    config/orbbec_rig.yaml                 role <-> serial
    config/orbbec_intrinsics_<serial>.json depth/color intrinsics +
                                           color->depth rigid transform

The live POC tracks in the **camera optical frame** (x right, y down, z
forward).  It deliberately does not turn the rig YAML's height/pitch/yaw
mount data into a robot-body transform: those values are incomplete (no
roll, no x/y offset), and silently assuming the missing components would
produce a pose that looks metric and body-relative while being wrong.  A
caller that needs a body pose must compose its own validated transform.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover - pyyaml is a project dep
    raise ImportError("pip install pyyaml") from exc

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
CONFIG_DIR = REPO_ROOT / "config"
FRAME_ID = "camera_optical"
POSE_FRAME = "world_from_camera_optical"

_DIST_KEYS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")


class SlamConfigError(ValueError):
    """A requested live-SLAM configuration cannot be served honestly."""


class CalibrationError(SlamConfigError):
    """Calibration is missing, malformed, or inconsistent with the frames."""


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics plus the 8-parameter rational distortion model."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = (0.0,) * 8

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width), matching numpy frame shapes."""
        return (int(self.height), int(self.width))

    def validate(self, name: str) -> None:
        if self.width <= 0 or self.height <= 0:
            raise CalibrationError(f"{name}: invalid size {self.width}x{self.height}")
        for key in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, key))
            if not math.isfinite(value) or (key in ("fx", "fy") and value <= 0):
                raise CalibrationError(f"{name}: invalid {key}={value!r}")
        if len(self.distortion) != 8:
            raise CalibrationError(f"{name}: expected 8 distortion coefficients")
        if not all(math.isfinite(float(v)) for v in self.distortion):
            raise CalibrationError(f"{name}: non-finite distortion coefficient")


@dataclass(frozen=True)
class RigCalibration:
    """One camera's live-odometry calibration (camera optical frame only)."""

    role: str
    serial: str
    depth: CameraModel
    color: CameraModel
    # Maps points in the color optical frame to the depth optical frame.
    # Translation is metres (the JSON stores millimetres).
    T_depth_from_color: np.ndarray
    source_path: str
    frame_id: str = FRAME_ID
    rig_source_path: str = ""

    @property
    def depth_shape(self) -> tuple[int, int]:
        return self.depth.shape

    @property
    def color_shape(self) -> tuple[int, int]:
        return self.color.shape

    @property
    def T_color_from_depth(self) -> np.ndarray:
        """Inverse of the factory color->depth transform (metres)."""
        R = self.T_depth_from_color[:3, :3]
        t = self.T_depth_from_color[:3, 3]
        out = np.eye(4, dtype=np.float64)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    def summary(self) -> dict:
        return {
            "role": self.role,
            "serial": self.serial,
            "depth_shape": [int(self.depth.width), int(self.depth.height)],
            "color_shape": [int(self.color.width), int(self.color.height)],
            "frame_id": self.frame_id,
            "source": self.source_path,
            "rig_source": self.rig_source_path,
        }


def _require(mapping: dict, key: str, where: str):
    if not isinstance(mapping, dict) or key not in mapping:
        raise CalibrationError(f"{where}: missing {key!r}")
    return mapping[key]


def _camera_model(data: dict, prefix: str, where: str) -> CameraModel:
    if prefix == "color":
        width = _require(data, "color_width", where)
        height = _require(data, "color_height", where)
        fx = _require(data, "color_fx", where)
        fy = _require(data, "color_fy", where)
        cx = _require(data, "color_cx", where)
        cy = _require(data, "color_cy", where)
        dist_raw = _require(data, "color_rgb_distortion", where)
    else:
        width = _require(data, "width", where)
        height = _require(data, "height", where)
        fx = _require(data, "fx", where)
        fy = _require(data, "fy", where)
        cx = _require(data, "cx", where)
        cy = _require(data, "cy", where)
        dist_raw = data.get("distortion")
    try:
        distortion = tuple(float(dist_raw[k]) for k in _DIST_KEYS) \
            if isinstance(dist_raw, dict) else (0.0,) * 8
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(f"{where}: malformed {prefix} distortion") from exc
    model = CameraModel(int(width), int(height), float(fx), float(fy),
                        float(cx), float(cy), distortion)
    model.validate(f"{where}:{prefix}")
    return model


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise CalibrationError(f"calibration file not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read calibration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"{path}: expected a JSON object")
    return data


def load_calibration(role: str, config_dir: Path | None = None,
                     rig_path: Path | None = None) -> RigCalibration:
    """Load and validate one role's intrinsics/extrinsics from config/.

    ``rig_path`` overrides the default ``<config_dir>/orbbec_rig.yaml``
    (e.g. an alternate rig YAML passed as ``--rig``); its camera entry
    selects the serial.  Intrinsics come from ``config_dir`` when given,
    otherwise from the rig YAML's directory.

    Raises CalibrationError for missing files, missing/NaN intrinsics, bad
    color->depth transforms, or a role that is not configured.  The returned
    object is always labelled ``frame_id="camera_optical"``.
    """
    if rig_path is not None:
        rig_path = Path(rig_path).expanduser()
        root = Path(config_dir) if config_dir is not None else rig_path.parent
    else:
        root = Path(config_dir) if config_dir is not None else CONFIG_DIR
        rig_path = root / "orbbec_rig.yaml"
    try:
        with open(rig_path, encoding="utf-8") as f:
            rig = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise CalibrationError(f"rig config not found: {rig_path}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"cannot read rig config {rig_path}: {exc}") from exc

    try:
        cameras = rig["robots"]["default"]["cameras"]
    except (KeyError, TypeError) as exc:
        raise CalibrationError(f"{rig_path}: missing robots.default.cameras") from exc

    serial = None
    for candidate, cfg in (cameras or {}).items():
        if isinstance(cfg, dict) and cfg.get("role") == role:
            serial = str(candidate)
            break
    if serial is None:
        raise CalibrationError(f"{rig_path}: no camera with role {role!r}")

    intr_path = root / f"orbbec_intrinsics_{serial}.json"
    data = _load_json(intr_path)
    where = str(intr_path)

    depth = _camera_model(data, "depth", where)
    color = _camera_model(data, "color", where)

    ext = _require(data, "color_to_depth_transform", where)
    try:
        rot = np.asarray(ext["rotation"], dtype=np.float64).reshape(3, 3)
        trans_mm = np.asarray(ext["translation"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError(
            f"{where}: color_to_depth_transform needs 9 rotation + 3 "
            f"translation values") from exc
    if not np.all(np.isfinite(rot)) or not np.all(np.isfinite(trans_mm)):
        raise CalibrationError(f"{where}: non-finite color_to_depth_transform")
    ortho_err = float(np.max(np.abs(rot @ rot.T - np.eye(3))))
    det = float(np.linalg.det(rot))
    if ortho_err > 1e-2 or det <= 0.5:
        raise CalibrationError(
            f"{where}: color_to_depth rotation is not a proper rotation "
            f"(orthonormal error {ortho_err:.3g}, det {det:.3g})")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = trans_mm / 1000.0

    return RigCalibration(
        role=role,
        serial=serial,
        depth=depth,
        color=color,
        T_depth_from_color=T,
        source_path=str(intr_path),
        frame_id=FRAME_ID,
        rig_source_path=str(rig_path),
    )
