"""Shared types for bebop lab.

Every other lab module imports from here and nowhere else inside the
package, so implementers can work on disjoint files against one contract.

Conventions
-----------
* Time is ``t_ns``: int64 nanoseconds since the epoch, the MCAP ``log_time``.
  A *tick* is the recorder's frame-set instant: every per-tick channel
  (``/depth_*``, ``/color_*`` AU, ``/cmd_vel``, ``/odom``) of one tick shares
  one ``log_time``. IMU, firmware and note channels keep their own times.
* Body frame: x forward, y left, z up, origin on the floor under the mast.
  Camera optical frame: x right, y down, z forward.
* Images are numpy ``uint8`` BGR ``[H, W, 3]`` (OpenCV order, what the
  recorder decoded) at the recorded resolution; depth is ``uint16`` mm
  ``[H, W]``. Models do their own resize / crop / normalisation: the dataset
  stays raw so any architecture can consume it.
* Modalities are the names in :data:`MODALITIES`. A model asks for a subset
  through :class:`InputSpec`; the runner materialises exactly those.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------
# Modalities
# --------------------------------------------------------------------------

#: Per-tick image modalities (one array per tick).
IMAGE_MODALITIES = ("color_near", "color_far", "depth_near", "depth_far")
#: High-rate series: a Sample carries a window ending at the tick.
IMU_MODALITIES = ("imu_accel_near", "imu_gyro_near", "imu_accel_far", "imu_gyro_far")
#: Per-tick state series (nearest message within tolerance).
STATE_MODALITIES = ("odom", "cmd_vel")
#: Sparse / latest-at series (the value in force at the tick).
LATEST_MODALITIES = ("runtime", "intent", "place")
#: Event markers (all events inside the history window).
EVENT_MODALITIES = ("event",)

MODALITIES: Tuple[str, ...] = (IMAGE_MODALITIES + IMU_MODALITIES + STATE_MODALITIES
                               + LATEST_MODALITIES + EVENT_MODALITIES)

#: MCAP topic for each modality (series modalities map 1:1).
TOPIC_FOR = {
    "color_near": "/color_near", "color_far": "/color_far",
    "depth_near": "/depth_near", "depth_far": "/depth_far",
    "imu_accel_near": "/imu_accel_near", "imu_gyro_near": "/imu_gyro_near",
    "imu_accel_far": "/imu_accel_far", "imu_gyro_far": "/imu_gyro_far",
    "odom": "/odom", "cmd_vel": "/cmd_vel", "runtime": "/runtime",
    "intent": "/intent", "place": "/place", "event": "/event",
}

#: Structured dtypes of the series modalities (``t_ns`` first, always).
SERIES_DTYPES: Dict[str, np.dtype] = {
    "odom": np.dtype([("t_ns", "i8"), ("x", "f8"), ("y", "f8"), ("theta", "f8")]),
    "cmd_vel": np.dtype([("t_ns", "i8"), ("vx", "f8"), ("wz", "f8")]),
    "imu_accel_near": np.dtype([("t_ns", "i8"), ("device_us", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8")]),
    "imu_gyro_near": np.dtype([("t_ns", "i8"), ("device_us", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8")]),
    "imu_accel_far": np.dtype([("t_ns", "i8"), ("device_us", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8")]),
    "imu_gyro_far": np.dtype([("t_ns", "i8"), ("device_us", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8")]),
    # Firmware telemetry flattened to the fields the lab needs; absent fields
    # (older firmware) are NaN / -1. Non-telemetry frames are not series.
    "runtime": np.dtype([("t_ns", "i8"), ("mode", "i2"), ("estop", "i1"),
                         ("operator_stale", "i1"), ("seat_held", "i1"),
                         ("wheel_l_vel", "f4"), ("wheel_l_target", "f4"),
                         ("wheel_r_vel", "f4"), ("wheel_r_target", "f4"),
                         ("battery_v", "f4"), ("motor_a", "f4"), ("soc_pct", "f4")]),
    # task / target / run_id / run_started_ns arrived with drive runs
    # (2026-10-25); older recordings decode them as "" / 0.
    "intent": np.dtype([("t_ns", "i8"), ("text", "U128"), ("from_place", "U64"), ("to_place", "U64"),
                        ("task", "U32"), ("target", "U128"), ("run_id", "U32"),
                        ("run_started_ns", "i8")]),
    "event": np.dtype([("t_ns", "i8"), ("kind", "U32"), ("note", "U256")]),
    "place": np.dtype([("t_ns", "i8"), ("place_id", "U64"), ("name", "U64"), ("score", "f4"), ("margin", "f4")]),
}


# --------------------------------------------------------------------------
# Series
# --------------------------------------------------------------------------

class Series:
    """A time-sorted structured numpy array with ``t_ns`` as its first field.

    Thin and deliberately boring: the alignment helpers every modality needs.
    """

    def __init__(self, data: np.ndarray, name: str = ""):
        if data.dtype.names is None or data.dtype.names[0] != "t_ns":
            raise ValueError(f"{name or 'series'}: structured dtype with t_ns first required")
        order = np.argsort(data["t_ns"], kind="stable")
        self.data = data[order] if not np.all(order == np.arange(len(data))) else data
        self.name = name

    # -- basics ------------------------------------------------------------
    def __len__(self) -> int:
        return int(len(self.data))

    def __getitem__(self, key):
        return self.data[key]

    @property
    def t_ns(self) -> np.ndarray:
        return self.data["t_ns"]

    @property
    def fields(self) -> Tuple[str, ...]:
        return tuple(self.data.dtype.names)

    @classmethod
    def empty(cls, name: str) -> "Series":
        return cls(np.empty(0, dtype=SERIES_DTYPES[name]), name)

    # -- alignment ---------------------------------------------------------
    def nearest(self, t_ns: int, tol_ns: Optional[int] = None):
        """The record nearest ``t_ns`` (within ``tol_ns`` if given), else None."""
        if len(self.data) == 0:
            return None
        i = int(np.searchsorted(self.t_ns, t_ns))
        best, best_dt = None, None
        for j in (i - 1, i):
            if 0 <= j < len(self.data):
                dt = abs(int(self.t_ns[j]) - int(t_ns))
                if best_dt is None or dt < best_dt:
                    best, best_dt = j, dt
        if best is None or (tol_ns is not None and best_dt > tol_ns):
            return None
        return self.data[best]

    def latest_at(self, t_ns: int):
        """The last record with ``t <= t_ns``, else None."""
        if len(self.data) == 0:
            return None
        i = int(np.searchsorted(self.t_ns, t_ns, side="right")) - 1
        return self.data[i] if i >= 0 else None

    def window(self, t0_ns: int, t1_ns: int) -> np.ndarray:
        """Records with ``t0 <= t < t1`` (``t0 < t1``)."""
        a = int(np.searchsorted(self.t_ns, t0_ns, side="left"))
        b = int(np.searchsorted(self.t_ns, t1_ns, side="left"))
        return self.data[a:b]

    def interp(self, t_ns, fields: Sequence[str]) -> Dict[str, np.ndarray]:
        """Linear interpolation of float ``fields`` at ``t_ns`` (scalar or array)."""
        t = np.asarray(t_ns, dtype=np.float64)
        src = self.t_ns.astype(np.float64)
        return {f: np.interp(t, src, self.data[f].astype(np.float64)) for f in fields}

    def rate_hz(self) -> float:
        if len(self.data) < 2:
            return 0.0
        span = (int(self.t_ns[-1]) - int(self.t_ns[0])) / 1e9
        return (len(self.data) - 1) / span if span > 0 else 0.0


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_dict(self) -> dict:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "width": self.width, "height": self.height}


def mount_rotation(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Camera-optical -> body rotation (float64). Same convention as
    ``bebop_vision.master_bev.mount_rotation``: optical x right, y down,
    z forward; body x forward, y left, z up; negative pitch looks down."""
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cp, sp = math.cos(-p), math.sin(-p)
    cy, sy = math.cos(y), math.sin(y)
    r0 = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ r0


@dataclass
class CameraModel:
    """A pinhole camera on the mast (no distortion), body <-> pixel."""

    role: str
    intrinsics: Intrinsics
    height_m: float
    pitch_deg: float
    yaw_deg: float = 0.0

    @property
    def R_opt2body(self) -> np.ndarray:
        return mount_rotation(self.pitch_deg, self.yaw_deg)

    def project(self, pts_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Body points ``[N, 3]`` -> ``(u, v, z_cam)``; ``z_cam <= 0`` is behind."""
        p = np.asarray(pts_body, dtype=np.float64).reshape(-1, 3) - np.array([0.0, 0.0, self.height_m])
        pc = p @ self.R_opt2body  # == R_body2opt @ p per row
        z = pc[:, 2]
        K = self.intrinsics
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K.fx * pc[:, 0] / z + K.cx
            v = K.fy * pc[:, 1] / z + K.cy
        return u, v, z

    def floor_point(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Pixels -> body points on the floor plane ``[N, 3]`` (NaN where the
        ray does not hit the floor)."""
        K = self.intrinsics
        xn = (np.asarray(u, dtype=np.float64) - K.cx) / K.fx
        yn = (np.asarray(v, dtype=np.float64) - K.cy) / K.fy
        rays = np.stack([xn, yn, np.ones_like(xn)], axis=-1) @ self.R_opt2body.T
        with np.errstate(divide="ignore", invalid="ignore"):
            s = -self.height_m / rays[..., 2]
        s = np.where(rays[..., 2] < 0, s, np.nan)
        pts = rays * s[..., None]
        pts[..., 2] = 0.0
        return pts


@dataclass
class Calib:
    """Everything a model needs to relate pixels to the body frame.

    Built from the session's ``/calib`` JSON (depth intrinsics, mounts, self
    masks) plus, when present on this machine, the colour intrinsics in
    ``config/orbbec_intrinsics_<serial>.json`` (the MCAP does not carry them).
    ``overrides`` apply measured corrections (e.g. the rig pitch) without
    editing either source; a model sees the corrected values.
    """

    roles: Dict[str, str] = field(default_factory=dict)              # role -> serial
    depth: Dict[str, Intrinsics] = field(default_factory=dict)       # role -> depth K (848x480)
    color: Dict[str, Intrinsics] = field(default_factory=dict)       # role -> colour K (1280x800), if known
    mounts: Dict[str, Dict[str, float]] = field(default_factory=dict)  # role -> {height_m, pitch_deg, yaw_deg}
    self_mask: Dict[str, Any] = field(default_factory=dict)          # role -> shapes in depth coords
    recorded: dict = field(default_factory=dict)                     # the raw /calib payload
    overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)  # role -> mount fields

    def mount(self, role: str) -> Dict[str, float]:
        m = dict(self.mounts.get(role, {}))
        m.update(self.overrides.get(role, {}))
        return m

    def camera(self, role: str, image: str = "color") -> CameraModel:
        K = (self.color if image == "color" else self.depth).get(role)
        if K is None:
            raise KeyError(f"no {image} intrinsics for role {role!r}")
        m = self.mount(role)
        return CameraModel(role=role, intrinsics=K, height_m=float(m["height_m"]),
                           pitch_deg=float(m["pitch_deg"]), yaw_deg=float(m.get("yaw_deg", 0.0)))

    def as_dict(self) -> dict:
        return {
            "roles": self.roles,
            "depth": {r: k.as_dict() for r, k in self.depth.items()},
            "color": {r: k.as_dict() for r, k in self.color.items()},
            "mounts": self.mounts, "self_mask": self.self_mask,
            "recorded": self.recorded, "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Calib":
        return cls(roles=dict(d.get("roles", {})),
                   depth={r: Intrinsics(**k) for r, k in d.get("depth", {}).items()},
                   color={r: Intrinsics(**k) for r, k in d.get("color", {}).items()},
                   mounts={r: dict(m) for r, m in d.get("mounts", {}).items()},
                   self_mask=dict(d.get("self_mask", {})),
                   recorded=dict(d.get("recorded", {})),
                   overrides={r: dict(m) for r, m in d.get("overrides", {}).items()})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Calib":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Input spec and samples
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class InputSpec:
    """What a model wants per tick.

    ``modalities``: names from :data:`MODALITIES`.
    ``history_s``: per-modality look-back in seconds. For image / state
    modalities a history yields a list of past ticks' values (oldest first,
    the current tick last); for IMU modalities it sets the window length
    (default 0.5 s); for ``event`` the look-back for markers (default 3 s).
    ``tol_ns``: nearest-match tolerance for state series (default 50 ms).
    """

    modalities: Tuple[str, ...]
    history_s: Mapping[str, float] = field(default_factory=dict)
    tol_ns: int = 50_000_000

    def __post_init__(self):
        bad = [m for m in self.modalities if m not in MODALITIES]
        if bad:
            raise ValueError(f"unknown modalities {bad}; valid: {MODALITIES}")
        for m in self.history_s:
            if m not in self.modalities:
                raise ValueError(f"history_s for {m!r} which is not in modalities")

    def history(self, modality: str) -> float:
        if modality in self.history_s:
            return float(self.history_s[modality])
        if modality in IMU_MODALITIES:
            return 0.5
        if modality in EVENT_MODALITIES:
            return 3.0
        return 0.0

    def topics(self) -> Tuple[str, ...]:
        return tuple(TOPIC_FOR[m] for m in self.modalities)


@dataclass
class Sample:
    """One tick as a model sees it.

    ``data[modality]`` is:
      * image modality, no history: ``np.ndarray`` (BGR uint8 or uint16 mm);
        with history: ``list[np.ndarray]`` oldest first, current last. A
        missing frame (the far camera drops colour under load) is ``None``.
      * state modality (odom, cmd_vel): a structured record (``rec["x"]``),
        or ``None`` if nothing within tolerance; with history: the structured
        array over the window.
      * IMU modality: the structured array over the window ending at the tick.
      * latest modality (runtime, intent, place): the structured record in
        force at the tick, or ``None``.
      * event: the structured array of markers in the look-back window.
    ``flags`` carries per-tick quality bits from the session (``has_color_near``,
    ``has_color_far``, ``far_hold_last``, ``stall_window``, ...).
    """

    t_ns: int
    session: str
    index: int
    data: Dict[str, Any]
    calib: Optional[Calib] = None
    flags: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, modality: str):
        return self.data[modality]

    def get(self, modality: str, default=None):
        return self.data.get(modality, default)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

@dataclass
class Path2D:
    """A timed body-frame path: ``pts[K, N, 2]`` (x forward, y left, metres)
    at ``rel_t[N]`` seconds after the tick, ``K`` hypotheses with ``weights[K]``.
    Optional ``psi[K, N]`` heading and ``v[K, N]`` speed."""

    rel_t: np.ndarray
    pts: np.ndarray
    weights: Optional[np.ndarray] = None
    psi: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None

    def __post_init__(self):
        self.rel_t = np.asarray(self.rel_t, dtype=np.float32)
        self.pts = np.asarray(self.pts, dtype=np.float32)
        if self.pts.ndim == 2:
            self.pts = self.pts[None]
        if self.weights is None:
            self.weights = np.full(self.pts.shape[0], 1.0 / self.pts.shape[0], np.float32)

    @property
    def best(self) -> np.ndarray:
        """``[N, 2]`` of the highest-weight hypothesis."""
        return self.pts[int(np.argmax(self.weights))]


@dataclass
class Mask:
    """Dense per-pixel output for one camera: ``values[H, W]`` class ids
    (int) or ``[C, H, W]`` probabilities / logits (float). ``stride`` relates
    it to the recorded image; ``crop`` is ``(row0, row1, col0, col1)`` of
    the recorded image it covers (None = whole image)."""

    camera: str
    values: np.ndarray
    classes: Tuple[str, ...] = ()
    stride: int = 1
    crop: Optional[Tuple[int, int, int, int]] = None


@dataclass
class Embedding:
    vector: np.ndarray            # [D] float
    source: str = ""              # backbone id


@dataclass
class Classes:
    logits: np.ndarray            # [C]
    names: Tuple[str, ...] = ()


@dataclass
class Grid:
    """A body-frame grid: ``values[C, H, W]``, cell size and origin (metres,
    body frame, of cell (0, 0)'s corner)."""

    values: np.ndarray
    cell_m: float
    origin_xy: Tuple[float, float]
    channels: Tuple[str, ...] = ()


@dataclass
class Scalars:
    values: Dict[str, float]


Output = Any  # one of Path2D | Mask | Embedding | Classes | Grid | Scalars
OUTPUT_KINDS = (Path2D, Mask, Embedding, Classes, Grid, Scalars)


# --------------------------------------------------------------------------
# Model protocol
# --------------------------------------------------------------------------

@runtime_checkable
class Model(Protocol):
    """What the runner, the trainer and the viewer need from a model.

    ``name``: short unique id for runs; ``spec``: the modalities it reads;
    ``outputs``: output name -> kind class (e.g. ``{"path": Path2D}``).
    ``prepare`` runs once per session (load weights, build LUTs from
    ``calib``); ``step`` maps one Sample to its outputs. ``batch_step`` is
    optional and defaults to a loop over ``step``.
    """

    name: str
    spec: InputSpec
    outputs: Mapping[str, type]

    def prepare(self, calib: Optional[Calib], cfg: Optional[Mapping[str, Any]] = None) -> None: ...

    def step(self, sample: Sample) -> Dict[str, Output]: ...


def batch_step(model: Model, samples: Iterable[Sample]) -> list:
    """``model.batch_step`` when it exists, else a loop over ``step``."""
    fn = getattr(model, "batch_step", None)
    samples = list(samples)
    if fn is not None:
        return list(fn(samples))
    return [model.step(s) for s in samples]


# --------------------------------------------------------------------------
# Identity / versioning helpers
# --------------------------------------------------------------------------

def file_identity(path: Path, head_bytes: int = 1 << 20) -> str:
    """Cheap stable identity for a (large) file: size + sha1 of the first MiB.
    Enough to detect a different recording; not a content hash."""
    import hashlib
    p = Path(path)
    h = hashlib.sha1()
    with open(p, "rb") as f:
        h.update(f.read(head_bytes))
    return f"{p.stat().st_size}-{h.hexdigest()[:16]}"


def git_hash(repo: Optional[Path] = None) -> Optional[str]:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
