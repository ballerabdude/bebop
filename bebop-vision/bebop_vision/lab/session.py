"""One recorded drive as lazy, time-aligned numpy series.

A :class:`Session` wraps one navd MCAP. The first access builds an index
under the lab cache (one streaming pass with ``mcap_tools.iter_messages``,
which also handles a footerless file): every series modality as a
structured array in ``series.npz``, colour access units / JPEGs
concatenated per role into ``video_<role>.bin`` with a structured index,
depth PNGs likewise, the calibration as ``calib.json`` and the raw tick
stamps with their quality flags in ``ticks.npy``. ``meta.json`` carries
the MCAP's identity and this module's producer version, so a stale index
is rebuilt instead of trusted.

:class:`Ticks` is the sampling surface a model or a trainer uses: the raw
tick stamps (the recorder's frame-set instants, ``/depth_near`` log times)
or those snapped to a Hz grid, plus an :class:`InputSpec` that turns one
tick into a :class:`Sample` with exactly the requested modalities.

Only ``bebop_vision.lab.types`` and ``bebop_vision.lab.frames`` (this
module's own payload readers) are imported from the lab package.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from bebop_vision.mcap_tools import McapInfo, ProtoDecoder, field_value, iter_messages, read_info
from bebop_vision.lab.frames import (DEPTH_INDEX_DTYPE, VIDEO_INDEX_DTYPE, DepthSeries,
                                     FrameSeries, is_keyframe)
from bebop_vision.lab.types import (EVENT_MODALITIES, IMAGE_MODALITIES, IMU_MODALITIES,
                                    LATEST_MODALITIES, SERIES_DTYPES, STATE_MODALITIES,
                                    TOPIC_FOR, Calib, InputSpec, Intrinsics, Sample, Series,
                                    file_identity)

#: Bump when the index layout or its semantics change; readers rebuild.
#: 2: meta.json carries ``payload_bytes`` so a missing or truncated ``.bin``
#: invalidates the cache instead of failing at frame access.
#: 3: the ``intent`` series carries ``task``, ``target``, ``run_id`` and
#: ``run_started_ns`` (drive runs); an older index lacks those columns.
PRODUCER = "session@3"

#: Per raw tick quality flags (``Session.flags``).
FLAGS_DTYPE = np.dtype([("t_ns", "i8"), ("has_color_near", "?"), ("has_color_far", "?"),
                        ("far_hold_last", "?"), ("gap_before_ms", "f4")])
_FLAG_FIELDS = ("has_color_near", "has_color_far", "far_hold_last", "gap_before_ms")

_ROLES = ("near", "far")
_COLOR_TOPICS = {"/color_near": "near", "/color_far": "far"}
_DEPTH_TOPICS = {"/depth_near": "near", "/depth_far": "far"}
_TOPIC_TO_SERIES = {TOPIC_FOR[m]: m for m in SERIES_DTYPES}
_PAYLOADS = tuple(f"{kind}_{role}" for kind in ("video", "depth") for role in _ROLES)
_REPO_ROOT = Path(__file__).resolve().parents[2]      # bebop-vision/
_DEFAULT_CACHE_ROOT = _REPO_ROOT / "datasets" / "lab"
_INTRINSICS_DIR = _REPO_ROOT / "config"
_RIG_YAML = _INTRINSICS_DIR / "orbbec_rig.yaml"


def cache_root() -> Path:
    """``$BEBOP_LAB_CACHE`` or ``bebop-vision/datasets/lab``."""
    env = os.environ.get("BEBOP_LAB_CACHE")
    return Path(env).expanduser() if env else _DEFAULT_CACHE_ROOT


def sample_stamps(stamps: Sequence[int], hz: float) -> np.ndarray:
    """Snap sorted raw stamps to a ``1/hz`` grid, picking the nearest raw
    stamp for each grid point and never repeating one. Same semantics as
    ``tools/mcap_extract._sample_stamps`` (re-implemented rather than
    imported: that module pulls in the recovery reader and is not a
    package). The recorder writes at the camera rate, not at a round Hz, so
    a minimum-interval filter would undersample; snapping keeps the
    intended rate and a real frame (never an interpolated one)."""
    stamps = np.asarray(stamps, dtype=np.int64)
    if len(stamps) == 0:
        return stamps
    step_ns = int(round(1e9 / max(float(hz), 1e-3)))
    targets = np.arange(int(stamps[0]), int(stamps[-1]) + 1, step_ns, dtype=np.int64)
    pos = np.searchsorted(stamps, targets)
    pos = np.clip(pos, 1, len(stamps) - 1)
    left, right = stamps[pos - 1], stamps[pos]
    # ties go to the later stamp, as the reference's `<=` does
    pick = np.where(np.abs(right - targets) <= np.abs(left - targets), pos, pos - 1)
    if len(stamps) == 1:
        pick = np.zeros_like(targets)
    chosen = stamps[pick]
    keep = np.r_[True, chosen[1:] != chosen[:-1]]
    return chosen[keep]


# --------------------------------------------------------------------------
# Index build
# --------------------------------------------------------------------------

def _runtime_row(t_ns: int, tel) -> tuple:
    """Flatten a ``TelemetryFrame`` into the ``runtime`` dtype; fields the
    recording's schema lacks (older firmware) are NaN / -1."""
    nan = float("nan")

    def sub(msg, name):
        v = field_value(msg, name)
        if v is None:
            return None
        try:
            return v if msg.HasField(name) else None
        except ValueError:
            return v

    def flag(msg, name):
        v = field_value(msg, name)
        return -1 if v is None else int(bool(v))

    def f32(msg, name):
        v = field_value(msg, name)
        return nan if v is None else float(v)

    mode = field_value(tel, "mode")
    drive = sub(tel, "drive")
    power = sub(tel, "power")
    wheels = list(field_value(tel, "wheels", []) or [])
    wl = wheels[0] if len(wheels) > 0 else None
    wr = wheels[1] if len(wheels) > 1 else None
    return (int(t_ns), -1 if mode is None else int(mode), flag(tel, "estop_latched"),
            flag(drive, "operator_stale") if drive is not None else -1,
            flag(drive, "has_active_operator") if drive is not None else -1,
            f32(wl, "velocity_rad_s"), f32(wl, "target_velocity_rad_s"),
            f32(wr, "velocity_rad_s"), f32(wr, "target_velocity_rad_s"),
            f32(power, "battery_voltage_v"), f32(power, "total_motor_current_a"),
            f32(power, "state_of_charge_pct"))


def _series_row(name: str, t_ns: int, msg) -> tuple:
    fv = field_value
    if name == "odom":
        return (t_ns, float(fv(msg, "x", 0.0)), float(fv(msg, "y", 0.0)), float(fv(msg, "theta", 0.0)))
    if name == "cmd_vel":
        return (t_ns, float(fv(msg, "vx", 0.0)), float(fv(msg, "wz", 0.0)))
    if name.startswith("imu_accel"):
        return (t_ns, int(fv(msg, "device_stamp_us", -1)), float(fv(msg, "ax", 0.0)),
                float(fv(msg, "ay", 0.0)), float(fv(msg, "az", 0.0)))
    if name.startswith("imu_gyro"):
        return (t_ns, int(fv(msg, "device_stamp_us", -1)), float(fv(msg, "gx", 0.0)),
                float(fv(msg, "gy", 0.0)), float(fv(msg, "gz", 0.0)))
    if name == "intent":
        # field_value is absent-safe: a recording made before drive runs has
        # an embedded schema without task / target / run_id / run_started_ns,
        # and those decode as "" / 0.
        return (t_ns, str(fv(msg, "text", "")), str(fv(msg, "from_place", "")),
                str(fv(msg, "to_place", "")), str(fv(msg, "task", "")), str(fv(msg, "target", "")),
                str(fv(msg, "run_id", "")), int(fv(msg, "run_started_ns", 0) or 0))
    if name == "event":
        return (t_ns, str(fv(msg, "kind", "")), str(fv(msg, "note", "")))
    if name == "place":
        return (t_ns, str(fv(msg, "place_id", "")), str(fv(msg, "name", "")),
                float(fv(msg, "score", 0.0)), float(fv(msg, "margin", 0.0)))
    raise KeyError(name)


def _roles_by_serial(recorded: Mapping[str, Any], rig_yaml: Optional[Path] = None) -> Dict[str, str]:
    """role -> serial for the serials the recorded ``/calib`` mentions.

    The payload keys intrinsics and mounts by serial and says nothing about
    roles. The rig YAML on this machine (``orbbec.load_rig_config``)
    resolves the real robot's serials; for anything else the steeper mount
    is the near camera (the rig has near at -60..-66 deg and far at -9..-17
    deg, by design). A rig YAML that is present but unreadable is reported
    with a warning rather than silently ignored: on the real rig both paths
    give the same roles, so the fallback would hide the misconfiguration."""
    serials = list(recorded.get("mounts", {}).keys()) or list(recorded.get("intrinsics", {}).keys())
    roles: Dict[str, str] = {}
    rig_yaml = Path(rig_yaml or _RIG_YAML)
    if rig_yaml.exists():
        import yaml   # orbbec imports it anyway; named here for its error class
        from bebop_vision.orbbec import load_rig_config
        try:
            cams = load_rig_config(rig_yaml)["robots"]["default"]["cameras"]
            for serial, cfg in cams.items():
                if serial in serials and isinstance(cfg, Mapping) and cfg.get("role") in _ROLES:
                    roles[cfg["role"]] = serial
        except (OSError, yaml.YAMLError, KeyError, TypeError, AttributeError) as exc:
            warnings.warn(f"{rig_yaml}: unreadable rig config ({exc!r}); camera roles fall back "
                          f"to the pitch heuristic", RuntimeWarning, stacklevel=2)
    rest = [s for s in serials if s not in roles.values()]
    if rest:
        mounts = recorded.get("mounts", {})
        rest.sort(key=lambda s: float(mounts.get(s, {}).get("pitch_deg", 0.0)))
        for role, serial in zip([r for r in _ROLES if r not in roles], rest):
            roles[role] = serial
    return roles


def color_intrinsics(serial: str, intrinsics_dir: Optional[Path] = None) -> Optional[Intrinsics]:
    """Colour intrinsics from ``config/orbbec_intrinsics_<serial>.json``
    (``orbbec.load_intrinsics``; the camera service dumps the file at open
    time) or None when the file is not on this machine. A file that is
    present but malformed warns, since silently losing the colour camera
    model is what a projection bug looks like downstream."""
    from bebop_vision.orbbec import intrinsics_path, load_intrinsics
    intrinsics_dir = Path(intrinsics_dir or _INTRINSICS_DIR)
    p = intrinsics_path(serial, intrinsics_dir)
    if not p.exists():
        return None
    try:
        j = load_intrinsics(serial, intrinsics_dir)
        if "color_fx" not in j:
            return None
        return Intrinsics(fx=float(j["color_fx"]), fy=float(j["color_fy"]), cx=float(j["color_cx"]),
                          cy=float(j["color_cy"]), width=int(j.get("color_width", 0)),
                          height=int(j.get("color_height", 0)))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        warnings.warn(f"{p}: unreadable colour intrinsics ({exc!r}); role has no colour camera model",
                      RuntimeWarning, stacklevel=2)
        return None


def fill_color_intrinsics(calib: Calib, intrinsics_dir: Optional[Path] = None) -> Calib:
    """Add colour intrinsics for every role that still lacks them, from the
    config files on this machine. Called when a Session's calibration is
    loaded (not only when the index is built), so a dump that appears later
    is picked up without ``index(force=True)``."""
    for role, serial in calib.roles.items():
        if role not in calib.color:
            k = color_intrinsics(serial, intrinsics_dir)
            if k is not None:
                calib.color[role] = k
    return calib


def build_calib(recorded: Mapping[str, Any], overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
                intrinsics_dir: Optional[Path] = None, rig_yaml: Optional[Path] = None) -> Calib:
    """A :class:`Calib` from the recorded ``/calib`` JSON, with colour
    intrinsics from ``config/orbbec_intrinsics_<serial>.json`` when that
    file exists here (the MCAP carries depth intrinsics only)."""
    recorded = dict(recorded or {})
    roles = _roles_by_serial(recorded, rig_yaml)
    depth: Dict[str, Intrinsics] = {}
    mounts: Dict[str, Dict[str, float]] = {}
    for role, serial in roles.items():
        k = recorded.get("intrinsics", {}).get(serial)
        if k:
            depth[role] = Intrinsics(fx=float(k["fx"]), fy=float(k["fy"]), cx=float(k["cx"]),
                                     cy=float(k["cy"]), width=int(k.get("width", 0)),
                                     height=int(k.get("height", 0)))
        m = recorded.get("mounts", {}).get(serial)
        if m:
            mounts[role] = {kk: float(v) for kk, v in m.items()}
    self_mask = {}
    for role in set(recorded.get("camera_self_mask_pixels", {})) | set(recorded.get("camera_self_mask_polygons", {})):
        self_mask[role] = {"rects": recorded.get("camera_self_mask_pixels", {}).get(role, []),
                           "polygons": recorded.get("camera_self_mask_polygons", {}).get(role, [])}
    calib = Calib(roles=roles, depth=depth, color={}, mounts=mounts, self_mask=self_mask,
                  recorded=recorded, overrides={r: dict(m) for r, m in (overrides or {}).items()})
    return fill_color_intrinsics(calib, intrinsics_dir)


class _Builder:
    """One streaming pass over the MCAP into the cache directory."""

    def __init__(self, path: Path, info: McapInfo, out: Path):
        self.path = path
        self.info = info
        self.out = out
        self.decoder = ProtoDecoder()
        self.rows: Dict[str, List[tuple]] = {n: [] for n in SERIES_DTYPES}
        self.calib: dict = {}
        self.video_idx: Dict[str, List[tuple]] = {r: [] for r in _ROLES}
        self.depth_idx: Dict[str, List[tuple]] = {r: [] for r in _ROLES}
        self.far_hold: List[Tuple[int, bool]] = []
        self.counts: Dict[str, int] = {}
        self.unknown_topics: set = set()
        self._tick_source = ""

    def run(self) -> dict:
        topics = [t for t in self.info.topics()]
        files = {}
        for role in _ROLES:
            files[f"video_{role}"] = open(self.out / f"video_{role}.bin", "wb")
            files[f"depth_{role}"] = open(self.out / f"depth_{role}.bin", "wb")
        offsets = {k: 0 for k in files}
        last_far: Optional[bytes] = None
        t_start = time.perf_counter()
        try:
            for schema, channel, message in iter_messages(self.path, topics, footerless=self.info.footerless):
                topic = channel.topic
                t_ns = int(message.log_time)
                self.counts[topic] = self.counts.get(topic, 0) + 1
                if topic == "/calib":
                    try:
                        self.calib = json.loads(message.data)
                    except (TypeError, ValueError):
                        pass
                    continue
                if channel.message_encoding != "protobuf" or schema is None:
                    self.unknown_topics.add(topic)
                    continue
                if topic in _COLOR_TOPICS or topic in _DEPTH_TOPICS:
                    msg = self.decoder.decode(schema, message.data)
                    data = bytes(field_value(msg, "data", b""))
                    fmt = str(field_value(msg, "format", "") or "").lower()
                    if topic in _COLOR_TOPICS:
                        role = _COLOR_TOPICS[topic]
                        if not fmt:
                            fmt = "jpeg" if schema.name.endswith("CompressedImage") else "h265"
                        if fmt in ("jpg", "mjpg", "mjpeg"):
                            fmt = "jpeg"
                        key = f"video_{role}"
                        self.video_idx[role].append((t_ns, offsets[key], len(data), is_keyframe(data, fmt), fmt))
                        if role == "far":
                            self.far_hold.append((t_ns, last_far is not None and data == last_far))
                            last_far = data
                    else:
                        role = _DEPTH_TOPICS[topic]
                        key = f"depth_{role}"
                        self.depth_idx[role].append((t_ns, offsets[key], len(data)))
                    files[key].write(data)
                    offsets[key] += len(data)
                    continue
                name = _TOPIC_TO_SERIES.get(topic)
                if name is None:
                    self.unknown_topics.add(topic)
                    continue
                msg = self.decoder.decode(schema, message.data)
                if name == "runtime":
                    # Only telemetry frames are a series; acks, snapshots and
                    # mode-change pushes on the same channel are skipped.
                    try:
                        which = msg.WhichOneof("payload")
                    except ValueError:
                        which = None
                    if which != "telemetry":
                        continue
                    self.rows[name].append(_runtime_row(t_ns, msg.telemetry))
                else:
                    self.rows[name].append(_series_row(name, t_ns, msg))
        finally:
            for f in files.values():
                f.close()
        elapsed = time.perf_counter() - t_start

        series = {}
        for name, dtype in SERIES_DTYPES.items():
            arr = np.array(self.rows[name], dtype=dtype) if self.rows[name] else np.empty(0, dtype=dtype)
            arr = arr[np.argsort(arr["t_ns"], kind="stable")]
            series[name] = arr
        np.savez(self.out / "series.npz", **series)
        for role in _ROLES:
            v = np.array(self.video_idx[role], dtype=VIDEO_INDEX_DTYPE) if self.video_idx[role] \
                else np.empty(0, dtype=VIDEO_INDEX_DTYPE)
            np.save(self.out / f"video_{role}.npy", v[np.argsort(v["t_ns"], kind="stable")])
            d = np.array(self.depth_idx[role], dtype=DEPTH_INDEX_DTYPE) if self.depth_idx[role] \
                else np.empty(0, dtype=DEPTH_INDEX_DTYPE)
            np.save(self.out / f"depth_{role}.npy", d[np.argsort(d["t_ns"], kind="stable")])

        flags = self._flags(series)
        np.save(self.out / "ticks.npy", flags)
        build_calib(self.calib).save(self.out / "calib.json")
        fmt = {r: (str(self.video_idx[r][0][4]) if self.video_idx[r] else None) for r in _ROLES}
        meta = {
            "source": str(self.path), "identity": file_identity(self.path), "producer": PRODUCER,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "index_seconds": round(elapsed, 3),
            "footerless": bool(self.info.footerless), "message_counts": self.counts,
            "ticks": int(len(flags)), "tick_source": self._tick_source,
            "color_fmt": fmt, "unknown_topics": sorted(self.unknown_topics),
            "series_counts": {n: int(len(a)) for n, a in series.items()},
            "keyframes": {r: int(sum(1 for x in self.video_idx[r] if x[3])) for r in _ROLES},
            "payload_bytes": {k: int(offsets[k]) for k in _PAYLOADS},
        }
        # meta.json last: a crash mid-build leaves no meta and the next
        # Session rebuilds.
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def _flags(self, series: Mapping[str, np.ndarray]) -> np.ndarray:
        depth_near = np.array(sorted({t for t, _, _ in self.depth_idx["near"]}), dtype=np.int64)
        if len(depth_near):
            ticks, self._tick_source = depth_near, "/depth_near"
        else:
            ticks, self._tick_source = np.unique(series["odom"]["t_ns"]), "/odom"
        out = np.zeros(len(ticks), dtype=FLAGS_DTYPE)
        out["t_ns"] = ticks
        near_t = np.array([x[0] for x in self.video_idx["near"]], dtype=np.int64)
        far_t = np.array([x[0] for x in self.video_idx["far"]], dtype=np.int64)
        out["has_color_near"] = np.isin(ticks, near_t)
        out["has_color_far"] = np.isin(ticks, far_t)
        if self.far_hold:
            held = {t for t, h in self.far_hold if h}
            out["far_hold_last"] = np.fromiter((int(t) in held for t in ticks), dtype=bool, count=len(ticks))
        gaps = np.diff(ticks).astype(np.float64) / 1e6 if len(ticks) > 1 else np.empty(0)
        out["gap_before_ms"] = np.r_[0.0, gaps].astype(np.float32)
        return out


# --------------------------------------------------------------------------
# Session
# --------------------------------------------------------------------------

class Session:
    """One drive MCAP with a lazily built, cached index (see the module
    docstring). ``overrides`` are mount corrections per role, applied to
    the calibration last (``{"near": {"pitch_deg": -59.0}}``)."""

    def __init__(self, path, cache_dir: Optional[Path] = None,
                 overrides: Optional[Mapping[str, Mapping[str, float]]] = None):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.name = self.path.stem
        self.cache_dir = Path(cache_dir) if cache_dir is not None else cache_root() / self.name
        self.overrides = {r: dict(m) for r, m in (overrides or {}).items()}
        self.info: McapInfo = read_info(self.path)
        self.topics: Tuple[str, ...] = tuple(self.info.topics())
        self._meta: Optional[dict] = None
        self._series: Optional[Dict[str, Series]] = None
        self._calib: Optional[Calib] = None
        self._flags: Optional[np.ndarray] = None
        self._grids: Dict[float, np.ndarray] = {}
        self._frames: Dict[str, FrameSeries] = {}
        self._depth: Dict[str, DepthSeries] = {}

    def __repr__(self) -> str:
        return f"Session({self.name!r}, {self.duration_s:.1f}s, {len(self.topics)} topics)"

    # -- time ---------------------------------------------------------------
    @property
    def start_ns(self) -> int:
        return int(self.info.start_ns or 0)

    @property
    def end_ns(self) -> int:
        return int(self.info.end_ns or 0)

    @property
    def duration_s(self) -> float:
        return self.info.duration_s

    # -- cache --------------------------------------------------------------
    _ARTIFACTS = ("series.npz", "ticks.npy", "calib.json", "video_near.npy", "video_far.npy",
                  "depth_near.npy", "depth_far.npy")

    def _cache_valid(self) -> Optional[dict]:
        meta_path = self.cache_dir / "meta.json"
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("producer") != PRODUCER or meta.get("identity") != file_identity(self.path):
            return None
        if not all((self.cache_dir / a).exists() for a in self._ARTIFACTS):
            return None
        # The payload stores are the big files an operator is tempted to
        # clean up; a missing or truncated one must rebuild, not raise at
        # the first frame access.
        sizes = meta.get("payload_bytes")
        if not isinstance(sizes, dict):
            return None
        for key in _PAYLOADS:
            p = self.cache_dir / f"{key}.bin"
            if not p.exists() or p.stat().st_size != int(sizes.get(key, -1)):
                return None
        return meta

    @property
    def indexed(self) -> bool:
        """A valid index exists (loaded, or on disk for this file and this
        producer). Cheap: reads ``meta.json`` and stats the artifacts; never
        builds."""
        return self._meta is not None or self._cache_valid() is not None

    def index(self, force: bool = False) -> Path:
        """Build the cache directory when missing, stale or ``force``; return it."""
        if not force and self._meta is not None:
            return self.cache_dir
        meta = None if force else self._cache_valid()
        if meta is None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            meta = _Builder(self.path, self.info, self.cache_dir).run()
            self._series = self._calib = self._flags = None
            self._grids.clear()
            self._frames.clear()
            self._depth.clear()
        self._meta = meta
        return self.cache_dir

    @property
    def meta(self) -> dict:
        self.index()
        return dict(self._meta or {})

    # -- data ---------------------------------------------------------------
    def series(self, name: str) -> Series:
        """Any name in ``SERIES_DTYPES``; empty when the topic is absent."""
        if name not in SERIES_DTYPES:
            raise KeyError(f"unknown series {name!r}; valid: {tuple(SERIES_DTYPES)}")
        if self._series is None:
            self.index()
            with np.load(self.cache_dir / "series.npz") as z:
                self._series = {n: Series(z[n], n) if n in z.files else Series.empty(n)
                                for n in SERIES_DTYPES}
        return self._series[name]

    @property
    def calib(self) -> Calib:
        """The recorded calibration with this machine's colour intrinsics
        (looked up on load, so a config dump added after the index was
        built is used) and the overrides applied last."""
        if self._calib is None:
            self.index()
            c = Calib.load(self.cache_dir / "calib.json")
            fill_color_intrinsics(c)
            c.overrides = {r: dict(m) for r, m in self.overrides.items()}
            self._calib = c
        return self._calib

    def frames(self, role: str) -> FrameSeries:
        """The role's colour stream. The raw tick grid is passed along so a
        tick without an AU taints the rest of its GOP (see frames.py)."""
        self._check_role(role)
        if role not in self._frames:
            self.index()
            self._frames[role] = FrameSeries(self.cache_dir / f"video_{role}.bin",
                                             self.cache_dir / f"video_{role}.npy", f"color_{role}",
                                             ticks=self.flags()["t_ns"])
        return self._frames[role]

    def depth(self, role: str) -> DepthSeries:
        self._check_role(role)
        if role not in self._depth:
            self.index()
            self._depth[role] = DepthSeries(self.cache_dir / f"depth_{role}.bin",
                                            self.cache_dir / f"depth_{role}.npy", f"depth_{role}")
        return self._depth[role]

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in _ROLES:
            raise KeyError(f"role must be one of {_ROLES}, got {role!r}")

    def flags(self) -> np.ndarray:
        """Structured per raw tick: ``t_ns, has_color_near, has_color_far,
        far_hold_last, gap_before_ms`` (0 for the first tick)."""
        if self._flags is None:
            self.index()
            self._flags = np.load(self.cache_dir / "ticks.npy")
        return self._flags

    def grid(self, hz: Optional[float] = None) -> np.ndarray:
        """The session-wide tick stamps at ``hz``: every raw tick for None,
        else :func:`sample_stamps` over the whole session. Windows and
        image history both index this grid, so a windowed ``ticks(hz, ...)``
        is a subset of ``ticks(hz)`` and labels computed on one are found
        by ``t_ns`` from the other."""
        t = self.flags()["t_ns"]
        if hz is None:
            return t
        key = float(hz)
        if key not in self._grids:
            self._grids[key] = sample_stamps(t, key)
        return self._grids[key]

    def ticks(self, hz: Optional[float] = None, start_ns: Optional[int] = None,
              end_ns: Optional[int] = None) -> "Ticks":
        """Raw ticks (``hz=None``) or the nearest raw tick per ``1/hz`` grid
        point anchored at the session start, restricted to
        ``[start_ns, end_ns]`` (inclusive)."""
        t = self.grid(hz)
        if start_ns is not None:
            t = t[t >= int(start_ns)]
        if end_ns is not None:
            t = t[t <= int(end_ns)]
        return Ticks(self, np.asarray(t, dtype=np.int64), hz=hz)

    def summary(self) -> dict:
        """What ``lab info`` prints."""
        meta = self.meta
        flags = self.flags()
        n = int(len(flags))
        span = (int(flags["t_ns"][-1]) - int(flags["t_ns"][0])) / 1e9 if n > 1 else 0.0
        out = {
            "name": self.name, "path": str(self.path), "size_bytes": int(self.info.size_bytes),
            "footerless": bool(self.info.footerless), "duration_s": round(self.duration_s, 3),
            "start_ns": self.start_ns, "end_ns": self.end_ns, "topics": dict(self.info.message_counts),
            "ticks": n, "tick_hz": round((n - 1) / span, 2) if span > 0 else 0.0,
            "tick_source": meta.get("tick_source"), "color_fmt": meta.get("color_fmt"),
            "keyframes": meta.get("keyframes"),
            "color_near_missing": int(np.count_nonzero(~flags["has_color_near"])),
            "color_far_missing": int(np.count_nonzero(~flags["has_color_far"])),
            "far_hold_last": int(np.count_nonzero(flags["far_hold_last"])),
            "max_gap_ms": float(flags["gap_before_ms"].max()) if n else 0.0,
            "series": meta.get("series_counts"), "calib_roles": self.calib.roles,
            "color_intrinsics": sorted(self.calib.color), "cache_dir": str(self.cache_dir),
            "index_seconds": meta.get("index_seconds"), "unknown_topics": meta.get("unknown_topics"),
        }
        return out


# --------------------------------------------------------------------------
# Ticks
# --------------------------------------------------------------------------

class Ticks:
    """Tick stamps of a session plus the :class:`InputSpec` that turns one
    into a :class:`Sample`. Without a spec a sample carries flags only."""

    def __init__(self, session: Session, t_ns: np.ndarray, hz: Optional[float] = None,
                 spec: Optional[InputSpec] = None):
        self.session = session
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.hz = hz
        self.spec = spec

    def __len__(self) -> int:
        return int(len(self.t_ns))

    def __repr__(self) -> str:
        return f"Ticks({self.session.name!r}, n={len(self)}, hz={self.hz}, spec={self.spec})"

    @property
    def dt_ns(self) -> int:
        """Nominal tick spacing: ``1/hz`` when snapped, else the median raw
        spacing (0 for fewer than two ticks)."""
        if self.hz:
            return int(round(1e9 / float(self.hz)))
        if len(self.t_ns) < 2:
            return 0
        return int(np.median(np.diff(self.t_ns)))

    def select(self, *modalities: str, history_s: Optional[Mapping[str, float]] = None,
               tol_ns: Optional[int] = None) -> "Ticks":
        kw = {"history_s": dict(history_s or {})}
        if tol_ns is not None:
            kw["tol_ns"] = int(tol_ns)
        spec = InputSpec(tuple(modalities), **kw)
        return Ticks(self.session, self.t_ns, hz=self.hz, spec=spec)

    def with_spec(self, spec: InputSpec) -> "Ticks":
        return Ticks(self.session, self.t_ns, hz=self.hz, spec=spec)

    # -- per tick -----------------------------------------------------------
    @property
    def _tol_ns(self) -> int:
        return int(self.spec.tol_ns if self.spec is not None else InputSpec(()).tol_ns)

    def flags(self, i: int) -> Dict[str, Any]:
        """``index``, ``t_ns``, ``raw_t_ns`` and the session's quality flags
        of the raw tick nearest this stamp within the spec's tolerance (an
        exact hit for ticks built by ``Session.ticks``). Every flag key is
        always present; the values are None when no raw tick is close
        enough, so a consumer can read them without guarding."""
        t = int(self.t_ns[i])
        fl = self.session.flags()
        out: Dict[str, Any] = {"index": int(i), "t_ns": t, "raw_t_ns": None}
        out.update({f: None for f in _FLAG_FIELDS})
        j = _nearest_index(fl["t_ns"], np.array([t], dtype=np.int64), self._tol_ns)[0]
        if j >= 0:
            row = fl[j]
            out["raw_t_ns"] = int(row["t_ns"])
            out.update(has_color_near=bool(row["has_color_near"]), has_color_far=bool(row["has_color_far"]),
                       far_hold_last=bool(row["far_hold_last"]), gap_before_ms=float(row["gap_before_ms"]))
        return out

    def __getitem__(self, i: int) -> Sample:
        if isinstance(i, slice):
            raise TypeError("Ticks supports integer indexing; slice t_ns and build a Ticks instead")
        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        t = int(self.t_ns[i])
        data: Dict[str, Any] = {}
        if self.spec is not None:
            for m in self.spec.modalities:
                data[m] = self._modality(m, i, t)
        return Sample(t_ns=t, session=self.session.name, index=int(i), data=data,
                      calib=self.session.calib, flags=self.flags(i))

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def _history_ticks(self, i: int, history_s: float) -> List[Optional[int]]:
        """Stamps of the past ticks covering ``history_s`` at the tick
        spacing, oldest first, the current tick last; None before the
        session's start so the list length is fixed per spec.

        The past ticks come from the session-wide grid at this Ticks' rate
        (``Session.grid``), not from this Ticks' own stamps, so the first
        samples of a ``start_ns`` window (a time-based split) keep the
        history the recording has. For a stamp off that grid the history is
        the grid ticks before it."""
        t = int(self.t_ns[i])
        grid = self.session.grid(self.hz)
        dt = self.dt_ns if self.hz else (int(np.median(np.diff(grid))) if len(grid) > 1 else 0)
        k = int(round(history_s * 1e9 / dt)) if dt > 0 and history_s > 0 else 0
        pos = int(np.searchsorted(grid, t, side="left"))     # grid[pos] == t for on-grid stamps
        out: List[Optional[int]] = []
        for j in range(pos - k, pos):
            out.append(int(grid[j]) if j >= 0 else None)
        out.append(t)
        return out

    def _image(self, m: str, t: Optional[int]) -> Optional[np.ndarray]:
        if t is None:
            return None
        role = m.split("_", 1)[1]
        if m.startswith("color"):
            return self.session.frames(role).at(t)
        return self.session.depth(role).at(t)

    def _modality(self, m: str, i: int, t: int):
        spec = self.spec
        h = spec.history(m)
        if m in IMAGE_MODALITIES:
            if h > 0:
                return [self._image(m, tj) for tj in self._history_ticks(i, h)]
            return self._image(m, t)
        s = self.session.series(m)
        if m in STATE_MODALITIES:
            if h > 0:
                return s.window(t - int(round(h * 1e9)), t + 1)
            return s.nearest(t, spec.tol_ns)
        if m in IMU_MODALITIES or m in EVENT_MODALITIES:
            return s.window(t - int(round(h * 1e9)), t + 1)
        if m in LATEST_MODALITIES:
            return s.latest_at(t)
        raise KeyError(m)

    # -- table --------------------------------------------------------------
    def frame(self, tol_ns: Optional[int] = None):
        """A ``polars.DataFrame`` of per-tick scalars: ``t_ns``, odom
        ``x/y/theta`` and cmd ``vx/wz`` (nearest within ``tol_ns``, null
        otherwise), the quality flags, the colour AU sizes per role
        (``au_size_near/far``, -1 without an AU; the far camera's held
        frames are re-encoded by NVENC on the robot and cannot be found by
        byte equality, so this is the measurement an AU-size detector would
        be tuned on), the runtime fields in force at the tick, the intent
        text and the place name."""
        import polars as pl
        tol = int(tol_ns if tol_ns is not None else self._tol_ns)
        t = self.t_ns
        cols: Dict[str, Any] = {"t_ns": t}
        fl = self.session.flags()
        pos = np.searchsorted(fl["t_ns"], t)
        pos = np.clip(pos, 0, max(len(fl) - 1, 0))
        hit = (len(fl) > 0) & (fl["t_ns"][pos] == t) if len(fl) else np.zeros(len(t), bool)
        for f in ("has_color_near", "has_color_far", "far_hold_last"):
            cols[f] = np.where(hit, fl[f][pos], False) if len(fl) else np.zeros(len(t), bool)
        cols["gap_before_ms"] = np.where(hit, fl["gap_before_ms"][pos], np.nan) if len(fl) else np.full(len(t), np.nan)
        for role in _ROLES:
            fs = self.session.frames(role)
            sizes = np.full(len(t), -1, dtype=np.int64)
            if len(fs):
                idx = _nearest_index(fs.t_ns, t, 0)
                sizes[idx >= 0] = fs.index["size"][idx[idx >= 0]]
            cols[f"au_size_{role}"] = sizes

        def nearest_cols(name: str, fields: Sequence[str], prefix: str):
            s = self.session.series(name)
            idx = _nearest_index(s.t_ns, t, tol)
            for f in fields:
                v = np.full(len(t), np.nan)
                ok = idx >= 0
                v[ok] = s.data[f][idx[ok]].astype(np.float64)
                cols[f"{prefix}{f}"] = v

        nearest_cols("odom", ("x", "y", "theta"), "odom_")
        nearest_cols("cmd_vel", ("vx", "wz"), "cmd_")

        rt = self.session.series("runtime")
        idx = np.searchsorted(rt.t_ns, t, side="right") - 1 if len(rt) else np.full(len(t), -1)
        for f in rt.fields[1:]:
            if rt.data.dtype[f].kind == "f":
                v = np.full(len(t), np.nan)
            else:
                v = np.full(len(t), -1, dtype=np.int64)
            ok = idx >= 0
            if len(rt):
                v[ok] = rt.data[f][idx[ok]]
            cols[f"rt_{f}"] = v

        def latest_text(name: str, field: str, col: str):
            s = self.session.series(name)
            if len(s) == 0:
                cols[col] = [None] * len(t)
                return
            j = np.searchsorted(s.t_ns, t, side="right") - 1
            cols[col] = [str(s.data[field][k]) if k >= 0 else None for k in j]

        latest_text("intent", "text", "intent")
        latest_text("place", "name", "place")
        return pl.DataFrame({k: (list(v) if isinstance(v, np.ndarray) and v.dtype.kind == "U" else v)
                             for k, v in cols.items()})


def _nearest_index(src: np.ndarray, t: np.ndarray, tol_ns: int) -> np.ndarray:
    """Index into sorted ``src`` of the stamp nearest each ``t`` within
    ``tol_ns``, else -1."""
    out = np.full(len(t), -1, dtype=np.int64)
    if len(src) == 0:
        return out
    pos = np.searchsorted(src, t)
    lo = np.clip(pos - 1, 0, len(src) - 1)
    hi = np.clip(pos, 0, len(src) - 1)
    d_lo = np.abs(src[lo].astype(np.int64) - t)
    d_hi = np.abs(src[hi].astype(np.int64) - t)
    best = np.where(d_hi < d_lo, hi, lo)
    d = np.minimum(d_lo, d_hi)
    out[d <= tol_ns] = best[d <= tol_ns]
    return out
