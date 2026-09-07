"""Orbbec Gemini 335Lg camera service for navd.

Owns the two depth cameras directly (bebop-vision is the sole owner of the
Orbbec rig; the firmware's OBSBOT webcam is unrelated and being retired —
docs/navd.md Section 9). One process holds a camera at a time: close
OrbbecViewer before running.

- Devices are matched by serial (never index — enumeration order is not
  stable with two identical devices on the same hub).
- The preferred depth profile is negotiated against what the device
  actually advertises: the far camera on the unfixed USB 2.0 cable only
  offers 848x480@10, and negotiation drops to it automatically. Profiles
  restore when the cable is swapped — no config change needed.
- Depth filters (SpatialModerate + Temporal + HoleFilling) run in the
  capture thread, per Section 3.3 of the plan.
- Depth preset (per-camera `depth_preset` in the rig YAML) is loaded at
  open time via `load_preset` — this replaces the earlier work-mode
  setting (a preset bundles its own work mode). Replaces "High Density"
  with the built-in "High Accuracy" preset (2026-09-07); the G33X custom
  preset bin is Gemini 330-only (firmware ERR_MISMATCH on pid 2059).
- Threading mirrors `camera.py`: one capture thread per camera feeding a
  latest-wins slot. Open failure at startup is a hard error (bench tool);
  a mid-run drop leaves the slot stale so the deadman stops the robot —
  the control loop never crashes on a camera exception.
"""

import collections
import dataclasses
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install pyyaml") from exc

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_RIG_YAML = CONFIG_DIR / "orbbec_rig.yaml"


def intrinsics_path(serial, config_dir=None):
    return Path(config_dir or CONFIG_DIR) / f"orbbec_intrinsics_{serial}.json"


def load_rig_config(path=None):
    with open(path or DEFAULT_RIG_YAML) as f:
        return yaml.safe_load(f)


def parse_self_mask_entries(entries):
    """Split rig-YAML self_mask_pixels entries into (rects, polygons).

    Two shape forms are accepted, in any order: a rect is four scalars
    [x0, y0, x1, y1]; a polygon is a list of [x, y] vertex pairs (the
    chassis silhouette is a trapezoid — a rect wastes floor around it).
    """
    rects, polys = [], []
    for e in (entries or []):
        scalars = [v for v in e if isinstance(v, (int, float))]
        if len(e) == 4 and len(scalars) == 4:
            rects.append(tuple(int(v) for v in e))
        else:
            try:
                polys.append([[int(p[0]), int(p[1])] for p in e])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(
                    f"self_mask_pixels entry must be [x0, y0, x1, y1] or "
                    f"[[x, y], ...]: {e!r}") from exc
    return rects, polys


def normalize_mask_shapes(rects, polys):
    """Coerce mask entries to int tuples / int vertex lists.

    Constructor hygiene for direct OrbbecCamera callers: YAML may hand us
    floats, and rect slices / fillPoly need ints. Expects the split output
    of parse_self_mask_entries.
    """
    return ([tuple(int(v) for v in r) for r in (rects or [])],
            [[[int(pt[0]), int(pt[1])] for pt in poly]
             for poly in (polys or [])])


def build_self_view_mask(rects, polys, width, height):
    """Static bool (H, W) mask covering the given rects + polygons.

    All coordinates are DEPTH-frame pixels; the mask is built once at
    camera open (the rigid mount keeps the self-view at fixed pixels) and
    applied per frame with a single boolean index. Shapes clamp to the
    frame; no shapes -> None (no masking). Rect slicing [y0:y1, x0:x1]
    matches the historical per-rect zeroing semantics.
    """
    if not rects and not polys:
        return None
    m = np.zeros((height, width), np.uint8)
    for x0, y0, x1, y1 in rects:
        m[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = 1
    for poly in polys:
        pts = np.array([[min(width - 1, max(0, int(x))),
                         min(height - 1, max(0, int(y)))] for x, y in poly],
                       np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


@dataclasses.dataclass
class StampedFrame:
    """Latest depth (+ optional color) frame from one camera.

    stamp_us is the SDK device timestamp of the depth frame; recv_ts is
    time.monotonic() at arrival — staleness is judged on recv_ts.
    """

    depth: np.ndarray          # uint16 (H, W), millimetres, 0 = invalid
    stamp_us: int
    recv_ts: float
    width: int
    height: int
    fps: float
    color: np.ndarray = None   # uint8 (H, W, 3) RGB, or None
    color_jpeg: bytes = None   # camera-encoded MJPEG bytes, or None
    serial: str = ""
    role: str = ""

    def age_s(self, now=None):
        return (now or time.monotonic()) - self.recv_ts


def _sdk():
    import pyorbbecsdk as ob
    return ob


def _set_depth_filters(ob):
    # Filter set proven in OrbbecViewer bring-up (plan Section 3.3).
    spatial = ob.SpatialModerateFilter()
    temporal = ob.TemporalFilter()
    hole = ob.HoleFillingFilter()
    return [spatial, temporal, hole]


def _negotiate_depth_profile(sensor, ob, preferred):
    """Pick the best advertised Y16 profile at or below (w, h, fps)."""
    pw, ph, pfps = preferred
    profiles = []
    plist = sensor.get_stream_profile_list()
    for i in range(plist.get_count()):
        vp = plist.get_stream_profile_by_index(i).as_video_stream_profile()
        if vp.get_format() == ob.OBFormat.Y16:
            profiles.append((vp.get_width(), vp.get_height(), vp.get_fps()))
    exact = [p for p in profiles if p[0] == pw and p[1] == ph and p[2] <= pfps]
    if exact:
        return max(exact, key=lambda p: p[2])
    smaller = [p for p in profiles if p[0] * p[1] <= pw * ph and p[2] <= pfps]
    if smaller:
        return max(smaller, key=lambda p: (p[0] * p[1], p[2]))
    raise RuntimeError(
        f"no depth profile at or below {preferred}; advertised: {sorted(set(profiles))}")


def _negotiate_color_profile(sensor, ob, preferred, want_format="rgb"):
    """Pick the best advertised color profile at or below (w, h, fps).

    Format preference order: `want_format` first, the other as fallback —
    raw RGB is the lossless default; MJPG is the camera's hardware JPEG
    encoder (zero CPU on the Jetson: the recorder can store the bytes
    verbatim instead of re-encoding RGB per tick). Returns
    (w, h, fps, format) or None when the sensor has no usable color.
    """
    pw, ph, pfps = preferred
    profiles = []
    plist = sensor.get_stream_profile_list()
    for i in range(plist.get_count()):
        vp = plist.get_stream_profile_by_index(i).as_video_stream_profile()
        profiles.append((vp.get_width(), vp.get_height(), vp.get_fps(),
                         vp.get_format()))
    first, second = ((ob.OBFormat.MJPG, ob.OBFormat.RGB)
                     if want_format == "mjpg" else
                     (ob.OBFormat.RGB, ob.OBFormat.MJPG))
    for fmt in (first, second):
        exact = [p for p in profiles
                 if p[3] == fmt and p[0] == pw and p[1] == ph and p[2] <= pfps]
        if exact:
            w, h, fps, _ = max(exact, key=lambda p: p[2])
            return w, h, fps, fmt
        smaller = [p for p in profiles
                   if p[3] == fmt and p[0] * p[1] <= pw * ph and p[2] <= pfps]
        if smaller:
            w, h, fps, _ = max(smaller, key=lambda p: (p[0] * p[1], p[2]))
            return w, h, fps, fmt
    return None


def _dump_intrinsics(pipeline, serial, config_dir=None):
    """Fetch depth intrinsics from the open pipeline and cache to JSON.

    The JSON is the source of truth for BEV (plan Section 6.2); it is
    normally written once by tools/orbbec_intrinsics.py — auto-provisioned
    here only so a fresh rig still boots.
    """
    param = pipeline.get_camera_param()
    intr = param.depth_intrinsic
    dist = param.depth_distortion
    data = {
        "serial": serial,
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.cx), "cy": float(intr.cy),
        "distortion": {"model": int(dist.model),
                       "k1": float(dist.k1), "k2": float(dist.k2),
                       "p1": float(dist.p1), "p2": float(dist.p2),
                       "k3": float(dist.k3), "k4": float(dist.k4),
                       "k5": float(dist.k5), "k6": float(dist.k6)},
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Color intrinsics + color->depth extrinsic: the semantic-fusion ray
    # LUTs (tools/fuse_navd_labels.py) project through these. The ISP
    # recalibrates them on firmware updates, so they belong in the same
    # cached snapshot (depth values are unaffected by ISP changes).
    rgb = getattr(param, "rgb_intrinsic", None)
    rgbd = getattr(param, "rgb_distortion", None)
    if rgb is not None and int(getattr(rgb, "width", 0)) > 0:
        data.update({
            "color_width": int(rgb.width), "color_height": int(rgb.height),
            "color_fx": float(rgb.fx), "color_fy": float(rgb.fy),
            "color_cx": float(rgb.cx), "color_cy": float(rgb.cy),
            "color_rgb_distortion": {
                "model": int(rgbd.model),
                "k1": float(rgbd.k1), "k2": float(rgbd.k2),
                "p1": float(rgbd.p1), "p2": float(rgbd.p2),
                "k3": float(rgbd.k3), "k4": float(rgbd.k4),
                "k5": float(rgbd.k5), "k6": float(rgbd.k6)},
        })
        # OBExtrinsic: rot = 3x3 rotation (9), transform = translation (3).
        ext = getattr(param, "transform", None)
        rot = None if ext is None else getattr(ext, "rot", None)
        tr = None if ext is None else getattr(ext, "transform", None)
        rot = np.asarray(rot, dtype=np.float32).reshape(-1) if rot is not None else []
        tr = np.asarray(tr, dtype=np.float32).reshape(-1) if tr is not None else []
        if rot.size == 9 and np.any(rot):
            data["color_to_depth_transform"] = {
                "rotation": [float(v) for v in rot],
                "translation": [float(v) for v in tr[:3]],
            }
    path = intrinsics_path(serial, config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[orbbec] intrinsics cached -> {path}")
    return data


def load_intrinsics(serial, config_dir=None):
    path = intrinsics_path(serial, config_dir)
    with open(path) as f:
        return json.load(f)


class OrbbecCamera:
    """One Gemini 335Lg: capture thread -> latest-wins StampedFrame slot."""

    def __init__(self, serial, role, depth_profile=(848, 480, 30),
                 color_profile=None, config_dir=None, mask_rects=None,
                 color_format="rgb", depth_work_mode=None, depth_preset=None,
                 mask_polys=None):
        self.serial = serial
        self.role = role
        self.depth_profile = tuple(depth_profile)
        self.depth_work_mode = depth_work_mode
        self.depth_preset = depth_preset
        self.color_profile = tuple(color_profile) if color_profile else None
        self.color_format = None
        self.color_format_want = color_format
        self.config_dir = config_dir
        # Self-view shapes (rects [x0, y0, x1, y1] and/or polygons
        # [[x, y], ...]) in DEPTH-frame pixels: the rigid mount means the
        # robot's own chassis always lands on the same pixels — zeroed
        # before the frame is published (body-frame boxes can't catch
        # cables and overhanging mounts that stick out past the measured
        # footprint). Polygons hug the trapezoid silhouette without
        # masking the floor a rect would cover.
        self.mask_rects, self.mask_polys = normalize_mask_shapes(
            mask_rects, mask_polys)
        self.read_fps = 0.0
        self._lock = threading.Lock()
        self._frame = None
        # Short arrival-ordered history (≈0.4 s at 15 fps) — lets consumers
        # pair frames across cameras by arrival time instead of trusting two
        # independent latest-wins slots (recorder_mcap NavdRecorder._tick).
        self._history = collections.deque(maxlen=6)
        self._running = False
        self._reader = None
        self._ctx = None
        self._pipeline = None
        self._open()
        # Built at the NEGOTIATED depth size; shape coordinates assume the
        # preferred profile and clamp if negotiation lands elsewhere.
        self._pix_mask = build_self_view_mask(
            self.mask_rects, self.mask_polys, self.width, self.height)
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name=f"orbbec-{role}")
        self._running = True
        self._reader.start()

    def _open(self):
        ob = _sdk()
        self._ob = ob
        ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
        self._ctx = ctx = ob.Context()
        try:
            dev = ctx.query_devices().get_device_by_serial_number(self.serial)
        except Exception as exc:
            raise RuntimeError(f"Orbbec device {self.serial} not found: {exc}") from exc
        if dev is None:
            raise RuntimeError(f"Orbbec device {self.serial} not found")
        # Hardware sync (Orbbec multi-camera sync hub): the pair runs
        # Primary (near) + Secondary-synced (far), all delays 0 — far frames
        # phase-lock to the near camera's vsync, so both views capture the
        # same instant (cross-device frame delta ~0-1 ms vs free-run drift).
        # Requires matched depth+color rates across the pair (rig YAML).
        _SYNC_MODE = {"near": ob.OBMultiDeviceSyncMode.PRIMARY,
                      "far": ob.OBMultiDeviceSyncMode.SECONDARY_SYNCED}
        if self.role in _SYNC_MODE and \
                hasattr(dev, "set_multi_device_sync_config"):
            cfg = dev.get_multi_device_sync_config()
            cfg.mode = _SYNC_MODE[self.role]
            cfg.depth_delay_us = 0
            cfg.color_delay_us = 0
            cfg.trigger_to_image_delay_us = 0
            cfg.trigger_out_delay_us = 0
            cfg.trigger_out_enable = self.role == "near"
            cfg.frames_per_trigger = 1
            dev.set_multi_device_sync_config(cfg)

        # Depth preset / work mode: set before the pipeline starts. Devices
        # power up in "Default", which over-smooths floor-level objects into
        # the ground plane (see module docstring). A preset (`depth_preset`)
        # bundles its own work mode and supersedes `depth_work_mode`.
        preset_loaded = False
        if self.depth_preset and hasattr(dev, "load_preset"):
            try:
                if dev.get_current_preset_name() != self.depth_preset:
                    dev.load_preset(self.depth_preset)
                preset_loaded = dev.get_current_preset_name() == self.depth_preset
                print(f"[orbbec] {self.role}({self.serial}): preset "
                      f"'{self.depth_preset}' active")
            except Exception as exc:
                print(f"[orbbec] {self.role}({self.serial}): depth preset "
                      f"'{self.depth_preset}' rejected: {exc}")
        if not preset_loaded and self.depth_work_mode and \
                hasattr(dev, "set_depth_work_mode"):
            try:
                current = dev.get_depth_work_mode()
                if getattr(current, "name", None) != self.depth_work_mode:
                    dev.set_depth_work_mode(self.depth_work_mode)
            except Exception as exc:
                print(f"[orbbec] {self.role}({self.serial}): depth work mode "
                      f"'{self.depth_work_mode}' rejected: {exc}")

        depth_sensor = None
        sensors = dev.get_sensor_list()
        for i in range(sensors.get_count()):
            s = sensors.get_sensor_by_index(i)
            if s.get_type() == ob.OBSensorType.DEPTH_SENSOR:
                depth_sensor = s
        if depth_sensor is None:
            raise RuntimeError(f"{self.serial}: no depth sensor")

        w, h, fps = _negotiate_depth_profile(depth_sensor, ob, self.depth_profile)
        if (w, h, fps) != self.depth_profile:
            print(f"[orbbec] {self.role}({self.serial}): profile {self.depth_profile} "
                  f"unavailable, using {(w, h, fps)}")
        self.width, self.height, self.fps = w, h, fps

        color_profile = None
        if self.color_profile:
            color_sensor = None
            for i in range(sensors.get_count()):
                s = sensors.get_sensor_by_index(i)
                if s.get_type() == ob.OBSensorType.COLOR_SENSOR:
                    color_sensor = s
            if color_sensor is not None:
                color_profile = _negotiate_color_profile(
                    color_sensor, ob, self.color_profile,
                    want_format=self.color_format_want)
            if color_profile is None:
                print(f"[orbbec] {self.role}({self.serial}): no usable color "
                      f"profile, depth only")
            else:
                self.color_format = color_profile[3]

        config = ob.Config()
        config.enable_video_stream(ob.OBStreamType.DEPTH_STREAM, w, h, fps,
                                   ob.OBFormat.Y16)
        if color_profile:
            cw, ch, cfps, cfmt = color_profile
            config.enable_video_stream(ob.OBStreamType.COLOR_STREAM, cw, ch,
                                       cfps, cfmt)
        self._pipeline = ob.Pipeline(dev)
        self._pipeline.enable_frame_sync()
        self._pipeline.start(config)
        self._filters = _set_depth_filters(ob)

        if not intrinsics_path(self.serial, self.config_dir).exists():
            try:
                _dump_intrinsics(self._pipeline, self.serial, self.config_dir)
            except Exception as exc:
                print(f"[orbbec] {self.role}: could not auto-provision intrinsics "
                      f"({exc}); BEV will fail until they exist")
        print(f"[orbbec] {self.role}({self.serial}): depth {w}x{h}@{fps} "
              f"color={'on' if self.color_profile else 'off'}"
              f"{' (' + str(self.color_format).split('.')[-1] + ')' if self.color_profile else ''}")

    def _read_loop(self):
        frames = 0
        t = time.monotonic()
        ob = self._ob
        while self._running:
            try:
                fs = self._pipeline.wait_for_frames(200)
                if fs is None:
                    continue
                depth = fs.get_depth_frame()
                if depth is None:
                    continue
                for f in self._filters:
                    depth = f.process(depth)
                    if depth is None:
                        break
                if depth is None:
                    continue
                arr = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(
                    depth.get_height(), depth.get_width()).copy()
                if self._pix_mask is not None:
                    arr[self._pix_mask] = 0
                color = None
                color_jpeg = None
                if self.color_profile:
                    cf = fs.get_color_frame()
                    if cf is not None:
                        cdata = cf.get_data()
                        if self.color_format == ob.OBFormat.MJPG:
                            # Hardware-encoded JPEG: pass the bytes through
                            # untouched (the recorder stores them verbatim;
                            # decoding here would burn capture-thread CPU).
                            color_jpeg = bytes(cdata)
                        else:
                            color = np.frombuffer(cdata, dtype=np.uint8).reshape(
                                cf.get_height(), cf.get_width(), 3).copy()
                stamp_us = int(depth.get_timestamp_us())
            except Exception as exc:
                # Mid-run drop: log, keep the slot stale (deadman stops the
                # robot), and retry — never take down the control loop.
                print(f"[orbbec] {self.role} capture error: {type(exc).__name__}: {exc}")
                time.sleep(0.5)
                continue
            frame = StampedFrame(
                depth=arr, stamp_us=stamp_us, recv_ts=time.monotonic(),
                width=self.width, height=self.height, fps=self.fps,
                color=color, color_jpeg=color_jpeg, serial=self.serial,
                role=self.role)
            with self._lock:
                self._frame = frame
                self._history.append(frame)
            frames += 1
            now = time.monotonic()
            if now - t >= 1.0:
                self.read_fps = frames / (now - t)
                frames = 0
                t = now

    def read(self):
        """Latest StampedFrame (copy) or None."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame

    def recent(self):
        """Arrival-ordered history (oldest→newest) of recent StampedFrames."""
        with self._lock:
            return list(self._history)

    def stop(self):
        self._running = False
        if self._reader is not None:
            self._reader.join(timeout=3.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None


class OrbbecRig:
    """Both cameras + the rig config; the BEV worker's capture front end.

    `roles` optionally restricts which configured cameras open (e.g.
    ("near",) for near-authoritative operation while the far camera's
    USB cable is unfixed or dropped off the bus).
    """

    def __init__(self, rig_path=None, color=False, config_dir=None, roles=None):
        cfg = load_rig_config(rig_path)
        cams = cfg["robots"]["default"]["cameras"]
        if roles:
            cams = {s: c for s, c in cams.items() if c["role"] in roles}
        self.cameras = {}
        for serial, c in cams.items():
            dp = tuple(c.get("depth_profile", (848, 480, 30)))
            rects, polys = parse_self_mask_entries(c.get("self_mask_pixels"))
            self.cameras[c["role"]] = OrbbecCamera(
                serial=serial,
                role=c["role"],
                depth_profile=dp,
                # color rate must match depth rate (hardware sync pairing)
                color_profile=(1280, 800, dp[2]) if color else None,
                config_dir=config_dir,
                mask_rects=rects,
                mask_polys=polys,
                color_format=c.get("color_format", "rgb"),
                depth_work_mode=c.get("depth_work_mode"),
                depth_preset=c.get("depth_preset"))

    def get(self, role):
        return self.cameras[role]

    def read_all(self):
        return {role: cam.read() for role, cam in self.cameras.items()}

    def wait_for_pair(self, timeout=10.0, max_age_s=0.3):
        """Block until every camera has a fresh frame (or timeout)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            frames = self.read_all()
            if all(f is not None and f.age_s() < max_age_s for f in frames.values()):
                return True
            time.sleep(0.05)
        return False

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()
