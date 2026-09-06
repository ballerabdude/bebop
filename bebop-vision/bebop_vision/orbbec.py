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
- Threading mirrors `camera.py`: one capture thread per camera feeding a
  latest-wins slot. Open failure at startup is a hard error (bench tool);
  a mid-run drop leaves the slot stale so the deadman stops the robot —
  the control loop never crashes on a camera exception.
"""

import dataclasses
import json
import threading
import time
from pathlib import Path

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


def _negotiate_color_profile(sensor, ob, preferred):
    """Pick the best advertised color profile at or below (w, h, fps).

    Prefers raw RGB (USB 3.x); falls back to MJPG, which is the only color
    the USB 2.0-cabled camera offers at high resolution (plan Section 3.2).
    Returns (w, h, fps, format) or None when the sensor has no usable color.
    """
    pw, ph, pfps = preferred
    profiles = []
    plist = sensor.get_stream_profile_list()
    for i in range(plist.get_count()):
        vp = plist.get_stream_profile_by_index(i).as_video_stream_profile()
        profiles.append((vp.get_width(), vp.get_height(), vp.get_fps(),
                         vp.get_format()))
    for fmt in (ob.OBFormat.RGB, ob.OBFormat.MJPG):
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
                 color_profile=None, config_dir=None, mask_rects=None):
        self.serial = serial
        self.role = role
        self.depth_profile = tuple(depth_profile)
        self.color_profile = tuple(color_profile) if color_profile else None
        self.color_format = None
        self.config_dir = config_dir
        # Self-view pixel rects (x0, y0, x1, y1): the rigid mount means the
        # robot's own chassis always lands on the same pixels — zeroed before
        # the frame is published (body-frame boxes can't catch cables and
        # overhanging mounts that stick out past the measured footprint).
        self.mask_rects = [tuple(r) for r in (mask_rects or [])]
        self.read_fps = 0.0
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._reader = None
        self._ctx = None
        self._pipeline = None
        self._open()
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True, name=f"orbbec-{role}")
        self._running = True
        self._reader.start()

    def _open(self):
        ob = _sdk()
        self._ob = ob
        ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
        self._ctx = ob.Context()
        try:
            dev = self._ctx.query_devices().get_device_by_serial_number(self.serial)
        except Exception as exc:
            raise RuntimeError(f"Orbbec device {self.serial} not found: {exc}") from exc
        if dev is None:
            raise RuntimeError(f"Orbbec device {self.serial} not found")

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
                    color_sensor, ob, self.color_profile)
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
              f"color={'on' if self.color_profile else 'off'}")

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
                for x0, y0, x1, y1 in self.mask_rects:
                    arr[y0:y1, x0:x1] = 0
                color = None
                if self.color_profile:
                    cf = fs.get_color_frame()
                    if cf is not None:
                        cdata = cf.get_data()
                        if self.color_format == ob.OBFormat.MJPG:
                            import cv2
                            jpg = np.frombuffer(cdata, dtype=np.uint8)
                            color = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
                            if color is not None:
                                color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
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
            with self._lock:
                self._frame = StampedFrame(
                    depth=arr, stamp_us=stamp_us, recv_ts=time.monotonic(),
                    width=self.width, height=self.height, fps=self.fps,
                    color=color, serial=self.serial, role=self.role)
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
            self.cameras[c["role"]] = OrbbecCamera(
                serial=serial,
                role=c["role"],
                depth_profile=c.get("depth_profile", (848, 480, 30)),
                color_profile=(1280, 800, 30) if color else None,
                config_dir=config_dir,
                mask_rects=c.get("self_mask_pixels"))

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
