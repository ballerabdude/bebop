"""Live observe-only visual odometry/SLAM over the recorder's camera slots.

``LiveSlam`` never opens a camera and never reads the robot runtime.  It
consumes the same latest-wins ``StampedFrame`` the recorder/video server
already publish, so it can run alongside an active recording session without
camera ownership conflicts or wheel-odometry input.

Scope of this POC:

* RGB-D only, camera optical frame (x right, y down, z forward).
* No wheel odometry, no IMU, no motion commands.
* Device timestamps only; a frame is fed to the tracker once.
* Pose is valid for the current tick only.  Missing/stale/skewed frames,
  backend errors, and tracking loss clear it.  A recovery starts a new
  trajectory segment because cuVSLAM's odometry frame can reset after loss.
* The optional SLAM-corrected pose is stored separately from the continuous
  odometry pose and is never substituted silently.
"""

from __future__ import annotations

import collections
import json
import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .backend import CuVSLAMTracker, Pose, TrackerBackend
from .calibration import (FRAME_ID, POSE_FRAME, CalibrationError,
                          RigCalibration, SlamConfigError, load_calibration)
from .depth import DepthRegistration
from .map import DepthGeometryMap
from .structure import StructureConfig, StructureMap

IMU_REJECT_REASON = (
    "RGB-D only: no validated camera-to-IMU calibration (rig_from_imu) or "
    "device/host timestamp alignment has been supplied, so live IMU is "
    "rejected rather than fabricating identity extrinsics"
)

_STATE_STARTING = "starting"
_STATE_TRACKING = "tracking"
_STATE_LOST = "lost"
_STATE_STALE = "stale"
_STATE_INVALID = "invalid"
_STATE_ERROR = "error"
_STATE_STOPPED = "stopped"


def json_safe(value):
    """Recursively convert numpy/scalars to JSON types; NaN/inf -> None."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, collections.deque)):
        return [json_safe(v) for v in value]
    return str(value)


@dataclass
class LiveSlamConfig:
    role: str = "far"
    config_dir: Path | None = None
    rig_path: Path | None = None
    enable_slam: bool = False
    use_gpu: bool = True
    map_path: str | None = None
    max_frame_age_s: float = 0.3
    # Color/depth come from one FrameSet.  Measured on Thor: SDK-synchronized
    # phase reached 16.6 ms; 17 ms covers a 30 Hz half period (16.667 ms)
    # plus timestamp rounding, while still rejecting a full 33 ms mismatch.
    max_pair_delta_us: int = 17_000
    poll_hz: float = 60.0
    trajectory_capacity: int = 900
    log_path: Path | None = None
    use_imu: bool = False
    status_log_interval_s: float = 5.0
    # Raw display cloud is opt-in; structural mapping is the default.
    enable_point_cloud: bool = False
    enable_structure: bool = True
    structure_config: StructureConfig | None = None
    structure_cache_min_interval_s: float = 0.25
    # /map geometry cache coalescing: rebuild at most once per revision and
    # no faster than this when requests arrive in bursts.  A segment change
    # always forces a rebuild.
    geometry_cache_min_interval_s: float = 0.25

    def validate(self) -> None:
        if self.use_imu:
            raise SlamConfigError(IMU_REJECT_REASON)
        if not math.isfinite(self.max_frame_age_s) or self.max_frame_age_s <= 0:
            raise SlamConfigError("max_frame_age_s must be positive")
        if int(self.max_pair_delta_us) <= 0:
            raise SlamConfigError("max_pair_delta_us must be positive")
        if not math.isfinite(self.poll_hz) or self.poll_hz <= 0:
            raise SlamConfigError("poll_hz must be positive")
        if int(self.trajectory_capacity) < 1:
            raise SlamConfigError("trajectory_capacity must be >= 1")
        if (not math.isfinite(self.geometry_cache_min_interval_s)
                or self.geometry_cache_min_interval_s < 0):
            raise SlamConfigError(
                "geometry_cache_min_interval_s must be >= 0")
        if (not math.isfinite(self.structure_cache_min_interval_s)
                or self.structure_cache_min_interval_s < 0):
            raise SlamConfigError(
                "structure_cache_min_interval_s must be >= 0")


class LiveSlam:
    """Poll one camera slot, track RGB-D, and publish bounded status."""

    def __init__(self, rig, config: LiveSlamConfig | None = None, *,
                 backend: TrackerBackend | None = None,
                 calibration: RigCalibration | None = None,
                 frame_source=None):
        self.config = config or LiveSlamConfig()
        self.config.validate()
        self.rig = rig
        self.calibration = calibration or load_calibration(
            self.config.role, self.config.config_dir,
            rig_path=self.config.rig_path)
        self.registration = DepthRegistration(self.calibration)
        self.session_id = str(uuid.uuid4())
        self.geometry = DepthGeometryMap(self.calibration, segment=0)
        self.structure = StructureMap(
            self.calibration, self.config.structure_config, segment=0)

        self._frame_source = frame_source
        self._camera = None
        if frame_source is None:
            cameras = getattr(rig, "cameras", None) or {}
            if self.config.role not in cameras:
                open_roles = ", ".join(sorted(cameras)) or "none"
                raise SlamConfigError(
                    f"live-slam camera role {self.config.role!r} is not open "
                    f"(open roles: {open_roles}); adjust --roles or "
                    f"--live-slam-camera")
            self._camera = cameras[self.config.role]

        self.backend = backend or CuVSLAMTracker(
            self.calibration,
            enable_slam=bool(self.config.enable_slam),
            use_gpu=bool(self.config.use_gpu),
            map_path=self.config.map_path,
        )

        self._lock = threading.RLock()
        self._running = False
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._state = _STATE_STARTING

        self._pose: dict | None = None
        self._pose_valid = False
        self._pose_recv_ts: float | None = None
        self._map_pose: dict | None = None
        self._map_valid = False

        self._segment = 0
        self._needs_new_segment = False
        self._last_seen_stamp: int | None = None
        self._last_processed_stamp: int | None = None
        self._clock_reset = False
        self._last_frame_age_s: float | None = None
        self._last_depth_scale_mm: float | None = None
        self._last_pair_skew_ms: float | None = None
        self._last_error: str | None = None

        self._frames_seen = 0
        self._frames_processed = 0
        self._frames_tracked = 0
        self._frames_lost = 0
        self._frames_rejected = 0
        self._trajectory = collections.deque(
            maxlen=int(self.config.trajectory_capacity))
        self._log_file = None
        self._last_status_log = 0.0
        self._last_geometry_mono: float | None = None
        self._map_error: str | None = None
        self._structure_error: str | None = None
        self._map_queue: queue.Queue = queue.Queue(maxsize=1)
        self._map_worker: threading.Thread | None = None

    # -- lifecycle --------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def start(self):
        with self._lock:
            if self._running:
                return self
            if self._stopped:
                raise SlamConfigError("LiveSlam cannot be restarted after stop()")
            if self.config.log_path is not None:
                path = Path(self.config.log_path).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._log_file = open(path, "a", encoding="utf-8", buffering=1)
            self._running = True
            self._started_at = time.monotonic()
            self._state = _STATE_STARTING
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="live-slam")
            self._thread.start()
            self._map_worker = threading.Thread(
                target=self._map_loop, daemon=True, name="live-slam-map")
            self._map_worker.start()
        print(f"[live-slam] started role={self.config.role} "
              f"backend={getattr(self.backend, 'name', type(self.backend).__name__)} "
              f"frame={self.calibration.frame_id} "
              f"slam={'on' if self.config.enable_slam else 'off'}",
              flush=True)
        return self

    def stop(self):
        with self._lock:
            thread = self._thread
            if not self._running and thread is None:
                self._state = _STATE_STOPPED
                return
            self._running = False
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)
        self._stop_map_worker()
        try:
            self.backend.close()
        except Exception:
            pass
        with self._lock:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                finally:
                    self._log_file = None
            self._thread = None
            self._stopped = True
            self._state = _STATE_STOPPED
            self._pose_valid = False
            self._pose = None
            self._map_valid = False
            self._map_pose = None
        print("[live-slam] stopped", flush=True)

    # -- loop -------------------------------------------------------------

    def _loop(self):
        period = 1.0 / float(self.config.poll_hz)
        while self._running:
            started = time.monotonic()
            try:
                self._tick()
            except Exception as exc:  # never let one bad tick kill the service
                self._set_invalid(
                    _STATE_ERROR,
                    f"unhandled live-slam tick error: {type(exc).__name__}: {exc}")
            self._maybe_log_status()
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, period - elapsed))

    def _read_frame(self):
        if self._frame_source is not None:
            return self._frame_source()
        # The driver publishes depth-only FrameSets alongside complete RGB-D
        # ones.  camera.read() can therefore hand us an incomplete sample and
        # invalidate an otherwise fresh pose.  Pick the newest *complete*,
        # in-age, timestamp-paired frame from the camera's bounded history;
        # keep its original recv_ts/device stamps (never refresh an old
        # frame).  If none qualifies, return the latest sample so status
        # reports the real rejection/staleness reason.
        recent_fn = getattr(self._camera, "recent", None)
        if callable(recent_fn):
            try:
                recent = list(recent_fn())
            except Exception:
                recent = []
            for frame in reversed(recent):
                if self._is_complete_pair(frame):
                    return frame
            if recent:
                return recent[-1]
        return self._camera.read()

    def _is_complete_pair(self, frame) -> bool:
        if frame is None or getattr(frame, "depth", None) is None:
            return False
        if getattr(frame, "color", None) is None \
                and not getattr(frame, "color_jpeg", None):
            return False
        try:
            stamp_us = int(getattr(frame, "stamp_us", None))
            color_stamp_us = int(getattr(frame, "color_stamp_us", None))
        except (TypeError, ValueError):
            return False
        if stamp_us <= 0 or color_stamp_us <= 0:
            return False
        if abs(color_stamp_us - stamp_us) > int(self.config.max_pair_delta_us):
            return False
        age = self._frame_age(frame)
        return (age is not None and math.isfinite(age) and age >= 0
                and age <= self.config.max_frame_age_s)

    @staticmethod
    def _frame_age(frame) -> float | None:
        age_fn = getattr(frame, "age_s", None)
        if callable(age_fn):
            try:
                return float(age_fn())
            except Exception:
                pass
        recv = getattr(frame, "recv_ts", None)
        if recv is None:
            return None
        return time.monotonic() - float(recv)

    def _rgb(self, frame) -> np.ndarray | None:
        """Return an RGB uint8 image from a StampedFrame.

        The rig config uses MJPEG for color, so ``frame.color`` is usually
        None and ``color_jpeg`` carries BGR-encoded JPEG bytes.  cv2 decodes
        BGR; cuVSLAM expects RGB, so convert explicitly.
        """
        color = getattr(frame, "color", None)
        if color is not None:
            arr = np.asarray(color)
            return arr if arr.ndim == 3 else None
        jpeg = getattr(frame, "color_jpeg", None)
        if jpeg:
            import cv2
            bgr = cv2.imdecode(np.frombuffer(bytes(jpeg), np.uint8),
                               cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return None

    def _tick(self):
        frame = self._read_frame()
        if frame is None:
            self._set_invalid(_STATE_STALE, "no frame from camera",
                              count_reject=False)
            return
        if self._clock_reset:
            self._set_invalid(
                _STATE_ERROR,
                self._last_error or "device clock reset; restart live-slam",
                count_reject=False)
            return

        stamp_us = getattr(frame, "stamp_us", None)
        try:
            stamp_us = int(stamp_us)
        except (TypeError, ValueError):
            self._frames_seen += 1
            self._set_invalid(_STATE_INVALID,
                              "frame has no valid device timestamp")
            return
        if stamp_us <= 0:
            self._frames_seen += 1
            self._set_invalid(
                _STATE_INVALID,
                f"invalid device timestamp {stamp_us}; must be positive")
            return

        with self._lock:
            if stamp_us == self._last_seen_stamp:
                age = self._frame_age(frame)
                if age is not None:
                    self._last_frame_age_s = age
                stale = (age is None or not math.isfinite(age) or age < 0
                         or age > self.config.max_frame_age_s)
                if stale:
                    self._set_invalid(
                        _STATE_STALE,
                        f"no fresh frame: age {age!r}s (last device stamp "
                        f"unchanged); pose invalidated", count_reject=False)
                return
            self._last_seen_stamp = stamp_us
            self._frames_seen += 1

        serial = getattr(frame, "serial", None)
        if not serial or str(serial) != self.calibration.serial:
            self._set_invalid(
                _STATE_INVALID,
                f"frame serial {serial!r} != calibration serial "
                f"{self.calibration.serial!r}; refusing cross-camera frame")
            return

        age = self._frame_age(frame)
        with self._lock:
            self._last_frame_age_s = age
        if age is None or not math.isfinite(age) or age < 0:
            self._set_invalid(
                _STATE_INVALID,
                f"invalid frame age {age!r}s; staleness cannot be validated")
            return
        if age > self.config.max_frame_age_s:
            self._set_invalid(
                _STATE_STALE,
                f"stale frame: age {age:.3f}s > "
                f"{self.config.max_frame_age_s:.3f}s")
            return

        depth = getattr(frame, "depth", None)
        if depth is None:
            self._set_invalid(_STATE_INVALID, "depth frame is missing")
            return
        color_stamp_us = getattr(frame, "color_stamp_us", None)
        try:
            color_stamp_us = int(color_stamp_us)
        except (TypeError, ValueError):
            self._set_invalid(
                _STATE_INVALID,
                "color_stamp_us is missing; refusing to pair depth with an "
                "unvalidated color frame")
            return
        if color_stamp_us <= 0:
            self._set_invalid(
                _STATE_INVALID,
                f"invalid color_stamp_us={color_stamp_us}; cannot validate "
                f"color/depth pairing")
            return
        skew_us = abs(color_stamp_us - stamp_us)
        # Record the observed skew even when the pair is rejected, so
        # telemetry shows the actual input rather than the last success.
        with self._lock:
            self._last_pair_skew_ms = skew_us / 1000.0
        if skew_us > int(self.config.max_pair_delta_us):
            self._set_invalid(
                _STATE_INVALID,
                f"color/depth device timestamp skew {skew_us / 1000.0:.1f} ms "
                f"> {self.config.max_pair_delta_us / 1000.0:.1f} ms; "
                f"dropping unpaired frame")
            return

        with self._lock:
            previous = self._last_processed_stamp
        if previous is not None and stamp_us <= previous:
            # cuVSLAM odometry assumes a monotonic device clock.  Feeding a
            # reset clock into the same Tracker would silently corrupt its
            # frame.  Refuse until the service/backend is recreated.
            with self._lock:
                self._clock_reset = True
            self._set_invalid(
                _STATE_ERROR,
                f"device clock moved backwards ({stamp_us} <= {previous}); "
                f"restart live-slam so cuVSLAM gets a fresh tracker")
            return

        scale = getattr(frame, "depth_scale_mm", None)
        try:
            scale = float(scale)
        except (TypeError, ValueError):
            scale = None
        if scale is None or not math.isfinite(scale) or scale <= 0:
            self._set_invalid(
                _STATE_ERROR,
                f"depth_scale_mm unavailable/invalid ({scale!r}); refusing "
                f"to assume the raw uint16 depth is millimetres")
            return

        rgb = self._rgb(frame)
        if rgb is None:
            self._set_invalid(_STATE_INVALID,
                              "color frame is missing or undecodable")
            return
        expected_color = (self.calibration.color.height,
                          self.calibration.color.width, 3)
        if tuple(rgb.shape) != expected_color:
            self._set_invalid(
                _STATE_INVALID,
                f"color shape {tuple(rgb.shape)} != calibrated "
                f"{expected_color} (h, w, 3)")
            return
        depth_arr = np.asarray(depth)
        if tuple(depth_arr.shape) != self.calibration.depth_shape:
            self._set_invalid(
                _STATE_INVALID,
                f"depth shape {tuple(depth_arr.shape)} != calibrated "
                f"{self.calibration.depth_shape} (h, w)")
            return

        try:
            aligned_mm = self.registration.align(depth_arr, scale)
        except (CalibrationError, ValueError) as exc:
            self._set_invalid(_STATE_ERROR, f"depth registration failed: {exc}")
            return

        timestamp_ns = int(stamp_us * 1000)
        try:
            result = self.backend.track(timestamp_ns, rgb, aligned_mm)
        except Exception as exc:
            self._set_invalid(
                _STATE_ERROR,
                f"tracker exception: {type(exc).__name__}: {exc}")
            return

        with self._lock:
            self._frames_processed += 1
            self._last_processed_stamp = stamp_us
            self._last_depth_scale_mm = scale
            self._last_pair_skew_ms = skew_us / 1000.0
            self._pose_recv_ts = float(getattr(frame, "recv_ts",
                                               time.monotonic()))

        self._apply_result(result, timestamp_ns)
        self._maybe_add_geometry(aligned_mm, rgb)
        self._write_log(frame, result)

    def _apply_result(self, result, timestamp_ns: int):
        state = str(getattr(result, "tracking_state", "lost") or "lost").lower()
        error = getattr(result, "error", None)
        pose = getattr(result, "pose", None)
        corrected = getattr(result, "corrected_pose", None)

        if state == "error" or error:
            with self._lock:
                self._frames_lost += 1
                self._needs_new_segment = True
            self._set_invalid(_STATE_ERROR,
                              str(error) or "tracker reported an error",
                              count_reject=False)
            return
        if pose is None or state == "lost":
            with self._lock:
                self._frames_lost += 1
                self._needs_new_segment = True
            self._set_invalid(_STATE_LOST,
                              "tracking lost; pose invalid until recovery",
                              count_reject=False)
            return

        with self._lock:
            if self._needs_new_segment:
                self._segment += 1
                self._needs_new_segment = False
                self.geometry.reset(self._segment)
                self.structure.reset(self._segment)
                self._last_geometry_mono = None
            segment = self._segment
            pose_dict = self._pose_dict(pose, segment, timestamp_ns)
            if pose_dict is None:
                self._frames_lost += 1
                self._needs_new_segment = True
                invalid = True
            else:
                invalid = False
                self._pose = pose_dict
                self._pose_valid = True
                self._state = _STATE_TRACKING
                self._last_error = None
                self._frames_tracked += 1
                map_dict = None
                if self.config.enable_slam and corrected is not None:
                    map_dict = self._pose_dict(corrected, segment, timestamp_ns)
                self._map_pose = map_dict
                self._map_valid = map_dict is not None
                self._trajectory.append({
                    "t_ns": int(timestamp_ns),
                    "translation": list(pose_dict["translation"]),
                    "rotation": list(pose_dict["rotation"]),
                    "segment": int(segment),
                    "map_pose": (
                        {"translation": list(map_dict["translation"]),
                         "rotation": list(map_dict["rotation"]),
                         "frame_id": map_dict["frame_id"],
                         "pose_frame": POSE_FRAME}
                        if map_dict is not None else None),
                })
        if invalid:
            self._set_invalid(_STATE_INVALID,
                              "tracker returned an unparseable/non-finite pose",
                              count_reject=False)

    def _maybe_add_geometry(self, depth_mm, rgb):
        """Queue the current tick's aligned depth + odometry pose.

        Rate-limited to <=1 Hz per *attempt*.  When the live service is
        running, point-cloud fusion and structure updates happen on a
        dedicated worker; the tracking thread only enqueues a latest-wins
        item.  Mapping never changes localization validity.
        """
        if not self.config.enable_point_cloud \
                and not self.config.enable_structure:
            return
        with self._lock:
            if (not self._pose_valid or self._pose is None
                    or self._state != _STATE_TRACKING):
                return
            now = time.monotonic()
            if (self._pose_recv_ts is not None
                    and now - self._pose_recv_ts
                    > self.config.max_frame_age_s):
                return
            rates = []
            if self.config.enable_point_cloud:
                rates.append(float(self.geometry.config.sample_hz))
            if self.config.enable_structure:
                rates.append(float(self.structure.config.sample_hz))
            hz = max(rates) if rates else 0.0
            if (hz <= 0 or (self._last_geometry_mono is not None
                            and now - self._last_geometry_mono < 1.0 / hz)):
                return
            # Count the attempt even if the frame has no valid depth or the
            # update raises; otherwise an unusable stream could sample faster.
            self._last_geometry_mono = now
            pose = dict(self._pose)
            segment = self._segment
        if self._running and self._map_worker is not None:
            item = (depth_mm, rgb, pose, segment)
            try:
                self._map_queue.put_nowait(item)
            except queue.Full:
                try:
                    self._map_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._map_queue.put_nowait(item)
                except queue.Full:
                    pass
            return
        # Tests and direct _tick() callers: fuse synchronously.
        self._add_geometry(depth_mm, rgb, pose, segment)

    def _add_geometry(self, depth_mm, rgb, pose, segment):
        if self.config.enable_point_cloud:
            try:
                self.geometry.add(depth_mm, rgb, pose, segment)
                with self._lock:
                    self._map_error = None
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._map_error = message
                print(f"[live-slam] point-cloud update skipped: {message}",
                      flush=True)
        if self.config.enable_structure:
            try:
                self.structure.add(depth_mm, pose, segment)
                with self._lock:
                    self._structure_error = None
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._structure_error = message
                print(f"[live-slam] structure update skipped: {message}",
                      flush=True)
        return True

    def _map_loop(self):
        while True:
            item = self._map_queue.get()
            if item is None:
                break
            self._add_geometry(*item)

    def _stop_map_worker(self):
        worker = self._map_worker
        if worker is None:
            return
        self._map_worker = None
        try:
            self._map_queue.put_nowait(None)
        except queue.Full:
            try:
                self._map_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._map_queue.put_nowait(None)
            except queue.Full:
                pass
        worker.join(timeout=5.0)

    @staticmethod
    def _pose_dict(pose, segment: int, timestamp_ns: int) -> dict | None:
        if isinstance(pose, dict):
            data = dict(pose)
            translation = data.get("translation")
            rotation = data.get("rotation")
        else:
            translation = getattr(pose, "translation", None)
            rotation = getattr(pose, "rotation", None)
            data = {}
        try:
            tr = [float(v) for v in translation]
            rot = [float(v) for v in rotation]
        except (TypeError, ValueError):
            return None
        if len(tr) != 3 or len(rot) != 4:
            return None
        if not all(math.isfinite(v) for v in tr + rot):
            return None
        norm = math.sqrt(sum(v * v for v in rot))
        if norm <= 1e-9:
            return None
        rot = [v / norm for v in rot]
        return {
            "valid": True,
            "t_ns": int(timestamp_ns),
            "translation": tr,
            "rotation": rot,
            "frame_id": data.get("frame_id", FRAME_ID),
            "pose_frame": POSE_FRAME,
            "segment": int(segment),
        }

    def _set_invalid(self, state: str, message: str,
                     count_reject: bool = True):
        with self._lock:
            self._state = state
            self._last_error = str(message)
            self._pose_valid = False
            self._pose = None
            self._map_valid = False
            self._map_pose = None
            if count_reject:
                self._frames_rejected += 1

    # -- logging ----------------------------------------------------------

    def _write_log(self, frame, result):
        if self._log_file is None:
            return
        with self._lock:
            record = {
                "wall_time_ns": time.time_ns(),
                "device_stamp_us": getattr(frame, "stamp_us", None),
                "color_stamp_us": getattr(frame, "color_stamp_us", None),
                "depth_scale_mm": self._last_depth_scale_mm,
                "pair_skew_ms": self._last_pair_skew_ms,
                "state": self._state,
                "pose_valid": self._pose_valid,
                "pose": self._pose,
                "map_pose": self._map_pose if self._map_valid else None,
                "segment": self._segment,
                "error": self._last_error,
            }
        try:
            self._log_file.write(
                json.dumps(json_safe(record), allow_nan=False) + "\n")
        except Exception:
            pass

    def _maybe_log_status(self):
        interval = float(self.config.status_log_interval_s)
        now = time.monotonic()
        if interval <= 0 or now - self._last_status_log < interval:
            return
        self._last_status_log = now
        status = self.status()
        pose = status.get("pose")
        if pose is not None:
            x, y, z = pose["translation"]
            where = f"pose=({x:.3f},{y:.3f},{z:.3f})"
        else:
            where = "pose=none"
        print(f"[live-slam] state={status['tracking_state']} "
              f"seg={status['segment']} frames={status['frames_seen']} "
              f"tracked={status['frames_tracked']} lost={status['frames_lost']} "
              f"rejected={status['frames_rejected']} {where} "
              f"err={status['last_error']}", flush=True)

    # -- read-only API ----------------------------------------------------

    def status(self) -> dict:
        with self._lock:
            now = time.monotonic()
            pose = None
            map_pose = None
            effective_state = self._state
            if self._pose_valid and self._pose is not None:
                age = (now - self._pose_recv_ts
                       if self._pose_recv_ts is not None else None)
                # A blocked worker/frozen capture must not leave a stale
                # pose looking current just because no tick ran.
                if age is None or age > self.config.max_frame_age_s:
                    effective_state = _STATE_STALE
                else:
                    pose = dict(self._pose)
                    pose["age_s"] = age
                    pose["pose_frame"] = POSE_FRAME
                    if self.config.enable_slam and self._map_valid \
                            and self._map_pose is not None:
                        map_pose = dict(self._map_pose)
                        map_pose["age_s"] = age
                        map_pose["pose_frame"] = POSE_FRAME
            status = {
                "ok": bool(self._running and effective_state != _STATE_ERROR),
                "service": "live-slam",
                "session_id": self.session_id,
                "role": self.config.role,
                "serial": self.calibration.serial,
                "frame_id": self.calibration.frame_id,
                "pose_frame": POSE_FRAME,
                "tracking_state": effective_state,
                "pose_valid": pose is not None,
                "pose": pose,
                "map_pose": map_pose,
                "slam_enabled": bool(self.config.enable_slam),
                "point_cloud_enabled": bool(self.config.enable_point_cloud),
                "structure_enabled": bool(self.config.enable_structure),
                "structure_error": self._structure_error,
                "segment": int(self._segment),
                "frames_seen": int(self._frames_seen),
                "frames_processed": int(self._frames_processed),
                "frames_tracked": int(self._frames_tracked),
                "frames_lost": int(self._frames_lost),
                "frames_rejected": int(self._frames_rejected),
                "trajectory_points": len(self._trajectory),
                "trajectory_capacity": int(self.config.trajectory_capacity),
                "last_error": self._last_error,
                "map_error": self._map_error,
                "last_frame_age_s": self._last_frame_age_s,
                "depth_scale_mm": self._last_depth_scale_mm,
                "pair_skew_ms": self._last_pair_skew_ms,
                "imu": {
                    "requested": bool(self.config.use_imu),
                    "enabled": False,
                    "reason": (IMU_REJECT_REASON if self.config.use_imu
                               else "disabled (RGB-D only POC)"),
                },
                "backend": getattr(self.backend, "name",
                                   type(self.backend).__name__),
                "uptime_s": ((now - self._started_at)
                             if self._started_at is not None else 0.0),
                "calibration": self.calibration.summary(),
            }
        return json_safe(status)

    def pose(self) -> dict:
        status = self.status()
        pose = status.get("pose")
        if pose is not None:
            return pose
        return {
            "valid": False,
            "frame_id": self.calibration.frame_id,
            "pose_frame": POSE_FRAME,
            "tracking_state": status["tracking_state"],
            "last_error": status["last_error"],
        }

    def trajectory(self, limit: int | None = None) -> dict:
        with self._lock:
            points = list(self._trajectory)
            if limit is not None:
                try:
                    n = max(0, int(limit))
                except (TypeError, ValueError):
                    n = 0
                points = points[-n:] if n else []
            return json_safe({
                "session_id": self.session_id,
                "frame_id": self.calibration.frame_id,
                "pose_frame": POSE_FRAME,
                "segment": int(self._segment),
                "capacity": int(self.config.trajectory_capacity),
                "count": len(self._trajectory),
                "returned": len(points),
                "points": points,
            })

    def _trajectory_for_map(self, limit: int = 300) -> dict:
        """Bounded trajectory for /map, preserving segment boundaries."""
        points = list(self._trajectory)
        total = len(points)
        if total <= limit:
            selected = points
        else:
            keep: set[int] = set()
            start = 0
            for i in range(1, total + 1):
                if i == total or points[i]["segment"] != points[start]["segment"]:
                    keep.add(start)
                    keep.add(i - 1)
                    start = i
            if len(keep) > limit:
                keep = set(sorted(keep)[-limit:])
            remaining = limit - len(keep)
            if remaining > 0:
                step = max(1, total // remaining)
                for i in range(total - 1, -1, -step):
                    keep.add(i)
                    if len(keep) >= limit:
                        break
                if len(keep) < limit:
                    for i in range(total - 1, -1, -1):
                        keep.add(i)
                        if len(keep) >= limit:
                            break
            selected = [points[i] for i in sorted(keep)]
        return json_safe({
            "session_id": self.session_id,
            "frame_id": self.calibration.frame_id,
            "pose_frame": POSE_FRAME,
            "segment": int(self._segment),
            "capacity": int(self.config.trajectory_capacity),
            "count": total,
            "returned": len(selected),
            "points": selected,
        })

    def _map_status_trajectory(self):
        with self._lock:
            return (self.status(), self._trajectory_for_map(300),
                    self.session_id)

    def map_snapshot(self) -> dict:
        """Map + status + trajectory as a dict (tests/tools).

        Geometry rebuild happens outside the LiveSlam status lock.  The HTTP
        path uses ``map_snapshot_bytes`` to reuse pre-serialized points.
        """
        geom = self.geometry.snapshot(0.0)
        status, trajectory, session_id = self._map_status_trajectory()
        if geom["segment"] != status["segment"]:
            geom = self.geometry.snapshot(0.0)
            status, trajectory, session_id = self._map_status_trajectory()
        return {
            "session_id": session_id,
            "revision": geom["revision"],
            "frame_id": geom["frame_id"],
            "pose_frame": geom["pose_frame"],
            "geometry_source": geom["geometry_source"],
            "corrected": geom["corrected"],
            "segment": geom["segment"],
            "points": geom["points"],
            "count": geom["count"],
            "capacity": geom["capacity"],
            "voxel_size_m": geom["voxel_size_m"],
            "map_points": geom["map_points"],
            "transmit_stride": geom["transmit_stride"],
            "updated_age_s": geom["updated_age_s"],
            "status": status,
            "trajectory": trajectory,
        }

    def map_snapshot_bytes(self) -> bytes:
        """Assemble /map bytes from cached points + small per-request fields.

        The immutable points array is serialized at most once per revision
        (and only when a reader asks).  Geometry work runs outside the
        LiveSlam lock; only status/trajectory copies happen under it.
        """
        geom, head, points = self.geometry.snapshot_bundle(
            float(self.config.geometry_cache_min_interval_s))
        status, trajectory, session_id = self._map_status_trajectory()
        if geom["segment"] != status["segment"]:
            # Segment changed while the lazy rebuild was running; never mix
            # old geometry with the new segment.
            geom, head, points = self.geometry.snapshot_bundle(0.0)
            status, trajectory, session_id = self._map_status_trajectory()
        tail = {
            "count": geom["count"],
            "capacity": geom["capacity"],
            "voxel_size_m": geom["voxel_size_m"],
            "map_points": geom["map_points"],
            "transmit_stride": geom["transmit_stride"],
            "updated_age_s": geom["updated_age_s"],
            "status": status,
            "trajectory": trajectory,
        }
        sid = json.dumps(session_id, separators=(",", ":"),
                         allow_nan=False).encode()
        tail_bytes = json.dumps(tail, separators=(",", ":"),
                                allow_nan=False).encode()
        return (b'{"session_id":' + sid + b',' + head
                + b',"points":' + points + b',' + tail_bytes[1:])

    def structure_snapshot(self) -> dict:
        """Phase-1 structural map snapshot as a dict (tests/tools)."""
        body = self.structure.snapshot(
            float(self.config.structure_cache_min_interval_s))
        with self._lock:
            status = self.status()
            trajectory = self._trajectory_for_map(300)
            session_id = self.session_id
        return {
            "ok": status["ok"],
            "service": "live-slam",
            "session_id": session_id,
            "frame_id": body["frame_id"],
            "pose_frame": body["pose_frame"],
            "structural": body["structural"],
            "persistent": body["persistent"],
            "loop_corrected": body["loop_corrected"],
            "gravity_source": body["gravity_source"],
            "gravity": body["gravity"],
            "floor": body["floor"],
            "grid": body["grid"],
            "diagnostics": body["diagnostics"],
            "walls": body["walls"],
            "wall_cells": body["wall_cells"],
            "path": body["path"],
            "pose": body["pose"],
            "trajectory": trajectory,
            "status": status,
            "revision": body["revision"],
            "updated_age_s": body["updated_age_s"],
        }

    def structure_snapshot_bytes(self) -> bytes:
        """Assemble /structure bytes from the cached structural body."""
        body, revision, age = self.structure.body_bundle(
            float(self.config.structure_cache_min_interval_s))
        with self._lock:
            status = self.status()
            trajectory = self._trajectory_for_map(300)
            session_id = self.session_id
        sid = json.dumps(session_id, separators=(",", ":"),
                         allow_nan=False).encode()
        status_bytes = json.dumps(status, separators=(",", ":"),
                                  allow_nan=False).encode()
        traj_bytes = json.dumps(trajectory, separators=(",", ":"),
                                allow_nan=False).encode()
        age_bytes = (b"null" if age is None else
                     str(round(float(age), 4)).encode())
        return (b'{"ok":' + (b"true" if status["ok"] else b"false")
                + b',"service":"live-slam","session_id":' + sid + b','
                + body
                + b',"trajectory":' + traj_bytes
                + b',"status":' + status_bytes
                + b',"revision":' + str(revision).encode()
                + b',"updated_age_s":' + age_bytes + b'}')
