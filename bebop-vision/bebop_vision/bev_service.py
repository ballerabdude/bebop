"""Live learned master-BEV worker (model-only).

Owns one 10 Hz loop over the rig's latest RGB-D frames, runs the TensorRT
student, and publishes the resulting grid to the operator video server
(``GET /bev`` JSON + ``/snapshot?stream=bev`` JPEG + the WHEP ``bev``
stream).  There is no geometric occupancy builder and no geometric
fallback; the model is the BEV.

The worker only reads the camera slots.  Processing is serial in this
thread (pyorbbecsdk stays out of pools, docs §2.8); the model preprocess
decodes the two camera MJPEG frames, then TensorRT runs the pair.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import cv2

from .bev_student_runtime import BevStudentRuntime
from .master_bev import render_bev
from .orbbec import load_rig_config


class BevService:
    """Latest-wins learned-BEV producer."""

    def __init__(self, rig, *, hz=10.0, video_server=None,
                 jpeg_quality=80, model_path=None, model_meta=None,
                 hazard_source="model", hazard_threshold=0.5,
                 runtime=None):
        self.rig = rig
        self.hz = max(1.0, float(hz))
        self.video_server = video_server
        self.jpeg_quality = int(jpeg_quality)
        self.hazard_source = hazard_source
        if runtime is None:
            if not model_path:
                raise ValueError(
                    "BevService is model-only; pass model_path or runtime")
            runtime = BevStudentRuntime(
                model_path, meta_path=model_meta,
                hazard_source=hazard_source,
                hazard_threshold=hazard_threshold)
        self.runtime = runtime
        self.range_m = float(runtime.range_m)
        self.width_m = float(runtime.width_m)
        self.cell_x_m = float(getattr(runtime, "cell_x_m", None)
                              or runtime.cell_m)
        self.cell_y_m = float(getattr(runtime, "cell_y_m", None)
                              or runtime.cell_m)
        self.cell_m = (self.cell_x_m if abs(self.cell_x_m - self.cell_y_m)
                       < 1e-9 else None)
        cfg = load_rig_config()["robots"]["default"]
        self.max_frame_age_s = float(
            (cfg.get("safety") or {}).get("max_frame_age_s", 0.3))

        self._latest = None
        self._jpeg = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_color = {}
        self._frames = 0
        self._fps = 0.0
        self._stats_ts = time.monotonic()
        self._last_error = None
        self._last_grid_ts = 0.0
        print(f"[bev] learned model {getattr(runtime, 'engine_path', '?')} "
              f"({getattr(runtime, 'in_shape', '?')}, hazard={hazard_source}, "
              f"threshold={hazard_threshold})", flush=True)

    # -- one tick -----------------------------------------------------------

    def tick_once(self):
        """Read the current frames once, run the model, publish."""
        frames, model_frames, ages = {}, {}, {}
        for role, cam in self.rig.cameras.items():
            frame = cam.read()
            if frame is None:
                frames[role], model_frames[role], ages[role] = (
                    None, None, None)
                continue
            age = frame.age_s()
            ages[role] = age
            frame = (frame if age <= self.max_frame_age_s else None)
            frames[role] = frame
            if frame is None:
                model_frames[role] = None
                continue
            if frame.color_jpeg is not None:
                self._last_color[role] = frame.color_jpeg
                model_frames[role] = frame
            elif frame.color is not None:
                model_frames[role] = frame
            elif self._last_color.get(role) is not None:
                # The Orbbec SDK drops color frames under load (~27-35% on
                # the far camera).  Hold the last JPEG and pair it with the
                # fresh depth.
                model_frames[role] = SimpleNamespace(
                    color_jpeg=self._last_color[role], color=None,
                    depth=frame.depth, depth_scale_mm=frame.depth_scale_mm,
                    stamp_us=frame.stamp_us)
            else:
                model_frames[role] = None

        missing = [r for r in ("near", "far")
                   if model_frames.get(r) is None]
        if missing:
            self._last_error = f"model input missing: {missing}"
            return None
        try:
            grid = self.runtime.infer(
                model_frames, geo_grid=None, ages_s=ages)
        except Exception as exc:  # noqa: BLE001 - model-only, no fallback
            self._last_error = f"model: {type(exc).__name__}: {exc}"
            print(f"[bev] model tick failed: {self._last_error}", flush=True)
            return None
        if grid is None:
            self._last_error = "model returned no grid"
            return None
        self._last_error = None

        jpeg = None
        try:
            # Upscale small grids for the operator stream; a 360-row
            # matched grid is already display sized.
            ok, buf = cv2.imencode(
                ".jpg", render_bev(grid, scale=max(1, 480 // grid.rows)),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                jpeg = buf.tobytes()
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"render: {type(exc).__name__}: {exc}"

        with self._lock:
            self._latest = grid
            self._jpeg = jpeg
            self._last_grid_ts = time.monotonic()
        if self.video_server is not None:
            self.video_server.publish_bev(grid, jpeg)
        return grid

    # -- worker -------------------------------------------------------------

    def _loop(self):
        interval = 1.0 / self.hz
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.tick_once()
            except Exception as exc:  # noqa: BLE001 - never kill the process
                self._last_error = f"tick: {type(exc).__name__}: {exc}"
            self._frames += 1
            now = time.monotonic()
            dt = now - self._stats_ts
            if dt >= 5.0:
                fps = self._frames / dt
                self._fps = fps
                self._frames = 0
                self._stats_ts = now
                g = self.latest()
                if g is not None:
                    c = g.counts()
                    model = self.runtime.status()
                    print(f"[bev] {fps:4.1f} Hz  {g.rows}x{g.cols}  "
                          f"occ={c['occupied']} haz={c['hazard']} "
                          f"unk={c['unknown']} "
                          f"infer={model.get('infer_ms')}ms", flush=True)
                elif self._last_error:
                    print(f"[bev] no grid: {self._last_error}", flush=True)
            sleep = interval - (time.monotonic() - t0)
            if sleep > 0:
                self._stop.wait(sleep)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bev",
                                        daemon=True)
        self._thread.start()
        print(f"[bev] started {self.range_m:.1f}x{self.width_m:.1f} m "
              f"@ {self.cell_x_m*100:.2g}x{self.cell_y_m*100:.2g} cm, "
              f"{self.hz:.0f} Hz (learned model-only)", flush=True)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- readers ------------------------------------------------------------

    def latest(self, max_age_s=None):
        with self._lock:
            grid = self._latest
            ts = self._last_grid_ts
        if grid is None:
            return None
        if max_age_s is not None and time.monotonic() - ts > max_age_s:
            return None
        return grid

    def latest_jpeg(self):
        with self._lock:
            return self._jpeg

    def status(self):
        with self._lock:
            grid = self._latest
            age = (time.monotonic() - self._last_grid_ts
                   if self._last_grid_ts else None)
            jpeg = self._jpeg
        return {
            "running": bool(self._thread is not None
                            and self._thread.is_alive()),
            "hz": self.hz,
            "fps": round(self._fps, 1),
            "last_error": self._last_error,
            "grid_age_s": round(age, 3) if age is not None else None,
            "has_jpeg": jpeg is not None,
            "model": self.runtime.status(),
            "grid": grid.to_dict() if grid is not None else None,
        }
