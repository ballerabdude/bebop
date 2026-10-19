"""Live master-BEV worker.

Owns one 10 Hz loop over the rig's latest depth frames, fuses them with
:mod:`bebop_vision.master_bev`, and publishes the resulting grid to the
operator video server (``GET /bev`` JSON + ``/snapshot?stream=bev`` JPEG).

The worker is deliberately independent of recording, driving, place memory
and the router: it only reads the camera slots.  Processing is serial in
this thread (pyorbbecsdk stays out of pools, docs §2.8); the numpy/cv2 work
is a few milliseconds per tick.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import cv2

from .bev_student_runtime import BevStudentRuntime
from .master_bev import MasterBevBuilder, render_bev
from .orbbec import load_rig_config


class BevService:
    """Latest-wins master-BEV producer."""

    def __init__(self, rig, *, builder=None, hz=10.0, video_server=None,
                 jpeg_quality=80, model_path=None, model_meta=None,
                 hazard_source="geometric"):
        self.rig = rig
        self.model_path = model_path
        self.builder = builder
        self.hz = max(1.0, float(hz))
        self.video_server = video_server
        self.jpeg_quality = int(jpeg_quality)
        self.hazard_source = hazard_source
        self.runtime = None
        self.model_error = None
        if model_path:
            try:
                self.runtime = BevStudentRuntime(
                    model_path, meta_path=model_meta,
                    hazard_source=hazard_source)
                print(f"[bev] learned model {model_path} "
                      f"({self.runtime.status()['input_shape']}, "
                      f"hazard={hazard_source})", flush=True)
            except Exception as exc:  # noqa: BLE001 - no old-BEV fallback
                self.model_error = f"{type(exc).__name__}: {exc}"
                print(f"[bev] learned model failed ({self.model_error}); "
                      f"BEV disabled (model-only mode)", flush=True)
            # Model-only: the geometric builder is not instantiated or run.
            self.builder = None
        elif self.builder is None:
            self.builder = MasterBevBuilder()
        if self.runtime is not None:
            self.range_m = self.runtime.range_m
            self.width_m = self.runtime.width_m
            self.cell_m = self.runtime.cell_m
            cfg = load_rig_config()["robots"]["default"]
            self.max_frame_age_s = float(
                (cfg.get("safety") or {}).get("max_frame_age_s", 0.3))
        elif self.builder is not None:
            self.max_frame_age_s = self.builder.max_frame_age_s
            self.range_m = self.builder.range_m
            self.width_m = self.builder.width_m
            self.cell_m = self.builder.cell_m
        else:
            # Model-only and the engine failed to load: keep the service
            # alive (no grid, no geometric fallback) so the rest of the
            # recorder/app still works.
            cfg = load_rig_config()["robots"]["default"]
            self.max_frame_age_s = float(
                (cfg.get("safety") or {}).get("max_frame_age_s", 0.3))
            self.range_m, self.width_m, self.cell_m = 6.0, 3.0, 0.05

        self._latest = None
        self._jpeg = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cache = {}
        self._last_color = {}
        self._frames = 0
        self._fps = 0.0
        self._stats_ts = time.monotonic()
        self._last_error = None
        self._last_grid_ts = 0.0

    # -- one tick -----------------------------------------------------------

    def _prepare_geometric(self, frames, ages):
        if self.builder is None:
            return None
        per_cam = {}
        for role, cam in self.rig.cameras.items():
            frame = frames.get(role)
            if frame is None:
                per_cam[role] = None
                continue
            cached = self._cache.get(role)
            if cached is not None and cached[0] == frame.stamp_us:
                per_cam[role] = cached[1]
                continue
            try:
                prepared = self.builder.prepare(frame)
            except Exception as exc:  # noqa: BLE001 - keep the feed alive
                self._last_error = f"{role}: {type(exc).__name__}: {exc}"
                per_cam[role] = None
                continue
            self._cache[role] = (frame.stamp_us, prepared)
            per_cam[role] = prepared
        try:
            return self.builder.fuse(per_cam, ages)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"fuse: {type(exc).__name__}: {exc}"
            return None

    def tick_once(self):
        """Read the current frames once, fuse, publish.  Deterministic/testable."""
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
                # the far camera).  The model needs both colors; hold the
                # last JPEG and pair it with the fresh depth.  The recorder
                # does the same for H.265.
                model_frames[role] = SimpleNamespace(
                    color_jpeg=self._last_color[role], color=None,
                    depth=frame.depth, depth_scale_mm=frame.depth_scale_mm,
                    stamp_us=frame.stamp_us)
            else:
                model_frames[role] = None

        grid = None
        if self.runtime is not None:
            missing = [r for r in ("near", "far")
                       if model_frames.get(r) is None]
            if missing:
                color = {}
                for r in ("near", "far"):
                    fr = frames.get(r)
                    color[r] = bool(
                        fr is not None
                        and (fr.color_jpeg is not None
                             or fr.color is not None))
                self._last_error = (
                    f"model input missing: {missing} color={color} "
                    f"age_s={ {r: ages.get(r) for r in ('near', 'far')} }")
                grid = None
            else:
                try:
                    grid = self.runtime.infer(
                        model_frames, geo_grid=None, ages_s=ages)
                except Exception as exc:  # noqa: BLE001 - no old-BEV fallback
                    self._last_error = (
                        f"model: {type(exc).__name__}: {exc}")
                    print(f"[bev] model tick failed: {self._last_error}",
                          flush=True)
                    grid = None
        elif self.builder is not None:
            grid = self._prepare_geometric(frames, ages)
        if grid is None:
            return None
        self._last_error = None

        jpeg = None
        try:
            ok, buf = cv2.imencode(
                ".jpg", render_bev(grid),
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
                    parts = " ".join(
                        f"{r}:{v.get('points', 0)}pts/{v.get('cells', 0)}c"
                        for r, v in g.contributions.items())
                    model = (" model="
                             f"{self.runtime.status()['infer_ms']}ms"
                             if self.runtime is not None else "")
                    print(f"[bev] {fps:4.1f} Hz  {g.rows}x{g.cols}  {parts}  "
                          f"occ={c['occupied']} haz={c['hazard']} "
                          f"unk={c['unknown']}{model}", flush=True)
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
              f"@ {self.cell_m*100:.0f} cm, {self.hz:.0f} Hz"
              f"{' (learned model-only)' if self.runtime is not None else ''}",
              flush=True)

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
            "model": (self.runtime.status() if self.runtime is not None
                      else None),
            "model_error": self.model_error,
            "grid": grid.to_dict() if grid is not None else None,
        }
