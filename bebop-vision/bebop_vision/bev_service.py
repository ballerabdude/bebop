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

import cv2

from .master_bev import MasterBevBuilder, render_bev


class BevService:
    """Latest-wins master-BEV producer."""

    def __init__(self, rig, *, builder=None, hz=10.0, video_server=None,
                 jpeg_quality=80):
        self.rig = rig
        self.builder = builder or MasterBevBuilder()
        self.hz = max(1.0, float(hz))
        self.video_server = video_server
        self.jpeg_quality = int(jpeg_quality)

        self._latest = None
        self._jpeg = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cache = {}
        self._frames = 0
        self._fps = 0.0
        self._stats_ts = time.monotonic()
        self._last_error = None
        self._last_grid_ts = 0.0

    # -- one tick -----------------------------------------------------------

    def tick_once(self):
        """Read the current frames once, fuse, publish.  Deterministic/testable."""
        per_cam, ages = {}, {}
        for role, cam in self.rig.cameras.items():
            frame = cam.read()
            if frame is None:
                per_cam[role], ages[role] = None, None
                continue
            age = frame.age_s()
            ages[role] = age
            if age > self.builder.max_frame_age_s:
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
            grid = self.builder.fuse(per_cam, ages)
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"fuse: {type(exc).__name__}: {exc}"
            grid = None
        if grid is None:
            return None

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
                        f"{r}:{v['points']}pts/{v['cells']}c"
                        for r, v in g.contributions.items())
                    print(f"[bev] {fps:4.1f} Hz  {g.rows}x{g.cols}  {parts}  "
                          f"occ={c['occupied']} haz={c['hazard']} "
                          f"unk={c['unknown']}", flush=True)
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
        print(f"[bev] started {self.builder.range_m:.1f}x"
              f"{self.builder.width_m:.1f} m @ {self.builder.cell_m*100:.0f} cm, "
              f"{self.hz:.0f} Hz", flush=True)

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
            "grid": grid.to_dict() if grid is not None else None,
        }
