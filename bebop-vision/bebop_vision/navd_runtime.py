"""Runtime BEV provider for the navd student model (plan §7.3).

Swaps the geometric BEV source for the exported NavdUNet ONNX at runtime:
raw sensor frames (near/far depth + near color) and the goal direction go
through one forward pass to a 60x60 class map that is wrapped in a
BevGrid the planner consumes unchanged ("the grid is the seam" —
GoalPlanner/GoalDriveNode never see the difference).

Auto-fallback to the geometric grid (plan §7.3): missing/stale camera
streams, NaN/Inf logits, `frac_navigable` outside [0.05, 0.95], or any
inference exception — the geometric grid computed in the same tick takes
over and the provider switch is logged once per transition. A fresh
model grid always wins; a single bad tick falls back for that tick only
(the next good prediction reclaims the provider, so a flaky model
degrades to "geometric with occasional model ticks" rather than
sticking).

Class mapping (model -> BEV vocabulary):
  navigable(1) -> FREE, blocked(0) -> OCCUPIED, caution(2) -> INFLATED in
  the planning grid (blocks polar rays and the near-cone check like any
  blockage, renders as the planning-block color) and -> HAZARD in the raw
  telemetry grid. No runtime inflation: the teacher labels the student
  was trained on already carry the margins.

Threading: predict() runs in the caller's BEV worker thread (never in a
camera capture thread — §2.8). onnxruntime releases the GIL inside Run;
preprocessing is numpy/cv2 only.
"""

import threading
import time

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("pip install opencv-python") from exc

from .bev import BevGrid, FREE, OCCUPIED, HAZARD, INFLATED
from .navd_pre import CELL_M, GRID, build_goal_raster, prep_color, prep_depth

BLOCKED, NAVIGABLE, CAUTION = 0, 1, 2

# Model class -> planning-grid class. HAZARD keeps caution distinct in the
# raw/telemetry map; INFLATED blocks planning without claiming "solid
# obstacle" the way OCCUPIED would.
_PLANNING = {BLOCKED: OCCUPIED, NAVIGABLE: FREE, CAUTION: INFLATED}
_RAW = {BLOCKED: OCCUPIED, NAVIGABLE: FREE, CAUTION: HAZARD}


def goal_raster_from_slot(goal, odom):
    """Planner goal (GoalHeading | GoalPoint | None) + odom tuple ->
    (60, 60) goal fan raster (duck-typed to avoid a goal_planner import)."""
    if goal is None:
        g = {"type": "none"}
    elif hasattr(goal, "heading_rad"):
        g = {"type": "heading", "heading_rad": goal.heading_rad}
    else:
        g = {"type": "point", "x": goal.x, "y": goal.y}
    return build_goal_raster(
        g, {"x": odom[0], "y": odom[1], "theta": odom[2]})


def gather_frames(rig, max_age_s):
    """Latest-wins read of every camera, freshness-filtered.

    Returns (frames, ages): frames[role] = StampedFrame or None (stale
    frames are reported as None, matching the BEV workers' per_cam
    convention), ages[role] = float or None. Read-only over the rig —
    safe to call from any worker thread (capture stays in the camera
    threads, §2.8).
    """
    frames, ages = {}, {}
    for role, cam in rig.cameras.items():
        f = cam.read()
        if f is None or f.age_s() > max_age_s:
            frames[role] = None
            continue
        frames[role] = f
        ages[role] = f.age_s()
    return frames, ages


class NavdModel:
    """ONNX NavdUNet session: raw frames -> (3, 60, 60) logits.

    Providers default to CUDA when available (Orin Nano), CPU otherwise;
    `providers` overrides (tests pin CPU). Input tensor names are taken
    from the session so the export's fixed-name contract is enforced by
    construction, not by string hope.
    """

    def __init__(self, path, providers=None):
        import onnxruntime as ort
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider",
                                     "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.output_name = self.sess.get_outputs()[0].name
        self.last_latency_ms = None

    def predict(self, depth_near_mm, depth_far_mm, color_rgb, goal_raster):
        """uint16 mm depths, RGB uint8 color, (60, 60) goal fan ->
        (3, 60, 60) float32 logits. Raises on malformed inputs (the
        source's fallback catches)."""
        dn, _ = prep_depth(depth_near_mm)
        df, _ = prep_depth(depth_far_mm)
        c = prep_color(color_rgb)
        feed = {name: t for name, t in zip(
            self.input_names,
            (dn[None].astype(np.float32), df[None].astype(np.float32),
             c[None].astype(np.float32),
             goal_raster[None, None].astype(np.float32)))}
        t0 = time.monotonic()
        logits = self.sess.run([self.output_name], feed)[0][0]
        self.last_latency_ms = (time.monotonic() - t0) * 1e3
        return logits


class NavdGridSource:
    """Model-first grid source with automatic geometric fallback.

    update(frames, geo_grid, goal, odom) -> (BevGrid, provider) where
    provider is "navd" or "geometric". `frames` is the fresh-only dict
    from gather_frames(); `geo_grid` is this tick's geometric BevGrid
    (fallback value + plane_ok carrier for the no_floor gate).

    The near camera's color arrives as camera-encoded MJPEG bytes
    (rig `color_format: mjpg`); decoding happens here, in the worker
    thread, cached per device stamp — never in the capture thread.
    """

    def __init__(self, model_path, max_frame_age_s=0.3,
                 frac_range=(0.05, 0.95), providers=None, log=print):
        self.model = NavdModel(model_path, providers=providers)
        self.max_frame_age_s = max_frame_age_s
        self.frac_lo, self.frac_hi = frac_range
        self._log = log
        self._provider = None
        self.last_reason = None   # why the model did not produce the last grid
        self._color_stamp = None
        self._color_rgb = None
        self._lock = threading.Lock()

    # --- inputs -------------------------------------------------------------

    def _color(self, near):
        if near.color is not None:
            return near.color
        if near.color_jpeg is None:
            return None
        if self._color_stamp != near.stamp_us:
            img = cv2.imdecode(
                np.frombuffer(near.color_jpeg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            self._color_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self._color_stamp = near.stamp_us
        return self._color_rgb

    # --- provider -----------------------------------------------------------

    def _try_model(self, frames, geo_grid, goal, odom, now):
        """(BevGrid, None) on success or (None, reason) on any failure."""
        try:
            near, far = frames.get("near"), frames.get("far")
            if near is None or far is None:
                return None, "no fresh near+far depth pair"
            if (near.age_s(now) > self.max_frame_age_s
                    or far.age_s(now) > self.max_frame_age_s):
                return None, "camera frame stale"
            color = self._color(near)
            if color is None:
                return None, "no decodable near color"
            raster = goal_raster_from_slot(goal, odom)
            logits = self.model.predict(near.depth, far.depth, color, raster)
            if not np.isfinite(logits).all():
                return None, "non-finite logits"
            cls = logits.argmax(axis=0).astype(np.uint8)
            frac = float((cls == NAVIGABLE).mean())
            if not (self.frac_lo <= frac <= self.frac_hi):
                return None, (f"frac_navigable {frac:.2f} outside "
                              f"[{self.frac_lo}, {self.frac_hi}]")
            occ = np.full((GRID, GRID), FREE, np.uint8)
            raw = np.full((GRID, GRID), FREE, np.uint8)
            for c in (BLOCKED, CAUTION):
                occ[cls == c] = _PLANNING[c]
                raw[cls == c] = _RAW[c]
            ages = {"near": near.age_s(now), "far": far.age_s(now)}
            return BevGrid(
                occ=occ, raw=raw, stamp_us=max(near.stamp_us, far.stamp_us),
                per_camera_age_s=ages,
                # The model knows nothing about ground planes; carry the
                # geometric fit health so the no_floor gate stays live.
                plane_ok=dict(geo_grid.plane_ok) if geo_grid is not None else {},
                roles=["navd"], cell_m=CELL_M, recv_ts=now), None
        except Exception as exc:   # noqa: BLE001 — any model failure falls back
            return None, f"{type(exc).__name__}: {exc}"

    def update(self, frames, geo_grid, goal, odom, now=None):
        """One tick: try the model, fall back to geometric. Returns
        (grid, provider); grid is None only when neither source has one.

        Fallback is per-tick and immediate — stricter than the spec's
        "> 0.5 s stale" window (which the drive node's recv_ts deadman
        enforces independently). A single bad tick serves geometric for
        that tick; the next good prediction reclaims the provider.
        """
        now = time.monotonic() if now is None else now
        with self._lock:
            grid, reason = self._try_model(frames, geo_grid, goal, odom, now)
            if grid is not None:
                provider = "navd"
                self.last_reason = None
            else:
                provider = "geometric"
                grid = geo_grid
                self.last_reason = reason
            if self._provider != provider:
                suffix = f" ({reason})" if reason else ""
                self._log(f"[navd-model] provider "
                          f"{self._provider or 'startup'} -> {provider}{suffix}")
            self._provider = provider
        return grid, provider

    @property
    def provider(self):
        return self._provider or "geometric"
