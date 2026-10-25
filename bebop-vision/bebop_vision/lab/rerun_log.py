"""Show a session, its labels and model runs in the ``bebop_navd`` Rerun app.

``show()`` loads the recording through the Rerun SDK under the dashboard's
application id (the same trick as ``rerun_dashboards/open.py``: a ``.mcap``
opened directly is named after the file and never matches the saved
blueprint), then logs the lab's outputs onto the recording's own timeline so
a path prediction, a mask or an embedding sits on the frame it was computed
from. The dashboard is the navd one with one more right-hand tab, ``lab``:

* ``near camera + paths`` / ``far camera + paths``: the colour videos with
  every run's path projected onto the floor and any masks overlaid;
* ``plan (body frame)``: a 2D view at ``/lab/plan`` with every run's path
  and the robot footprint;
* one time-series view per run that produced embeddings or scalars
  (``/lab/<run>/...``).

Timeline. Rerun's MCAP loader (0.31) puts every message on the
``message_log_time`` timeline (plus ``message_publish_time``), not on
``log_time``, so the lab logs there too (:data:`TIMELINE`): nothing lines up
otherwise. Outputs go in columns (``send_columns`` with a ``TimeColumn`` of
the ticks' ``t_ns``), one call per entity, which also keeps the SDK's
wall-clock ``log_time`` off the lab's rows.

Body-frame plan view. Rerun's 2D views are image-like: ``u`` grows to the
right and ``v`` grows downwards. The body frame is x forward, y left, so a
path is drawn with x upwards and y leftwards by logging ``(u, v) = (-y, -x)``
(:func:`plan_xy`). The footprint is the rig config's measured chassis
rectangle (``bev_student.ego.x_range_m`` / ``y_range_m`` in
``config/orbbec_rig.yaml``, read through ``bebop_vision.orbbec.load_rig_config``;
the body sits about 10 cm ahead of and 5 cm right of the mast, so it is not
centred on the origin), logged once as a static strip so a path's first
points land relative to the real body. :data:`FOOTPRINT_XY` is the fallback
when the config is not readable on this machine.

Entity layout (``<run>`` is ``run.meta["name"]``; the leaf is the run name
when the output is named like its kind, ``path`` / ``mask`` / ``embedding``
/ ``scalars`` / ``classes``, else ``<run>.<output>``):

* ``Path2D``  -> ``/lab/plan/<leaf>`` (plan) and ``/color_<role>/lab/<leaf>``
  for both cameras (projected onto the floor plane through
  ``calib.camera(role).project``; the far camera does not see the first
  ~1.1 m of floor, so short paths only show in the near image; skipped
  silently per camera when its colour
  intrinsics are unknown, which is the case when
  ``config/orbbec_intrinsics_<serial>.json`` is not on this machine);
* ``Mask``    -> ``/color_<camera>/lab/<leaf>/mask`` as a ``SegmentationImage``
  at the full image resolution (upsampled by ``stride``, placed at ``crop``;
  class ids are shifted by one so 0 stays Rerun's transparent id and marks
  the uncovered area), with a static ``AnnotationContext`` for the names;
* ``Embedding`` -> PCA to three components over the run (numpy SVD) as
  ``Scalars`` at ``/lab/<run>/<output>/pc0..2``;
* ``Scalars`` -> ``/lab/<run>/<output>/<key>``; ``Classes`` -> one scalar per
  class name. ``Grid`` outputs are not drawn (a body-frame raster would need
  a transform the plan view does not have yet).

Every overlay under ``/color_<role>`` declares the image's coordinate
frame (``rr.CoordinateFrame("<frame_id>_image_plane")``, the name Rerun's
MCAP loader gives the video's frame; :func:`image_frame_id`). Without it
the camera view finds no transform from the overlay's implicit
``tf#/color_near/lab/...`` frame and skips it.

Labels are shown like a run named ``label:<variant>`` (``label:hindsight``).

Rerun is imported lazily so the package imports on the robot without it;
the SDK and the viewer must both be 0.31.3 (``pip install rerun-sdk==0.31.3``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from bebop_vision.lab.types import CameraModel, Classes, Embedding, Mask, Path2D, Scalars

#: Where the rerun_dashboards scripts live (not a package: imported by path).
DASHBOARDS_DIR = Path(__file__).resolve().parents[3] / "rerun_dashboards"
#: The timeline Rerun's MCAP loader uses for message log times (0.31).
TIMELINE = "message_log_time"
PLAN_ORIGIN = "/lab/plan"
#: Fallback chassis rectangle ``(x0, x1, y0, y1)`` in the body frame, metres:
#: the ``ego`` rectangle of config/orbbec_rig.yaml as measured 2026-10-19
#: (used when the rig config cannot be read; see :func:`footprint_rect`).
FOOTPRINT_XY = (-0.10, 0.29, -0.27, 0.17)
#: Bytes of SegmentationImage buffer per ``send_columns`` call. Full-res
#: canvases are T x H x W bytes: at 1280x800 one minute of 20 Hz ticks is
#: 1.2 GB. Measured 2026-10-24 on 300 such ticks (307 MB of canvas): one
#: send peaks at +1052 MB RSS (the numpy -> Arrow -> SDK copies), 64 MB
#: chunks still at +524 MB, 8 MB chunks with a flush after each at +69 MB
#: and the same for 600 ticks, so this is a bound, not a per-session total.
MASK_CHUNK_BYTES = 8 << 20
#: Plan view extent as (u_range, v_range) in plan coordinates; v grows
#: downwards so forward is negative v. A 3 s horizon at 1 m/s fits.
PLAN_BOUNDS = ((-2.0, 2.0), (-3.5, 0.75))
#: Colour cameras a path is projected into (floor plane, z = 0).
IMAGE_ROLES = ("near", "far")
INSTALL_HINT = "pip install rerun-sdk==0.31.3"
LABEL_COLOR = (40, 200, 90)
#: Run colours in order of appearance (labels take LABEL_COLOR).
PALETTE = ((0, 120, 255), (255, 140, 0), (230, 60, 60), (150, 90, 255),
           (0, 190, 190), (255, 80, 200), (170, 170, 50), (120, 120, 120))
#: Output names that mean "the run's <kind>", so the leaf is the run name alone.
_KIND_NAMES = {"path", "mask", "embedding", "scalars", "classes", "grid"}


# --------------------------------------------------------------------------
# Lazy imports
# --------------------------------------------------------------------------

def _rerun():
    """``(rerun, rerun.blueprint)`` or an ImportError with the install hint."""
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(f"rerun_log needs the Rerun SDK matching the viewer: {INSTALL_HINT}") from e
    return rr, rrb


_DASH_CACHE: Dict[str, Any] = {}


def _dashboard_module(name: str):
    """Import ``rerun_dashboards/<name>.py`` by path (it is not a package).

    ``navd_blueprint`` imports rerun at its top, so this is as lazy as rerun.
    Cached per process; a module already in ``sys.modules`` (e.g. tests that
    inserted the directory into ``sys.path``) is reused rather than loaded
    twice, which would duplicate the view / container classes.
    """
    if name in _DASH_CACHE:
        return _DASH_CACHE[name]
    mod = sys.modules.get(name)
    if mod is None:
        path = DASHBOARDS_DIR / f"{name}.py"
        if not path.is_file():
            raise FileNotFoundError(f"rerun_dashboards/{name}.py not found at {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    _DASH_CACHE[name] = mod
    return mod


# --------------------------------------------------------------------------
# Pure helpers (no rerun needed; unit-tested directly)
# --------------------------------------------------------------------------

def plan_xy(pts: np.ndarray) -> np.ndarray:
    """Body ``(x forward, y left)`` -> plan view ``(u, v) = (-y, -x)`` so that
    forward is up and left is left on screen."""
    p = np.asarray(pts, dtype=np.float32)
    out = np.empty_like(p)
    out[..., 0] = -p[..., 1]
    out[..., 1] = -p[..., 0]
    return out


def footprint_rect(rig: Optional[dict] = None, robot: str = "default") -> Tuple[float, float, float, float]:
    """The chassis rectangle ``(x0, x1, y0, y1)`` in the body frame, metres.

    From ``robots.<robot>.bev_student.ego.{x_range_m,y_range_m}`` of the rig
    config (``rig`` when given, else ``config/orbbec_rig.yaml`` through
    ``bebop_vision.orbbec.load_rig_config``); :data:`FOOTPRINT_XY` when the
    file or the keys are missing, so the plan view always has a body.
    """
    if rig is None:
        try:
            from bebop_vision.orbbec import load_rig_config
            rig = load_rig_config()
        except Exception:
            return FOOTPRINT_XY
    try:
        ego = rig["robots"][robot]["bev_student"]["ego"]
        x0, x1 = (float(v) for v in ego["x_range_m"])
        y0, y1 = (float(v) for v in ego["y_range_m"])
    except (KeyError, TypeError, ValueError):
        return FOOTPRINT_XY
    return (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))


def footprint_strip(rect: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """Closed box ``[5, 2]`` in plan coordinates of the chassis rectangle
    (:func:`footprint_rect` when ``rect`` is None)."""
    x0, x1, y0, y1 = footprint_rect() if rect is None else rect
    body = np.array([[x1, y1], [x1, y0], [x0, y0], [x0, y1], [x1, y1]], np.float32)
    return plan_xy(body)


def project_floor(camera: CameraModel, pts_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Body-frame floor points ``[N, 2]`` -> pixel ``(uv [M, 2], keep [N] bool)``.

    Points behind the camera (``z_cam <= 0``) and outside the image are
    dropped; ``keep`` says which inputs survived so a strip can be cut where
    it leaves the frame instead of drawing a line across the image.
    """
    xy = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    pts = np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)
    u, v, z = camera.project(pts)
    K = camera.intrinsics
    with np.errstate(invalid="ignore"):
        keep = (z > 0) & np.isfinite(u) & np.isfinite(v) \
            & (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    uv = np.stack([u[keep], v[keep]], axis=1).astype(np.float32)
    return uv, keep


def _runs_of_true(keep: np.ndarray) -> List[Tuple[int, int]]:
    """``[(start, stop), ...]`` of the consecutive True runs in ``keep``."""
    out: List[Tuple[int, int]] = []
    start = None
    for i, k in enumerate(list(keep) + [False]):
        if k and start is None:
            start = i
        elif not k and start is not None:
            out.append((start, i))
            start = None
    return out


def mask_canvas(mask: Mask, image_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """A ``Mask`` as a full-resolution ``uint8`` class-id image for Rerun.

    Class ids are shifted by one (``values + 1``) because Rerun renders id 0
    transparent; 0 therefore marks pixels the mask does not cover (outside
    ``crop``, or the remainder when ``stride`` does not divide the crop).
    Float ``[C, H, W]`` values are reduced with argmax first. ``image_hw``
    defaults to the mask's own extent.
    """
    vals = np.asarray(mask.values)
    if vals.ndim == 3:
        vals = np.argmax(vals, axis=0)
    if vals.ndim != 2:
        raise ValueError(f"Mask.values must be [H, W] or [C, H, W], got {vals.shape}")
    ids = (vals.astype(np.int64) + 1).clip(0, 255).astype(np.uint8)
    s = max(int(mask.stride), 1)
    if s > 1:
        ids = np.repeat(np.repeat(ids, s, axis=0), s, axis=1)
    if mask.crop is not None:
        r0, r1, c0, c1 = (int(c) for c in mask.crop)
    else:
        r0, c0 = 0, 0
        r1, c1 = ids.shape
    if image_hw is None:
        image_hw = (r1, c1)
    H, W = int(image_hw[0]), int(image_hw[1])
    canvas = np.zeros((H, W), np.uint8)
    h = min(ids.shape[0], r1 - r0, H - r0)
    w = min(ids.shape[1], c1 - c0, W - c0)
    if h > 0 and w > 0:
        canvas[r0:r0 + h, c0:c0 + w] = ids[:h, :w]
    return canvas


def _mask_extent(mask: Mask) -> Tuple[int, int]:
    """``(rows, cols)`` of the recorded image a mask covers: its ``crop``
    bounds, else its upsampled size."""
    if mask.crop is not None:
        return int(mask.crop[1]), int(mask.crop[3])
    vals = np.asarray(mask.values)
    h, w = vals.shape[-2:]
    s = max(int(mask.stride), 1)
    return int(h) * s, int(w) * s


def _masks_extent(masks: Sequence[Optional[Mask]]) -> Tuple[int, int]:
    """The smallest canvas that holds every mask in ``masks`` (Nones skipped)."""
    ext = [_mask_extent(m) for m in masks if m is not None]
    return (max(e[0] for e in ext), max(e[1] for e in ext)) if ext else (1, 1)


def _mask_class_count(masks: Sequence[Optional[Mask]]) -> int:
    """Class count for unnamed masks: the channel count of ``[C, H, W]``
    values, else one past the largest id seen over the run (so a class
    absent from the first tick still gets an entry)."""
    n = 0
    for m in masks:
        if m is None:
            continue
        vals = np.asarray(m.values)
        if vals.ndim == 3:
            return int(vals.shape[0])
        if vals.size:
            n = max(n, int(np.max(vals)) + 1)
    return n


def pca3(vectors: np.ndarray) -> np.ndarray:
    """``[T, D]`` -> ``[T, 3]`` principal-component scores (centred, SVD).

    Fewer than three usable components (``T`` or ``D`` small) are padded with
    zeros so the three ``pc`` series always exist.
    """
    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        return np.zeros((X.shape[0] if X.ndim == 2 else 0, 3), np.float64)
    X = np.nan_to_num(X - X.mean(axis=0, keepdims=True))
    k = min(3, *X.shape)
    out = np.zeros((X.shape[0], 3), np.float64)
    if k == 0:
        return out
    # Full SVD of [T, D] is fine at the lab's sizes (thousands x <=1024).
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    out[:, :k] = U[:, :k] * S[:k]
    return out


# --------------------------------------------------------------------------
# Runs and labels seen the same way
# --------------------------------------------------------------------------

@dataclass
class RunView:
    """What the logger needs from a ``Run`` (or a ``LabelSet`` dressed as
    one): a name, tick stamps and per-output lists aligned with ``t_ns``."""

    name: str
    t_ns: np.ndarray
    outputs: Dict[str, list] = field(default_factory=dict)

    def leaf(self, output: str) -> str:
        return self.name if output in _KIND_NAMES else f"{self.name}.{output}"

    def kinds(self) -> Dict[str, type]:
        """output name -> output class, from the first non-None value."""
        out = {}
        for name, values in self.outputs.items():
            first = next((v for v in values if v is not None), None)
            if first is not None:
                out[name] = type(first)
        return out


def run_view(run: Any) -> RunView:
    meta = getattr(run, "meta", None) or {}
    name = str(meta.get("name") or getattr(run, "name", None) or "run")
    return RunView(name=name, t_ns=np.asarray(run.t_ns, dtype=np.int64),
                   outputs={k: list(v) for k, v in dict(run.outputs).items()})


def labels_as_run(labels: Any) -> RunView:
    """A ``LabelSet`` as the run ``label:<variant>`` with one ``path`` output
    (None where the label is invalid)."""
    variant = str(getattr(labels, "variant", None) or "hindsight")
    t_ns = np.asarray(labels.t_ns, dtype=np.int64)
    paths = []
    for t in t_ns:
        d = labels.get(int(t))
        ok = d is not None and bool(d.get("valid", True)) and d.get("path") is not None
        paths.append(d["path"] if ok else None)
    return RunView(name=f"label:{variant}", t_ns=t_ns, outputs={"path": paths})


def _unique_names(views: Sequence[RunView]) -> None:
    """Two runs of one model would overwrite each other's entities."""
    seen: Dict[str, int] = {}
    for v in views:
        n = seen.get(v.name, 0)
        seen[v.name] = n + 1
        if n:
            v.name = f"{v.name}#{n + 1}"


# --------------------------------------------------------------------------
# Image frames
# --------------------------------------------------------------------------

def image_frame_id(session, role: str) -> str:
    """The ``frame_id`` of the session's first ``/color_<role>`` message,
    else the recorder's convention ``<role>_color``."""
    default = f"{role}_color"
    path = getattr(session, "path", None)
    if path is None:
        return default
    try:
        from bebop_vision.mcap_tools import ProtoDecoder, iter_messages
        dec = ProtoDecoder()
        for schema, _, msg in iter_messages(path, [f"/color_{role}"]):
            fid = getattr(dec.decode(schema, msg.data), "frame_id", "")
            return fid or default
    except Exception:  # an unreadable file keeps the convention
        return default
    return default


# --------------------------------------------------------------------------
# Blueprint
# --------------------------------------------------------------------------

def build_blueprint(series_runs: Sequence[str] = (), active_tab: str = "lab"):
    """The navd dashboard plus a ``lab`` tab: the plan view and one
    time-series view per run in ``series_runs`` (``/lab/<run>``).

    ``navd_blueprint.build()`` is composed, not copied: its left column and
    the existing tabs are reused as built, so dashboard edits there show up
    here. ``active_tab`` is ``"lab"`` or one of ``navd_blueprint.TABS``.
    """
    rr, rrb = _rerun()
    navd = _dashboard_module("navd_blueprint")
    base = navd.build()
    left, tabs = base.root_container.contents
    plan = rrb.Spatial2DView(
        origin=PLAN_ORIGIN, name="plan (body frame)",
        visual_bounds=rrb.VisualBounds2D(x_range=PLAN_BOUNDS[0], y_range=PLAN_BOUNDS[1]))
    series = [rrb.TimeSeriesView(origin=f"/lab/{run}", name=run) for run in series_runs]
    # The camera views share the image entities with the left column's
    # tiles, so the projected paths and masks draw over the recorded video.
    cams = rrb.Vertical(
        *[rrb.Spatial2DView(origin=f"/color_{role}", name=f"{role} camera + paths")
          for role in IMAGE_ROLES],
        name="cameras")
    right = rrb.Vertical(plan, *series, row_shares=[2.0] + [1.0] * len(series)) if series else plan
    lab = rrb.Horizontal(cams, right, column_shares=[1.3, 1.0], name="lab")
    names = tuple(navd.TABS) + ("lab",)
    index = names.index(active_tab) if active_tab in names else len(names) - 1
    new_tabs = rrb.Tabs(*tabs.contents, lab, active_tab=index)
    panels = [p for p in (base.blueprint_panel, base.selection_panel, base.time_panel) if p is not None]
    return rrb.Blueprint(
        rrb.Horizontal(left, new_tabs, column_shares=base.root_container.column_shares),
        *panels,
    )


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

class _Logger:
    """Logs RunViews into a RecordingStream, one ``send_columns`` per entity.

    Columnar sends carry exactly the index given (:data:`TIMELINE`); row by
    row ``log`` would also stamp every row with the SDK's wall-clock
    ``log_time`` / ``log_tick``, which the viewer may then pick as its
    default timeline, where the lab's rows sit at "now" and the MCAP's are
    absent. Holds the per-session caches (camera model, image sizes) shared
    by every run.
    """

    def __init__(self, rec, rr, session) -> None:
        self.rec = rec
        self.rr = rr
        self.session = session
        self._image_hw: Dict[str, Optional[Tuple[int, int]]] = {}
        self._camera: Dict[str, Optional[CameraModel]] = {}
        self._frames_set: set = set()

    def image_plane(self, role: str) -> str:
        """The coordinate frame Rerun's MCAP loader puts ``/color_<role>``
        in: ``<frame_id>_image_plane``, ``frame_id`` being the Foxglove
        message's (``near_color`` / ``far_color`` from the recorder). An
        overlay must declare it, or the camera view reports "no transform
        path" and draws nothing (seen in 0.31.3)."""
        return f"{image_frame_id(self.session, role)}_image_plane"

    def attach_image_frame(self, entity: str, role: str) -> None:
        """Put ``entity`` (an overlay under ``/color_<role>``) in that
        camera's image-plane frame, once per entity."""
        if entity in self._frames_set:
            return
        self.rec.log(entity, self.rr.CoordinateFrame(self.image_plane(role)), static=True)
        self._frames_set.add(entity)

    # -- session facts -----------------------------------------------------
    def camera(self, role: str) -> Optional[CameraModel]:
        """The colour camera model, or None when the intrinsics are unknown."""
        if role not in self._camera:
            try:
                self._camera[role] = self.session.calib.camera(role, "color")
            except (KeyError, AttributeError, TypeError, ValueError):
                self._camera[role] = None
        return self._camera[role]

    def image_hw(self, role: str, t_ns: Sequence[int] = ()) -> Optional[Tuple[int, int]]:
        """Recorded colour size for ``role``: from the colour intrinsics when
        known, else from one decoded frame; None if neither is available.

        The frame probe tries the series' own first stamp (which has a frame
        by construction) before the run's ticks in ``t_ns``: the far camera
        drops colour frames under load, so a single tick can miss.
        """
        if role in self._image_hw:
            return self._image_hw[role]
        hw: Optional[Tuple[int, int]] = None
        K = getattr(getattr(self.session, "calib", None), "color", {}).get(role)
        if K is not None:
            hw = (int(K.height), int(K.width))
        else:
            try:
                frames = self.session.frames(role)
                own = np.asarray(getattr(frames, "t_ns", ()), dtype=np.int64).ravel()[:1]
                probes = [int(t) for t in own] + [int(t) for t in list(t_ns)[:8]]
            except Exception:
                frames, probes = None, []
            for t in probes:
                try:
                    frame = frames.at(t)
                except Exception:
                    frame = None
                if frame is not None:
                    hw = (int(frame.shape[0]), int(frame.shape[1]))
                    break
        self._image_hw[role] = hw
        return hw

    # -- helpers -----------------------------------------------------------
    def _index(self, t_ns: np.ndarray) -> list:
        times = np.asarray(t_ns, dtype=np.int64).astype("datetime64[ns]")
        return [self.rr.TimeColumn(TIMELINE, timestamp=times)]

    def _send_scalars(self, path: str, t_ns: np.ndarray, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(values)
        if ok.any():
            self.rec.send_columns(path, indexes=self._index(t_ns[ok]),
                                  columns=self.rr.Scalars.columns(scalars=values[ok]))

    def _send_strips(self, path: str, t_ns: np.ndarray, strips: list, colors: list,
                     radii: list, lengths: list) -> None:
        """Strips partitioned per tick; a length of 0 clears the entity at
        that tick instead of leaving the previous path on screen."""
        if not strips:
            return
        cols = self.rr.LineStrips2D.columns(strips=strips, colors=colors, radii=radii)
        self.rec.send_columns(path, indexes=self._index(t_ns),
                              columns=[c.partition(lengths) for c in cols])

    # -- static -----------------------------------------------------------
    def log_static(self, views: Sequence[RunView]) -> None:
        rr = self.rr
        self.rec.log(f"{PLAN_ORIGIN}/footprint",
                     rr.LineStrips2D([footprint_strip()], colors=[(200, 200, 200)],
                                     radii=[rr.components.Radius.ui_points(1.5)]),
                     static=True)
        for view in views:
            for output, kind in view.kinds().items():
                if kind is Mask:
                    first = next(v for v in view.outputs[output] if v is not None)
                    names = tuple(first.classes) or tuple(
                        f"class {i}" for i in range(_mask_class_count(view.outputs[output])))
                    # ids shifted by one: 0 is the transparent, uncovered area
                    ctx = [(i + 1, n) for i, n in enumerate(names)]
                    path = f"/color_{first.camera}/lab/{view.leaf(output)}/mask"
                    self.rec.log(path, rr.AnnotationContext(ctx), static=True)
                    self.rec.log(path, rr.SegmentationImage.from_fields(opacity=0.5), static=True)

    # -- per run ----------------------------------------------------------
    def log_run(self, view: RunView, color: Tuple[int, int, int]) -> None:
        t_ns = np.asarray(view.t_ns, dtype=np.int64)
        for output, kind in view.kinds().items():
            values = list(view.outputs[output])[:len(t_ns)]
            values += [None] * (len(t_ns) - len(values))
            if kind is Path2D:
                self._log_paths(view.leaf(output), t_ns, values, color)
            elif kind is Mask:
                self._log_masks(view.leaf(output), t_ns, values)
            elif kind is Embedding:
                self._log_embeddings(f"/lab/{view.name}/{output}", t_ns, values)
            elif kind is Scalars:
                keys = sorted({k for v in values if v is not None for k in v.values})
                for key in keys:
                    col = np.array([v.values.get(key, np.nan) if v is not None else np.nan
                                    for v in values], np.float64)
                    self._send_scalars(f"/lab/{view.name}/{output}/{key}", t_ns, col)
            elif kind is Classes:
                first = next(v for v in values if v is not None)
                names = tuple(first.names) or tuple(f"class {j}" for j in range(len(first.logits)))
                for j, name in enumerate(names):
                    col = np.array([float(np.asarray(v.logits).ravel()[j]) if v is not None else np.nan
                                    for v in values])
                    self._send_scalars(f"/lab/{view.name}/{output}/{name}", t_ns, col)
            # Grid: not drawn (see the module docstring)

    def _log_embeddings(self, base: str, t_ns: np.ndarray, values: list) -> None:
        vecs = [np.asarray(e.vector, np.float64).ravel() if e is not None else None for e in values]
        D = next(len(v) for v in vecs if v is not None)
        X = np.stack([v if v is not None and len(v) == D else np.full(D, np.nan) for v in vecs])
        valid = ~np.isnan(X).any(axis=1)
        scores = np.full((len(X), 3), np.nan)
        scores[valid] = pca3(X[valid])
        for k in range(3):
            self._send_scalars(f"{base}/pc{k}", t_ns, scores[:, k])

    def _log_masks(self, leaf: str, t_ns: np.ndarray, values: list) -> None:
        """Full-resolution class-id canvases, sent in chunks of about
        :data:`MASK_CHUNK_BYTES` so a long session never materialises in RAM
        at once (several ``send_columns`` on one entity simply add chunks)."""
        rr = self.rr
        first = next(v for v in values if v is not None)
        # every tick shares one canvas size: the recorded image when known,
        # else the union of the masks' own extents (crops can differ per tick)
        hw = self.image_hw(first.camera, t_ns[[i for i, v in enumerate(values) if v is not None]])
        if hw is None:
            hw = _masks_extent(values)
        H, W = hw
        fmt = rr.components.ImageFormat(width=W, height=H, channel_datatype="U8")
        path = f"/color_{first.camera}/lab/{leaf}/mask"
        self.attach_image_frame(path, first.camera)
        rows = max(1, MASK_CHUNK_BYTES // max(H * W, 1))
        for a in range(0, len(values), rows):
            chunk = values[a:a + rows]
            buf = np.zeros((len(chunk), H * W), np.uint8)
            for i, v in enumerate(chunk):
                # a None leaves an all-zero canvas: it clears the overlay at
                # ticks without a mask instead of holding the previous one
                if v is not None:
                    buf[i] = mask_canvas(v, hw).ravel()
            self.rec.send_columns(path, indexes=self._index(t_ns[a:a + len(chunk)]),
                                  columns=rr.SegmentationImage.columns(buffer=buf, format=[fmt] * len(chunk)))
            del buf
            # without this the SDK batcher keeps every chunk until the final
            # flush and the peak grows with the session again (measured 4x)
            self.rec.flush()

    def _log_paths(self, leaf: str, t_ns: np.ndarray, values: list,
                   color: Tuple[int, int, int]) -> None:
        plan_strips, plan_colors, plan_radii, plan_len = [], [], [], []
        cams = {role: self.camera(role) for role in IMAGE_ROLES}
        img = {role: ([], [], [], []) for role, cam in cams.items() if cam is not None}
        for path in values:
            n_plan = 0
            n_img = {role: 0 for role in img}
            if path is not None:
                pts = np.asarray(path.pts, np.float32)      # [K, N, 2]
                if pts.ndim == 2:
                    pts = pts[None]
                K = pts.shape[0]
                best = int(np.argmax(path.weights)) if path.weights is not None else 0
                for k in [best] + [j for j in range(K) if j != best]:
                    # the best hypothesis is opaque and thicker; the others fade
                    c = (*color, 255) if k == best else (*color, 110)
                    r = -2.5 if k == best else -1.2           # negative radius = UI points
                    plan_strips.append(plan_xy(pts[k]))
                    plan_colors.append(c)
                    plan_radii.append(r)
                    n_plan += 1
                    for role, (strips, colors, radii, _) in img.items():
                        uv, keep = project_floor(cams[role], pts[k])
                        offsets = np.cumsum(keep) - 1
                        # one strip per in-image stretch, so a path leaving the
                        # frame is cut there rather than bridged across it
                        for a, b in _runs_of_true(keep):
                            if b - a >= 2:
                                strips.append(uv[offsets[a]:offsets[b - 1] + 1])
                                colors.append(c)
                                radii.append(r)
                                n_img[role] += 1
            plan_len.append(n_plan)
            for role in img:
                img[role][3].append(n_img[role])
        self._send_strips(f"{PLAN_ORIGIN}/{leaf}", t_ns, plan_strips, plan_colors, plan_radii, plan_len)
        for role, (strips, colors, radii, lens) in img.items():
            if strips:
                self.attach_image_frame(f"/color_{role}/lab/{leaf}", role)
            self._send_strips(f"/color_{role}/lab/{leaf}", t_ns, strips, colors, radii, lens)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def show(session, runs: Sequence[Any] = (), labels: Any = None, spawn: bool = True,
         save: Optional[str | Path] = None, blueprint: Any = "navd",
         extra_views: bool = True) -> None:
    """Open ``session`` in the Rerun viewer with ``runs`` and ``labels`` overlaid.

    ``save=path.rrd`` writes a self-contained recording (data + dashboard)
    instead of spawning the viewer; ``spawn=False`` without ``save`` logs
    into a stream nobody reads (useful for a dry run). ``blueprint`` is
    ``"navd"`` (the dashboard, extended with the lab tab unless
    ``extra_views=False``), ``None`` (send none) or a ready ``rrb.Blueprint``.
    """
    rr, rrb = _rerun()
    navd = _dashboard_module("navd_blueprint")
    mcap_prep = _dashboard_module("mcap_prep")

    views = [run_view(r) for r in runs]
    if labels is not None:
        views.append(labels_as_run(labels))
    _unique_names(views)
    series_runs = [v.name for v in views
                   if any(k in (Embedding, Scalars, Classes) for k in v.kinds().values())]

    if isinstance(blueprint, str):
        if blueprint != "navd":
            raise ValueError(f"unknown blueprint {blueprint!r}; use 'navd', None or an rrb.Blueprint")
        bp = build_blueprint(series_runs) if extra_views else navd.build()
    else:
        bp = blueprint

    rec = rr.RecordingStream(navd.NAME, recording_id=str(uuid.uuid4()))
    if save is not None:
        rec.save(str(save))
    elif spawn:
        rec.spawn(default_blueprint=bp)
    if bp is not None:
        # The viewer keeps a per-app-id blueprint and prefers it over a
        # spawn-time default, so send it as the active one (as open.py does).
        rec.send_blueprint(bp, make_active=True, make_default=True)

    fd, tmp = tempfile.mkstemp(suffix=".mcap")
    os.close(fd)
    try:
        try:
            load_path, repaired = mcap_prep.prepare(str(session.path), tmp)
        except BaseException:
            # an unreadable capture must not leave an empty .mcap per retry
            _remove(tmp)
            raise
        if repaired is None:
            _remove(tmp)
        try:
            rec.log_file_from_path(load_path)
            logger = _Logger(rec, rr, session)
            logger.log_static(views)
            palette = iter(PALETTE * (len(views) // len(PALETTE) + 1))
            for view in views:
                color = LABEL_COLOR if view.name.startswith("label:") else next(palette)
                logger.log_run(view, color)
        finally:
            _remove(repaired)
    finally:
        # Flush and close whatever was logged even when a run failed half
        # way: a save= .rrd then holds a complete recording of the rest, and
        # a spawned viewer is not left with an orphaned connection. The
        # exception still propagates.
        rec.flush()
        rec.disconnect()


def _remove(path: Optional[str]) -> None:
    if path is not None and os.path.exists(path):
        os.remove(path)
