"""FastAPI backend for the ML review dashboard (bebop-vision/dashboard/).

Wraps the headless DatasetDashboard core (tools/dashboard_core.py) —
sessions/ticks/overlays/hand-edits — and adds the model-validation
surface:

  GET  /api/sessions
  GET  /api/session/<name>/ticks
  GET  /api/tick/<name>/<stamp>
  GET  /api/overlay/<name>/<stamp>?role&src&alpha
  GET  /api/sam/<name>/<stamp>?role&mode&alpha     raw SAM mask / +depth gate
  POST /api/tick/<name>/<stamp>/hand        {"grid"} | {"clear": true}
  GET  /api/pipeline                        per-session stage artifact status
  GET  /api/pipeline/train                  runs / ONNX / checkpoint artifacts
  GET  /api/model                           loaded model info | null
  POST /api/model/load                      {"path": weights/navd.onnx}
  GET  /api/validate/<name>/<stamp>         one-tick replay vs teacher
  POST /api/validate/<name>/run             background session sweep
  GET  /api/validate/<name>/status          sweep progress + results
  GET  /api/validate/<name>/results         cached results file listing

Validation runs the ONNX with the EXACT runtime preprocessing path
(navd_pre.py via navd_runtime goal-raster construction) so what the
dashboard shows is what the deployed model saw — the whole point is
explaining runtime behavior ("the why"), not a parallel pipeline.

Inference prefers the CUDA execution provider. The CUDA/cuDNN shared
libraries ship inside torch's nvidia-* pip packages; they are pre-loaded
with RTLD_GLOBAL here so onnxruntime-gpu finds them without
LD_LIBRARY_PATH gymnastics in the user's shell. CPU is the fallback.
"""

import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard_core import DatasetDashboard  # noqa: E402
from dashboard_core import DatasetDashboard  # noqa: E402


def _self_disc_mask():
    """(60, 60) bool: cells inside the rig's min_range dead disc —
    inlined from the (excised) navd_runtime; pure rig-geometry numpy."""
    from bebop_vision.orbbec import load_rig_config
    cfg = load_rig_config()["robots"]["default"]["bev"]
    min_r = float(cfg.get("min_range_m", 0.55))
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    x = RANGE_M - (rows + 0.5) * CELL_M
    y = (cols + 0.5) * CELL_M - WIDTH_M / 2.0
    return np.hypot(x, y) < min_r


# --- shared navd-preprocessing math (was bebop_vision/navd_pre.py) --------
# Inlined so the dashboard depends on nothing but this file: the grid
# geometry (60x60 @ 5 cm, 3 m forward x 3 m wide) is the BEV model's
# native output frame, and the color/depth normalization must stay
# byte-identical to what the exported models were trained on.

IMG_H, IMG_W = 240, 424
GRID = 60
RANGE_M, WIDTH_M, CELL_M = 3.0, 3.0, 0.05
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _build_goal_raster(goal, odom):
    """goal dict from the manifest {type: heading|point|none, ...} ->
    (60, 60) float32 fan: 1 along the goal bearing from the robot origin,
    fading with angular distance."""
    g = np.zeros((GRID, GRID), np.float32)
    if goal.get("type", "none") == "heading":
        bearing = float(goal["heading_rad"])
    elif goal.get("type") == "point":
        gx, gy = float(goal["x"]), float(goal["y"])
        ox, oy, oth = float(odom["x"]), float(odom["y"]), float(odom["theta"])
        bearing = math.atan2(gy - oy, gx - ox) - oth
    else:
        return g
    rows, cols = np.mgrid[0:GRID, 0:GRID]
    x = RANGE_M - (rows + 0.5) * CELL_M
    y = (cols + 0.5) * CELL_M - WIDTH_M / 2.0
    ang = np.arctan2(y, np.maximum(x, 1e-6))
    d = np.abs(np.angle(np.exp(1j * (ang - bearing))))
    return np.clip(1.0 - d / (math.pi / 2.0), 0.0, 1.0).astype(np.float32)


def _prep_depth(d_mm):
    """Raw uint16 mm depth -> (meters f32 [1,H,W], validity mask [1,H,W]).

    Invalid (0) pixels stay 0 after clipping; values clip to [0.3, 6.0] m.
    """
    import cv2
    d = cv2.resize(d_mm, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    m = (d > 0).astype(np.float32)
    out = np.clip(d.astype(np.float32) * 1e-3, 0.3, 6.0) * m
    return out[None], m[None]


def _prep_color(rgb):
    """RGB uint8 image -> ImageNet-normalized f32 [3,H,W] (training norm)."""
    import cv2
    c = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    c = (c.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return c.transpose(2, 0, 1)


app = FastAPI(title="bebop-vision dashboard", version="0.1.0")
dash = DatasetDashboard(_REPO_ROOT / "datasets" / "navd-v0")

# --- model state (workstation replay inference) --------------------------

_model = {"sess": None, "path": None, "providers": None, "kind": None}
_TRAJ_INPUTS = {"color", "color_far", "goal_vec", "last_twist"}
_validate_jobs = {}   # session name -> {"running", "done", "total", "error"}
_validate_lock = threading.Lock()

# self dead disc — the runtime (navd_runtime.NavdGridSource) carves these
# cells to FREE after the class mapping; replay must mirror that so what
# the dashboard shows is what the drive node's planner consumes
_SELF_DISC = _self_disc_mask()

# pipeline artifact roots (step 1 MCAPs + step 6 training outputs);
# overridable in tests — None means "sibling `sessions/` of the data dir"
_MCAP_DIR = None
_TRAIN_ROOT = _REPO_ROOT / "runs"
_WEIGHTS_ROOT = _REPO_ROOT / "weights"


def _get_sess():
    if _model["sess"] is None:
        raise HTTPException(400, "no model loaded (POST /api/model/load)")
    return _model["sess"]


def _preload_cuda_libs():
    """RTLD_GLOBAL-load torch's bundled CUDA/cuDNN libs so onnxruntime's
    CUDA EP can dlopen them (site-packages/nvidia/*/lib). Idempotent,
    best-effort — failures are fine (system libs may already be loaded)."""
    import ctypes
    import glob
    import site
    libs = []
    for sp in site.getsitepackages() + [str(Path(sys.prefix) / "lib")]:
        libs += glob.glob(f"{sp}/nvidia/*/lib/*.so*")
    for so in sorted(set(libs)):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


def _make_session(path: str):
    """(InferenceSession, providers) with CUDA preference + CPU fallback."""
    import onnxruntime as ort
    want = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()]
    _preload_cuda_libs()
    try:
        sess = ort.InferenceSession(str(path), providers=want)
        return sess, sess.get_providers()
    except Exception:
        if want == ["CPUExecutionProvider"]:
            raise
        # CUDA libs present but unusable (driver/distro mismatch) — CPU it is
        sess = ort.InferenceSession(str(path),
                                    providers=["CPUExecutionProvider"])
        return sess, ["CPUExecutionProvider"]


def _warmup(sess):
    """One zero-input pass: the first CUDA run pays ~20-30 s of cuDNN/arena
    init — pay it at load time, not in the reviewer's first click."""
    import numpy as np
    feed = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        feed[i.name] = np.zeros(shape, np.float32)
    t0 = time.monotonic()
    sess.run(None, feed)
    return round((time.monotonic() - t0) * 1e3, 1)


def _prep_frames(session: str, stamp: int):
    """Raw tick files -> (depth_near, depth_far, color_rgb, goal_raster).

    Uses navd_pre.py (shared with training AND runtime) so preprocessing
    cannot drift from what the robot runs. Mirrors NavdGridSource's
    MJPEG decode for color (extracted sessions store plain JPEGs).
    """
    import cv2

    d = dash._session_dir(session)
    s20 = f"{int(stamp):020d}"
    dep = d / "depth" / f"{s20}.npz"
    jpg = d / "color" / f"{s20}.jpg"
    lab = d / "labels" / f"{s20}.npz"
    if not dep.exists():
        raise HTTPException(404, f"no depth npz for stamp {stamp}")
    if not jpg.exists():
        raise HTTPException(404, f"no near color jpg for stamp {stamp}")
    with np.load(dep) as z:
        if "near" not in z.files or "far" not in z.files:
            raise HTTPException(404, "depth npz missing near/far arrays")
        dn, _ = _prep_depth(z["near"])
        df, _ = _prep_depth(z["far"])
    c = _prep_color(cv2.cvtColor(cv2.imread(str(jpg)), cv2.COLOR_BGR2RGB))
    row = dash._manifest_row(session, stamp)
    if row is None:
        raise HTTPException(404, f"no manifest row for stamp {stamp}")
    raster = _build_goal_raster(row["goal"], row["odom"])
    label = None
    if lab.exists():
        with np.load(lab) as z:
            # same preference the training loader uses (hand > fused)
            key = "hand" if "hand" in z.files else "fused"
            if key in z.files:
                label = z[key].astype(np.uint8)
    return dn, df, c, raster, label


def _predict(dn, df, c, raster):
    """(argmax grid, per-class probability grids, latency_ms).

    Raises HTTPException with the same failure vocabulary the drive node
    sees at runtime (navd_runtime.NavdGridSource.last_reason). The
    frac_navigable gate is NOT enforced here — a rejected tick is exactly
    the case the dashboard must visualize, so the route returns the grid
    plus the verdict the drive node would have applied.
    """
    sess = _get_sess()
    feed = {"depth_near": dn[None].astype(np.float32),
            "depth_far": df[None].astype(np.float32),
            "color": c[None].astype(np.float32),
            "goal": raster[None, None].astype(np.float32)}
    t0 = time.monotonic()
    logits = sess.run(None, feed)[0][0]
    latency_ms = (time.monotonic() - t0) * 1e3
    if not np.isfinite(logits).all():
        raise HTTPException(422, "non-finite logits")
    mx = logits.max(axis=0, keepdims=True)
    p = np.exp(logits - mx)
    p /= p.sum(axis=0, keepdims=True)
    cls = logits.argmax(axis=0).astype(np.uint8)
    frac = float((cls == 1).mean())       # NAVIGABLE (raw — gate input)
    # mirror the runtime's self dead-disc carve (navd_runtime.py):
    # what the planner consumes never contains the robot itself
    cls[_SELF_DISC] = 1                  # NAVIGABLE
    return cls, p, latency_ms, frac


def _gate_verdict(frac):
    """The runtime's frac_navigable plausibility gate (navd_runtime.py):
    outside [0.05, 0.95] the drive node holds. Returns (rejected, reason)."""
    if not (0.05 <= frac <= 0.95):
        return True, (f"frac_navigable {frac:.2f} outside [0.05, 0.95] "
                      "(drive node would hold)")
    return False, None


# --- original dashboard routes (thin shim over DatasetDashboard) ---------

class HandBody(BaseModel):
    grid: list | None = None
    clear: bool = False


class ModelLoadBody(BaseModel):
    path: str = "weights/navd.onnx"


@app.get("/api/sessions")
def sessions():
    return {"sessions": dash.list_sessions()}


@app.get("/api/session/{name}/ticks")
def ticks(name: str):
    try:
        return dash.list_ticks(name)
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")


@app.get("/api/tick/{name}/{stamp}")
def tick(name: str, stamp: int, grids: bool = True):
    """Full tick payload. `?grids=false` skips the 9 label grids (the
    player only needs the media while it advances at 2 ticks/s)."""
    try:
        out = dash.tick_payload(name, stamp)
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")
    if not grids:
        out["grids"] = {}
    return out


@app.get("/api/gridtex/{name}/{stamp}")
def grid_tex(name: str, stamp: int):
    """The camera image projected ONTO the 60x60 grid (inverse of the
    label overlay): every BEV cell is painted with the mean color of the
    stride-4 color pixels whose flat-floor ray lands in it, near camera
    first, far camera filling cells the near camera does not cover.

    Caveat inherited from the flat-floor model: a pixel hitting a TALL
    obstacle lands (on the ground plane) BEHIND the obstacle, so tall
    objects smear toward the far edge of the grid. Floor cells — the ones
    being painted — are exact.
    """
    try:
        out = dash.build_texture(name, int(stamp))
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


@app.get("/api/overlay/{name}/{stamp}")
def overlay(name: str, stamp: int, role: str = "near", src: str = "auto",
            alpha: float = 0.5):
    out = dash.overlay_payload(name, stamp, role=role, src=src, alpha=alpha)
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


@app.get("/api/sam/{name}/{stamp}")
def sam_overlay(name: str, stamp: int, role: str = "near",
                mode: str = "sam", alpha: float = 0.5):
    """Step-3 artifact view: the raw SAM floor mask over the camera
    image (`mode=sam`, green) or the SAM+depth per-pixel gate
    (`mode=gate`) that reproduces fuse_navd_labels.py's decision —
    green = floor confirmed by depth (navigable), red = blocked
    evidence (off-ground / no depth), transparent = unconfirmed /
    sky / the robot's own chassis."""
    if role not in ("near", "far"):
        raise HTTPException(400, "role must be near|far")
    if mode not in ("sam", "gate"):
        raise HTTPException(400, "mode must be sam|gate")
    try:
        out = dash.sam_overlay_payload(name, int(stamp), role=role,
                                       mode=mode, alpha=alpha)
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


@app.post("/api/tick/{name}/{stamp}/hand")
def hand(name: str, stamp: int, body: HandBody):
    if body.clear:
        ok, msg = dash.clear_hand(name, stamp)
    elif body.grid is not None:
        ok, msg = dash.save_hand(name, stamp, body.grid)
    else:
        raise HTTPException(400, "need 'grid' or 'clear'")
    if not ok:
        raise HTTPException(400, msg)
    return {"ok": True, "message": msg}


# --- pipeline artifact status (stages 1-7) --------------------------------

@app.get("/api/pipeline")
def pipeline():
    """Per-session artifact status for the data stages: record (raw
    MCAP), extract (ticks + manifest), sam (mask counts + sampled
    coverage per camera), fuse (labels carrying `fused`), review (hand
    edits). Train/export/runtime are global — see /api/pipeline/train."""
    return {"sessions": dash.pipeline_status(mcap_dir=_MCAP_DIR)}


@app.get("/api/pipeline/train")
def pipeline_train():
    """Step-6 artifacts: training runs (parsed from runs/*/ AND
    weights/*/train_log.jsonl — epochs, best val_miou + per-class IoUs),
    exported ONNX files in weights/, and torch checkpoint dirs. The
    export parity-gate result is stdout-only (never persisted), so it
    cannot be reported here."""
    runs = []
    seen = set()
    for root in (_TRAIN_ROOT, _WEIGHTS_ROOT):
        if not root.is_dir():
            continue
        for d in sorted(root.iterdir()):
            log = d / "train_log.jsonl"
            if not log.is_file() or d in seen:
                continue
            seen.add(d)
            if not log.is_file():
                continue
            epochs, best_miou, best_ious = 0, None, None
            try:
                for line in log.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if "val_miou" not in row:
                        continue
                    epochs += 1
                    if best_miou is None or row["val_miou"] > best_miou:
                        best_miou = round(float(row["val_miou"]), 4)
                        best_ious = [round(float(x), 4)
                                     for x in row.get("ious", [])]
            except OSError:
                continue
            runs.append({"name": d.name, "epochs": epochs,
                         "best_val_miou": best_miou,
                         "best_ious": best_ious,
                         "mtime": int(log.stat().st_mtime)})
    onnx, ckpts = [], []
    if _WEIGHTS_ROOT.is_dir():
        for p in sorted(_WEIGHTS_ROOT.glob("*.onnx")):
            st = p.stat()
            onnx.append({"name": p.name,
                         "size_mb": round(st.st_size / 1e6, 1),
                         "mtime": int(st.st_mtime)})
        for d in sorted(_WEIGHTS_ROOT.iterdir()):
            if d.is_dir():
                files = sorted(p.name for p in d.iterdir() if p.is_file())
                if files:
                    ckpts.append({"name": d.name, "files": files})
    return {"runs": runs, "onnx": onnx, "checkpoints": ckpts}


# --- model validation -----------------------------------------------------

@app.get("/api/model")
def model_info():
    if _model["sess"] is None:
        return {"loaded": False}
    return {"loaded": True, "path": _model["path"],
            "providers": _model["providers"], "kind": _model["kind"]}


@app.post("/api/model/load")
def model_load(body: ModelLoadBody):
    """Load an ONNX navd model. Prefers the CUDA EP (with torch's bundled
    CUDA libs pre-loaded) and falls back to CPU. Includes a warmup pass so
    the first replay click is fast. Path resolves repo-root-relative."""
    p = Path(body.path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    if not p.exists():
        raise HTTPException(404, f"no such model file: {p}")
    try:
        sess, providers = _make_session(str(p))
        warmup_ms = _warmup(sess)
    except Exception as exc:  # noqa: BLE001 — surface export/EP problems
        raise HTTPException(400, f"onnx load failed: {exc}")
    _model["sess"] = sess
    _model["path"] = str(p)
    _model["providers"] = providers
    _model["kind"] = ("traj" if {i.name for i in sess.get_inputs()}
                      == _TRAJ_INPUTS else "bev")
    return {"ok": True, "path": str(p), "providers": providers,
            "kind": _model["kind"], "warmup_ms": warmup_ms}


# NOTE: /run, /status, /results must be declared BEFORE /validate/{name}/{
# stamp} — FastAPI matches in declaration order and "status"/"results"
# would otherwise hit the int `stamp` converter and 422.


@app.post("/api/validate/{name}/run")
def validate_run(name: str, body: ModelLoadBody):
    try:
        dash._session_dir(name)
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")
    with _validate_lock:
        job = _validate_jobs.get(name)
        if job and job["running"]:
            raise HTTPException(409, "sweep already running")
        _validate_jobs[name] = {"running": True, "done": 0,
                                "total": None, "error": None,
                                "results": None}
    threading.Thread(target=_run_sweep, args=(name, body.path),
                     daemon=True, name=f"validate-{name}").start()
    return {"ok": True, "started": True}


@app.get("/api/validate/{name}/status")
def validate_status(name: str):
    job = _validate_jobs.get(name)
    if job is None:
        # surface any persisted results from previous runs
        try:
            d = dash._session_dir(name)
        except FileNotFoundError:
            raise HTTPException(404, "unknown session")
        files = sorted(d.glob("validation_*.json"))
        return {"running": False, "results_files":
                [f.name for f in files]}
    out = dict(job)
    try:
        d = dash._session_dir(name)
        files = sorted(d.glob("validation_*.json"))
        out["results_files"] = [f.name for f in files]
    except FileNotFoundError:
        pass
    return out


@app.get("/api/validate/{name}/results")
def validate_results(name: str, file: str):
    try:
        d = dash._session_dir(name)
    except FileNotFoundError:
        raise HTTPException(404, "unknown session")
    if "/" in file or "\\" in file or not file.startswith("validation_") \
            or not file.endswith(".json"):
        raise HTTPException(400, "bad results filename")
    p = d / file
    if not p.exists():
        raise HTTPException(404, "no such results file")
    return json.loads(p.read_text())


@app.get("/api/validate/{name}/{stamp}")
def validate_tick(name: str, stamp: int):
    """One-tick replay: runtime-effective model grid vs effective teacher
    (hand > fused).

    The returned model grid is CARVED (self dead disc navigable) exactly
    like navd_runtime.NavdGridSource — what the planner consumes. The
    frac_navigable gate verdict applies to the model's RAW output, same
    as the runtime. Agreement is computed over scene cells only (the
    disc is excluded — the teacher's disc labels are bookkeeping, not
    scene).
    """
    dn, df, c, raster, label = _prep_frames(name, stamp)
    cls, p, latency_ms, frac = _predict(dn, df, c, raster)
    rejected, reason = _gate_verdict(frac)
    scene = ~_SELF_DISC
    out = {
        "model": cls.tolist(),
        "prob": (p * 255).astype(np.uint8).tolist(),   # [3][60][60]
        "latency_ms": round(latency_ms, 1),
        "frac_navigable": round(frac, 4),
        "gate_rejected": rejected,
        "gate_reason": reason,
        "self_carved_cells": int(_SELF_DISC.sum()),
        "goal_raster": raster.tolist(),
        "label_key": None,
    }
    if label is not None:
        agree = np.zeros((60, 60), np.uint8)
        diff = (cls != label) & scene
        agree[diff] = (cls[diff] + 1).astype(np.uint8)   # 1..3 = model class
        out.update({
            "label_key": "hand" if _label_is_hand(name, stamp) else "fused",
            "label": label.tolist(),
            "agreement": agree.tolist(),
            "agreement_pct": round(float((~diff)[scene].mean()) * 100, 2),
        })
    return out


def _label_is_hand(name, stamp):
    lab = dash._session_dir(name) / "labels" / f"{int(stamp):020d}.npz"
    try:
        with np.load(lab) as z:
            return "hand" in z.files
    except Exception:
        return False



def _run_sweep(name: str, model_path: str):
    """Background: replay every tick, aggregate + persist results."""
    p = Path(model_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    sess, providers = _make_session(str(p))
    _warmup(sess)
    old_sess, old_path, old_prov = \
        _model["sess"], _model["path"], _model["providers"]
    _model["sess"], _model["path"], _model["providers"] = sess, str(p), providers
    job = _validate_jobs.get(name)
    ticks = dash.list_ticks(name)
    scene = ~_SELF_DISC
    confusion = np.zeros((3, 3), np.int64)   # [label, model]
    iou_hits = np.zeros(3, np.int64)
    iou_preds = np.zeros(3, np.int64)
    worst = []
    failures = []      # exceptions / missing data (not scoreable)
    rejected = []      # gate rejections — scored, but the runtime would hold
    done = 0
    try:
        for t in ticks:
            stamp = int(t["stamp"])
            try:
                dn, df, c, raster, label = _prep_frames(name, stamp)
                if label is None:
                    done += 1
                    job["done"] = done
                    continue
                cls, probs, lat, frac = _predict(dn, df, c, raster)
                # scene cells only: the self dead disc is carved by the
                # runtime and carries no scene signal
                cm = cls[scene]
                lm_all = label[scene]
                for l in range(3):
                    lm = lm_all == l
                    confusion[l] += np.bincount(cm[lm], minlength=3)
                for k in range(3):
                    iou_hits[k] += int(((cm == k) & (lm_all == k)).sum())
                    iou_preds[k] += int((cm == k).sum()) + \
                        int((lm_all == k).sum()) - \
                        int(((cm == k) & (lm_all == k)).sum())
                agree = float((cm == lm_all).mean())
                worst.append({"stamp": t["stamp"], "agreement": agree,
                              "frac_navigable": round(frac, 4),
                              "latency_ms": round(lat, 1)})
                is_rej, reason = _gate_verdict(frac)
                if is_rej:
                    rejected.append({"stamp": t["stamp"], "reason": reason})
            except HTTPException as exc:
                failures.append({"stamp": t["stamp"],
                                 "reason": exc.detail})
            except Exception as exc:  # noqa: BLE001 — one bad tick != abort
                failures.append({"stamp": t["stamp"],
                                 "reason": f"{type(exc).__name__}: {exc}"})
            done += 1
            job["done"] = done
        iou = [round(float(h / max(preds, 1)), 4)
               for h, preds in zip(iou_hits, iou_preds)]
        worst.sort(key=lambda w: w["agreement"])
        results = {
            "model": str(p), "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ticks_total": len(ticks), "ticks_scored": len(worst),
            "failures": failures, "gate_rejected": rejected,
            "mean_agreement": round(float(np.mean(
                [w["agreement"] for w in worst])) if worst else 0.0, 4),
            "confusion_label_x_model": confusion.tolist(),
            "per_class_iou": iou,
            "worst": worst[:50],
        }
        out = dash._session_dir(name) / f"validation_{p.stem}.json"
        out.write_text(json.dumps(results, indent=1))
        job.update(done=len(ticks), running=False, results=str(out.name),
                   error=None)
    except Exception as exc:  # noqa: BLE001
        job.update(running=False, error=f"{type(exc).__name__}: {exc}")
    finally:
        _model["sess"], _model["path"], _model["providers"] = \
            old_sess, old_path, old_prov





# --- static frontend (built React bundle) --------------------------------

_dist = _REPO_ROOT / "dashboard" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True),
              name="dashboard")


def main():  # pragma: no cover — manual launch path
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser(description="navd ML review dashboard API")
    ap.add_argument("--data", default=str(_REPO_ROOT / "datasets" / "navd-v0"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()
    dash.root = Path(args.data)
    dash._manifests.clear()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
