"""FastAPI backend for the review dashboard (bebop-vision/dashboard/).

Wraps the headless DatasetDashboard core (tools/dashboard_core.py):
sessions, ticks, camera/SAM overlays, and hand-edited label corrections
for the navd dataset.
"""

import json
import math
import sys
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

_MCAP_DIR = None
_TRAIN_ROOT = _REPO_ROOT / "runs"
_WEIGHTS_ROOT = _REPO_ROOT / "weights"


app = FastAPI(title="bebop-vision dashboard", version="0.2.0")
dash = DatasetDashboard(_REPO_ROOT / "datasets" / "navd-v0")


class HandBody(BaseModel):
    grid: list | None = None
    clear: bool = False


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
