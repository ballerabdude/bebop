# bebop-vision ML review dashboard

The place ML engineers go to review the navd training dataset, hand-correct
teacher labels, and understand **why a trained model behaved the way it did**
on a recorded run.

The legacy inline-UI tool (`tools/dataset_dashboard.py`) was removed
2026-09-07. Its headless core lives on as `tools/dashboard_core.py` — the
single source of truth for npz/manifest/audit logic — wrapped by the
FastAPI backend (`tools/dashboard_api.py`).

## Layout

```
dashboard/                   this app (Vite + React 19 + TS + Tailwind 4)
tools/dashboard_api.py       FastAPI backend: dataset routes + model validation
tests/test_dashboard_api.py  HTTP contract tests (synthetic session tree)
```

## Run (dev)

Two processes:

```bash
# 1. API backend (port 8099; 9090-9092 are taken on the robot)
cd bebop-vision
python tools/dashboard_api.py --data datasets/navd-v0 --port 8099

# 2. Frontend dev server with HMR + /api proxy
cd bebop-vision/dashboard
npm install
npm run dev            # http://localhost:5173
```

## Run (production / robot)

Build once; the API server then serves the bundle as static files:

```bash
cd bebop-vision/dashboard
npm run build          # -> dashboard/dist/

cd bebop-vision
python tools/dashboard_api.py --host 0.0.0.0 --port 8099
# open http://<workstation>:8099
```

Install deps:

```bash
pip install -e "bebop-vision[dashboard]"     # fastapi/uvicorn/pydantic
```

For model validation, add onnxruntime — GPU build on NVIDIA workstations
(1.22.x = CUDA 12 build, matching the torch-shipped CUDA libs; the API
pre-loads them and falls back to CPU if unusable):

```bash
pip uninstall onnxruntime && pip install "onnxruntime-gpu==1.22.*"
```

## UI tour

**Sidebar (shared across tabs)** — session list with per-session
hand-review progress bars, a filter box, and the loaded-model status pill.

### Tick review

- **Toolbar**: tick navigation, jump filters (disagree / caution /
  unreviewed), the displayed grid (auto = hand-else-fused / hand / fused),
  mining-channel overlays, the SAM label overlay with alpha slider, and
  the SAM mask opacity slider.
- **SAM segmentation (step-3 artifact view)**: each color card (near +
  far) has an `off / sam / gate` selector.
  - `sam` — the raw SAM 3.1 floor mask from `sam_floor{,_far}/` in green.
  - `gate` — the SAM+depth per-pixel decision fuse_navd_labels.py applies:
    **green** = floor confirmed by the measured depth landing on the
    flat-ground prediction (became navigable), **red** = blocked evidence
    (off-ground surface or no depth at all), **transparent** =
    depth-consistent ground SAM missed (stays unconfirmed), sky, and the
    robot's own chassis. Gate needs the ray LUT + rig config; when they
    are unavailable only raw mode is offered.
  - Opacity is applied client-side (the slider never refetches); the card
    footer shows mask coverage and, in gate mode, navigable/blocked px.
- **Filmstrip**: every tick as a tiny class-balance bar (red/green/amber),
  with green top marks for hand-reviewed ticks and red marks for
  disagree ticks — scan a whole session at a glance and click to jump.
- **Media pane**: near color (with overlay), depth near, far color, depth far.
- **Editor pane**: camera-aligned 60×60 paint canvas with **projected
  image underlay** — the near/far camera image projected onto the grid
  through the ray LUT (each cell shows the mean color of the pixels whose
  flat-floor ray lands there; near camera wins, far fills the rest) — with
  a paint-opacity slider, brush-footprint preview, hover cell readout;
  class picker, brush size, undo / revert / save / clear-hand; stacked
  class-balance bars for fused vs hand; tick context (cmd_vel, odom, goal).
- Saving writes `hand` into `labels/<stamp>.npz` — additive, teacher keys
  never touched, every edit audited to `hand_edits.jsonl`.
- Full keyboard model: `←/→` nav · `1/2/3` classes · `[ ]` brush · `u` undo
  · `s` save · `r` revert · `t` grid source · `o` overlay · `d/c/n/f` jumps
  · `?` shortcuts modal.

### Pipeline (the artifact browser)

One live row per session across the seven pipeline stages: **record**
(raw MCAP in `datasets/sessions/`, size), **extract** (tick count +
manifest), **SAM** (mask counts + sampled mean coverage per camera),
**fuse** (`fused` labels per tick), **review** (hand edits) — click the
session name to open it in tick review. Below: the global **train +
export** artifacts (every `train_log.jsonl` under `runs/` and `weights/`
with epochs + best val_miou, the exported ONNX files with sizes, and the
torch checkpoint dirs) and a jump into model validation. Refresh rescans.

### Model validation (the "why" tab)

- Load any exported ONNX (default `weights/navd.onnx`). Inference runs on
  the CUDA execution provider when available (torch's bundled CUDA libs
  are pre-loaded; CPU is the automatic fallback), with a warmup pass at
  load so the first replay is instant.
- **Replay player**: press play and the session plays back tick-by-tick —
  every shown tick is run through the model with the EXACT runtime
  preprocessing path (`bebop_vision/navd_pre.py`) and scored against the
  effective teacher, **next to the camera views** (near/far color + depth,
  fetched light without the label grids). Transport: play/pause (space),
  skip (←/→), 0.5×/1×/2× speed, filmstrip scrubbing, worst-tick rows jump
  into the player. Then inspect:
  - model grid vs teacher/hand label
  - disagreement map colored by the model's class
  - per-class probability heatmaps (confidence, not just argmax)
  - the goal raster the model received
  - the `frac_navigable` gate verdict — rejected ticks still render, with
    the reason the drive node would hold (a rejection is exactly the
    behavior you're debugging).
- **Session sweep**: batch over every tick → mean agreement, confusion
  matrix, per-class IoU, runtime-rejected list, worst-ticks table (click a
  row to replay that tick). Persisted to `<session>/validation_<model>.json`
  so re-runs after a retrain are comparable.

## Stamps are strings

19-digit `stamp_ns` exceeds JS `Number.MAX_SAFE_INTEGER`; the client passes
`stamp` (20-digit zero-padded string, matches filenames) everywhere.

## Ports

Default 8099. Do NOT use 9090 (firmware), 9091 (bebop-agent), 9092
(bebop-vision videoserver).
