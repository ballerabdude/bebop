# bebop-vision

Vision stack for the bebop robot: a **teacher pipeline** that uses SAM 3.1
text-prompted segmentation to auto-label datasets, distills them into a
lightweight SegFormer navigable-path model (**student**), and a sector
planner that drives the robot from the predicted mask.

The teacher never runs on the robot. Only the distilled student
(`weights/navseg`, ~15 MB SegFormer-B0) does.

## How it fits together

The firmware owns the camera exclusively and republishes it as MJPEG on
`GET /video` (see `firmware/bebop-linux/src/video.rs`); the operator app
plays the same stream, and bebop-vision is just another subscriber.
Robot control speaks the same protobuf-over-WebSocket runtime API as the
operator app (`ws://bebop.local:9090/ws`). See `bebop_vision/camera.py`
and `bebop_vision/robot.py` for the two halves.

```
 record             SAM 3.1 masks           labelnav                train_nav
 robot MJPEG ----> frames + concept  ---->  0/1/2 nav labels ---->  SegFormer-B0 student
 (main.py         masks                   (concept -> nav      (class-weighted CE,
 --record-...)                             distillation)         val-mIoU checkpoint)
                                            teacher-only             runtime model
```

Nav classes: `0 = blocked`, `1 = navigable` (floor, clear of obstacles),
`2 = caution` (floor within a safety margin of obstacles). The planner
consumes `navigable` only (`planner.py`); weak caution predictions simply
shrink the drivable area, which fails conservative.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Entry point: record datasets, run the pipeline, drive the robot |
| `train_nav.py` | Train the SegFormer-B0 student |
| `bebop_vision/camera.py` | Threaded MJPEG/RTSP/file consumer with reconnect |
| `bebop_vision/sam3_concepts.py` | SAM 3.1 text-prompted concept segmenter (teacher) |
| `bebop_vision/recorder.py` | Dataset recorder: frames + synchronized masks |
| `bebop_vision/labelnav.py` | Concept masks -> nav label distillation |
| `bebop_vision/navseg.py` | Student inference (runtime, robot-safe deps) |
| `bebop_vision/planner.py` | Sector planner + drive node (mask -> twist) |
| `bebop_vision/robot.py` | Protobuf-over-WS runtime client |
| `bebop_vision/download_sam3.py` | Fetch gated SAM weights from Hugging Face |
| `tools/` | One-shot SAM 3.1 encoder ONNX export + TensorRT build |
| `weights/` | Checkpoints (gitignored) |
| `datasets/` | Recorded datasets (gitignored) |

## Install

Runtime side (all you need on the robot or for inference):

```sh
python -m venv .venv
.venv/bin/pip install -e .
```

Teacher side (dataset recording + labeling, workstation with NVIDIA GPU):

```sh
.venv/bin/pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git" psutil
# sam3 pulls torchvision from PyPI, which will NOT match a cu128 torch build.
# Reinstall it from the same index torch came from, e.g.:
.venv/bin/pip install --force-reinstall --no-deps "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128
```

Packaging notes (as of sam3 0.1.0): its metadata omits `psutil` and
`torchvision` despite importing both, and it pins `numpy<2`. The numpy
downgrade coexists fine with opencv-python in practice; pip's metadata
warning about it can be ignored.

### SAM weights

SAM 3.x checkpoints are gated. Request access at
<https://huggingface.co/facebook/sam3.1>, put `HF_TOKEN=...` in `.env`
(gitignored), then:

```sh
.venv/bin/python -m bebop_vision.download_sam3          # both versions
.venv/bin/python -m bebop_vision.download_sam3 sam3.1   # one version
```

## The teacher pipeline

### 0. Reach the robot

Defaults point at `http://bebop.local:9090/video` / `ws://bebop.local:9090/ws`
(`bebop_vision/config.py`). mDNS can be flaky; if resolution fails, pass the
LAN IP explicitly: `--source http://<robot-ip>:9090/video`. Test with
`getent hosts bebop.local` or `curl -I http://<robot-ip>:9090/video`.

### 1. Record a teacher dataset

```sh
.venv/bin/python main.py --record-dataset datasets/nav-v0 \
    --seconds 60 --record-rate 2 --display \
    --concepts "floor,wall,person,chair,table,door"
```

Writes `images/*.jpg`, SAM 3.1 concept masks `masks/*.npz`, and
`manifest.jsonl` (per-sample concept pixel counts). Concepts default to
`config.RECORD_CONCEPTS` if omitted; `--conf` sets the mask confidence
threshold (default 0.5).

**Record while moving.** The recorder is a plain video subscriber — it is
not mode-gated — so teleop the robot with the app's direct controllers while
recording. A stationary 60 s take yields ~120 near-identical frames, which
trains a model that knows exactly one room. Aim for 1–2k frames across
5–10 scenes (different rooms, obstacle layouts, lighting). Camera gimbal
sweeps (`RobotClient.set_camera_pose`) also add viewpoint diversity from a
stationary robot.

### 2. Distill masks into nav labels

```sh
.venv/bin/python -m bebop_vision.labelnav datasets/nav-v0 --floor floor --margin 25
```

`navigable` = the floor concept(s) minus a `--margin`-pixel dilation around
every other concept (the robot-radius safety band becomes `caution`;
everything else is `blocked`). Writes `labels/*.png` keyed to image stems.
Cheap to re-run with a different margin — no re-recording needed.

### 3. Train the student

```sh
.venv/bin/python train_nav.py datasets/nav-v0 --epochs 60 --out weights/navseg
```

SegFormer-B0 from scratch at 512 px, class-weighted cross-entropy
(caution is a tiny fraction of pixels), flip/brightness augmentation,
OneCycle LR, best-val-mIoU checkpointing. On a modern GPU expect
~1.5 s/epoch at ~120 samples.

Interpreting metrics: mean mIoU is dominated by the caution class, whose
thin band is inherently hard to nail (IoU 0.1 is normal early on). Judge
the model by `blocked`/`navigable` IoU (~0.85+ on a sane dataset) and by
eye (next step).

### 4. Export the student for the robot

The robot's firmware (`bebop-linux`) runs the nav model itself, in
process, via ONNX Runtime — same pattern as the locomotion policy. This
step produces the firmware artifact:

```sh
.venv/bin/python tools/export_navseg_onnx.py
```

- Writes `weights/navseg.onnx` (graph) + `weights/navseg.onnx.data`
  (external weights) — the same two-file drop-in convention as
  `policy.onnx` / `policy.onnx.data`
- Parity-gates the export: torch vs ONNX Runtime 1.23.0 must agree on
  ≥99% of labels across sampled dataset frames (matching the dylib
  version the firmware pairs with)

### 5. Deploy to the robot

On the Jetson, ship both files next to the robot YAML (e.g.
`firmware/bebop-linux/config/` when running from the checkout,
`/etc/bebop/` for installed releases) and enable the runner with a
`nav:` block in the YAML:

```yaml
nav:
  rate_hz: 10
```

The firmware's nav runner (`src/nav.rs`) subscribes to the camera hub,
JPEG-decodes + preprocesses each frame (identical math to
`navseg.py`), runs the ONNX session with the **CUDA execution provider
when the installed `libonnxruntime.so` supports it** (CPU EP as
fallback), and publishes `NavState` in telemetry plus subscribe-gated
`NavMaskFrame` pushes — the operator app's video overlay ("Labels"
toggle) consumes the latter. A missing model or `nav:` block changes
nothing else about the robot: soft-fail, `NavState.present = false`.

GPU note: the aarch64 CUDA build of onnxruntime must match the `ort`
crate version the firmware uses (1.23.0 for `ort 2.0.0-rc.11`) — no
official release ships one, so it's built from source on the Jetson
(`./build.sh --config Release --build_shared_lib --use_cuda
--cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
--cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native ...`). Deployment
is **three files** into `/usr/local/lib` (keep the CPU original as
`*.cpu-backup`): `libonnxruntime.so` *plus*
`libonnxruntime_providers_shared.so` (the provider bridge) and
`libonnxruntime_providers_cuda.so` (the 80+ MB CUDA EP itself). Ship
only the main lib and the CUDA EP silently fails to load — the by-name
append path logs a warning and runs everything on CPU while still
*reporting* the CUDA session. Diagnose with the firmware's
`nav_probe` bin and `tegrastats` (GR3D should show bursts).

Also: on a battery-powered robot, 25 W MAXN transients during CUDA
init can brown out the supply (hard power-off / SIGSEGV mid-init).
Run `sudo nvpmodel -m 0` (15 W) — costs ~15% of GPU throughput and
stabilizes the rails.

### 6. Evaluate live

```sh
.venv/bin/python main.py --display --source http://<robot-ip>:9090/video
```

Watch the overlay: walk in front of the camera and the mask should track
you as blocked while the floor stays navigable. `--record out.mp4` saves
the annotated stream; `--nav-model` points at a different checkpoint;
`--source` also accepts any video file or RTSP URL for offline eval.

On the robot itself the app's video screen has a **"Labels"** toggle —
the same model output, overlaid on the live playback by the firmware
(no Python involved).

### 7. Drive (workstation bench only — do not use on the robot yet)

```sh
.venv/bin/python main.py --drive
```

Pipeline frames -> `SectorPlanner` (image sectors scored by navigable
corridor depth; rotate toward the best sector) -> `SetVelocityCommand` at
`--command-hz` (default 10 Hz). Safety behavior, by design:

- requires firmware mode `MODE_RUN_POLICY` (`--drive-any-mode` overrides
  this — bench use only)
- e-stop telemetry latches a stop (callback in `main.py`)
- 0.5 s deadman: stale nav results stop commands
- robot disconnected -> zero twist

## Dataset layout

```
datasets/nav-v0/
  images/     *.jpg     frames as recorded
  masks/      *.npz     per-concept boolean masks (compressed)
  labels/     *.png     0/1/2 nav labels (generated by labelnav)
  manifest.jsonl        per-sample concept pixel counts
```

## SAM 3.1 TensorRT encoder (optional)

For faster teacher inference, export the SAM 3.1 vision encoder to ONNX
and build an fp16 engine, then pass `--sam-trt weights/sam31_encoder_fp16.engine`
to `main.py --record-dataset`:

```sh
.venv/bin/python tools/export_sam31_encoder.py   # needs the sam3 package
.venv/bin/python tools/build_trt_engine.py       # needs tensorrt
```

Both scripts use hard-coded paths under `weights/`. The TRT encoder is
teacher-side only; the robot runs the distilled student on plain torch.

## License note

SAM 3.x code and weights (teacher tooling, `weights/sam3*.pt`) are Meta's
"SAM License" — commercial use and fine-tuning are permitted and you own
derivative works, but redistribution of SAM 3.x materials or derivatives
must carry the SAM License. The distilled student model is a plain
torch/transformers artifact with no such restriction. See the header of
`bebop_vision/sam3_concepts.py`.
