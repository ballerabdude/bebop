# bebop-vision

Data-collection and dataset-labeling stack for the bebop robot: the
Orbbec RGB-D rig is recorded into **navd MCAP sessions** (dual-camera
color + lossless depth + cmd_vel/odom/goal + a geometric BEV teacher
grid), auto-labeled with SAM 3.1 floor concepts fused by measured depth,
and reviewed/corrected in a dashboard — the dataset any navigation
approach (learned or classical) trains and evaluates from.

This tree intentionally carries **no driving code**. Drive the robot
from the app (manual teleop); the recorder captures whatever you do.
The navigation research (BEV student, end-to-end trajectory student,
TensorRT runtime) lives on the `exp/navd-ai-20260909` branch.

## How a session becomes training data

```
 record            extract              SAM 3.1 labels          fuse                    review
 teleop + app ----> navd MCAP  --------> tools/sam_floor_label -> tools/fuse_navd_labels -> dashboard
 goals + odom      (tools/             (floor/carpet/rug/     (SAM floor gated by     (paint corrections
 (app Navigate)    mcap_extract.py)    ground per camera)     measured depth -> 60x60  as `hand` labels)
                                                              teacher per tick)
```
(The recorded MCAP carries no BEV grid — labels are generated purely
from SAM × measured depth.)

- **Record** — `main.py --record-navd <dir> --auto` opens/closes MCAP
  segments with the drive state (wheels armed, no estop) and prunes under
  a disk budget. Navigation goals from the app are recorded per tick —
  **teleop toward an active waypoint**: goals in the data are the signal
  goal-following models train on.
- **Extract** — `tools/extract_all_navd.py` unpacks MCAPs into
  `datasets/navd-v0/<session>/{color,color_far,depth,labels,manifest.jsonl}`.
- **Label** — `tools/sam_floor_label.py near|far` prompts SAM 3.1 with
  drivable-ground concepts per frame; `tools/fuse_navd_labels.py` gates
  SAM's floor by measured depth into a 60×60 teacher grid per tick
  (`blocked / navigable / caution`).
- **Review** — `tools/dashboard_api.py` (dashboard UI): replay ticks,
  compare fused teacher vs prediction overlays, paint corrections as
  additive `hand` labels. A session counts when it passes review.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Entry point: navd MCAP recorder (rig + operator video + goals) |
| `bebop_vision/orbbec.py` | Orbbec rig: cameras, sync, intrinsics, rig YAML |
| `bebop_vision/camera.py` | Threaded MJPEG/RTSP/file consumer with reconnect |
| `bebop_vision/robot.py` | Protobuf-over-WS runtime client (telemetry, goals) |
| `bebop_vision/videoserver.py` | Operator WebRTC (WHEP) streams (`:9092/whep?stream=...`) |
| `bebop_vision/whep.py` | webrtcbin H.264 pipeline + offer/answer per client |
| `bebop_vision/recorder_mcap.py` | navd MCAP writer (10 Hz ticks, shared log_time) |
| `bebop_vision/sam3_concepts.py` | SAM 3.1 text-prompted concept segmenter (teacher) |
| `bebop_vision/goals.py` | Navigation-goal slot (heading / odom point) |
| `tools/mcap_extract.py` | MCAP session -> navd-v0 training layout |
| `tools/extract_all_navd.py` | Extract + audit every mirrored session |
| `tools/sam_floor_label.py` | SAM 3.1 floor labels per extracted session |
| `tools/fuse_navd_labels.py` | SAM × depth -> fused 60×60 teacher grids |
| `tools/dashboard_api.py` | FastAPI backend for the review dashboard |
| `dashboard/` | Review dashboard (React): replay, overlay, hand painting |
| `tools/set_self_mask.py` | Chassis self-view mask painter |
| `tools/orbbec_imu_probe.py` | Camera IMU (accel/gyro) profiles + depth-clock offset probe |
| `weights/` | Checkpoints (gitignored) |
| `datasets/` | Extracted datasets (gitignored) |

Navigation research (BEV student, trajectory student, TensorRT runtime,
drive paths) lives on the `exp/navd-ai-20260909` branch — including its
own README additions and `docs/navd-traj.md`.

## Install

Robot + workstation runtime:

```sh
python -m venv .venv
.venv/bin/pip install -e .
```

Labeling side (workstation with NVIDIA GPU — SAM 3.1 teacher):

```sh
.venv/bin/pip install "sam3 @ git+https://github.com/facebookresearch/sam3.git" psutil
# sam3 pulls torchvision from PyPI, which will NOT match a cu128 torch build.
# Reinstall it from the same index torch came from, e.g.:
.venv/bin/pip install --force-reinstall --no-deps "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128
```

### Model weights

The robot does not ship weights. `config/models.yaml` is the catalog (shared
with the firmware): each entry is either `kind: hf` (download from Hugging
Face) or `kind: local` (trained on the robot, or copied from a workstation
with `scp`), and carries a `purpose` (e.g. `segmentation`, `backbone`,
`navigation`). Models sharing a purpose are alternatives. Add a model by
appending a catalog entry — no code changes.

**From the app (recommended):** *Configure models* on the Dashboard, the
Motor Bench footer (or the direct-connect screens) opens the **Models**
screen. It lists the catalog grouped by purpose, downloads on demand, and
lets the operator pick the active model per purpose. Gated entries need a
Hugging Face token, which the app sends to the robot (stored root-only at
`/etc/bebop/hf_token`, never echoed back). The token's account must have
accepted each model's license first, e.g.
<https://huggingface.co/facebook/sam3.1>. The purpose→model choice is
persisted at `/etc/bebop/model_selection.json` for the runtime to read. The
robot must be in Known Network (client) mode with internet during the
download; the firmware drives `bebop-model-download@<id>.service` and streams
progress into telemetry.

**On the workstation / by hand:**

```sh
# Put HF_TOKEN in .env (gitignored) or export it, then:
.venv/bin/python -m bebop_vision.download_model sam3.1   # one catalog id
.venv/bin/python -m bebop_vision.download_model --all    # every hf entry

# Legacy compatibility wrapper (downloads sam3 + sam3.1):
.venv/bin/python -m bebop_vision.download_sam3
```

## Collecting data

1. On the Jetson: `.venv/bin/python main.py --record-navd /var/lib/bebop-captures --auto`
2. From the app: enable wheels, teleop. Segments open/close with the
   drive state automatically.
3. Follow `docs/data-collection.md` for the per-session protocol —
   especially **goal-conditioned legs** (app Navigate waypoint + teleop
   toward it), which are the rarest and most valuable kind of tick.
4. Pull to the workstation (`just pull-data`), extract + audit
   (`just extract`), then review in the dashboard
   (`tools/dashboard_api.py`, port 8099).

## License note

SAM 3.x code and weights (teacher tooling, `weights/sam3*.pt`) are Meta's
"SAM License" — commercial use and fine-tuning are permitted and you own
derivative works, but redistribution of SAM 3.x materials or derivatives
must carry the SAM License. Downstream artifacts you train from the
labels (e.g. distilled students) are plain torch artifacts with no such
restriction. See the header of `bebop_vision/sam3_concepts.py`.
