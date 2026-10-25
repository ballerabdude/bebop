# bebop lab: model development on recorded drives

`bebop-vision/bebop_vision/lab/` is a Python package for developing
perception and navigation models from the robot's drive recordings. It is
not a UI: it is driven from a notebook, a script or
`python -m bebop_vision.lab`, and it uses Rerun as its only screen.

What it does, in the order a developer meets it:

1. **Sessions.** Pull drive MCAPs straight from the robot's firmware
   (`GET /captures`) and open any of them as typed, time-aligned numpy
   series plus random-access colour and depth frames. No second dataset
   layout: the recorder's MCAP is the dataset.
2. **Modalities.** A model declares what it reads with an `InputSpec`
   (colour, depth, IMU windows, odometry, commanded twist, firmware
   telemetry, intent, events, place sightings, with per-modality history).
   The dataset stays raw, so any architecture can consume it; ablating a
   modality is editing one tuple.
3. **Labels from the recording.** Hindsight paths (where the robot actually
   went over the next 3 s, in the body frame of the tick, heading from the
   camera gyro), with reverse windows truncated and flagged. Other label
   variants plug in beside it.
4. **Architectures.** Any timm model, the DINOv3 ViTs through Hugging Face,
   or a custom module, behind one `Backbone` interface (pyramid stages or
   tokens, pooled vector, preprocessing). Frozen features are cached once
   per (backbone, geometry) so heads train in minutes.
5. **Training modes.** `frozen`, `lpft` (linear probe, then unfreeze the
   last block groups) or `full`, with a run directory that records the git
   hash, config, backbone id, label versions and split.
6. **Runs, scores, views.** Run any model over a session, score it against
   the labels and the constant-velocity baseline, and see inputs, labels and
   outputs on the recording's own timeline in Rerun. Export to ONNX, check
   parity, and measure latency on the Thor with trtexec.

The whole package imports on the robot too (same venv rules), so an
experiment can be replayed on-device for engine parity.

## Setup

```sh
cd bebop-vision
.venv/bin/pip install -e '.[lab]'      # av, polars, timm, onnx, onnxruntime, rerun-sdk==0.31.3, marimo
python -m bebop_vision.lab ls
```

`rerun-sdk` must match the installed viewer (`scripts/setup-rerun.sh`,
`RERUN_VERSION`, 0.31.3 today). Sessions are found under
`datasets/sessions/` and `datasets/bev-student/raw/mcap/`; the per-session
cache lives under `datasets/lab/<session>/` (override with
`BEBOP_LAB_CACHE` or `--cache`). Both are gitignored.

## The loop from a shell

```sh
L='.venv/bin/python -m bebop_vision.lab'
$L remote                                  # what the robot has (finalizes the health log first)
$L pull 'navd_session_20261024_*'          # Range-resumed mirror over HTTP, skips active / young files; also mirrors the run log
$L runs                                    # drive runs: task, target, sessions, outcome (data-collection.md › Runs)
$L info 20261018_191941                    # channels, tick rate, keyframes, calibration; builds the index
$L ticks 20261018_191941 --hz 10 --head 20 # the per-tick table (odom, cmd, flags, runtime latest-at)
$L label 20261018_191941                   # hindsight paths -> datasets/lab/<s>/labels/hindsight_v2/
$L run 20261018_191941 --model cv          # constant-velocity baseline -> datasets/lab/<s>/runs/cv-<stamp>/
$L score datasets/lab/navd_session_20261018_191941/runs/cv-*/
$L show 20261018_191941 --run <run-dir> --labels hindsight       # Rerun, or --save x.rrd
$L backbones                               # the registry
$L features 20261018_191941 --backbone timm:convnext_tiny.dinov3_lvd1689m --role far --size 416x256 --what pooled,c3
```

Session arguments accept a stem, a unique fragment (`20261018_191941`), a
glob or a path. `--pitch-near` / `--pitch-far` override the recorded mount
pitch for projection (the rig YAML is stale: measured -59.5 / -8.6 deg
against -66 / -17.4; see `path-model.md` section 4).

Measured on the workstation (RTX 5090, 2026-10-24, the 67 s session
`navd_session_20261018_191941`, 1524 ticks, 1280x800 H.265 colour):

| Step | Time |
|---|---|
| index (one pass over the 624 MB MCAP) | 0.9 s |
| hindsight labels at 10 Hz (636 ticks) | 0.26 s |
| colour frame, random access (decode from the previous keyframe) | 49 ms |
| colour frame, sequential | 3.1 ms |
| depth frame (PNG) | 4.9 ms |
| DINOv3 ConvNeXt-T pooled + c3 features, far 416x256, 636 ticks | 15 s incl. weight download |
| frozen path probe, 3 epochs over 446 ticks (frames decoded each epoch) | 63 s |

## The loop from Python

`notebooks/lab_quickstart.py` (marimo) walks it; the pieces:

```python
from bebop_vision.lab import InputSpec
from bebop_vision.lab.catalog import Catalog
from bebop_vision.lab.session import Session
from bebop_vision.lab import labels, backbones, models, train, runner, metrics, rerun_log
from bebop_vision.lab.dataset import LabDataset, make_splits

s = Session(Catalog().resolve("20261018_191941"))
ticks = s.ticks(hz=10.0)
sample = ticks.select("color_near", "depth_near", "imu_gyro_near", "odom", "cmd_vel",
                      history_s={"imu_gyro_near": 1.0})[300]      # a types.Sample

hind = labels.label_session(s, variant="hindsight", ticks=ticks)   # cached, versioned

bb = backbones.get("timm:convnext_tiny.dinov3_lvd1689m")           # or hf:..., custom:module:attr
head = models.PathHead(in_dim=bb.embed_dim, n_scalars=2, n_points=15)
pm = models.HeadModel(bb, head, modality="color_far", size=(416, 256),
                      scalar_modalities=("cmd_vel",), mode="lpft", unfreeze_blocks=2)
splits = make_splits([...session names...], holdout=1)             # by drive, never by frame
ds_tr = LabDataset([...train Ticks...], pm.spec, labels={"hindsight": hind})
rd = train.fit(pm, ds_tr, ds_va, train.TrainConfig(epochs=10, mode="lpft"), train.path_loss,
               run_dir=s.index() / "train" / "cnxt_path_v1")       # manifest.json, metrics.jsonl, best.pt

run = runner.run(s, pm, ticks=va)                                  # outputs.npz + run.json
board = metrics.leaderboard([({"model": "cnxt"}, metrics.score_paths(run, hind)),
                             ({"model": "cv"}, metrics.score_paths(runner.run(s, models.ConstantVelocity(), ticks=va), hind))])
rerun_log.show(s, runs=[run], labels=hind)                         # the recording + labels + outputs
```

Swapping the architecture is the `backbones.get` string. Changing the
inputs is the `InputSpec`. Changing how much of the backbone trains is
`mode` / `unfreeze_blocks`. Everything else stays.

## Modules

| Module | Does |
|---|---|
| `types.py` | `Series` (structured numpy with `t_ns` first; nearest / latest-at / window / interp), `Calib` and `CameraModel` (body frame <-> pixels), `InputSpec`, `Sample`, output kinds (`Path2D`, `Mask`, `Embedding`, `Classes`, `Grid`, `Scalars`), the `Model` protocol. |
| `session.py`, `frames.py` | `Session`: one pass over the MCAP builds a cache (`series.npz`, raw AU / PNG byte stores with keyframe indexes, `calib.json`, tick flags) keyed by file identity and producer version. `Ticks` snaps to a Hz grid (nearest raw tick, never interpolated) and materialises `Sample`s. `FrameSeries` decodes H.265 from the previous keyframe; `DepthSeries` decodes PNG; `materialize()` writes a uint8 memmap for fast passes. |
| `catalog.py` | Local roots, the robot's `/captures`, the session ledger; `pull()` reuses `tools/pull_captures.py`; `runs()` groups segment files into drive runs by `run_id`. |
| `labels.py` | `fused_pose` (odom distance + gyro heading, bias from still periods, gap flags), `hindsight_paths`, `LabelSet` (npz + meta, versioned), `constant_velocity_path`. |
| `backbones.py`, `features.py` | The registry (`timm:`, `hf:`, `custom:`; `available()` lists the curated set with licence notes) and the per-session feature cache. |
| `dataset.py`, `models.py`, `train.py` | `LabDataset` over `(Ticks, InputSpec, labels)`, `make_splits` by drive, `make_run_splits` by run, `ConstantVelocity`, `EmbeddingModel`, `PathHead`, `SegHead`, `HeadModel` with LP-FT staging, `fit` with run manifests. |
| `runner.py`, `metrics.py` | `run()` with per-tick timing and saved outputs; ADE/FDE per horizon, mask IoU, `leaderboard`, `worst`. |
| `rerun_log.py` | `show()`: the `bebop_navd` dashboard plus a lab tab (body-frame plan view, per-run series), paths projected into the near image, masks as segmentation overlays, embeddings as PCA series. |
| `bench.py` | `export_onnx` (opset 17), `onnx_parity`, `thor_latency` (scp + trtexec over ssh; refuses while the recorder runs; never drives), `parse_trtexec`. |
| `cli.py` | `python -m bebop_vision.lab ...` |

Tests: `cd bebop-vision && .venv/bin/python -m pytest tests/lab -q` (282
tests, no network, no weights, no GPU, no real recordings). `tests/lab/synth.py`
writes synthetic sessions with the recorder's exact channels from a known
arc trajectory, with the tick index painted into every colour frame and
depth frame, so alignment and decoding are checked against ground truth.

## Conventions and gotchas

- Time is `t_ns` (MCAP `log_time`). A tick is the recorder's frame-set
  instant; `/depth_near` stamps define the ticks (`/odom` when depth is
  absent). IMU and firmware messages keep their own stamps.
- Body frame: x forward, y left, z up, origin on the floor under the mast.
  The plan view draws x upwards and y leftwards.
- Images are BGR uint8 at the recorded resolution (1280x800 colour,
  848x480 depth in mm). Models resize, crop and normalise themselves
  (`Backbone.preprocess`).
- A missing far colour frame (the far camera drops frames under load) is
  `None` in a `Sample` and `has_color_far=False` in the flags. On the H.265
  recorder path the held frame is re-encoded, so `far_hold_last` only fires
  for byte-identical JPEG repeats.
- Colour intrinsics are not in the MCAP; they come from
  `config/orbbec_intrinsics_<serial>.json` on this machine. Without them
  `Calib.camera(role)` raises and the image overlays are skipped.
- The far camera does not see the first ~1.1 m of floor, so short slow
  paths project only into the near image.
- The 2026-10-18 recordings predate `/runtime`, `/intent`, `/event` and
  `/place`; those series are empty there and the runtime columns are NaN / -1.
- Caches rebuild when the MCAP identity or the producer version changes;
  `index --force` rebuilds by hand.
