# Exact input/output BEV grid experiment — 2026-10-19

> **Retired 2026-10-23.** The BEV student runtime, model, training and evaluation tools were removed
> from the tree: the navigation redo does not use a BEV (see [`path-model.md`](path-model.md)).
> The dataset extractor, the SAM x depth teacher, the learned chassis footprint and the geometry
> helpers stay. This document is kept as the measurement record; file paths in it may be gone
> (`git log -- bebop-vision/bebop_vision/bev_service.py`).

The operator explicitly selected **360 rows × 640 columns** for both camera
input and BEV output, replacing the previous training queue. This is a new
SegFormer-B5 experiment, not an upscaled rendering of the 120×60 model.

## Contract

| Item | Value |
|---|---|
| Two-camera runtime input | `[2, 4, 360, 640]`, RGB-D |
| Semantic output | `[4, 360, 640]`, floor/wall/obstacle/hazard logits |
| Ego / coverage outputs | `[1, 360, 640]` each |
| Internal feature lift | 360×640 (`--full-res-lift`) |
| Physical extent | 6 m forward × **4 m** sideways (first trial: 3 m, see below) |
| Forward cell spacing | 6/360 m = 1.6667 cm |
| Sideways cell spacing | 4/640 m = 0.625 cm (3 m trial: 0.46875 cm) |
| Output cells | 230,400, **32×** the old grid |

The map has **rectangular metric cells**. Metadata explicitly records
`cell_x_m` and `cell_y_m`, and sets the legacy scalar `cell_m` to null.
Coordinate projection, chassis masking, wall component extents and the
historical 5 cm neighborhood radii use the appropriate physical axis.

The live map contract now carries rectangular cells: `BevGrid` has
`cell_x_m`/`cell_y_m` (and `cell_fwd_m`/`cell_side_m` accessors that fall
back to the scalar `cell_m`), the `/bev` JSON emits both plus a null `cell_m`
for non-square grids, the runtime (`BevStudentRuntime.configure_grid`) builds
the static ego rectangle and the inflation ellipse per axis, and the renderer
draws metric range rings as pixel ellipses. The firmware and the app do not
read the cell size (the app only shows the rendered `bev` video stream), so
no change was needed there. The previous 10 Hz measurements in
[bev-segformer.md](bev-segformer.md) used 120×60 output; see the deployment
section below for this model's measured speed.

## Labels

- Source: the same 2,100 frame pairs selected by `labels_base2/` across the
  same six drives and train/validation split.
- Destination: each session's `labels_grid360x640/`.
- Fresh SAM 3.1 category maps plus original calibrated native-depth samples;
  depth sampling stride is **1×1** rather than the older subsampling.
- Labels are rasterized directly at 360×640, without resizing old labels or
  filling unobserved cells as floor. The original `labels/` and
  `labels_base2/` sets are preserved.
- Stair/drop policy remains the floor-confirmed, below-plane hazard rule;
  stair prompts cannot become floor. Ego remains the physical chassis rectangle.
- `grid.json` records metric geometry, teacher settings and depth stride.
  Training rejects a label directory with mismatched grid metadata.
- Label writes are atomic; `--resume` validates and reuses completed frames.
  Preview images and per-frame diagnostics accompany the new labels.

The first frame from each drive passed label validation. Coverage on those
six samples was 17.8–39.1%; labeling took about 2 seconds per pair, implying
roughly 70 minutes for the full set. These fractions are observations from
the samples, not a density guarantee for the complete dataset.

### Why the labels moved to 4 m wide with a footprint splat

Reviewing the first 3 m-wide, point-projected labels on 2026-10-19 showed two
things the sensor data makes unavoidable at this cell size:

1. **Horizontal stripes of unknown.** Each depth pixel landed in exactly one
   cell. The floor distance between consecutive depth rows grows with the
   square of the range (far camera, 1.32 m high, 17.4° down, f = 414 px):
   about 0.5 cm at 1 m, 1.0 cm at 2 m, 2.0 cm at 3 m, 3.2 cm at 4 m, 6.9 cm
   at 6 m. The forward cell is 1.67 cm, so beyond ~2.5 m most cells between
   depth rows got no sample. Measured observed fraction by band on drive
   191614 (old 120×60 vs the 3 m point labels): 0–1 m 52% → 48%, 1–2 m
   65% → 62%, 2–3 m 45% → 32%, 3–4 m 28% → 16%, 4–6 m 12% → 6%. The
   semantic loss ignores unknown cells, so the semantic head was only
   under-supervised, but the coverage head would have learned the sampling
   pattern as if it were observed area. The recorded depth is the raw
   848×480 16-bit sensor output with 74–82% valid pixels; the stripes come
   from the grid being finer than the ground sampling, not from the data.
2. **Width.** The cameras cover 91° horizontally, so at 3 m they see 6 m
   across. Share of depth evidence (within 6 m forward, below 1.5 m height,
   160 frames across all drives) outside a ±1.5 m map: 1% at 0–1 m, 11% at
   1–2 m, 31% at 2–3 m, 51% at 3–4 m. At ±2 m: 0%, 1%, 15%, 38%.

The fix is a **footprint splat** (`LabelConfig.splat_footprint`, relabel flag
`--splat`): a depth pixel observes the patch of surface between itself and
its right and down pixel neighbours, so every cell inside the body-frame
bounding box of those three points gets the sample's vote, with the same
range weight. A neighbour more than `splat_max_m` (12 cm) away on any body
axis is across a discontinuity and does not extend the patch, so walls and
object edges stay one sample thick. Plane fitting still uses the un-splatted
points. This is what the sensor saw, not interpolation: a pixel at 4 m
genuinely covers a 3 cm × 1 cm patch. Coverage became solid on the floor and
the covered-but-unclassified cells of the point projection mostly resolved
to a class.

The width went to **4 m** (`--width-m 4.0`, training `--bev-width 4.0`),
keeping 640 columns, which also brings the sideways cell (0.625 cm) closer
to what the sensor resolves at 2 m. Labels for the 4 m splat run live in
`labels_grid360x640w4/`; the 3 m point-projected set in `labels_grid360x640/`
is complete and kept as the control. The eval tooling can crop to the shared
3 m when scoring against the old 120×60 model.

Relabeling is now parallel in the wrapper (one worker per drive, `WORKERS`,
~5.6 GB VRAM each). Serial labeling was one process on one core with the
GPU half idle (1.85 s/frame); five workers saturate the GPU at ~1.2 s/frame
aggregate. The RANSAC floor fit also had a latent crash on sparse near-floor
bands (fewer than 288 band points with 96 triplets drawn without
replacement), hit by the stride-1 labels on a frame facing a wall; it now
samples with replacement there (`master_bev.fit_ground_plane`, regression
test in `tests/test_bev_student.py`).

### Chassis: dead disc replaced by a measured rectangle (2026-10-19)

The round unknown region around the ego box in every label was not the
chassis: `geometry.min_range_m` culled every depth point within 0.55 m of
the body origin (the camera mast) before labeling, a leftover from the old
geometric BEV. Over 16 frames the near camera had ~375k floor-level points
inside that disc but outside the chassis, after its pixel self-mask, i.e.
observed floor beside the wheels thrown away.

The chassis was measured from the near camera's depth across all six drives
(persistence of above-floor points in body coordinates; the robot's own
parts sit at the same body coordinates in every frame while the scene
changes): the frame top at 0.58 m spans x -0.10..+0.29, y -0.20..+0.09, so
the body is 0.40 m long by 0.29 m wide, centred 10 cm ahead of and 5 cm to
the right of the mast. The previous YAML ego rectangle (x 0..0.55,
y ±0.22) extended 25 cm past the front rails over floor the near camera
cannot see past the body (unobserved, not chassis) and was centred on the
mast rather than the body.

`config/orbbec_rig.yaml` now carries, as the single source for the labeler
**and** the live runtime (which ORs the static ego rectangle onto the
predicted ego every frame):

- `bev_student.ego`: x -0.10..0.29, y -0.27..0.17 (measured body, widened
  to the 0.44 m the old entry carried for wheels outside the frame).
- `bev_student.label.cull_rect_m`: x -0.30..0.37, y -0.35..0.25 (ego plus
  ~8 cm margin), replacing the disc. Unset it to get the disc back.

These are camera-derived fallbacks: the base and wheels are below what the
camera sees. **Verify with a tape measure** (overall width incl. wheels,
mast-to-front and mast-to-rear distance, lateral mast offset) and update the
YAML; labels and runtime then agree by construction.

All older label sets (`labels/`, `labels_base2/`, `labels_fixed/`,
`labels_local/`, `labels_nohaz/`, `labels_prompt/`, the 3 m
`labels_grid360x640/` control and the first `labels_grid360x640w4/`) were
deleted on 2026-10-19 together with the two superseded matched-grid run
directories. The frame selection no longer depends on an older label set:
`bev_relabel_grid.py --stride 3` labels every third index row, which is
exactly the 2,100-frame selection the base sets used (`sessions.json`
records the counts). `grid.json` records `frame_stride`.

## Run

```bash
bash bebop-vision/tools/train_bev_matched_grid.sh            # 4 m, splat (default)
WIDTH=3.0 SPLAT=0 LABELS=labels_grid360x640 \
RUN=runs/bev-student/exp-base2-b5-grid360x640 bash bebop-vision/tools/train_bev_matched_grid.sh
```

The pipeline relabels in parallel (`WORKERS`, default 5), refuses to train if
any drive's label log lacks `GRID_LABELS_DONE`, then starts training,
evaluation and ONNX export automatically. It uses a lock to prevent
duplicate launches.

- Run: `bebop-vision/runs/bev-student/exp-base2-b5-grid360x640w4/`
  (the 3 m point-label control run directory is `exp-base2-b5-grid360x640/`).
- Progress: `pipeline.log`; per-drive label logs in `relabel/`; training
  also writes `train.log` and `tb/`.
- Recipe: 40 epochs, 5 frozen-encoder epochs, microbatch 1 with accumulation
  over 8 pairs, horizontal flip, hazard weight 3, Tversky weight 0.5 with
  alpha/beta 0.5/0.5, balanced checkpoint score.
- Initial weights: the ADE20K-pretrained SegFormer-B5 encoder; new BEV heads.
- A full-backbone forward/backward, gradient accumulation and validation
  smoke test passed on the RTX 5090 using real newly generated labels.

Validation uses grid-derived forward-distance bands and bounded-size review
images. Evaluation loss is now a true per-sample average (previous code
divided the sample-weighted sum by the number of batches); old logged loss
values at batch >1 should not be directly compared. Other class metrics are
unchanged by that correction. IoU across the old and new teacher grids is
also a different benchmark; inspect both against common physical scenes.

## B5 run cancelled at epoch 18; DINOv3 at 256×448 replaces it (2026-10-19)

The B5 run plateaued: validation mIoU sat at 0.557–0.564 from epoch 11 to
17 while train loss kept falling (0.84 → 0.70) and validation loss drifted
up (1.35 → 1.49), i.e. the remaining epochs were overfitting a 2,100-frame,
one-apartment dataset. Hazard recall stayed at 0.28–0.33 as in every
earlier experiment. The run was stopped during epoch 18 (epoch 11 remains
its best checkpoint, deployed below); the run directory is kept.

The follow-up run shortens the BEV to **4 m forward × 4 m wide, 240×640**
(the operator asked for a shorter grid; the 4–6 m rows were observed in
only ~9 % of cells) at the same 1.667 cm × 0.625 cm cells, and switches the
encoder to DINOv3 ViT-S/16 (22.6 M params, pretrained weights in
`weights/dinov3-vits16`) at a **256×448** camera input (DINO wants
multiples of 16), microbatch 4 with accumulation 2, 30 epochs, 5 frozen.
The labels were not regenerated: `tools/bev_crop_labels.py` drops the 120
far rows of every 6 m label cell-for-cell into `labels_grid240x640w4r4/`
(`grid.json` records `cropped_from`). Smoke-tested at batch 4 before
launch; the live run takes about 0.7 s per step (484 steps, ~6 min per
epoch, ~15 GB VRAM).

```bash
RUN=runs/bev-student/exp-dino-grid240x640w4r4-h256 LABELS=labels_grid240x640w4r4 \
BEV_ROWS=240 BEV_RANGE=4.0 ENCODER=dinov3 HEIGHT=256 WIDTH_PX=448 \
EPOCHS=30 STAGE1=5 BATCH=4 ACCUM=2 NUM_WORKERS=8 SKIP_RELABEL=1 \
bash bebop-vision/tools/train_bev_matched_grid.sh
```

The wrapper now takes `ENCODER`, `HEIGHT`/`WIDTH_PX`, `BEV_ROWS`/`BEV_RANGE`,
`EPOCHS`/`STAGE1`, `BATCH`/`ACCUM`, `NUM_WORKERS` and `SKIP_RELABEL`
overrides. (A first DINO launch on the full 6 m grid was stopped within
its first epoch once the operator clarified that the BEV, not the camera
input, was to shrink; its run directory was removed.)

### Chassis learned from the recordings (self-supervised ego)

`tools/bev_ego_from_data.py` replaces the hand-entered ego rectangle with a
footprint measured from the robot's own data. Over 633 frames (every 10th
index row of all six drives, both cameras at depth stride 1, pixel self
mask off) it accumulates two body-frame rasters at 1 cm:

- `p_body`: fraction of frames in which a cell holds depth 3 cm–1.2 m above
  the floor. The scene changes between frames; the robot's own parts do
  not, so cells above 0.85 are the chassis the near camera can see (the
  frame top, 0.58 m up): x −0.11..+0.30, y −0.21..+0.10 m, the same body
  the earlier one-off measurement found.
- `p_obs`: fraction of frames in which a cell is observed at all. Cells
  that are never observed (below 0.03) within 15 cm beside that core, and
  within its rear..front span, are floor the body hides from the cameras
  (wheels, underbody). The strip the frame hides *ahead* of the bumper is
  occluded floor, not body, and is excluded.
- The convex hull of core plus shadow is the footprint: x −0.13..+0.32,
  y −0.34..+0.19 m (0.45 m long, 0.53 m wide, 0.229 m²). The 0.402 m wheel
  track implies about 0.45 m overall, so the learned mask is up to 4 cm
  generous per side; those cells can never be observed from the robot, so
  nothing is lost by calling them ego.

The raster lives in `config/ego_mask.npz` (preview `config/ego_mask.png`:
p_obs, p_body, core/shadow/hull with the old rectangle and the cull
rectangle overlaid) and `bev_student.ego.mask_file` in the rig YAML points
to it. `master_bev.rasterize_ego` samples it onto any grid; the labeler
(`ego_mask`) and the live runtime (`configure_grid`) both use it, so labels
and runtime agree by construction. The rectangle stays as the fallback
when the file is absent. `--apply-labels labels_grid360x640w4` rewrote the
ego layer of the existing 2,100 labels in place (1,518 ego cells per
label; chassis cells become semantic 255 and coverage False) without
touching SAM, and recorded `ego_mask_file` in each `grid.json`. Rebuild the
raster after any chassis or camera-mount change.

## Option C: 160×320 over 4 m × 4 m, reflection-safe labels, single head (2026-10-19 evening)

Reviewing the input data with the labels showed that **44 % of all hazard
cells came from 62 frames** (mostly drives 193224 and 193619) where the far
camera reads a glossy laminate floor as a pit 0.3–0.8 m below the fitted
plane: stereo matching on the floor's reflection. The labeler's drop rule
("below the plane on SAM floor") had no lower bound, so it painted the
whole room orange. `LabelConfig.hazard_deep_drop_m` (0.25 m) now requires
the SAM stairs/step/hole prompt for any depth-only drop deeper than one
stair riser; deeper unconfirmed samples are reflections and are removed
from *all* evidence (they are not floor observations either). On the worst
frames hazard cells fell from 77k to 3k while a real staircase frame was
unchanged (1,666 → 1,668 cells); the per-frame `reflections` count is in
the label stats. Synthetic tests use an 18 cm riser (a 30 cm depth-only
drop is now, correctly, not a hazard).

The DINO run on the 240×640 labels confirmed the damage: hazard F1 peaked
at 0.40 by epoch 7 and collapsed to 0.26 by epoch 29 as the model fit the
reflection "hazards" (val loss 1.55 → 1.80). Its eval/export step did not
run because the wrapper script was edited while that pipeline was still
executing it (bash reads scripts incrementally; never edit a running
wrapper). The checkpoint is kept for a speed reference only.

Grid options A–E (240×640 down to 80×80 over 4 m × 4 m) were rendered at
native pixel size for the same frames; the operator chose **C: 160×320,
2.5 cm × 1.25 cm**, which matches what the depth sensor resolves at 2.5–3 m
and keeps thin obstacles one to two cells wide at a third of the cost.
Labels are regenerated at that grid (`labels_grid160x320w4r4/`, reflection
rule on, learned ego raster) rather than resampled.

**Single head.** The ego head is gone and coverage is no longer a separate
sigmoid: `--single-head` trains one 6-class output
(floor, wall, obstacle, hazard, **ego**, **unknown**). The chassis is a class
the model predicts, so its own body can never be floor or an obstacle; the
runtime still ORs the learned raster onto the predicted ego as a belt.
Contract: input `[2, 4, 256, 448]`, output `[6, 160, 320]`; no sigmoids,
one tensor each way. The export writes `single_head: true` and the class
list, the runtime derives ego (= class 4) and coverage (≠ class 5) from the
argmax, and the eval tools and the hazard sweep do the same when the extra
heads are absent. The older three-head engines still load unchanged.

```bash
RUN=runs/bev-student/exp-dino-grid160x320w4r4-h256 LABELS=labels_grid160x320w4r4 \
BEV_ROWS=160 BEV_COLS=320 BEV_RANGE=4.0 ENCODER=dinov3 HEIGHT=256 WIDTH_PX=448 \
EPOCHS=30 STAGE1=5 BATCH=4 ACCUM=2 NUM_WORKERS=8 \
EXTRA_TRAIN_ARGS="--single-head" bash bebop-vision/tools/train_bev_matched_grid.sh
```

(`RELABEL_ONLY=1` stops after the labels; `SKIP_RELABEL=1` trains on
existing ones; `BEV_COLS` joins the overrides.)

### Results on the option C labels (2026-10-19, 23:20)

| Run (same labels, same grid, same recipe) | Best epoch | mIoU (labelled cells) | floor / wall / obstacle / hazard IoU | Hazard F1 | Coverage P / R | Ego |
|---|---|---|---|---|---|---|
| single head, 6 classes | 17 | 0.455 | 0.76 / 0.43 / 0.51 / 0.12 | 0.18 | 0.82 / 0.80 | class IoU 0.94 |
| **three heads (control)** | 11 | **0.538** | 0.83 / 0.56 / 0.53 / 0.23 | **0.377** | 0.84 / 0.88 | head, 8 % FP |

The single head lost 0.08 mIoU on the same data: 33 % of wall-labelled and
35 % of hazard-labelled cells were predicted "unknown", because the
majority class competes in the same softmax and wins wherever the evidence
for a thin wall or a stair edge is weak. Re-scoring the checkpoint with the
class chosen among the four real classes and "unknown" used only as a
coverage gate did not help (0.459), so it is the training formulation, not
the decision rule. The 6-class logs also used a stricter IoU (predictions
on unobserved cells counted as false positives); `evaluate()` now reports
the labelled-cells IoU for every layout and keeps the strict one as
`miou_strict`. The three-head control is the deployed model; the
single-head code path stays available (`--single-head`) as a negative
result. `tools/bev_eval_diagnostics.py` now tolerates class ids ≥ 4.

Comparable earlier numbers: the 240×640 DINO run on the reflection-polluted
labels peaked at 0.546 / hazard F1 0.40 and collapsed to 0.26 by its end;
on the cleaned labels the control holds 0.53 and 0.33–0.38 to the last
epoch (train loss 0.55, val 1.70, so still overfitting a one-apartment
dataset, just no longer to reflections).

### Deployed: option C three-head DINO, epoch 11 (2026-10-19 23:20)

`weights/bev_student_c160_fp16.engine` on the Thor (66 MB, TensorRT 10.16.2,
`trtexec --fp16`, built in ~3 min; ONNX parity: semantic argmax agreement
0.9999). `/etc/bebop/vision.env` points `--bev-model` at it (the B5 e11
setting is backed up as `vision.env.bak-<stamp>`). Code, rig YAML and the
learned ego raster were synced first; robot tests 311 passed.

| Measurement on the Thor (recorder + WebRTC streams running) | Value |
|---|---|
| trtexec GPU compute, median | 20 ms |
| Recorded-pair benchmark, 10 Hz pace, p50 / p95 tick | 78 / 107 ms (9.9 Hz) |
| Live service: preprocess + infer | 38 + 24 ms |
| Live grid rate (213 polls of `/bev` in 5 s) | **9.3 Hz**, median 100 ms, p95 133 ms |

Live grid: 160×320 over 4 m × 4 m, ego 533 cells from the learned raster
OR the ego head, grid age 0.08 s. The preprocess (two 1280×800 JPEG decodes
and resizes, ~38 ms) is now the largest term; decoding at half resolution
is the next speed lever if a true 10 Hz is wanted. Note: rate measurements
must poll faster than the grid updates; a 4 Hz poll reads as "4 Hz".

### Cull follows the footprint (2026-10-20)

Reviewing the live map showed obstacles painted right up to the chassis.
In the validation ring just outside the old cull rectangle the deployed
model called 2.3 % of cells obstacle against 1.7 % in the teacher, and a
third of those were labelled floor. Cause: `cull_rect_m` was 8–17 cm
larger than the learned body on every side, so a band of cells around the
chassis was unobserved in every label and the model learned to extend
neighbouring obstacles into it. `label.cull_ego_margin_m` (0.02) now drops
depth only inside the learned raster dilated by 2 cm (`ego_cull_mask` /
`points_in_raster`); `cull_rect_m` remains the fallback without a raster,
and unsetting both restores the disc. Labels regenerated into
`labels_grid160x320w4r4c/`; run `exp-dino-grid160x320w4r4c-h256` (three
heads, same recipe) is the candidate to replace the deployed engine.

Result (both checkpoints, epoch 11 each, scored on the footprint-cull
validation labels so the comparison is like for like):

| Validation, `labels_grid160x320w4r4c` | deployed (`w4r4`) | footprint cull (`w4r4c`) |
|---|---|---|
| labelled-cells mIoU | 0.539 | 0.539 |
| wall / obstacle / hazard IoU | 0.559 / 0.534 / 0.238 | 0.548 / 0.534 / 0.245 |
| hazard P / R / F1 | 0.459 / 0.331 / 0.384 | 0.472 / 0.338 / 0.394 |
| coverage P / R | 0.838 / 0.881 | 0.829 / 0.889 |
| ring beside chassis: predicted obstacle (teacher 1.73 %) | 2.29 % | 2.01 % |
| of those, labelled floor / unknown / obstacle | 36 / 38 / 25 % | 33 / 37 / 29 % |

The band the old rectangle used to hide (old cull minus footprint + 2 cm,
129 cells per frame) is still 88 % unobserved in the teacher: the near
camera simply cannot see most of it past the frame, so the cull change
only recovers the fringe. Both models predict almost nothing there
(obstacle on labelled floor 0.03 % → 0.01 %). The gain is therefore
modest: equal mIoU, hazard F1 +0.01, and 12 % fewer obstacle calls in the
ring; coverage trades 1 % precision for 1 % recall. The footprint-cull
model was deployed because it is marginally better and its labels match
the current rig config. The remaining red at the chassis edge in the live
view is real clutter or model over-extension, not the cull.

Deployed 2026-10-20 02:00 as `weights/bev_student_c160fc_fp16.engine`
(66 MB, `trtexec --fp16`, ONNX parity agree 1.0000); `/etc/bebop/vision.env`
backed up as `vision.env.bak-<stamp>` and repointed; code, rig YAML and
tests synced (robot suite 342 passed; `tools/mcap_extract.py` had drifted
and was re-synced too).

| Measurement on the Thor | Value |
|---|---|
| Recorded-pair benchmark, 10 Hz pace, p50 / p95 tick | 69 / 94 ms (10.0 Hz) |
| Live service: preprocess + infer (median) | 42 + 23 ms |
| Live grid rate (258 polls of `/bev` in 10 s, 30 Hz poll) | **9.7 Hz**, median 112 ms, p95 121 ms |

Live grid: 160×320 over 4 m × 4 m, ego 527 cells, age 0.11 s. Polling
`/bev` at ~120 Hz loads the service enough to read 8.2 Hz; poll at 20–30 Hz.
The hazard patch at the far doorway is still present and still unverified.

### Hazard strip beside the chassis (2026-10-20)

The live map still showed thin orange (hazard) strips hugging the chassis
sides. The teacher does the same: in the footprint-cull labels, 83 % of
frames carry ≥10 hazard cells within 10 cm of the footprint, almost all on
the **left** side (6–10 cm band: 5.8 % of cells hazard, 8 531 left vs 46
right; nothing front or back). Tracing one frame to the near image shows
the cause: depth pixels on the silhouette edge of the left chassis rail mix
the rail (0.6 m away) with the floor behind it and land 5–8 cm below the
floor plane, 4–6 cm outside the learned footprint at y ≈ +0.25 m; SAM calls
that floor, so the hazard rule marks a drop. The right rail's edge pixels
fall inside the (wider) right side of the footprint and are culled.

Two fixes, both configured in `orbbec_rig.yaml`:

- `label.hazard_ego_margin_m: 0.15` — in `camera_evidence`, below-plane
  samples inside the footprint dilated by 15 cm are not hazard
  (`ego_dilated_mask`, counted as `edge_hazard` per camera). Floor and
  obstacle evidence beside the body is unchanged. Labels
  `labels_grid160x320w4r4e/`, run `exp-dino-grid160x320w4r4e-h256`
  (same recipe) are the candidate for the next engine.
- `ego.hazard_pad_m: 0.15` — runtime guard for engines trained before the
  rule: a predicted drop in the 15 cm ring around the static chassis takes
  the model's best other class (`edge_hazard_cells` in the model stats).
  Live on the robot since 07:45: hazard cells in the ring 21 → 0 on
  consecutive grids, 268 ring cells rewritten, rate unchanged (10 Hz,
  infer 23 ms). Robot tests 345 passed.

Synthetic check: a 12 cm sunken band 3–12 cm ahead of the body loses its
hazard cells within 15 cm of the body and keeps them beyond 45 cm sideways.

Result of the retrain (both epoch 11, scored on the hazard-margin
validation labels):

| Validation, `labels_grid160x320w4r4e` | footprint cull (`w4r4c`) | hazard margin (`w4r4e`) |
|---|---|---|
| labelled-cells mIoU | 0.533 | 0.534 |
| hazard P / R / F1 | 0.428 / 0.318 / 0.365 | 0.479 / 0.302 / 0.371 |
| coverage P / R | 0.829 / 0.889 | 0.831 / 0.886 |
| hazard cells predicted in the 15 cm ring (164 frames; teacher 94) | 2 418 | 0 |
| ring beside chassis: predicted obstacle (teacher 1.73 %) | 2.01 % | 2.16 % |

The strip is gone at the source: the new model predicts no drop in the
ring without any runtime pad, hazard precision rises 5 points at a small
recall cost, everything else is within noise. The 94 teacher cells left in
the ring come from the splat and the fused local-drop rule, not from the
camera evidence.

Deployed 2026-10-20 09:37 as `weights/bev_student_c160e_fp16.engine`
(66 MB, ONNX parity agree 1.0000; previous env backed up). Recorded-pair
benchmark 69 / 91 ms p50 / p95 at a 10 Hz pace; live 9.8 Hz (264 polls in
10 s), preprocess 35 ms + infer 23 ms. The live grid shows no orange beside
the chassis and 13 hazard cells in total. **The runtime pad is still doing
work**: `edge_hazard_cells` is ~210 per frame on the live scene even with
the new engine, although the same model predicts none in the validation
ring. The live scene (robot parked on the charger, different lighting)
is outside the drives, so keep `ego.hazard_pad_m` on; it is not a
cosmetic knob yet. Previous engines `bev_student_c160_fp16.engine` and
`bev_student_c160fc_fp16.engine` remain on the robot as fallbacks.

## Live deployment of epoch 11 (2026-10-19, while training continued)

The epoch-11 checkpoint (balanced score 0.525: val mIoU 0.557, wall IoU
0.645, obstacle recall 0.772, hazard F1 0.366, coverage P/R 0.86/0.85) was
snapshotted to `deploy-e11/best_e11.pt`, exported on the CPU
(`bev_student_e11.onnx`, 498 MB, ONNX parity: semantic argmax agreement
0.9999, cosine 1.0) and built on the Thor with `trtexec --fp16`
(`weights/bev_student_e11_fp16.engine`, 255 MB, TensorRT 10.16.2, ~4 min).
`/etc/bebop/vision.env` now points `--bev-model` at it (previous env backed
up as `vision.env.bak-<stamp>`); `bebop-vision.service` restarted cleanly and
publishes a 360×640 grid over 6 m × 4 m. The robot's test suite passed after
the file sync (303 passed).

Measured speed on the Thor, with the recorder and WebRTC streams running:

| Stage | Time per tick |
|---|---|
| trtexec GPU compute, median / p95 | 70 / 82 ms |
| Recorded-pair benchmark, 10 Hz pace, JPEG decode → postprocess, p50 / p95 | 152 / 184 ms (6.5 Hz) |
| Live service: preprocess + infer | 41 + 66 ms |
| Live service rate (its own 5 s average) | **5.0 Hz**, ~200 ms per tick |

So the matched-grid model runs at about half the 10 Hz target. The engine
alone takes two-thirds of the budget (66 ms vs 2.4 ms for the DINO engine);
the remaining ~90 ms is numpy postprocessing over 230k cells (softmax,
sigmoids, dilation), the 360×640 render and JPEG encode. Moving the
softmax/argmax/sigmoid onto the GPU (the outputs are already torch tensors)
and decoding the colour JPEGs at half resolution are the obvious savings,
but 10 Hz is not reachable with this engine on the Thor; 8 Hz is the
realistic ceiling. The BEV is display/telemetry only (goal-drive uses the
navd model), so the lower rate is not a safety issue.

Deploying also exposed that the committed `main.py` passed
`args.bev_hazard_threshold` without defining the `--bev-hazard-threshold`
argument (the robot had an older `main.py` without the call); the flag is
now defined (default 0.5).

## Cleanup and stopped work

The previous queue was terminated during the 424×240 B5 control run. The
848×480 / 120×60-output B5 run had already completed training/evaluation/export.

Retained reference runs and their **best** checkpoints:

- `dino-vits16-full`: original DINO baseline.
- `exp-base2-flip-tvbal`: currently deployed clean-label DINO model.
- `exp-base2-b5-480-lift5cm`: completed B5 comparison.

Removed obsolete experiments, smoke checkpoints, duplicate last checkpoints,
TensorBoard/log directories for discarded runs, and untrained benchmark
weights/ONNX files. Compact configs, metrics and benchmark JSON results were
archived under `runs/bev-student/archive/<old-run-name>/` before deletion.
The cleanup manifest is `archive/cleanup-20261019.json`.

The main cleanup reduced the run tree from **19.32 GB to 2.21 GB**, reclaiming
**17.11 GB** (decimal), before this new experiment's artifacts. Temporary
benchmark engines/data on the Thor freed another **5.4 GiB**. Local temporary
benchmark data/logs and the new smoke-test checkpoints were also removed;
the smoke-test summary is archived. Model source weights and datasets were
retained.

Verification: **358 passed, 1 skipped**, including the single-head model (forward, checkpoint, export, loss, eval, runtime), the reflection rule, the label crop tool, the learned chassis footprint (derivation on synthetic persistence maps, rasterisation, config and runtime wiring, label rewrite), the rectangular-cell runtime geometry (per-axis ego/inflation, JSON payload, render), rectangular metric
projection/ego/drop-hazard tests, the footprint splat (fills sampling gaps,
keeps a vertical face thin, bounded patch size), the width override, the
rectangular near-body cull and its YAML validation, the sparse-plane-fit
regression, matching-input/output training and checkpoint
reload, and ONNX equivalence of the TensorRT-compatible spatial SegFormer MLP.
