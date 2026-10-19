# Learned master-BEV student (SAM 3.1 teacher → DINOv3 student)

Status: **M1 implemented (2026-10-18)** — label extraction, SAM 3.1 ×
depth label fusion, review artifacts, dataset index/split/report, and the
synthetic test suite.  No training yet; M2 is the student model.

This is the learned semantic complement to the geometric master BEV in
`docs/master-bev.md`.  The geometric grid stays as the fallback and as a
verification source.  SAM 3.1 is offline-only; the deployed runtime will be
a DINOv3 ViT-S/16 (or ConvNeXt-T fallback) student exported to ONNX/TensorRT.

## 1. Data / run layout

All large artifacts are gitignored (`bebop-vision/datasets/`,
`bebop-vision/runs/`).  The committed files are the tools, package modules,
tests and this doc.

```text
bebop-vision/
  datasets/bev-student/
    raw/mcap/<session>.mcap           original robot MCAPs (never modified)
    sessions/<session>/
      color_near/<stamp>.jpg          1280x800, as recorded
      color_far/<stamp>.jpg
      depth_near/<stamp>.png          uint16 mm, lossless
      depth_far/<stamp>.png
      labels/<stamp>.npz              semantic/ego/coverage
      labels/diagnostics.jsonl        per-label plane + class census
      index.jsonl                     one row per extracted 10 Hz frame
      calib.json                      recorded + config intrinsics/extrinsics
      meta.json                       source size, tick rate, extracted rate
      preview/                        review artifacts (see §5)
    splits/drive_split.json           train/val by drive, never by frame
    meta/label_stats.json             aggregate report
  runs/bev-student/<run_id>/
    tb/                               TensorBoard event files
    checkpoints/
    export/                           ONNX / TensorRT / metadata
    train_log.jsonl
    config.json
```

`<stamp>` is the source MCAP `log_time` in nanoseconds, zero-padded to 20
digits.  The extractor snaps each source tick to a 10 Hz target grid (nearest
real frame; never interpolated), because the recorder writes at the
frame-set rate (~21-24 Hz on the 2026-10-18 sessions), not at exactly 10 Hz.

The six 2026-10-18 sessions unpacked on 2026-10-18:

| Session | Duration | Source ticks | Source Hz | 10 Hz frames |
|---|---:|---:|---:|---:|
| `navd_session_20261018_191614` | 195.1 s | 4,371 | 22.39 | 1,920 |
| `navd_session_20261018_191941` | 67.0 s | 1,524 | 22.73 | 633 |
| `navd_session_20261018_192050` | 65.2 s | 1,576 | 24.17 | 644 |
| `navd_session_20261018_193224` | 122.8 s | 2,925 | 23.82 | 1,194 |
| `navd_session_20261018_193619` | 157.4 s | 3,359 | 21.34 | 1,415 |
| `navd_session_20261018_193932` | 53.8 s | 1,157 | 21.47 | 492 |
| **Total** | **661 s** | **14,912** | | **6,298** |

## 2. Grid convention (unchanged from master BEV)

- Body frame: x forward, y left, z up; origin on the floor under the robot.
- `range_m=6.0`, `width_m=3.0`, `cell_m=0.05` → `rows=120`, `cols=60`.
- `row = (range_m - x) / cell_m`; `col = (y + width_m/2) / cell_m`.
- Row 0 is the far edge; col 0 is the right edge.
- Mounts/intrinsics are the same files the geometric BEV uses:
  `config/orbbec_rig.yaml`, `config/orbbec_intrinsics_<serial>.json`.

Label ids: `0 floor`, `1 wall`, `2 obstacle`, `3 hazard`, `255 ignore`.
`ego` and `coverage` are separate bool layers.  Ego is the fixed physical
chassis rectangle (`x=0.0–0.55 m` visible forward, `y=±0.22 m`), **not** the
fan-shaped near self-mask: the self-mask still excludes self pixels from
scene evidence, but it includes the mast/rails and nearby floor and must
not define the ego footprint.  The planner-facing semantic ids are **not**
`master_bev.FREE/OCCUPIED/HAZARD/INFLATED`; M5 maps them.

## 3. Teacher: SAM 3.1 × depth

Prompts (curated; `bebop_vision/bev_student_labels.py`):

| Class | Prompts |
|---|---|
| floor | floor, carpet, rug, ground |
| wall | wall, door, door frame, pillar |
| obstacle | box, bag, chair, table, person, cable, trash can, shelf, cardboard box, bed, sofa, couch, cabinet, dresser, wardrobe, desk, counter, appliance |
| hazard support | stairs, staircase, step, hole, ledge, drop |

The furniture prompts (`bed`, `sofa`, `couch`, `cabinet`, `dresser`,
`wardrobe`, `desk`, `counter`, `appliance`) extend the brief's curated
box/chair/table set: the first bedroom review showed beds/dressers were
otherwise unlabeled, and these home-tour drives are dominated by furniture.

Per labeled keyframe and camera:

1. Deproject every sampled depth pixel with the rig mount/intrinsics
   (`master_bev` convention; near stride 4×2, far 4×1).
2. Project the 3D point into the **color** frame through the cached
   depth→color extrinsic (`color_to_depth_transform`, millimetres) and
   sample the SAM category confidence at that pixel.  This is exact, not
   the zero-baseline approximation the old `fuse_navd_labels.py` used.
3. Fit the camera's floor plane on a near-floor band (RANSAC, then refine),
   with a joint-plane fallback and 0.5× weight for a failed camera.
4. Accumulate weighted evidence per cell:
   - floor: SAM floor/ground AND depth on the local floor plane; plus a
     weak depth-only vote (`depth_floor_weight=0.25`) where SAM returned no
     mask at all.
   - wall: SAM wall/door/pillar AND point height ≥ 0.8 m; compact, short
     components (<0.6 × 0.6 m and <2 m tall) are reclassified as obstacles
     (bookshelf edges), tall pillars stay walls.
   - obstacle: SAM object prompt AND point height 0.03–1.5 m.
   - hazard: depth below the local floor plane by >5 cm (range-scaled),
     gated by SAM floor/ground or stairs/hole on the same point; local
     floor-height drops are opt-in and off by default indoors.  Points
     below the plane under furniture are suppressed by an obstacle-adjacency
     check.
   - stairs/step/hole prompts are **never floor**: at/above the local floor
     plane they become obstacles, below it hazards.  This catches ascending
     stairs (e.g. the 193224 hallway) as well as drops, independent of small
     plane-fit variation.
5. Fuse both cameras and resolve with priority
   `hazard > wall > obstacle > floor`; ego and unobserved cells become 255.
6. `coverage` is any valid depth return from either camera (ego cells are
   excluded from coverage and marked `ego=True`).

Effective label rate: ~9-10 Hz extraction × `--stride 3` = ~3.0-3.3 Hz
(the exact per-session rate is in `preview/stats.json`).  Adjacent 10 Hz
frames are nearly identical; this is intentional.

## 4. Commands

From `bebop-vision/` unless noted.

```bash
# One-time venv addition on the workstation: PyAV for H.265 decode.
.venv/bin/pip install 'av==12.3.0'

# 1. Download the sessions from the robot (organized raw dir).
mkdir -p datasets/bev-student/raw/mcap
for s in 20261018_191614 20261018_191941 20261018_192050 \
         20261018_193224 20261018_193619 20261018_193932; do
  curl -fSsS -o datasets/bev-student/raw/mcap/navd_session_$s.mcap \
    http://bebop.local:9090/captures/dl/navd_session_$s.mcap
done

# 2. MCAP -> 10 Hz frames + index (both cameras, depth PNG).
.venv/bin/python tools/bev_student_dataset.py extract --sample-hz 10

# 3. SAM 3.1 labels + previews (GPU; ~1.5 s per keyframe pair).
.venv/bin/python tools/bev_student_labels.py --stride 3 --preview-every 10

# 4. Validate, split by drive, report.
.venv/bin/python tools/bev_student_dataset.py index
.venv/bin/python tools/bev_student_dataset.py split \
    --val-sessions navd_session_20261018_193932
.venv/bin/python tools/bev_student_dataset.py report

# 5. A TensorBoard run directory (M2 fills it).
.venv/bin/python tools/bev_student_dataset.py init-run --run-id dino-vits16-m1
tensorboard --logdir runs/bev-student --port 6006
```

`tools/mcap_extract.py` also has the legacy `navd-v0` layout as its default;
the bev-student path is `--layout bev-student` or the dataset tool above.
Use `--session NAME` to process one session; `--resume` skips existing
label npz files; `--max-frames` is for a quick review pass.

## 5. Review artifacts

Per session under `sessions/<session>/preview/`:

- `<stamp>_color_masks.jpg` — near|far color with SAM masks overlaid
  (green floor, blue wall, red obstacle; the depth-gated hazard is shown
  in the BEV label, not the color overlay).
- `<stamp>_bev_labels.png` — 120×60 semantic grid, mirrored horizontally
  like `render_bev`, with class percentages and the ego footprint outlined.
- `contact_sheet.png` — grid of BEV label tiles.
- `contact_sheet_colors.jpg` — grid of color-mask tiles.
- `stats.json` — frames, label rate, effective label Hz, per-class cells
  and fractions, coverage/ego fractions, plane-ok fraction.

Do **not** start training until a drive with a wall, a box/chair and a
step/hazard has been reviewed.  The 2026-10-18 home-tour drives contain
walls, doors, chairs, tables, beds, cabinets, sofas and a staircase.  The
living-room drive `navd_session_20261018_193932` includes the staircase
frame `01792366817349875004` (explicit `_color_masks.jpg` +
`_bev_labels.png` pair in that session's `preview/`); use it for the
hazard review.  A short drive with a cardboard box in the path is still
worth recording before M3 sign-off.

## 6. M1 acceptance checks

- [x] `tools/mcap_extract.py` / `tools/bev_student_dataset.py extract`
      produces the 10 Hz layout from the six MCAPs.
- [x] `tools/bev_student_labels.py` writes `semantic/ego/coverage` npz +
      previews; effective label rate is documented.
- [x] `tools/bev_student_dataset.py index/split/report` validates shapes,
      dtypes, class ids, ego/coverage invariants and splits whole drives.
- [x] Synthetic label projection (floor/box/step-down) and conflict-priority
      tests pass without SAM or hardware.
- [ ] Human review of a full drive's contact sheets (this doc + the
      `preview/` artifacts are the review material).
- [x] A drive with a visible step/hazard is available (staircase in
      `navd_session_20261018_193932`); tune the hazard gate after review.

## 6b. M1 verification (2026-10-18)

- `cd bebop-vision && .venv/bin/python -m pytest tests/ -q` →
  **324 passed, 1 skipped** (13 new in `tests/test_bev_student.py`).
- Extraction → **6,298 frames** across six drives (table in §1); all
  `index.jsonl` rows validate, no missing color/depth files.
- DINOv3 ViT-S/16 fetched from the gated HF repo
  (`facebook/dinov3-vits16-pretrain-lvd1689m`, 21.6 M params) into
  `weights/dinov3-vits16/`; verified at 240×424 (395 patch tokens) and
  480×848 (1,595 tokens).
- Two full drives are labeled: `navd_session_20261018_191614` (640
  keyframes, 3.28 Hz effective) and `navd_session_20261018_193932` (164,
  3.06 Hz effective); a killed exploratory pass left 4 valid labels in
  `navd_session_20261018_191941`. Aggregate: **808 labeled frames**,
  floor 21.8%, wall 4.2%, obstacle 3.6%, hazard 0.7%, ignore 69.7%.
- Split: **val = `navd_session_20261018_193932`** (staircase living room),
  train = `navd_session_20261018_191614` + the 4 exploratory frames in
  `navd_session_20261018_191941`. Re-run `split` after labeling more.
- The other four drives are extracted and ready; run the same label
  command after reviewing the two full drives. `preview/stats.json`
  records the exact rate per drive.

## 7. M2+ plan (not implemented yet)

- `bebop_vision/bev_student.py`: DINOv3 ViT-S/16 (~21 M) shared encoder,
  4th depth channel, multi-layer patch tokens → FPN 60×106×128, learned
  near/far embedding, multi-plane IPM at z = 0.0/0.3/0.8/1.5 m, fixed
  bilinear LUT (`Gather`/`Mul`/`Add`, no `grid_sample`), U-Net decoder to
  120×60, heads semantic[4], ego[1], coverage[1].
- `train_bev_student.py`: class-weighted CE/focal + Dice for wall/hazard,
  BCE ego/coverage, AMP/AdamW, stage 1 frozen encoder / stage 2 low-LR
  unfreeze, drive-level split, TensorBoard scalars/images, best checkpoint
  by safety-weighted score (obstacle/hazard recall + wall IoU).
- `tools/export_bev_student.py`: whole-graph ONNX (opset 17/18, static
  `[2,4,240,424]`), ONNX parity, TensorRT FP16 engine + metadata.
- `tools/bench_bev_student.py`: PyTorch/ONNX/TRT parity and Thor latency
  (target ≤20 ms/camera, ≤40 ms for the pair at 10 Hz).
- Live integration: `main.py --bev-model`, `bev_service` model producer,
  `BevGrid` semantic/wall/ego RLE, `/bev` JSON, render palette, planner
  mapping, geometric fallback.
- DINOv3 ViT-L/16 is an optional feature-distillation teacher only; never
  deployed.

## 8. Licenses

- SAM 3.x: Meta SAM License; commercial use/fine-tuning permitted,
  redistributed SAM materials/derivatives must carry the license.  Teacher
  side only.
- DINOv3: DINOv3 License; commercial use permitted with the license notice
  on redistribution.  `bebop-vision/weights/dinov3-vits16/` was fetched from
  the gated `facebook/dinov3-vits16-pretrain-lvd1689m` repo with the
  operator's HF read token; keep `LICENSE.md` with the weights.
- Do not add SAM/DINOv3 weights to git or release artifacts without the
  license files.
