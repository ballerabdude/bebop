# SegFormer-B5 and higher-resolution BEV — 2026-10-19

**Update:** the operator stopped this queue and selected the
[exact 360×640 input/output grid experiment](bev-matched-grid.md). Discarded
run metrics and benchmark JSON files now live under
`bebop-vision/runs/bev-student/archive/<old-run-name>/`; their large checkpoint
and benchmark binaries have been removed. The completed high-resolution B5
run's best checkpoint is retained.

Follow-up to [the clean-label experiments](bev-experiments.md). The operator
reviewed the deployed DINOv3 base2 model and requested a B5 / resolution trial.

## What resolution means here

Sizes in this table are **width × height** (CLI arguments are named explicitly).

| Stage | Current student | High-resolution experiment |
|---|---|---|
| Per-camera RGB-D input | 424 × 240 | 848 × 480 |
| Native encoder features | DINO: 26 × 15 tokens, stride 16 | SegFormer: four scales, finest 212 × 120, stride 4 |
| Fused image feature map | 106 × 60, interpolated from DINO tokens | 212 × 120, fused from SegFormer scales |
| BEV lift | 30 × 60, 10 cm cells | 60 × 120, 5 cm cells |
| Semantic output | 60 × 120, 5 cm cells | 60 × 120, 5 cm cells |
| Physical map | 3 m wide × 6 m forward | 3 m wide × 6 m forward |

There are two independent switches: `--height/--width` and `--full-res-lift`.
The latter removes the coarse 10 cm projection bottleneck; it does not change
the published BEV dimensions. Increasing input size requires rebuilding the
calibrated RGB/depth sampling LUTs, retraining, and exporting a new engine.
Changing only the ONNX input shape or resizing a displayed map is insufficient.

A **2.5 cm output** would be 120 × 240 cells (four times as many), with the
same physical extent. That requires regenerating teacher labels at that cell
size and generalizing the remaining fixed-grid training/evaluation code. It
is a separate follow-up, not an option implemented in this experiment. Start
by improving the information feeding the existing 5 cm output.

## Architecture and compatibility

- `--encoder segformer-b5`: the MiT-B5 encoder from
  `nvidia/segformer-b5-finetuned-ade-640-640`, adapted to RGB-D with a
  zero-initialized depth channel. Its ADE20K image decoder is discarded;
  the existing learned BEV fusion/semantic/ego/coverage heads are trained.
- **82,304,550 parameters**, versus 22,618,790 for the current DINO student.
- Four native feature scales are projected to 128 channels and fused, then
  passed through the calibrated LUT and BEV decoder.
- `--encoder dinov3` remains available at either input/lift resolution, for
  a resolution-only control experiment.
- New checkpoints contain the backbone config, architecture, input size,
  and training LUT metadata/buffers. Evaluation/export reconstruct them
  without downloading pretrained weights. Old DINO checkpoints still load.
- The runtime reads the static engine input size and checks metadata rather
  than always resizing to 424 × 240. This runtime change is required before
  deploying a high-resolution engine.
- BEV hazard prediction remains learned; the ego mask is the chassis rectangle.

## Timing procedure

Benchmark fixtures (gitignored under `bebop-vision/runs/bev-student/`):

- `benchmark-b5-240/`
- `benchmark-b5-360-lift5cm/` (640 × 360 intermediate resolution)
- `benchmark-b5-480/`
- `benchmark-b5-480-lift5cm/`

These have pretrained encoders and **untrained BEV heads**. They are only
latency/export fixtures, never accuracy candidates or deployable models.
FP32 ONNX exports passed parity: maximum absolute error < 7e-6,
semantic argmax agreement 1.0 on the export test inputs.

### TensorRT export gotcha (resolved by a spatial MLP)

On Thor / TensorRT 10.16.2, the initial standard SegFormer graph gave badly
incorrect results in **FP32 as well as FP16**. Neither eager attention,
FP32 normalization/softmax, basic ONNX constant folding, nor builder
optimization level 0 fixed it. Input preprocessing matched across machines
apart from small JPEG decoding rounding differences (<= 1 RGB byte).

Intermediate-output bisection isolated the first error to stage 1 block 0's
MLP: exposing its first Linear output restored correct results; exposing
the following depthwise Conv output did not. The default compiled block's
maximum FP32 error was 1.872, versus ~7.4e-6 when that boundary was exposed.

`tools/export_bev_student.py` now replaces SegFormer's Mix-FFN with an
equivalent **spatial implementation** during export: reshape tokens to BCHW,
1×1 Conv using the first Linear's weights, depthwise Conv, GELU, 1×1 Conv
using the second Linear's weights, then reshape back to tokens. Training
and checkpoint weights are unchanged. CPU tests check equivalence to the
original module and ONNX parity. The first quick-build FP16 engine with
this rewrite gave semantic agreement 0.9971 and max logit error 0.0105 on
the saved pair, instead of agreement ~0.67 / error ~1.54.

Use the **current exporter** (metadata `spatial_segformer_mlp: true`). The
earlier `bev_student.onnx` / `bev_student.engine` benchmark fixtures are
retained for diagnosis only. Corrected benchmark artifacts use
`bev_student_spatial.onnx` and `bev_student_spatial_opt3.engine`.

Build on Thor (the exported graph is static; do not pass `--shapes`):

```bash
trtexec --onnx=bev_student.onnx --saveEngine=bev_student.engine --fp16
```

Place `bev_student.json` beside the engine. Measure JPEG decoding, RGB-D
resize/normalization, transfers, inference, output transfers and postprocess:

```bash
.venv/bin/python tools/bev_benchmark_engine.py \
  --engine /path/to/bev_student.engine \
  --frames /path/to/saved-pair --out benchmark.json --hz 10
```

Saved pair: `near.jpg`, `far.jpg`, `near.png`, `far.png`; depth is uint16
millimetres. `--reference reference.npz` compares raw TensorRT outputs to
ONNX outputs from the same pair. `--runtime-file` allows testing a staged
candidate runtime without installing it. The tool opens no cameras.

Default: 20 warmups, then 200 paced ticks. Report p50/p95/p99/max total
latency and ticks exceeding 100 ms, not only `trtexec` kernel timing. A
recorded-pair benchmark demonstrates compute feasibility; live sustained
10 Hz still requires validation after training. Recheck trained-engine
numerics and timing before deployment.

The recorded pair is from `navd_session_20261018_193932`, stamp
`1792366772240729308`. The existing live DINO service ran concurrently, with
the operator's permission, throughout timing. Baseline with the same saved
pair: 47.859 ms median, 50.282 ms p95, zero ticks over 100 ms out of 200.

### Corrected FP16 results (default builder optimization level 3)

| Architecture / camera input | Lift cell | Total p50 | Total p95 | Total p99 | >100 ms / 200 |
|---|---:|---:|---:|---:|---:|
| Deployed DINO / 424 × 240 | 10 cm | 47.859 ms | 50.282 ms | 51.033 ms | 0 |
| B5 / 424 × 240 | 10 cm | 55.093 ms | 62.683 ms | 67.007 ms | 0 |
| B5 / 640 × 360 | 5 cm | 69.953 ms | 78.562 ms | 93.661 ms | 0 |
| B5 / 848 × 480 | 5 cm | 86.506 ms | 101.222 ms | 118.907 ms | 13 |

The corrected B5 graphs have semantic argmax agreement **0.99833 / 0.99486 /
0.99736**, respectively, against the original ONNX graph on the recorded
pair; maximum logit differences across all heads are <= 0.0129. That comparison
also includes the small cross-platform JPEG rounding differences.

640 × 360 is the strongest current 10 Hz timing candidate: 2.26× the input
pixels, full-resolution 5 cm projection, and no missed deadlines in this
short test. 848 × 480 has substantially less headroom and missed 6.5% of
deadlines under this concurrent workload. These are feasibility results
from pretrained-encoder / untrained-head fixtures, not live deployment or
accuracy validation of the trained B5 checkpoint.

Per-run results are saved as `benchmark_spatial_opt3.json` in the local
benchmark directories; the baseline is `benchmark-b5-240/baseline_thor.json`.
Initial numerically incorrect results are named `benchmark_invalid_original.json`
and must not be used as deployment evidence.

## Training recipe

Use the same six drives, preserved original labels, `labels_base2`, split,
flip augmentation, and balanced hazard loss as the deployed winner:

```bash
cd bebop-vision
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python train_bev_student.py \
  --encoder segformer-b5 --height 480 --width 848 --full-res-lift \
  --run runs/bev-student/exp-base2-b5-480-lift5cm \
  --labels-subdir labels_base2 --epochs 40 --stage1-epochs 5 \
  --batch-size 4 --accum-steps 2 --num-workers 4 \
  --hazard-weight 3 --hazard-tversky-weight 0.5 \
  --hazard-tversky-alpha 0.5 --hazard-tversky-beta 0.5 --score balanced
```

The effective training batch is 8 pairs. A full-resolution, full-lift,
unfrozen-backbone smoke run passed on the RTX 5090, including an incomplete
accumulation group and validation. For the architecture-only comparison use
`--height 240 --width 424`, omit `--full-res-lift`, and use a separate run.
TensorBoard uses the existing `runs/bev-student` log tree.

Evaluate wall/obstacle IoU, hazard precision/recall/F1, coverage, and visual
boundary quality against `exp-base2-flip-tvbal`. Higher capacity is not
evidence of better generalization: this remains a small dataset with few
real drop examples. The repeatedly used validation drive is a development
benchmark, not an untouched final test set.

### Training status snapshot

At the operator's status request, the 848 × 480 / 5 cm-lift run was in epoch
37 of 40 (zero-based epoch 36); epochs took about 81 seconds. Its best
balanced-score checkpoint was zero-based epoch 23:

| Raw validation metric | Deployed DINO | B5 best so far |
|---|---:|---:|
| mIoU | 0.5451 | 0.5474 |
| Floor IoU | 0.8351 | 0.8507 |
| Wall IoU | 0.6291 | 0.6336 |
| Obstacle IoU | 0.5222 | 0.5319 |
| Hazard IoU | 0.1939 | 0.1732 |
| Hazard precision | 0.4387 | 0.3948 |
| Hazard recall | 0.2579 | 0.2358 |
| Hazard F1 | 0.3248 | 0.2953 |
| Coverage precision | 0.8162 | 0.8672 |
| Coverage recall | 0.9116 | 0.9089 |
| Balanced score | 0.5084 | 0.5043 |

This is a trade-off, not a clear improvement over the deployed model.
The 424 × 240 B5 control run is queued after high-resolution training,
evaluation, and export. Both are 40 epochs; logs and TensorBoard events are
in their run directories. Final live-masked diagnostics are generated after
training and are needed before deciding which checkpoint to deploy.

The operator also proposed increasing the *output* grid. A recommended next
experiment is **240 rows × 120 columns at 2.5 cm/cell**, supervised with
newly generated labels while preserving `labels/` and `labels_base2/`.
Matching the rectangular camera image dimensions directly would not preserve
square metric cells over the 6 m × 3 m BEV. This output-grid experiment has
not yet been implemented or started.
