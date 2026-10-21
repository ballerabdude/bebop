#!/usr/bin/env bash
# Matched-grid trial: explicit camera-input and BEV-output tensor shapes on
# separate native-depth labels.  Relabels in parallel (one worker per
# drive), then trains, evaluates and exports.
#
# Environment overrides (defaults = the 4 m-wide footprint-splat B5 run):
#   RUN      run directory                (runs/bev-student/exp-base2-b5-grid360x640w4)
#   ENCODER  segformer-b5 | dinov3        (segformer-b5)
#   HEIGHT / WIDTH_PX  camera input size  (360 / 640; DINOv3 wants multiples of 16)
#   EPOCHS / STAGE1    schedule           (40 / 5)
#   BATCH / ACCUM      microbatch, accumulation (1 / 8)
#   NUM_WORKERS  dataloader workers       (4)
#   SKIP_RELABEL  1 = labels already complete; go straight to training (0)
#   RELABEL_ONLY  1 = stop after the labels are complete (0)
#   EXTRA_TRAIN_ARGS  appended to train_bev_student.py (e.g. --single-head)
#   LABELS   label subdir per session     (labels_grid360x640w4; must start with labels_grid)
#   WIDTH    BEV width in metres          (4.0)
#   BEV_ROWS / BEV_COLS / BEV_RANGE  output rows, columns and forward range (360 / 640 / 6.0; the
#            labels must already have this grid, e.g. tools/bev_crop_labels.py)
#   SPLAT    1 = footprint splat labels   (1)
#   WORKERS  parallel relabel processes   (5; ~5.6 GB VRAM each)
#   CULL_RECT  x0,x1,y0,y1 body-frame chassis+margin rectangle replacing the
#              0.55 m dead disc (unset = disc)
#   EGO_RECT   x0,x1,y0,y1 physical chassis rectangle for the ego layer
#              (unset = rig config)
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
run=${RUN:-runs/bev-student/exp-base2-b5-grid360x640w4}
labels=${LABELS:-labels_grid360x640w4}
encoder=${ENCODER:-segformer-b5}
height=${HEIGHT:-360}
width_px=${WIDTH_PX:-640}
epochs=${EPOCHS:-40}
stage1=${STAGE1:-5}
batch=${BATCH:-1}
accum=${ACCUM:-8}
num_workers=${NUM_WORKERS:-4}
skip_relabel=${SKIP_RELABEL:-0}
relabel_only=${RELABEL_ONLY:-0}
extra_train_args=${EXTRA_TRAIN_ARGS:-}
width=${WIDTH:-4.0}
bev_rows=${BEV_ROWS:-360}
bev_cols=${BEV_COLS:-640}
bev_range=${BEV_RANGE:-6.0}
splat=${SPLAT:-1}
workers=${WORKERS:-5}
cull_rect=${CULL_RECT:-}
ego_rect=${EGO_RECT:-}
mkdir -p "$run"
exec 9> "$run/.pipeline.lock"
flock -n 9 || { echo "matched-grid pipeline already running"; exit 1; }
if [[ -e "$run/checkpoints/last.pt" || -e "$run/checkpoints/best.pt" ]]; then
    echo "Existing training checkpoints in $run; resume training explicitly."
    exit 1
fi
echo "$$" > "$run/pipeline.pid"
exec > >(tee -a "$run/pipeline.log") 2>&1
py=.venv/bin/python
relabel_args=(--rows "$bev_rows" --cols "$bev_cols" --width-m "$width" --range-m "$bev_range" --out-labels "$labels" --resume --stride 3)
[[ "$splat" == 1 ]] && relabel_args+=(--splat)
[[ -n "$cull_rect" ]] && relabel_args+=(--cull-rect "$cull_rect")
[[ -n "$ego_rect" ]] && relabel_args+=(--ego-rect "$ego_rect")
if [[ "$skip_relabel" == 1 ]]; then
echo "PHASE: relabeling skipped (SKIP_RELABEL=1); using existing $labels"
else
echo "PHASE: native-depth relabeling (${bev_rows}x${bev_cols} over ${bev_range} x ${width} m, splat=$splat, cull_rect=${cull_rect:-config}, ego_rect=${ego_rect:-config}) x$workers workers"
mkdir -p "$run/relabel"
"$py" - <<'PY' | xargs -P "$workers" -I{} sh -c \
    "$py -u tools/bev_relabel_grid.py ${relabel_args[*]} --sessions {} > '$run/relabel/{}.log' 2>&1 || { echo 'RELABEL FAILED: {}'; exit 1; }"
from bebop_vision.bev_student_data import DEFAULT_ROOT, session_dirs
# Largest drives first so the long pole starts immediately.
for s in sorted(session_dirs(DEFAULT_ROOT), key=lambda p: -sum(1 for _ in (p / "depth_far").glob("*.png"))):
    print(s.name)
PY
if grep -L GRID_LABELS_DONE "$run"/relabel/*.log | grep -q .; then
    echo "relabel incomplete; see $run/relabel/"; exit 1
fi
if [[ "$relabel_only" == 1 ]]; then
    echo "MATCHED_GRID_RELABEL_DONE $labels"; exit 0
fi
fi
echo "PHASE: $encoder training (${height}x${width_px} input, ${bev_rows}x${bev_cols} output over ${bev_range} x ${width} m, $epochs epochs)"
"$py" -u train_bev_student.py \
    --encoder "$encoder" --height "$height" --width "$width_px" \
    --bev-rows "$bev_rows" --bev-cols "$bev_cols" --bev-width "$width" --bev-range "$bev_range" --full-res-lift \
    --run "$run" --labels-subdir "$labels" \
    --epochs "$epochs" --stage1-epochs "$stage1" --batch-size "$batch" --accum-steps "$accum" --num-workers "$num_workers" \
    --hazard-weight 3 --hazard-tversky-weight 0.5 \
    --hazard-tversky-alpha 0.5 --hazard-tversky-beta 0.5 --score balanced \
    $extra_train_args \
    | tee "$run/train.log"
echo "PHASE: evaluation and export"
"$py" -u tools/bev_eval_run.py --run "$run" --labels-subdir "$labels" \
    --batch-size 1 > "$run/eval.log" 2>&1
"$py" -u tools/bev_eval_diagnostics.py --run "$run" --labels-subdir "$labels" \
    --batch-size 1 --out "$run/eval/diagnostics" > "$run/diagnostics.log" 2>&1
"$py" -u tools/export_bev_student.py --checkpoint "$run/checkpoints/best.pt" \
    > "$run/export.log" 2>&1
echo "MATCHED_GRID_TRAIN_EVAL_EXPORT_DONE $run"
