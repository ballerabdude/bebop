# Master BEV — one fused grid from both depth cameras

Status: **step 1 implemented (2026-10-18)** — the grid, the live worker and
the operator feeds. No planner, no drive commands, no object semantics yet.

The grid is deliberately **not square**: the near camera sees the floor close
to the robot, the far camera sees much further ahead than the robot is wide.
Default: **6.0 m forward × 3.0 m wide at 5 cm = 120 × 60 cells**.

## Why a new builder

The old navd BEV (`docs/navd.md` §6.3) did read both cameras, but its result
was near-dominant:

- the grid was only 3 × 3 m, and the near camera was authoritative below
  1.5 m, so the far camera only owned the top half;
- each camera fitted its own floor plane, so a small far-mount error changed
  the ground and turned far points into noise;
- fusion was `np.maximum` of class ids plus a hard authority switch — one
  stray point could create an obstacle;
- there was no source/confidence layer, so a missing far contribution was
  invisible.

The master builder keeps the proven geometry (deprojection, mount rotation,
batched RANSAC, synthetic tests) and replaces the fusion.

## Pipeline

Per tick:

1. **Prepare per camera** (`MasterBevBuilder.prepare`): depth → optical
   points → body frame using the measured mount (height, pitch, yaw, optional
   x/y offset).  Invalid/out-of-range samples are dropped, then the body-frame
   dead disc (`min_range_m`) and self-view box are applied.  No floor decision
   is made here.
2. **Per-camera floor plane** (`_fit_camera_plane`): a RANSAC plane fitted on
   points within `floor_band_m` of the body floor (default 0.40 m), gated to
   a normal within `plane_normal_deg` of vertical (default 10°) and an origin
   height inside `[-0.20, +0.20] m`; then refined on every point close to the
   fitted plane.  This keeps walls, desks and tall boxes out of the sample.
   A joint plane is still fitted for diagnostics.  A camera whose own fit
   fails uses the joint plane at `fallback_weight` (0.5) and contributes no
   drop hazards; `plane_ok` is false for it.
3. **Per-camera evidence**: points are classified against the joint plane:
   - `occupied`: height `[occ_min_m, occ_max_m]` (default 3–30 cm), with the
     lower bound raised by `noise_scale × range` at distance;
   - `free`: floor within 2 cm below the lower bound;
   - `hazard`: height below `-max(hazard_drop_m, noise_scale × range)`.
     Adjacent-cell drop detection (stairs) is opt-in via
     `enable_drop_hazard`; it is noisy on cluttered floors, so it is off by
     default and points genuinely below the floor still mark HAZARD;
   - overhang above `occ_max_m` is ignored.
4. **Range weighting**: each point carries a confidence from its **forward
   distance**, not Euclidean camera range (the near camera is accurate a
   metre ahead even though its mount height makes the Euclidean range
   larger).
   - near: full to 1.2 m, fades to zero at 2.0 m;
   - far: ramps 1.0→2.0 m, full 2.0–4.5 m, fades to zero at 7.0 m.
5. **Fusion**: per cell, weighted occupied/free/hazard evidence.  Occupied
   requires `occ_w >= occ_min_w` and dominance over free evidence; hazard
   requires `haz_w >= haz_min_w` and wins over occupied.  A cell is *known*
   when at least one valid return landed in it; the weighted evidence sets
   the class.
6. **Gap fill**: 1–2 cell interior gaps between observed free cells are
   filled, but never the outer boundary and never across an occupied/hazard
   cell.  This removes the dappling caused by sparse far returns.
7. **Inflation**: occupied/hazard cells are dilated by the robot radius into
   the planning layer (`INFLATED`); `raw` keeps the pre-inflation grid.

## Grid convention

Body frame: x forward, y left, z up; origin on the floor under the robot.

- row 0 = far edge (`x = range_m`), rows grow toward the robot;
- col 0 = right edge (`y = -width_m/2`), cols grow leftward;
- class ids: `0 FREE`, `1 OCCUPIED`, `2 HAZARD`, `3 INFLATED`;
- `coverage` is separate: unobserved cells are `FREE` by id (planner
  compatibility) but `coverage=False`;
- `source`: `0 none`, `1 near`, `2 far`, `3 both`;
- `confidence`: weighted evidence strength, 0–255.

`BevGrid` also carries `stamp_us`, `recv_ts`, `per_camera_age_s`,
`plane_ok`, the plane fit (normal, offset, inliers, residuals, per-camera
checks), and per-camera contribution counters.

## Files

| Path | Role |
|---|---|
| `bebop_vision/master_bev.py` | geometry, joint floor, weighted fusion, rendering, JSON |
| `bebop_vision/bev_service.py` | 10 Hz latest-wins worker + `/bev` publish |
| `bebop_vision/videoserver.py` | `GET /bev`, `/snapshot?stream=bev`, `/whep?stream=bev` |
| `main.py` | `--bev` standalone; auto-start during `--record-navd`; `--no-bev`, `--bev-hz` |
| `tests/test_master_bev.py` | synthetic-depth tests (both cameras, boxes, step down, source, rendering, JSON, per-camera planes) |
| `tests/test_bev_service.py` | worker/cache/stale-camera tests |
| `tools/bev_compare.py` | capture a color+depth pair and write a color→BEV / occupancy comparison montage |
| `config/orbbec_rig.yaml` | `master_bev:` block (the legacy `bev:` 3×3 block is untouched); near `zero_self_pixels: false` |

## Run it

Standalone view (no robot connection, no recording):

```bash
cd ~/bebop/bebop-vision
sudo .venv/bin/python -u main.py --bev
```

Then:

```bash
curl -s localhost:9092/bev | jq '.counts, .contributions'
curl -s localhost:9092/snapshot?stream=bev -o /tmp/bev.jpg
```

The operator app's Video/Teleop screens list **BEV** as a WebRTC stream
(`:9092/whep?stream=bev`).  The JSON endpoint is the richer view: the app can
poll it at 10 Hz to draw coverage/source/confidence rather than a video.

During `--record-navd` the worker starts automatically; `--no-bev` disables
it.  `--bev` with no other mode opens the depth cameras only (`color=False`).

## Bring-up checks

1. Flat floor, both cameras: `/bev` should show near cells (source 1), far
   cells (source 2), an overlap (source 3), and `plane_ok` true for both.
2. Box at ~1 m: occupied cells at the near face; box at ~2.5 m: occupied
   cells from the far camera.
3. Cover/unplug one camera: the other region keeps updating; `ages_s` and
   `plane_ok` show the degradation.
4. Stair/step down: hazard cells at the drop.
5. Compare `--roles near` and `--roles far`: the far run should show the
   1.2–5 m region, the near run 0.55–1.75 m.
6. `main.py --bev --seconds 30` should sustain ~10 Hz on the Thor.

If a box lands in the wrong cell in one camera but not the other, re-check
that camera's `height_m`/`pitch_deg`/`yaw_deg`; the joint plane and the
per-camera residual in `/bev` make a mis-calibrated mount visible.

The per-camera plane offset `d` is a direct mount-height check.  In the
first live test the far plane sat about 6 cm below the near plane
(`far d≈-0.059`, `near d≈-0.006`), and both planes were tilted ~5° with the
robot on a cluttered carpet.  The per-camera fits absorb that; if the
offset grows or the tilt persists on a known-flat floor, re-measure the
mount.  Clutter on the floor is expected to render as occupied/inflated
cells, not free space.

The near camera keeps its chassis polygon for label consumers, but
`zero_self_pixels: false` stops zeroing the published depth.  The body
`robot.self_mask` removes chassis points before the BEV, so the robot is
still not an obstacle.  Use `tools/bev_compare.py` to project the color
image into the same grid and check occupied cells against the objects
(the first run showed the clear floor free, boxes/bags/desk/wall occupied,
and the thin orange streak matching a red cable on the floor).

## Known limits / next

- No temporal filter: one noisy frame can still flicker a cell.  The
  confidence layer is there to add a decayed evidence layer later.
- Body-frame only, no world-frame map and no relocalization.
- Far coverage thins beyond ~5 m; the grid reaches 6 m, but the last cells
  can stay unknown.
- No planner or drive commands.  Step 2 consumes `raw`/`occ` + `coverage`
  for obstacle avoidance.
- The legacy navd 3 × 3 m `bev:` geometry is untouched for the exp branch;
  the master block is separate.
