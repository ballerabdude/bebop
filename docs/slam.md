# slam — live visual localization (RGB-D POC)

Status: **opt-in, observe-only POC** (2026-10). One Orbbec camera feeds
NVIDIA cuVSLAM RGB-D odometry from inside the existing recorder process. The
estimate is read-only JSON on **`:9096`**. Nothing here commands the wheels,
reads wheel odometry, or fuses the IMU.

## 1. Scope

In scope: RGB-D odometry from one camera (`far` by default) inside the navd
recorder; poses in the **camera optical frame only** (x right, y down,
z forward), labelled `frame_id="camera_optical"`,
`pose_frame="world_from_camera_optical"`; current-tick validity (stale,
incomplete, skewed, wrong-serial, lost, or failed tracking invalidates
`pose`/`map_pose`); a bounded in-memory trajectory; read-only HTTP status;
Phase-1 structural mapping with depth-estimated gravity/floor, a
gravity-aligned occupancy grid, free-space carving, and 2D wall line
segments; hardware-free tests.

Not claimed: wheel-odometry input/fusion (the offline replay may compare
against `/odom`; live never reads it); IMU fusion or IMU-measured gravity
(no validated camera↔IMU extrinsic/noise model, so `--live-slam-imu` is
rejected, not defaulted); a camera-optical → body transform (the rig YAML
lacks roll and x/y offset, so a body pose would be fabricated); a persistent
structural map, loop closure, or relocalization (the structure feed is
per-session and odometry-based, and drifts); safety rating or motion commands.

## 2. Hardware and process model

Jetson AGX Thor, JetPack 7.2 / L4T R39.2.1, CUDA 13.2, Python 3.12.
Orbbec Gemini 335Lg: 848×480 depth @ 30 fps, 1280×800 color, shared device
timestamps. The recorder is the **only camera owner**; `--live-slam` is a
flag on that same process and never opens a camera itself. Before a manual
camera-owning run, stop the existing `bebop-vision` service/process (and any
OrbbecViewer), then restore it afterward. Ports: firmware `:9090`, agent
`:9091`, video `:9092`, voice `:9093`, models `:9094`, places `:9095`,
**live SLAM `:9096`**.

## 3. Dependency

Install the official NVIDIA Isaac cuVSLAM v17.0.1 Thor wheel into the
existing `bebop-vision/.venv`:

```bash
cd ~/bebop/bebop-vision
.venv/bin/pip install --no-deps \
  'cuvslam @ https://github.com/nvidia-isaac/cuVSLAM/releases/download/v17.0.1/cuvslam-17.0.1%2Bcu13-cp312-abi3-manylinux_2_39_aarch64.whl'
.venv/bin/python -c 'import cuvslam; print(cuvslam.__version__)'
```

`requirements.lock.txt` carries the same wheel URL/hash. Keep `--no-deps`:
the wheel is ABI-built for this platform and must not pull a second
NumPy/OpenCV/CUDA stack into the venv. Upstream Multisensor is
experimental/pinhole-only; this POC uses RGB-D.

## 4. Running

```bash
cd ~/bebop/bebop-vision
sudo .venv/bin/python -u main.py \
  --record-navd /var/lib/bebop-captures --auto \
  --live-slam
```

Actual options:

```text
--live-slam                       enable the observe-only tracker
--live-slam-camera ROLE           far|near (default: far)
--live-slam-port PORT             status port (default: 9096)
--live-slam-max-pair-skew-ms MS   color/depth pairing tolerance (default 17)
--live-slam-map                   request corrected/map pose (experimental)
--live-slam-log PATH              append per-frame JSON diagnostics
--live-slam-imu                   rejected (no calibration)
```

`--live-slam-map`/`--live-slam-log` require `--live-slam`. The POC needs no
`--goal-drive`, `--navd-model`, armed wheels, or Policy mode.

For the app Map page, `/map` is available whenever `--live-slam` is on.
Deployments can opt in without changing the unit by putting
`BEBOP_VISION_EXTRA_ARGS="--live-slam --live-slam-map"` in
`/etc/bebop/vision.env` (the unit reads that EnvironmentFile); restart the
vision service only while the robot is idle. `--live-slam-map` additionally
exposes the experimental corrected pose; the map feed itself always uses
uncorrected odometry.

If live-SLAM initialization fails (missing wheel, bad calibration, port in
use), the opt-in run cleans up and exits; it does **not** silently continue
recording. Run without `--live-slam` for the normal recorder.

Check status:

```bash
curl -s http://127.0.0.1:9096/status | python3 -m json.tool
curl -s http://127.0.0.1:9096/healthz
```

## 5. Status and HTTP API

`GET /status` is the full status. Key fields:

| field | meaning |
|---|---|
| `ok`, `service`, `backend`, `uptime_s` | liveness / tracker identity |
| `session_id` | UUID per `LiveSlam`, shared by `/status`, `/trajectory`, `/map` |
| `role`, `serial` | tracked role and calibrated serial |
| `frame_id`, `pose_frame` | `camera_optical`, `world_from_camera_optical` |
| `tracking_state` | `starting` / `tracking` / `lost` / `stale` / `invalid` / `error` / `stopped` |
| `pose_valid`, `pose` | current pose only; `pose=null` when invalid |
| `map_pose`, `slam_enabled` | optional corrected pose, kept separate |
| `segment` | increments on recovery after loss/reset |
| `frames_seen/processed/tracked/lost/rejected` | counters |
| `trajectory_points/capacity` | bounded ring state |
| `last_error` | last validation/tracker error |
| `map_error` | last geometry-update error (does not invalidate pose) |
| `structure_error` | last structural-map update error (does not invalidate pose) |
| `point_cloud_enabled` | raw display-cloud accumulation (default `false`) |
| `structure_enabled` | 2D structure/floor-plan mapping (default `true`) |
| `last_frame_age_s`, `pair_skew_ms` | observed age and color/depth skew |
| `depth_scale_mm` | SDK scale used for raw depth |
| `imu` | `{requested, enabled:false, reason}` |
| `calibration` | role/serial/shapes/frame/source/rig_source |

`pose` contains `t_ns`, `translation`, `rotation` (xyzw), `frame_id`,
`segment`, `age_s`. `status()` independently re-checks pose age against
`max_frame_age_s`, so a blocked worker or frozen capture cannot leave a stale
pose looking current. Recovery starts a new `segment`; poses are not stitched.

HTTP is read-only:

```text
GET /healthz                  liveness + tracking_state/pose_valid
GET /status                   full status
GET /pose                     current pose (or valid=false)
GET /map                      optional raw depth-geometry snapshot
GET /structure                2D floor plan: gravity, floor, grid, walls
GET /trajectory?limit=N       bounded recent trajectory
GET /                         service/endpoint list
```

POST, PUT, PATCH, and DELETE return 405; unknown GET paths return 404. There are no
start/stop endpoints; `--live-slam` at process start and normal shutdown own
the lifecycle.

### 5.1 Map page feed (`GET /map`)

The raw display point cloud is **disabled by default**
(`LiveSlamConfig.enable_point_cloud=False`): accumulating 100k voxels and
serializing tens of thousands of points cost GIL time and bandwidth that the
app cannot draw. `/map` remains schema-compatible when the gate is off —
`points: []`, `count: 0`, `map_points: 0`, `transmit_stride: 1` — and
enabling the gate restores the existing cloud path. Structural mapping
(`/structure`, below) is the default map product. When the cloud is enabled,
geometry is built inside the same
live process from the **already color-aligned depth** and the matching
**continuous odometry pose** for that tick; no camera is opened and no wheel
odometry is read. Sampling is capped at **1 Hz**. Internal storage keeps up to
**100,000 voxels** (5 cm default, LRU eviction), while the app receives at
most **30,000 points** (deterministic even stride at rebuild time; 100k
internally transmits 25k with stride 4). Returns only depth within
**6 m**. Points are unprojected with calibrated color rays (the rational
distortion is inverted, not ignored) and transformed by
`world_from_camera_optical`.

```json
{
  "session_id": "uuid",
  "revision": 12,
  "frame_id": "camera_optical",
  "pose_frame": "world_from_camera_optical",
  "geometry_source": "depth_odometry",
  "corrected": false,
  "segment": 0,
  "points": [[0.12, -0.34, 1.56, 120, 180, 220]],
  "count": 1,
  "capacity": 100000,
  "voxel_size_m": 0.05,
  "map_points": 1,
  "transmit_stride": 1,
  "updated_age_s": 0.31,
  "status": {"...": "existing /status dict"},
  "trajectory": {"...": "existing /trajectory dict"}
}
```

Coordinates are metres in the camera-optical world frame; RGB is 0–255
(zeros when a frame has no usable color). `count == len(points)` is the
transmitted point count; `map_points` is the internal (pre-decimation)
voxel total and `transmit_stride` is the deterministic stride used. `capacity`
remains the internal 100,000-voxel LRU bound. `session_id` is generated per
`LiveSlam` and also appears in `/status` and `/trajectory`. `updated_age_s` is
`null` before the first successful sample. The map is cleared whenever the
tracker starts a new `segment`, and the snapshot's `segment`, `status`, and
`trajectory` are taken under one lock.

This is an **experimental odometry reconstruction**, not an occupancy grid,
not a loop-corrected map, and not a persistent relocalization map. It uses
only uncorrected odometry poses (`corrected=false`); SLAM-corrected poses are
never mixed into old geometry. Drift, tracking resets, and depth-scale error
therefore move the cloud. `GET /map` is read-only, CORS-enabled, and
`Cache-Control: no-store`.

In **bebop-app**, open **Map** from the dashboard or motor bench. Drag to
In **bebop-app**, open **Map** from the dashboard or motor bench. Drag to
orbit, wheel/pinch to zoom, **Fit** to recenter, and **Top view** for the
camera-relative X/Z projection. Cloud and path layers can be toggled.
**Pause feed** freezes incoming data; disconnected/paused views retain the
last cloud but hide the camera marker and label telemetry as last reported.
The page polls `/map` without overlapping requests. Its embedded trajectory
is downsampled to 300 points; `/trajectory` retains the original 900-point
bound. Geometry is materialized lazily **only when `/map` is read**, at most
once per revision and coalesced by a small interval; point JSON is serialized
outside the geometry lock, and map fusion runs on a dedicated worker, so
capture/tracking is not blocked by serialization. The app can only draw
30,000 points, so the internal 100k map is deterministically decimated to
<=30k at rebuild time; serializing all 100k was pure GIL and bandwidth waste.
At the 30k transmit cap the response is ~0.7–0.9 MB, paid as bandwidth on
every poll.

Local workstation measurement with a populated 100,000-voxel map (25,000
transmitted, stride 4): rebuild ~77 ms, point serialization ~15 ms, cached
bytes assembly ~3.4 ms, dict snapshot ~2.8 ms, HTTP median ~15 ms (max ~42 ms
including the 0.72 MB client read/parse), cached transmitted list + bytes
~7.4 MB. Thor is slower; treat these as relative costs, not robot guarantees.

**Robot POV** switches the viewer to perspective at the tracked camera's
position and orientation. It follows the same observed odometry pose and
retains the existing feed rate. The virtual lens has a 70° vertical field of
view, with points behind the camera or nearer than 0.1 m clipped. This is the
camera-optical viewpoint through the reconstructed cloud, not a live video
image or a calibrated robot-body/eye transform. POV waits when a pose is
stale, lost, disconnected, or paused. Toggle it off, or choose **Top view**
or **Fit**, to return to the overview.

Map-page validation on 2026-10-17: frontend production build passed; browser
checks passed at 1280 px and 390 px widths with real robot geometry, orbit,
zoom, top view, Fit, pause/resume, and simulated connection loss (no page
errors or horizontal overflow). Final robot suite: **216 passed**. With the
10,000-point cloud, median HTTP response time was **77 ms** across 42 samples
after warm-up; 40 reported valid tracking and two reported stale poses.
There were zero tracker losses or rejected inputs in that run. The current
robot has `/etc/bebop/vision.env` configured to start live SLAM with Vision.

### 5.2 Structural map (`GET /structure`, Phase 1)

This is the top-down floor-plan feed. It is **per-session and odometry-based**,
not persistent: there is no loop closure, relocalization, or cross-session
storage. Gravity is **estimated from the depth floor plane**, never from an
IMU. The floor fit is constrained to a normal within 35° of the camera `-Y`
(up) prior and needs both a minimum inlier count and floor area. Before that
support exists, `frame_id`/`pose_frame` are `null`, `gravity.valid=false`,
`floor.valid=false`, and no up vector is fabricated.

`GET /structure` (read-only, CORS, no-store) returns:

```json
{
  "ok": true, "service": "live-slam", "session_id": "uuid",
  "frame_id": "map", "pose_frame": "map_from_camera_optical",
  "structural": true, "persistent": false, "loop_corrected": false,
  "gravity_source": "estimated_from_floor",
  "gravity": {"valid": true, "up": [x,y,z], "roll_deg": n,
              "pitch_deg": n, "inlier_ratio": r},
  "floor": {"valid": true, "normal": [x,y,z], "offset_m": n,
            "camera_height_m": n, "inliers": n, "rmse_m": n,
            "refinements": n},
  "grid": {"origin": [x,y], "resolution_m": 0.05, "width": W, "height": H,
           "occupied_count": n, "free_count": n, "encoding": "rle_base64",
           "occupied_rle": "...", "free_rle": "..."},
  "diagnostics": {"occupied_cells": n, "wall_cells": n,
                  "wall_components": n, "wall_components_elongated": n,
                  "wall_components_rejected_shape": n,
                  "wall_components_rejected_boundary": n,
                  "wall_segments": n, "free_cells": n,
                  "floor_rmse_m": n, "floor_refinements": n},
  "walls": [[x0,y0,x1,y1], "..."],
  "wall_cells": [[ix,iy], "..."],
  "path": [[x,y], "..."],
  "pose": {"valid": true, "x": n, "y": n, "yaw_deg": n, "t_ns": n,
           "segment": n},
  "trajectory": {"...": "bounded trajectory dict"},
  "status": {"...": "existing /status dict"},
  "revision": n, "updated_age_s": n
}
```

Conventions and bounds:

* All map quantities are metres in the gravity-aligned **`map`** frame:
  `z=0` is the estimated floor plane, `+x` is the initial camera forward
  projected onto the floor, and `+y` completes a right-handed frame with up.
* `gravity.up` is the estimated up unit vector in the **camera-optical** frame
  (x right, y down, z forward). `gravity.roll_deg`/`pitch_deg` are camera
  corrections: `pitch_deg > 0` means the camera is pitched down;
  `roll_deg > 0` means the up vector leans toward camera +x.
* `floor.normal` is that same camera-optical up vector; `floor.offset_m` is
  the camera-frame plane offset (numerically the camera height because the
  camera origin is the plane-equation point) and `camera_height_m` is the
  same distance for convenience.
* `pose.yaw_deg` is the camera-forward heading in the map frame, measured
  counter-clockwise about the up axis from map `+x` (right-handed positive).
* `grid.origin` is the map-frame near corner of the cell-index bounding box;
  `width`/`height` are cell counts. RLE is row-major, starts with a 0-run, and
  uses little-endian **uint16** run lengths (`rle_base64`). Decode
  `occupied_rle`/`free_rle` to `width*height` booleans.
* **Occupied is the obstacle layer.** `occupied_rle`/`occupied_count` contain
  every cell with a hit above `occupied_min_height_m` (0.28 m): walls, desks,
  chairs, boxes and people are all rendered. A cell is a *wall candidate* only
  when its accumulated depth span is at least `wall_min_extent_m` (0.40 m)
  **and** its top clears `wall_min_top_m` (0.8 m, just above desk height).
  That height floor is only a low-clutter filter; it is deliberately modest
  because a chest-height camera pitched down often sees a real wall top out
  below 1.4 m (the old floor rejected ~99.8% of occupied structure live).
  Candidates are connected-componented after a one-cell morphological closing
  (grid gaps are not walls), and a component is a wall only when it is
  **elongated and thin**: PCA extent ratio at least `wall_min_aspect_ratio`
  (3) with a robust thickness at most `wall_max_thickness_m` (0.5 m), or a
  thin boundary whose two extents are both at least
  `wall_min_loop_extent_m` (1.5 m, a room/L/T network). Compact blobs never
  become walls. The diagnostics expose `wall_candidate_cells`,
  `wall_cells_rejected_extent`, `wall_cells_rejected_height`,
  `wall_components_rejected_shape` and `wall_components_rejected_boundary`
  so a zero-wall map can be attributed to the right stage.
* **A wall must bound observed space.** Each sampled wall cell is probed
  along its local normal: carved free cells must lie on one side and not on
  the other (`wall_boundary_check`, probes 3 cells, 25% of sampled cells).
  An obstacle in open floor has free space on both sides and is rejected.
  Limitation: a wall genuinely seen from both sides also has free on both
  sides and is rejected; disable the check if that trade-off is unacceptable
  for a run.
* `walls` are fitted 2D line segments from accepted wall cells only (greedy
  2D growing, collinear neighbours merged, segments shorter than
  `min_wall_segment_m` = 0.8 m dropped). `wall_cells` is capped at 2,000
  cells, walls at 500 segments.
* Hits below `occupied_min_height_m` (0.28 m) never become occupancy. Free
  space is carved along rays to **every** depth hit, including floor hits, so
  the walkable floor is marked free rather than left blank or occupied.
* `diagnostics.occupied_cells` counts the obstacle layer;
  `diagnostics.wall_cells` is the accepted wall subset that feeds `wall_cells`
  and `walls`; `wall_components` / `wall_components_elongated` /
  `wall_components_rejected_shape` / `wall_components_rejected_boundary`
  explain why candidate clusters did or did not become walls; `free_cells` is
  the carved free-space count. A tilted floor can still appear in the obstacle
  layer, but its cells have no vertical extent/top and never become a wall.
  `floor.rmse_m`/`diagnostics.floor_rmse_m` is the inlier residual of the
  least-squares-refitted floor plane and `floor_refinements` is the number of
  refit passes (0-3).
* The floor is re-fit incrementally as the robot moves (RANSAC followed by up
  to `floor_refine_iterations` least-squares passes over the inliers); the
  gravity-aligned grid frame is locked at the first valid estimate, so later
  floor refinement does not rotate old cells.
* Sampling is 1 Hz and runs on the map worker. Structure snapshots are lazy
  and cached per revision; the tracking thread is never blocked. Internal
  cells are capped at 60,000; eviction prefers free-only cells so walls are
  not displaced by floor carving. A worst-case synthetic 20,000-cell
  alternating grid measures ~130 KB; realistic contiguous floor/wall maps
  are ~1-2 KB (measured 1.2-1.7 KB), far below the 200 KB budget.
  `trajectory` is the bounded 300-point map trajectory.
  Local synthetic measurements: structure snapshot build ~1.5 ms for a
  tilted-floor scene and ~1.1 ms for a floor+wall scene, ~19 ms for a
  3,721-cell wall grid (18 KB payload), cached HTTP ~1 ms; the wall-grid
  update itself is ~110 ms on the worker (Thor is slower).

The raw display point cloud is a separate opt-in path; see 5.1. This
structural feed does not use SLAM-corrected poses, wheel odometry, or IMU.

A snapshot is also the input to the places service's per-place submap
endpoint (`POST /places/{id}/map` on `:9095`, see
[`internal-map.md`](internal-map.md)): the payload's gravity-aligned grid,
walls and `pose` are accumulated into one local patch, with the first
capture's pose pinned as the place anchor. That store assumes snapshots share
a session's `map` frame; frames from different sessions are unrelated until
relocalization exists.


## 6. Frame path

The camera publishes depth-only FrameSets alongside complete RGB-D ones.
`camera.read()` can therefore be depth-only even when `camera.recent()` holds
a fresh complete pair. The live path scans `recent()` newest→oldest and picks
the newest **complete** frame: depth + color (or color JPEG), positive paired
device stamps within `max_pair_delta_us`, finite age in
`[0, max_frame_age_s]`. It preserves the frame's original `recv_ts` and
device stamps. If none qualifies it returns the latest sample so status
exposes the real rejection reason. Injected `frame_source` is unchanged.

Per selected frame:

1. **Pair stamps.** `stamp_us` (depth) and `color_stamp_us` (color) must both
   be positive and differ by ≤ `max_pair_delta_us`. The SDK-synchronized
   color/depth phase reaches **16.6 ms** at 30 Hz; the default is **17 ms**
   (half period 16.667 ms + timestamp rounding), which still rejects a full
   33 ms adjacent-frame mismatch. A 15 ms cap produced a 9.7 s gap and 250
   input rejections while the engine itself tracked 1192/1192 frames
   (`segment=0`). Actual skew is published as `pair_skew_ms`.
2. **Check serial and age.** Frame serial must equal the calibrated serial;
   age must be finite and ≤ `max_frame_age_s`.
3. **Apply depth scale once.** Raw depth is uint16 in SDK units, not assumed
   millimetres. `get_depth_scale()` supplies mm/unit; scale before
   unprojection. The extrinsic translation is already metric.
4. **Register depth to color.** Unproject with depth intrinsics, apply the
   inverse factory `color_to_depth_transform`, project with color intrinsics
   and 8-parameter rational distortion. Output is uint16 mm in the color
   frame, 0 = invalid.
5. **RGB, not BGR.** Camera MJPEG decodes via cv2 as BGR; convert to RGB
   before cuVSLAM.
6. **Track.** Pass the RGB image, aligned uint16 mm depth
   (`depth_scale_factor=1000`), and `stamp_us * 1000` ns.
7. **Validate.** `None`/non-finite pose, tracker error, or loss invalidates
   the current pose. A repeated device stamp is never fed twice. A backwards
   device clock is not fed to the same tracker: status becomes `error` and
   live-slam must be restarted with a fresh tracker.

## 7. Calibration

Sources: `config/orbbec_rig.yaml` (role↔serial) and
`config/orbbec_intrinsics_<serial>.json` (depth/color intrinsics,
distortion, `color_to_depth_transform`). The loader validates positive
shapes/focals, finite intrinsics/distortion, a proper 3×3 orthonormal
rotation (`max|RRᵀ−I| < 1e-2`, `det > 0`), finite translation, and a role
that resolves to a serial. Translation is stored in mm and converted to
metres. Mount height/pitch/yaw are not used for the pose.

## 8. Failure modes and limitations

Stale/incomplete/skewed/wrong-serial frames: `pose=null`, state
`stale`/`invalid`; never reuse an older color image. Frozen duplicates are
not tracked twice and clear once their age exceeds the limit. Tracking loss
is `lost` with `pose=null`; recovery increments `segment`. Tracker
exceptions set `error`/`last_error` without killing the recorder. A backwards
clock sets `error` and requires a restart (never feed a reset clock to the
same tracker). Drift, pure rotation, blank walls, glass, blur, and fast turns
commonly lose tracking; the nominal mount is not a full extrinsic
calibration. Depth-scale errors produce plausible but metrically wrong
paths. GPU contention increases latency. Map/loop closure/relocalization are
unvalidated; observe only.

## 9. Offline replay and tests

`tools/slam_replay.py` evaluates recorded MCAP sessions on a workstation;
its wheel-odometry CSV is an evaluation reference only. Its legacy
`--depth-scale` workaround does not apply to the live path, which reads the
SDK scale.

Hardware-free tests:

```bash
cd ~/bebop/bebop-vision
.venv/bin/python -m pytest tests/test_slam.py -q
```

Coverage: calibration validation, depth scale/extrinsic order, same-tick
pairing, recent-history complete-pair selection, stale/frozen/duplicate
frames, serial/skew rejection, tracking loss/error/recovery segments,
backwards-clock lockout, camera-optical frame, separate odom/map pose,
bounded trajectory, no wheel dependency, IMU rejection, read-only HTTP
lifecycle/405, the bounded depth-geometry map (distortion ray inverse,
metric scale/transform/bounds, voxel capacity, 100k internal / 30k transmit
decimation determinism, segment reset, 1 Hz throttle,
lazy 100k-point cache/serialization outside the map lock,
point-cloud-off-by-default), and the Phase-1 structural map (no estimate
before floor support, synthetic floor+wall gravity/line fit, aligned-frame
and yaw math, above-floor occupancy, wall-like verticality, the tilted-floor
zero-wall regression, free carving to floor hits, floor RMSE improvement
under least-squares refit, RLE grid, wall line segments, sub-200 KB payload,
and `/structure` schema/CORS/diagnostics).

## 10. Manual turn diagnostic

`tools/slam_turn_test.py` reads the `:9096` pose, commands zero linear
velocity and a capped angular speed, and stops on stale/lost/segment
change, jump, wrong direction, no progress, or timeout. It requires
`--execute`, refuses to move with a latched estop (and never resets it),
checks fresh telemetry/mode/armed state each tick, and in cleanup waits for
**post-command fresh telemetry** confirming zero twist, disarmed wheels, and
restored mode before disconnecting. The yaw conversion lives in the tool
(relative camera rotation conjugated by the nominal mount), not in the
package.

Hardware result (2026-10-17, 90 s run with video + places + continuous MCAP): visual
yaw ranged **-46.109° … +46.997°**, final **-45.682°**. Both turns completed
across an interrupted run: the diagnostic aborted on an invalid RGB-D input;
the remaining return/right legs were then explicitly relaunched with adjusted
visual-relative targets on the same live map/`segment=0`. The tool does not
automatically resume after an abort. No wheel odometry was used. The recorder wrote 2440 frames /
964 MB; pose median interval was **66.669 ms (~15 Hz)**. The engine reported
`frames_lost=0`; the input interruption was traced to the earlier 15 ms pairing
limit. Cleanup confirmed fresh-telemetry zero twist, disarmed wheels,
`MODE_IDLE`, and estop false. The standard vision service was restored.

The final **17 ms** configuration passed a separate 90 s stationary soak with
video, places, and continuous recording active:

- **1344 tracker frames**, all valid, one segment, zero input rejections.
- After 20 s warm-up: median observation interval **66.669 ms**, maximum gap
  **133.338 ms**, and zero gaps above the 300 ms freshness limit.
- **236/236 HTTP status samples** after warm-up had a valid pose; median pose
  age **112 ms**, maximum **204 ms**.
- Observed color/depth timestamp skew reached **16.997 ms**; the SDK depth
  scale was **1.0 mm per raw unit**.
- Recorder: **2450 frames / 984 MB**. Final on-robot test suite: **210 passed**.

Diagnostics and MCAPs are on the robot under `/tmp/opencode/live-slam-poc/`
(`live-history.jsonl` / `run-history.log` for motion, `live-final.jsonl` /
`run-final.log` for the final soak). These tests establish the live RGB-D POC;
they do not validate translation accuracy, persistent relocalization, or
place-to-place navigation. These measurements preceded the Map-page change;
the current robot's normal Vision service now enables live SLAM through
`/etc/bebop/vision.env` as described above.
