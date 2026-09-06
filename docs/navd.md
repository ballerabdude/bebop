# navd — Depth-based goal-conditioned obstacle avoidance

Status: **Phase A + recorder v2 shipped** (2026-09-06, main @ ad631c6, CI
green). Phase A (§6) is implemented and bench-verified except the formal
§6.7 acceptance demo; the recorder v2 (§7.1) is shipped and verified
end-to-end (auto-segmented MCAP sessions + web-app download + extractor).
**Phase B (§7.2–7.4) is not started** — the handoff brief for it is
[`navd-b-handoff.md`](navd-b-handoff.md) (implementation deltas, environment
facts, work plan). ED cable swap still pending (§11.1).
Scope: bebop-vision (Python, Jetson), firmware-adjacent but firmware mostly
untouched (one listing-filter change in `server/ws.rs`, deployed).

---

## 1. Overview

navd gives the Bebop robot the ability to navigate toward a commanded
**direction of travel** (a heading offset, or an x/y waypoint resolved through
wheel odometry) while **avoiding obstacles** perceived by two Orbbec Gemini
335Lg depth cameras.

The system is built and delivered in three stages, each independently useful:

- **Phase A** — geometric autonomy: depth → bird's-eye-view (BEV) occupancy
  grid → goal-conditioned polar planner → `SetVelocityCommand` twists. No
  machine learning. Deterministic, debuggable, and it doubles as the
  auto-labeling teacher for Phase B.
- **Phase B** — the learned model: a small student network that predicts BEV
  navigability from the raw depth images + the goal direction, trained against
  the geometric teacher (and operator teleop for imitation). Handles the cases
  pure geometry cannot: glass, dark/reflective objects, sensor noise.
- **Phase C** — hardening: packaging/deployment, telemetry overlays, app
  integration, docs.

### Non-goals (v1)

- Mapping / localization beyond the firmware's existing wheel odometry.
- Slopes, ramps, or non-flat floors (flat-floor assumption is explicit).
- Dynamic-obstacle motion prediction (obstacles are treated as static per tick).
- Speeds above the existing planner limit (0.4 m/s).
- Keeping the legacy RGB pipeline alive on the robot: the OBSBOT USB webcam
  and the firmware nav runner are being retired (Section 9). `navseg`
  survives as workstation-side training/recording tooling only; on-robot,
  navd supersedes it.

---

## 2. Current system (what we build on)

Facts about the existing stack that navd depends on or reuses. File paths are
authoritative — read them before changing any contract.

### 2.1 Perception → control loop today (RGB — legacy, being retired: Section 9)

```
firmware video.rs (owns OBSBOT /dev/video0, MJPG 1280x720@30)
  → GET http://bebop.local:9090/video        (multipart MJPEG)
bebop_vision/camera.py                        (threaded freshest-frame reader)
bebop_vision/navseg.py                        (SegFormer-B0 → 3-class label map)
bebop_vision/planner.py::SectorPlanner        (15 vertical bands → clearance → twist)
bebop_vision/planner.py::DriveNode            (10 Hz, mode/estop/deadman gating)
  → SetVelocityCommand {linear_x, angular_z}  (proto field 14, over ws://:9090/ws)
firmware supervisor.rs::drive_command         (operator arbitration + watchdogs)
  → differential drive (2x ODrive S1, wheel r=0.05 m, track 0.30 m)
```

- `SectorPlanner` semantics (planner.py:33-72): navigable = label 1 only
  (caution 2 counts as blocked); clearance per band = fraction of consecutive
  bottom rows ≥ `row_coverage` (0.85); near-zone check; score =
  `clearance^1.5 * (1 + center_bias(1.5)*exp(-off^2/0.18))`; states =
  drive / rotate-in-place / search with `v_max=0.4`, `wz_max=1.2`,
  `wz_turn=1.8`, `min_clearance=0.12`, `turn_threshold=0.5`.
- `DriveNode` gating (planner.py:75-115): mode must be `MODE_RUN_POLICY`,
  estop callback latches zero twist, `deadman_s=0.5` on stale perception,
  commands rate-limited to `command_hz=10`.
- **navd reuses all of these defaults.** The goal planner is a generalization
  of this state machine; the gating node is shared.

### 2.2 Firmware safety kernel (unchanged by navd)

- `supervisor.rs::drive_command` (398-440): zero twist always accepted; a
  non-zero twist requires the **active operator** seat (first non-zero client
  claims it; 2000 ms grace); others are rejected while the seat is held.
- **Link-loss watchdog** (supervisor.rs:1613-1645): a non-zero twist not
  refreshed within `operator_timeout_ms` (500 ms) is zeroed. → any autonomy
  client must stream at **≥ 2 Hz**; navd streams at 10 Hz.
- WS disconnect / E-stop / mode change release the seat and zero the twist.
- Firmware nav (`nav.rs`) is **observe-only** by design ("it observes; it
  never commands"). navd keeps that split: the safety kernel stays small and
  verifiable; the autonomy brain is a replaceable client.

### 2.3 Teacher→student pattern (reused for Phase B)

- Recorder: `bebop_vision/recorder.py` (images/ + masks/*.npz + manifest.jsonl).
- Distillation: `bebop_vision/labelnav.py` (concept masks → 3-class label with
  obstacle-dilation margin band).
- Training: `train_nav.py` (SegFormer-B0 from scratch, class-weighted CE,
  OneCycleLR, best-val-mIoU checkpointing).
- Export: `tools/export_navseg_onnx.py` — two-artifact ONNX (`*.onnx` +
  `*.onnx.data`), fixed tensor names (`pixel_values` [1,3,512,512] → `logits`
  [1,3,128,128], opset 17), normalization NOT baked in (consumer-side), parity
  gate ≥ 0.99 argmax agreement vs torch.
- navd copies this pattern wholesale for the depth student (Section 7).

### 2.4 Orbbec camera bring-up (done)

- `pyorbbecsdk2==2.1.2` declared in `bebop-vision/pyproject.toml` (single
  source of dependency truth; install via `.venv/bin/pip install -e .`).
- udev rules vendored at `scripts/orbbec-99-obsensor-libusb.rules`, installed
  by `scripts/install-jetson.sh --setup-orbbec` (non-root access, no group
  membership needed — devices are 0666).
- Devices (Jetson `bebop.local`):
  - `CPBLC53000PE` — **near** camera. On a USB 3.x lane (5000 Mbps) after a
    cable swap. Full profile set available.
  - `CPBLC53000ED` — **far** camera. **Still linked at 480 Mbps (old cable) —
    blocker**: at 480 Mbps the firmware only advertises crippled profiles
    (max depth 640x360@10; color RGB only at 424x240@10). Usable for a
    reduced far field at 10 fps, but swap the cable for the real horizon;
    profiles restore automatically.
- One process holds a camera at a time — close OrbbecViewer before running.
- Units stream on the *same* Realtek hub; each camera has its own SuperSpeed
  lane (verified: 5G through the hub works).

---

## 3. Hardware setup

### 3.1 Mounting

Both cameras are on a forward-looking mast, stacked vertically:

| Camera | Serial | Role | Pitch | Coverage |
|---|---|---|---|---|
| near | `CPBLC53000PE` | near field / negative obstacles | angled down more | ~0.3–1.5 m ahead |
| far  | `CPBLC53000ED` | planning horizon | slightly down | ~1.5–3 m ahead |

There is partial vertical overlap in the middle range. Exact angles/heights
are measured once and recorded in the rig config (Section 6.2); the config —
not this document — is the source of truth for extrinsics.

### 3.2 Streams (Phase A)

| Stream | Format | Rate | Purpose |
|---|---|---|---|
| depth (both cams) | 848x480 @ 30 fps, Y16 (uint16, mm) | 30 Hz capture, ~10 Hz processed | BEV geometry |
| color (PE) | 1280x800 @ 30 fps, RGB | capture only | dataset recording, debugging |
| color (ED) | MJPG until re-cabled → RGB after | capture only | dataset recording |

Bandwidth: 2x depth Y16 @ 848x480x30 ≈ 48 MB/s total — fits on separate
SuperSpeed lanes; do **not** put both cameras on a shared USB 2.0 path.

Frame sync: no hardware sync. Fusion tolerates the skew: at 0.4 m/s a 50 ms
timestamp difference is 2 cm. Each camera's BEV contribution carries its own
timestamp; the grid is re-fused every tick.

Camera hardware encode: **none** — the 335Lg exposes no H.264/H.265 UVC
profiles (verified on-device 2026-09-05; color sensor offers raw/MJPG only),
and the **Orin Nano has no NVENC hardware encoder** either (board string
"Jetson Orin Nano ... Super"; no `nvv4l2h265enc` element, no encoder device —
NVDEC decode-only board). Operator-stream encoding is therefore **software
(libx264 via PyAV, already a transitive dep)**, sized to what the 6x
Cortex-A78AE cores sustain — measured on-device (zerolatency/superfast,
4 threads, ~2 Mbps, synthetic motion):

| Config | Encode rate | Verdict |
|---|---|---|
| x264 848x480–1280x800 | 45 fps | real-time at full res — **chosen** |
| x265 848x480 | 44 fps | viable, but no advantage worth the fleet HEVC requirement |
| x265 1280x720 | 11.6 fps | not real-time at 30 |
| x265 1280x800 | 10.6 fps | not real-time |

NVIDIA's own app note for this exact situation ("Software Encode in Orin
Nano", JetPack r35.3.1 docs) confirms the premise — *"the NVIDIA Jetson Orin
Nano does not have the NVENC engine"* — and documents software encoding with
libav/**x264 only**. Its tuning tables are the reason x264 wins here:
encoding 30 fps with a hardware-style GOP (IDR/I interval 30, `ref=1`,
`bframes=0`, AQ off) cut x264 superfast cost from 41% to **18% of one core**.
~2–3 Mbps vs 15–24 Mbps MJPEG at the same resolution — the bandwidth case for
H.265 (~2 Mbps) was not worth its fleet HEVC-decode requirement and ~4x
encoder CPU cost, so the stream is **H.264**.

See the videoserver spec (Section 9.2, Stage 1).

### 3.3 SDK processing

Use `pyorbbecsdk` frame-sync on each camera pipeline (color/depth alignment
within a device) and enable the depth filters: SpatialModerateFilter +
TemporalFilter + HoleFillingFilter (settings proven in OrbbecViewer bring-up).
Do not use cross-device sync.

---

## 4. Runtime decision (ADR)

**Decision**: navd runs in **Python inside the existing `bebop-vision`
process** on the Jetson, talking to the firmware as a WS client.

Alternatives considered:

1. **Inside the Rust firmware binary** — rejected: the firmware's own design
   keeps nav observe-only so a perception bug can never command the motors or
   take down the safety kernel. Embedding autonomy would grow the blast radius
   of every perception iteration.
2. **Second Rust binary (`bebop-nav`)** — viable and keeps the single-binary
   packaging model, but slows ML iteration (no pyorbbecsdk; BEV math,
   calibration, and student serving all need reimplementation). Depth capture
   in Rust is still possible without the Orbbec SDK: the 335Lg depth stream is
   standard UVC Y16 on `/dev/video*` (readable `rscam`-style like `video.rs`
   does), and the student runs through the existing `ort` crate pattern from
   `nav.rs`.
3. **Python now (chosen)** — pyorbbecsdk integration, teacher/training
   tooling, and the DriveNode control path already exist here. Performance is
   a non-issue at 10 Hz (~10 ms/frame-pair for vectorized numpy BEV).

**Migration path**: the student ships as a two-artifact ONNX with a parity
gate and the planner is ~100 lines of geometry — if Python ops pain ever
materializes, a Phase-C Rust port is mechanical (V4L2 depth + same ONNX +
same math), not a redesign.

**Packaging obligation created by this choice**: bebop-vision currently runs
manually from a checkout. Phase C adds a systemd unit + an install step in
`scripts/install-jetson.sh` so the robot boots into autonomy without a
developer shell. Until then, navd runs on the bench only.

---

## 5. Architecture

```
        +----------------------+        +----------------------+
        | OrbbecCamera near PE |        | OrbbecCamera far  ED |
        | depth 848x480@30 Y16 |        | depth 848x480@30 Y16 |
        +----------+-----------+        +----------+-----------+
                   | (10 Hz stride each)          |
                   v                              v
        +-----------------------------------------------+
        | bev.py: deproject (stride 4) → body frame      |
        | → RANSAC ground plane → per-cam cell classes   |
        | → fuse (near authoritative < 1.5 m, union)     |
        | → inflate by robot radius                      |
        +----------------------+------------------------+
                               v
                     BEV occupancy 60x60 (3x3 m @ 5 cm, body frame, 10 Hz)
                               |
   goal heading --(body frame)-v
        +-----------------------------------------------+
        | goal_planner.py: polar rays -60..+60 @ 10 deg  |
        | clearance per ray; score = clearance^1.5       |
        |   * (1 + goal_bias * exp(-dpsi^2/0.18));       |
        | states: drive / rotate / search; hard-stop cone|
        +----------------------+------------------------+
                               v
                    DriveNode (shared): 10 Hz twist, mode gate,
                    estop latch, 0.5 s stale-BEV deadman
                               v
        SetVelocityCommand → firmware supervisor → wheels
```

Color streams are recorded but do not participate in Phase A control.

---

## 6. Phase A — geometric autonomy (spec)

### 6.1 `bebop_vision/orbbec.py` — camera service

- `OrbbecCamera(serial, role, depth_profile=(848,480,30), color_profile=None)`.
  Opens via `ob.Context().query_devices()` matched by serial (never index —
  enumeration order is not stable with two identical devices).
- Depth filters enabled per Section 3.3. Optional color stream (recording).
- Threading: capture thread per camera, latest-wins slot + lock (copy of the
  `camera.py` pattern). Public surface: `read() -> StampedFrame | None` with
  `stamp_us` from the SDK frame timestamp, plus `age_s(now)` for staleness.
- `OrbbecRig` owns both cameras + the rig config; exposes
  `wait_for_pair(timeout)` used by the BEV worker (near+far, both fresh).
- Failure semantics: any camera open failure at startup = hard error (bench
  tool); mid-run camera drop = stale slot → deadman stop (below). Never
  crash the control loop on a camera exception.

### 6.2 `config/orbbec_rig.yaml` + intrinsics

```yaml
# config/orbbec_rig.yaml — measured once per robot; source of truth for extrinsics
robots:
  default:
    cameras:
      CPBLC53000PE: { role: near, height_m: 0.55, pitch_deg: -35.0, yaw_deg: 0.0 }
      CPBLC53000ED: { role: far,  height_m: 0.75, pitch_deg: -12.0, yaw_deg: 0.0 }
    bev:
      range_m: 3.0        # forward horizon
      width_m: 3.0        # lateral span (-1.5..+1.5)
      cell_m: 0.05        # 60x60 cells
      near_authority_m: 1.5
```

Convention: `pitch_deg` negative = camera center points below horizon.
Intrinsics live per-serial in `config/orbbec_intrinsics_<serial>.json`,
written once by `tools/orbbec_intrinsics.py` (depth camera params via
`pipeline.get_camera_param()`, `VideoIntrinsic` fx/fy/cx/cy + distortion).
The BEV code fails loudly if the JSON for a mounted serial is missing.

Extrinsics verification procedure (bench, once after mounting): run the BEV
debug view with a flat floor and a single box at a known spot; the box must
land within one cell of its measured position in both cameras' regions;
re-measure pitch/height if the floor line is bowed or the box is displaced.

### 6.3 `bebop_vision/bev.py` — occupancy grid

Per camera, per tick (10 Hz):

1. **Subsample** depth stride 4 (212x120 samples ≈ 25 k rays — enough at 5 cm
   cells for a 3 m grid) and drop invalid (0) / out-of-range (> 6 m) pixels.
2. **Deproject** with per-serial intrinsics to camera-frame points.
3. **Transform** to robot body frame using extrinsics (x forward, y left, z
   up; R = Rz(yaw) · Ry(pitch), t = (0, 0, height)).
4. **Ground plane**: start from the mount-height prior (z = 0 plane); refine
   with a RANSAC plane fit over the tick's points (residual gate 15 mm).
   Fit failure (insufficient inliers — e.g. camera briefly pointing at a
   wall) → fall back to the prior for that camera that tick.
5. **Cell classification** (per point, into the 60x60 grid):
   - height above local ground in **[0.03, 0.30] m** → `occupied`
   - ground-plane **drop > 0.05 m** between adjacent cells → `hazard`
     (negative obstacle: stairs, hole)
   - height > 0.30 m → ignore (overhang; robot clears under it)
   - height in [0, 0.03) m → floor

   Hardware additions (2026-09-05, found necessary on the real rig):
   - **Self-view mask** — the mast cameras see the robot's own chassis at
     fixed body-frame positions; points inside the footprint box
     (`robot.self_mask` in the rig YAML) above the floor are dropped before
     classification, or the robot blocks itself permanently.
   - **Range-scaled plane residual** — max(15 mm, 1% of point range); the
     fixed 15 mm gate is tighter than stereo noise beyond ~2.5 m and
     starved the far camera's floor fit.
   - `plane_min_frac` is 0.10 (far camera is wall-heavy in cluttered
     scenes); the plane-fit window is `[-(mount_height + 0.5), +0.05]` m.
6. **Fuse**: for cells covered by both cameras, the **near camera wins below
   1.5 m** (closer = better depth accuracy), far camera wins beyond; elsewhere
   union. Both cameras must be fresh (< 0.3 s) for a fused grid; a single
   stale camera degrades to the other's region only.
7. **Inflate** occupied/hazard cells by the robot radius (0.20 m ≈ 4 cells)
   for planning (the raw grid is kept for telemetry/dataset labels).

Output: `BevGrid` dataclass — `occ` (uint8 60x60: 0 free, 1 occupied,
2 hazard, 3 inflated), `stamp_us`, `per_camera_age_s`, `plane_ok` flags.

### 6.4 `bebop_vision/goal_planner.py` — goal-conditioned planner

- **Goal input**: `GoalHeading` (body-frame heading offset, rad) or
  `GoalPoint` (odom x, y — converted to body-frame heading each tick using
  `DriveState.odom_x/y/theta` from telemetry; bearing =
  `atan2(gy - odom_y, gx - odom_x) - odom_theta`).
- **Polar scoring**: 13 rays over −60°..+60° (10° steps). Along each ray,
  march the inflated grid from r=0.35 m to 3.0 m; `clearance_i` = first
  occupied/hazard hit range (or 3.0 m if clear), normalized.
- **Score**: `score_i = clearance_i^1.5 * (1 + goal_bias * exp(-dpsi_i^2/0.18))`
  with `dpsi_i = ray_angle - goal_heading`, `goal_bias = 1.5` — the existing
  `SectorPlanner` curve with center bias replaced by goal bias.
- **State machine** (mirrors `SectorPlanner` semantics):
  - best `clearance < 0.12` → **search**: rotate in place at `wz_turn=1.8`
    toward the goal side (persisted direction; flip on goal side change).
  - near-cone (±20° at r < 0.5 m) contains an obstacle → **hard stop**
    (vx = 0, wz = 0). Distinct from search: something is actually in front.
  - `|best_dpsi| > turn_threshold (0.5)` or best ray's near-field (first
    0.5 m) blocked → **rotate in place** toward the goal (`vx = 0`,
    `wz = clamp(k * dpsi, wz_max)`).
  - else → **drive**: `vx = v_max * clip(best_clearance / 0.5, 0, 1)`,
    `wz = clamp(k * dpsi, ±wz_max)` with `k = 2.0`.
- **Goal reach**: for `GoalPoint`, stop (zero twist, state "reached") when
  horizontal distance < 0.3 m. `GoalHeading` never "reaches" — it is a
  heading hold until re-issued or cleared.
- All limits/defaults match `SectorPlanner` (v_max 0.4, wz_max 1.2) and stay
  CLI-tunable (`--v-max`, `--wz-max`).

### 6.5 Waypoint interface + `main.py --goal-drive`

```
python main.py --goal-drive \
    [--goal-heading-deg 25 | --goal-xy 1.5 0.5] \
    [--v-max 0.4] [--wz-max 1.2] [--display]
```

- Runtime re-issue (bench): stdin lines `heading <deg>` / `xy <x> <y>` /
  `stop`. Parsed on the main thread; shared with the planner via a small
  lock-free slot (same latest-wins style).
- Telemetry: odom comes from `ServerRuntimeMessage` telemetry
  `TelemetryFrame.drive.odom_*` (already published by firmware; add an
  accessor on `RobotState` in `robot.py` if missing when implemented).

### 6.6 Safety (Phase A)

Inherited from `DriveNode`, unchanged:
- Mode gate: twists only in `MODE_RUN_POLICY` (bench override flag exists).
- E-stop: telemetry estop → latch zero twist (requires manual reset).
- Deadman: BEV older than `deadman_s=0.5` → zero twist ("waiting").
- Stream rate 10 Hz ≫ 2 Hz arbitration minimum; stopping always wins (zero
  twist is never rejected).
- New for navd: camera process crash → no twists at all (watchdog stops the
  robot within 500 ms); BEV `plane_ok == False` on **both** cameras for >
  1 s → zero twist (floor estimate untrustworthy).

### 6.7 Phase A tests + acceptance

- Unit (synthetic depth): flat floor → all free; box at 1 m → occupied cells
  at the right BEV location ±1 cell; stair drop → hazard; overlap region →
  near-camera authority; stale camera → region degradation. Run offline, no
  hardware.
- Bench acceptance demo:
  1. Waypoint 3 m ahead of the robot with a chair offset ~0.7 m in the path
     → robot arcs around it, arrives within 0.3 m, stops.
  2. Human steps into the path at ~1.5 m → robot stops before 0.5 m.
  3. BEV debug overlay (`--display`) shows the chair as inflated occupied
     cells and the floor free.
  4. E-stop during a run → immediate zero twist.

---

## 7. Phase B — dataset + student model (spec)

### 7.1 Recorder v2 → MCAP sessions → `datasets/navd-v0/`

> **SHIPPED 2026-09-06** — `bebop_vision/recorder_mcap.py`,
> `main.py --record-navd [--auto]`, `tools/mcap_extract.py`, Foxglove
> layout; verified end-to-end on-device (8 Hz, web-app download).
> Implementation deltas and operational facts: `docs/navd-b-handoff.md`.

Recording format: **one MCAP file per session**, written on-robot by
`bebop_vision/recorder_mcap.py` and copied to the workstation (scp) as the
single data artifact. MCAP is already the firmware's capture format (policy
capture), indexed, and opens directly in Foxglove for review.

Channels (JSON-encoded, ~10 Hz):

| Channel | Payload |
|---|---|
| `/color_near` | `foxglove.CompressedImage` (JPEG, q85, base64 JSON) — renders in Foxglove Image panel |
| `/depth_near`, `/depth_far` | raw PNG (uint16 mm, lossless — **training channels**) |
| `/depth_near_preview` | `foxglove.RawImage` 106x60 16uc1 — dashboard depth view |
| `/bev_map` | `foxglove.RawImage` 60x60 rgb8 top-down teacher map + goal arrow |
| `/cmd_vel` | `{vx, wz}` — the operator's teleop twist from firmware telemetry |
| `/odom` | `{x, y, theta}` |
| `/goal` | `{type, heading_rad \| xy}` — current goal slot |
| `/bev_teacher` | 60x60 uint8 geometric grid (raw + plane_ok) — computed online by the Phase A code |
| `/calib` | intrinsics + rig extrinsics, once at session start |

Review sessions in Foxglove with `foxglove/bebop_navd_layout.json`
(generate via `python3 foxglove/make_foxglove_layout.py --layout navd`).

Teacher labels = **models, not hand rules** (revised 2026-09-06, supersedes
the geometry-only teacher):

- **YOLO-seg** auto-label pass over recorded color (workstation GPU,
  `yolo26l-seg`) — semantic obstacles (person/chair/box/...), bulk.
- **Hand labeling** on top of YOLO: human review/correction of the fused
  grid + masks for hard frames (grid-level paint tool; SAM3 assist where
  masks are needed).
- **Geometric BEV** (`/bev_teacher`) stays in the file as a third opinion —
  it sees walls/negative obstacles RGB misses, and doubles as the runtime
  fallback.

Fusion follows `labelnav.py`: semantic masks → obstacle class + dilation
margin band, unioned with geometric occupancy; teacher/model disagreement
lands in the *caution* class (free hard-negative mining).

Target: 5–10 k frames over 3–5 teleop sessions (varied obstacles, lighting,
goal directions; include the failure cases: glass door/table, black bag,
reflective floor).

### 7.2 Student model (`bebop_vision/navd.py`)

- **Inputs**
  - `depth_near` [1,1,240,424] f32, meters, clipped [0.3, 6.0], 0 = invalid
  - `depth_far`  [1,1,240,424] f32, same
  - `color`      [1,3,240,424] f32 (PE camera) — added 2026-09-06: teacher
    labels now carry RGB semantics; without a color input the student
    cannot predict dark/low-texture obstacles at runtime (depth sees
    neither glass nor dark objects well, so there is no depth-only signal
    to distill them from)
  - `goal`       [1,1,60,60]  f32, BEV-shaped raster of the goal direction
    (unit-gradient fan from robot origin toward the goal heading)
- **Output**: `logits` [1,3,60,60] — same class semantics as navseg:
  0 blocked, 1 navigable, 2 caution (caution = within the inflation margin).
- **Backbone**: shallow SegFormer-style encoder (SegFormer-B0 variant with
  multi-modal stem, or a 4-level UNet; pick whichever trains better at
  ~3–5 M params — decide by val mIoU, same metric as `train_nav.py`).
- **Loss**: `CE(logits, teacher_BEV)` (class-weighted, inverse frequency —
  reuse `class_weights_from`) `+ λ * imitation term` (cross-entropy between
  the planner-score argmax over the predicted grid and the operator twist
  direction bin; λ ≈ 0.2).
- **Augmentations**: horizontal flip (flip goal channel too!), depth dropout
  blobs (simulates IR dropouts), per-pixel depth noise σ=10 mm, brightness/
  color jitter (color input now present), goal-heading resampling (same
  scene reused with many goals — free diversity).
- **Coded runtime safety envelope (not learned)**: deadman, e-stop latch,
  mode gate, final near-cone check on whatever grid the model outputs, and
  auto-fallback to the geometric BEV below.

### 7.3 Export + runtime swap

- `tools/export_navd_onnx.py`: two-artifact ONNX (opset 17), fixed names
  above, **parity gate ≥ 0.99 cell-wise argmax agreement vs torch** over
  sampled dataset frames (same pattern as `export_navseg_onnx.py`).
- Runtime: `--navd-model weights/navd` switches the BEV source from
  geometric to student (runs via onnxruntime CUDA EP, ~10 Hz). The planner
  and DriveNode are untouched — the grid is the seam.
- **Auto-fallback to geometric** when: student output stale (> 0.5 s), any
  NaN/Inf, or `frac_navigable` outside [0.05, 0.95] (implausible — floor or
  wall everywhere). Log which provider produced each grid (mirror
  `NavState.provider`).

### 7.4 Phase B acceptance

- Student ≥ geometric teacher on the failure-case suite (glass/black/reflective
  scenes: teacher marks them navigable+collision-prone, student must not).
- No regression vs teacher on normal scenes (val mIoU + bench runs).
- Bench demo: full goal-drive run with the student grid; force a fallback
  mid-run (kill the model) → robot stops safely, reverts, continues.

---

## 8. Phase C — hardening (spec sketch, scoped at Phase B exit)

- **Packaging**: `bebop-vision.service` systemd unit (After=bebop-linux,
  Restart=on-failure) + `install-jetson.sh` step shipping the code + a pinned
  wheel set (or the venv). Autonomy as a managed service, not a dev shell.
- **Telemetry**: BEV grid push over WS mirroring `NavMaskFrame`
  (grid bytes + fracs + provider); app/Foxglove overlay.
- **App integration**: proto extension — client oneof field 22
  `SetNavigationGoal { oneof goal { float heading_rad; Vec2 point_odom; } }`
  consumed by the navd process via its existing WS client; app UI = tap on
  video → heading, or map click → odom point.
- **Docs**: flat-floor limitation, operator-arbitration behavior with
  autonomy in the seat, tuning guide for the planner constants.

---

## 9. Legacy OBSBOT webcam retirement plan

The OBSBOT Tiny 2 PTZ (the firmware's `/dev/video0` USB webcam) was prototype
hardware for the RGB nav experiments. With navd, the Orbbec 335Lg pair owned
by bebop-vision becomes the only vision sensors, and the OBSBOT pipeline is
removed end-to-end. Constraint: **the operator must never lose the live video
feed**, so the replacement stream exists before anything is deleted.

### 9.1 Inventory (everything attached to the old webcam)

**Firmware (`firmware/bebop-linux/`) — delete:**
- `src/video.rs` — `VideoHub`, `rscam` MJPG V4L2 capture, PTZ pose stamping
- `src/ptz.rs` — OBSBOT UVC pan/tilt (CIDs 0x009a0908/09)
- `src/nav.rs` — navseg runner (subscribes the video hub; observe-only
  telemetry) + `src/bin/nav_probe.rs`
- `config/bebop_wheeled.yaml` `video:` + `nav:` blocks;
  `config/navseg.onnx` + `navseg.onnx.data`
- `Cargo.toml`: `rscam`, `zune-jpeg` (both single-consumer)

**Firmware — edit:**
- `lib.rs` module decls; `main.rs` (`--nav` CLI, `resolve_nav_path`,
  VideoHub/nav spawn wiring); `config.rs` (`VideoConfig`, `NavConfig`, the
  `nav:`-requires-`video:` validation); `server/ws.rs` (`/video` route +
  multipart handler, nav-mask push pump, `SubscribeNav` plumbing);
  `server/handlers.rs` (`SetCameraPose` → PTZ, `SubscribeNav` ack, snapshot
  fields); `server/telemetry.rs` (`build_camera_state`, `build_nav_state`)

**bebop-vision:** `config.py::DEFAULT_SOURCE` (repoint, Stage 1);
`robot.py::set_camera_pose` + camera-pose fields (deprecate).
`camera.py` stays (source-agnostic reader; used by recorder/training).

**bebop-app:** `VideoFeed` source swap (Stage 1); "Labels" nav overlay +
`subscribeNav` machinery; `PtzJoystick`, `useCameraPtz`, gamepad right-stick
PTZ aiming; `CameraView`/`NavView`/`NavMaskView` types;
`wsTransport.setCameraPose`/`subscribeNav`.

**Infra:** CI `libv4l-dev` install + `navseg.onnx(.data)` release staging;
`install-jetson.sh` `v4l-utils` prereq block (only consumer was OBSBOT PTZ
jogging), OBSBOT sentence in the `setup_orbbec` comment, `nav_gpu_check`
(nav-runner CUDA health).

**Proto (`bebop_runtime.proto`) — deprecate, do not delete** (wire compat;
field numbers get `reserved` only in a dedicated bindings-refresh PR):
`SetCameraPose` (client 19), `SubscribeNav`/`UnsubscribeNav` (20/21),
`NavMaskFrame` (server 8), `CameraState`/`nav` telemetry fields (17/18) and
the `NavState`/`NavMaskFrame` messages.

**Untouched:** the entire safety/control stack (supervisor, watchdogs, CAN,
policy runner, IMU, powerboard), `/healthz`, `/ws`, `/captures`, all other
app screens, jetson-agent, `bebop_v2.yaml` (never had `video:`).

### 9.2 Staging (order matters)

- **Stage 1 — replacement stream.** `bebop_vision/videoserver.py`:
  threaded HTTP server on `:9091`, route `/video` serving **H.264
  fragmented-MP4** (`video/mp4`, chunked transfer) — played in the Tauri
  app via MSE in a `<video>` element (VideoFeed is rewritten from the
  legacy MJPEG `<img>` to an MSE player).
  - **Encoding**: software **libx264 via PyAV** (verified available in the
    venv) at **1280x800@30**, preset superfast / tune zerolatency, ~2–3
    Mbps, hardware-style GOP (`keyint=30`, `ref=1`, `bframes=0`) — measured
    45 fps encode on the Orin Nano, and NVIDIA's app-note tables put the
    tuned config at ~18% of one core at 30 fps (the board has no NVENC; see
    Section 3.2 table). Full native resolution of the near camera, no
    operator-side requirements (H.264 decodes everywhere via MSE).
  - **Encoder isolation**: encode runs in its own thread (PyAV
    `CodecContext`), fed the latest color frame at the stream rate; if the
    BEV worker contends for CPU, the stream drops to 15 fps before BEV ever
    drops (control-critical path wins).
  - The H.264 bitstream is also what navd-v0 recording can archive later
    (bitstream passthrough instead of per-frame JPGs) — synergy, not a v1
    requirement.
  - Acceptance: teleop video shows the Orbbec view while the firmware
    `/video` still runs (both live = zero-downtime cutover); live-camera
    H.264 encode sustains 30 fps at 1280x800 with total encoder CPU ≤ ~25%
    of the board (measured acceptance replaces the synthetic benchmark).
- **Stage 2 — firmware removal** (one PR): full inventory above. Acceptance:
  `cargo build`/CI green, robot boots with all control/telemetry surfaces
  intact, `/video` returns 404, `navseg.onnx` out of the release bundle.
- **Stage 3 — client cleanup:** app PTZ/Labels removal; `robot.py`
  deprecations; docs (`bebop-vision/README.md` camera-ownership + URL
  sections, `install-jetson.sh` comments).
- **Stage 4 — proto hygiene:** deprecate/reserve the fields listed above,
  regenerate TS + Python bindings, drop dead generated code.

### 9.3 Decisions

- **Firmware nav runner: retired with Stage 2.** It was observe-only RGB
  telemetry; navd supersedes it. The `ort` + CUDA EP loading pattern from
  `nav.rs` remains the reference for any future firmware-side student port.
- **PTZ: no replacement.** The Orbbecs are fixed-mount; the app aiming UI is
  removed rather than remapped.
- **`navseg` Python tooling: kept** (workstation training/recording only).
  Its recorder can source from the Stage-1 stream or the Orbbec rig
  directly — the RGB student remains useful as a Phase B ablation baseline
  and teacher-adjacent tooling.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Flat-floor assumption breaks (ramps, carpet transitions) | v1 documented limitation; RANSAC residual → caution band; dual `plane_ok` gate stops the robot if both cameras lose the floor |
| ED camera (far) still on USB 2.0 cable | Blocker tracked; far field limited to 640x360@10 until swapped — Phase A can run near-cam-authoritative (PE, full speed) in the meantime; profiles restore automatically at 5 G |
| Depth noise at 3 m (335Lg ≈ 1% of range) | Temporal filter + inflation margin; caution class absorbs the band |
| Software H.264 encoder CPU cost (no NVENC on Orin Nano) | Measured: x264 1280x800@30 ≈ 45 fps untuned; NVIDIA app-note GOP tuning (keyint=30, ref=1, bframes=0 — §3.2) brings 30 fps to ~18% of one core; encoder thread degrades to 15 fps before BEV/control ever drop |
| GPU contention (navseg CUDA + navd student) | Stagger rates if needed (navd 10 Hz, navseg 10 Hz is the budget to verify with `nav_probe` + `tegrastats`); navd can run CPU EP for the planner-critical path if GPU saturates |
| Arbitration conflict (teleop operator vs navd) | Documented: first non-zero client holds the seat; navd acquires it when started, operator gamepad takes over by simply moving the stick; navd zero-twist never blocks a stop |
| Glass / reflective / black obstacles invisible to IR | Explicit Phase B motivation; geometric teacher knowingly fails these; student trained on teleop human avoidance of them |
| Two cameras drift (mount knocks) | Extrinsics verification procedure (Section 6.2) is a 5-minute bench check; rig YAML is the only thing to re-measure |
| Single-process autonomy crash | Watchdog (500 ms) stops the robot; systemd auto-restart (Phase C) |

---

## 11. Open items / prerequisites

1. **Swap `CPBLC53000ED` USB cable** (same fix as PE) — required for the full
   far-field horizon (it is the planning-horizon camera). Profiles restore
   automatically (verified: negotiation picks 848x480@10 today, 30 fps after).
2. Jetson `bebop-vision/.venv`: Phase A deps installed (pyorbbecsdk2, numpy
   1.26 pinned, opencv 4.11, pyyaml, websockets, protobuf, mcap, pytest —
   2026-09-05/06). `pip install -e .` for torch etc. still pending before
   Phase B training (long install — start overnight).
3. Extrinsics **measured from live floor fits** (2026-09-05, auto-estimator):
   near `CPBLC53000PE` = 1.27 m / pitch −66°; far `CPBLC53000ED` = 1.32 m /
   pitch −17.4°; written to `config/orbbec_rig.yaml` (the doc's earlier
   0.55/−35 and 0.75/−12 numbers were stale placeholders). The bench
   verification procedure (flat floor + box at a known spot) still to run.
4. One-time intrinsics dump per serial — **done** 2026-09-05
   (`tools/orbbec_intrinsics.py` → `config/orbbec_intrinsics_<serial>.json`).
5. ~~Decide final near/far serial-role mapping~~ — resolved: **PE = near,
   ED = far** (confirmed from the mounted cameras' views, 2026-09-05).
6. **OBSBOT retirement** (Section 9): Stage 1 video server (`videoserver.py`
   + app port default) lands before any firmware removal; then Stage 2–4.
7. ~~Videoserver H.265 track: NVENC~~ — **resolved 2026-09-05**: Orin Nano
   has no hardware encoder; verified libx264/libx265 via PyAV on-device and
   benchmarked (Section 3.2 table). **Decision: H.264 stream** — x264
   1280x800@30, superfast/zerolatency + tuned GOP, ~2–3 Mbps. H.265 dropped:
   its only win (bitrate) was irrelevant at 2 Mbps, and it cost ~4x encoder
   CPU plus a fleet HEVC-decode requirement.

---

## 12. References

- `bebop-vision/bebop_vision/`: `camera.py` (threading pattern), `navseg.py`
  (model I/O), `planner.py` (`SectorPlanner`, `DriveNode`), `robot.py`
  (WS client, twist + telemetry), `recorder.py`, `labelnav.py`, `config.py`.
- navd Phase A modules: `bebop_vision/orbbec.py` (camera service + profile
  negotiation + intrinsics cache), `bebop_vision/bev.py` (BEV grid),
  `bebop_vision/goal_planner.py` (planner + `GoalDriveNode` + goal slot),
  `main.py --goal-drive` (wiring), `config/orbbec_rig.yaml` (extrinsics),
  `tools/orbbec_intrinsics.py`, `tests/` (35 synthetic unit tests).
- `bebop-vision/train_nav.py`, `tools/export_navseg_onnx.py` (export contract).
- `firmware/bebop-linux/src/`: `nav.rs` (ORT pattern, CUDA EP — retired per
  Section 9, kept as reference), `video.rs` (V4L2 UVC capture — retired per
  Section 9; the Rust-port path for depth), `supervisor.rs`
  (arbitration + watchdog), `drive.rs` (twist kinematics), `config.rs`
  (`nav:` block — retired per Section 9).
- `jetson-agent/bebop-proto/proto/bebop_runtime.proto` (message schemas;
  next free client field: 22).
- `scripts/install-jetson.sh` + `scripts/orbbec-99-obsensor-libusb.rules`
  (camera bring-up).
- Local bring-up artifacts (not in repo): viewer script
  `/tmp/opencode/orbbec_viewer.py`, probe script
  `/tmp/opencode/orbbec_profiles.py`.
