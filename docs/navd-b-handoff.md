# navd Phase B — handoff brief

For the agent/person picking up Phase B (learned student model). Read this
together with `docs/navd.md` (the spec) — this file covers **what already
shipped, the decisions that deviated from the spec, environment facts, and
the exact remaining work**. State as of 2026-09-06, `main` @ `ad631c6`,
CI green.

---

## 1. What shipped (all on `main`, deployed to the robot)

### Phase A — geometric autonomy (docs/navd.md §6) — DONE

| Piece | File | Notes |
|---|---|---|
| Camera service | `bebop-vision/bebop_vision/orbbec.py` | serial-matched, profile negotiation, filter chain, intrinsics cache, pixel self-mask |
| BEV grid | `bebop-vision/bebop_vision/bev.py` | deproject → batched RANSAC → classify → fuse → inflate; float32/bincount hot path |
| Planner + gating | `bebop-vision/bebop_vision/goal_planner.py` | `GoalHeading`/`GoalPoint`, 13-ray polar, search-latch, `GoalDriveNode` (mode/estop/deadman/no-floor) |
| Entry point | `bebop-vision/main.py --goal-drive` | stdin goal slot, BEV worker thread, `--display` overlay |
| Rig config | `bebop-vision/config/orbbec_rig.yaml` | **measured extrinsics** (near 1.27 m / −66°, far 1.32 m / −17.4°), self-mask rects, dead disc |
| Intrinsics dump | `bebop-vision/tools/orbbec_intrinsics.py` | per-serial JSON in `config/` (already dumped on the robot) |
| Tests | `bebop-vision/tests/test_bev.py`, `test_goal_planner.py`, `test_orbbec.py` | 26 synthetic tests, no hardware needed |

Verified live on the Jetson: 8–9 Hz fused grids, WS gating, correct
hard-stop before obstacles. **One §6.7 acceptance gap**: the formal demo
(3 m waypoint arc-around + human-steps-in + e-stop-during-run) was
interrupted twice — only stop-before-obstacle was observed. Worth
re-running when the floor is clear, but it does not block Phase B.

### Recorder v2 (§7.1) — DONE

| Piece | File | Notes |
|---|---|---|
| MCAP writer | `bebop-vision/bebop_vision/recorder_mcap.py` | 10 Hz channels, shared log_time per tick (ns!), Foxglove-schema image channels |
| Auto-segment CLI | `bebop-vision/main.py --record-navd [--auto]` | segments follow DialIn/RunPolicy + armed; roll 400 MB / 10 min; prune oldest `navd_session_*` under budget; single-instance lock |
| Extractor | `bebop-vision/tools/mcap_extract.py` | session → `datasets/navd-v0/<name>/{color,depth,labels,manifest.jsonl}` |
| Foxglove | `foxglove/layout_navd.py` → `bebop_navd_layout.json` | Image panels (`imageMode.imageTopic`!), twist/odom/goal/plane plots |
| Firmware | `firmware/bebop-linux/src/server/ws.rs` | `GET /captures` lists `navd_session_*` too (**deployed**, sha `716a2be`) |
| Installer | `scripts/install-jetson.sh` | pre-creates `/var/lib/bebop-captures` as the invoking user |

## 2. Decisions that deviate from / extend the spec (important)

The spec doc was revised where noted, but several implementation facts
live only here:

1. **Self-view suppression is two-layered** (§6.3 revision): a body-frame
   footprint box **plus per-camera pixel rects** (`self_mask_pixels` in the
   rig YAML — the chassis occupies fixed pixels because the mount is
   rigid). Without both, the robot sees its own frame/foam boards as
   obstacles. Plus a **0.55 m body-frame dead disc** (`bev.min_range_m`) —
   chassis-silhouette stereo artifacts sit just outside the mask and
   nothing inside the disc is actionable (rays start at 0.35 m).
2. **Plane residual is range-scaled**: `max(15 mm, 1% of point range)` —
   the fixed 15 mm gate starved the far camera's floor fit beyond 2.5 m.
   Plane-fit window is `[-(mount_height + 0.5), +0.05]` m.
3. **Planner state priority is NOT the spec's literal order**: search
   (latched, direction fixed at entry, hysteresis exit) → rotate when
   `|dpsi| > 0.5` → hard-stop when the near cone is blocked → rotate when
   best-corridor near-blocked → drive. Hard-stopping before rotate
   deadlocked the robot facing a small obstacle with a clear corridor
   beside it (bench run 2). Search latching fixed a left/right thrash
   (the original oscillation the user e-stopped).
4. **RunPolicy is a no-op on the wheeled robot** — the bundled policy.onnx
   is for the legged bebop_v2 and fails to load ("policy not loaded; will
   be a no-op"). Wheel motion comes from `tick_drive` executing cmd_vel
   whenever wheels are armed. So DialIn and RunPolicy are equivalent for
   wheeled teleop; the recorder triggers on **either** mode + armed +
   non-estop.
5. **MCAP specifics**: log_time/publish_time are **nanoseconds**; schema
   names must be exactly `foxglove.CompressedImage` / `foxglove.RawImage`
   (registry names like `foxglove.image.messages.*` are NOT recognized in
   MCAP — panels report "topic not available"); image `encoding` values
   are case-sensitive (`16UC1`, `rgb8`); **chunk compression must be off**
   (`CompressionType.NONE`) — mcap 1.4.0's zstd chunk path silently loses
   data past the 1 MB chunk boundary. Payloads are pre-compressed anyway.
6. **Foxglove layout**: Image panel topic binding is `imageMode.imageTopic`
   (current builds); legacy keys included for older builds. BEV map renders
   un-rotated (row 0 = 3 m ahead, robot at bottom-center).
7. **Camera exclusivity**: one process per camera. The recorder takes a
   single-instance lock (`/tmp/navd_recorder.lock`) — a second instance
   fails fast with the holder's PID. Stale orphans holding the camera
   otherwise surface as `uvc_open failed: -6`.
8. **Do not use thread pools around pyorbbecsdk capture**: pool-based
   encode/BEV was slower (GIL) and triggered `malloc(): unaligned fastbin
   chunk` corruption. Serial tick in the recorder thread is ~8 Hz and
   stable.
9. **Installer**: the capture dir `/var/lib/bebop-captures` is pre-created
   by `install-jetson.sh` as the invoking user (the root-run firmware only
   creates it if missing). Two `set -u` unbound-variable bugs
   (`SETUP_ORBBEC_ONLY`, `SETUP_ORBBEC`) were fixed — if you add flags,
   initialize their variables in the defaults block.

## 3. Environment & operational facts

- Robot: `ssh bebop@bebop.local` (pw `bebop`), repo `~/bebop`, synced to
  `origin/main`. Firmware `bebop-linux` runs as root (systemd), config
  `/etc/bebop/bebop_wheeled.yaml`, capture dir `/var/lib/bebop-captures`
  (web app: `http://bebop.local:9090/captures`, download
  `/captures/dl/<name>`).
- Jetson venv `bebop-vision/.venv`: pyorbbecsdk2 2.1.2, numpy **1.26.4
  (pinned — 2.x breaks pyorbbecsdk)**, opencv 4.11, pyyaml, websockets,
  protobuf 7.35.1, mcap, pytest. **No torch/transformers yet.**
- Hardware: near cam `CPBLC53000PE` (USB3, 848x480@30 + color), far cam
  `CPBLC53000ED` (**unfixed USB 2.0 cable — drops off the bus
  occasionally; currently usually run with `--roles near`**). Left wheel
  needed a manual calibration once (MISSING_ESTIMATE estop).
- Data on the robot: 7 navd sessions (2026-09-06, ~240 MB, near-only) in
  `/var/lib/bebop-captures`, all DialIn/static-to-slow-drive — usable for
  pipeline bring-up, **not** the real training set.
- Workstation has the sessions mirrored in `bebop-vision/datasets/sessions/`
  and one extracted example in `bebop-vision/datasets/navd-v0/`.
- Foxglove layout import: `foxglove/bebop_navd_layout.json` (regenerate via
  `python3 foxglove/make_foxglove_layout.py --layout navd --force`).
- 40 Python tests green (workstation + Jetson); firmware fmt/clippy/test
  green; CI on `main` green (runs on every push).

## 4. Phase B work plan (docs/navd.md §7.2–7.4)

In order; 1–4 are data plumbing, 5–7 are the model:

1. **Jetson venv full install** (§11.2): `pip install -e .` for
   torch/transformers — long install, do it early/overnight.
2. **YOLO-seg auto-label pass** (workstation GPU; `yolo26l-seg.pt`):
   instance masks over extracted color frames. Bulk labeling.
3. **Hand-label tool**: grid-level (60×60) review/paint over fused labels;
   writes `labels/{stamp}.npz` `hand` array. Training prefers `hand`,
   falls back to `teacher`.
4. **Fusion → navd-v0 labels**: YOLO masks + dilation margin ∪ geometric
   `/bev_teacher`; disagreement → caution class. Emit via the extractor.
5. **Data collection** (user drives; recorder is ready): 5–10k frames,
   3–5 sessions, varied obstacles/lighting/goals + the failure cases
   (glass, black bag, reflective floor). ED camera cable reseat first so
   far-depth is in the data (it is a model input).
6. **Student model** `bebop_vision/navd.py` (§7.2): inputs
   `depth_near [1,1,240,424]`, `depth_far [1,1,240,424]`,
   `color [1,3,240,424]`, `goal [1,1,60,60]` → `logits [1,3,60,60]`
   (0 blocked / 1 navigable / 2 caution). SegFormer-B0-style multi-modal
   vs 4-level UNet, ~3–5 M params — decide by val mIoU. Loss: weighted CE
   + λ≈0.2 imitation term (twist-direction bin). Augmentations: h-flip
   (goal channel too), depth dropout, depth noise σ=10 mm, color jitter,
   goal resampling.
7. **Export + runtime swap** (§7.3): `tools/export_navd_onnx.py` mirroring
   `export_navseg_onnx.py` (two-artifact ONNX, fixed tensor names, parity
   gate ≥ 0.99 vs torch); `--navd-model` flag swaps the BEV source
   (onnxruntime CUDA EP, ~10 Hz); auto-fallback to geometric on stale
   > 0.5 s / NaN / `frac_navigable` outside [0.05, 0.95]; log provider.
8. **Phase B acceptance** (§7.4): student ≥ teacher on the failure-case
   suite; no regression on normal scenes (val mIoU); bench demo with forced
   fallback (kill the model mid-run → stop safely → revert → continue).

## 5. Guardrails

- **The grid is the seam.** `GoalPlanner` + `GoalDriveNode` do not change
  for the student; only the grid provider swaps. Do not rewrite them.
- The **coded safety envelope stays coded**: deadman, e-stop latch, mode
  gate, near-cone final check, geometric fallback (§7.2 revision).
- Keep the test pattern: synthetic unit tests, hardware paths mocked/faked.
- Don't spin thread pools around pyorbbecsdk capture (see §2.8).
- If you add CLI flags to `scripts/install-jetson.sh`, initialize the
  variables in the defaults block (see §2.9).

## 6. Out of scope for Phase B

- §9 OBSBOT retirement (stages 1–4) — separate stream, do after/parallel.
- §8 Phase C (systemd unit for the recorder, BEV telemetry push over WS,
  app `SetNavigationGoal` field 22, tuning docs).
- User-owned items: ED cable reseat, driving the data-collection sessions,
  wheel re-calibration if the estop reappears.
- Nice-to-haves discussed but unbuilt: live `foxglove-sdk` WebSocket view
  (`ws://bebop.local:8765`) during teleop; `SetRecording` firmware flag
  (tag 22) for app-driven record control — the `--auto` trigger covers the
  workflow today.
