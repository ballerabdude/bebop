# nav — depth stop-cone + bearing waypoint driver

Status: **landed 2026-10-21**, bench-tested on the robot's cameras (idle
clearance map live on `:9097`); the first live move is pending a working
drive bus (see §6). This is the *fast layer* of the vision-first navigation
plan: a higher layer (place recognition, VLM) picks waypoints as a bearing +
distance; this module gets the robot there without hitting anything, using
depth only. It is written fresh and depends on nothing from the archived
navd / BEV driving stack.

## 1. Pieces

| File | Role |
|---|---|
| `bebop_vision/nav/depth_cone.py` | body-frame clearance map from the mast depth cameras |
| `bebop_vision/nav/bearing.py` | odom-closed-loop waypoint controller, one twist per tick |
| `bebop_vision/nav/pose.py` | gyro + wheel-odometry pose estimate (the goal frame) |
| `bebop_vision/nav/drive.py` | `NavDrive` loop (arming, deadman, abort rules) + `NavServer` HTTP |
| `bebop_vision/nav/synth.py` | synthetic depth renderer for tests/tools |
| `tools/nav_cone_preview.py` | offline check on recorded depth PNGs, renders a top-down view |
| `bebop_vision/nav/visual.py` | DINOv3 patch matching -> heading correction |
| `bebop_vision/nav/follow.py` | `RouteFollower`: teach-and-repeat over `NavDrive` |
| `tests/test_nav_*.py`, `tests/test_routes.py` | 69 hardware-free tests |

## 2. Depth stop-cone

Every valid depth pixel on a stride is unprojected with the depth
intrinsics and placed in the body frame (x forward, y left, z up, origin at
the mast on the floor) using the rig YAML height/pitch. A **per-frame floor
plane fit** (RANSAC within 15° of the prior, low-passed) then measures
heights from the real floor: on 2026-10-21 both cameras sat **~6-7° off**
the YAML pitch, enough to turn flat carpet 1 m ahead into "drops" without
the fit. The fit result is published (`floor_fit` in `/clearance`).

The near camera is the **reference**: it fits against the nominal floor
(within 15°, 15 cm). Every other camera must agree with the reference plane
to within 3° / 6 cm or it follows the reference plane plus a slowly learned
per-camera offset (live 2026-10-21, facing a desk, the far camera's own
RANSAC locked onto a plane 31 cm below the carpet and turned the floor into
obstacles). A fit also needs ≥ 45 % of its candidates as inliers.

Points classify by height: floor (|z| ≤ 6 cm), obstacle (10 cm ≤ z ≤
1.8 m), drop (z ≤ -10 cm, widened by 3 cm per metre of range). **Drops
are taken from the near camera only** (`drop_roles`): the far camera's
residual plane error produced phantom drops at 1.5-2.5 m, and the near
camera sees any real drop in time at 0.3 m/s. A corridor drop needs at
least 4 agreeing points. Chassis returns are removed by the rig's depth
self-mask polygon and by the learned footprint box (`config/ego_mask.npz`
extent, dilated 6 cm; a 15 cm ring around it ignores drops, which there are
rail-silhouette noise).

Output is a polar map over 15 sectors spanning ±60°: per sector the nearest
obstacle/drop range and the **first contiguous run of observed floor** (a
25 cm radial gap ends it, so floor seen beyond a hole or over a low bar does
not count). Clearance is the minimum of the two, capped at 2.5 m: unobserved
space is never free. A chassis-width corridor straight ahead is summarised
separately (`corridor_min_m`, `drop`). Cost: ~3.5 ms per tick on the
workstation, ~7 ms on the Thor for both cameras.

## 3. Pose: gyro heading, odometry distance

Wheel odometry over-reads in-place rotation on this carpet: on 2026-10-21
the firmware reported a 170° turn for a physical ~82° (both Orbbec camera
gyros agree once rotated into the body frame; the robot ended up facing the
desk while odometry said it faced the room). The driver therefore keeps its
own pose (`bebop_vision/nav/pose.py`, `GET /pose`): heading integrates the
camera gyros (200 Hz, rad/s, taps on the camera IMU buffers so the recorder
keeps its own stream), with a per-camera bias learned while the robot is
demonstrably still; distance is the wheel-odometry displacement projected on
the odom heading (signed, so reverse works). Until bias is learned, or when
the gyro stream stalls for 0.5 s, heading falls back to odometry deltas and
`heading_source` says so. Goals and `history` are in this pose frame; it
starts at the first firmware odom pose seen and is local to the process.
The firmware IMU (`ImuStats`) is not configured on this robot.

## 4. Bearing controller

A goal is a pose-frame point, set from a bearing + distance at the moment
of the request (or as x, y). Each tick: if the goal lies within the
observed clearance along its bearing, head straight for it; otherwise score
sectors by `clearance^1.5 · (1 + 2·exp(-(Δ/20°)²/2))` and take the best
with clearance ≥ `stop_m`. Heading error above 45° rotates in place
(refused when any return is closer than 0.5 m, the footprint corner sweep);
otherwise drive with speed scaled by clearance (full above 1.3 m, zero at
`stop_m` = 0.6 m from the mast, ~0.28 m from the bumper) and by the
remaining distance. Goals behind the camera FOV turn first. No map, a stale
map (> 0.4 s) or an untrusted frame holds still with the goal kept. Blocked
for 4 s aborts the goal. Defaults: `v_max` 0.3 m/s, `w_max` 0.9 rad/s,
reach tolerance 0.15 m, max goal 6 m.

## 5. Drive loop and HTTP

`NavDrive` runs inside the recorder process (the cameras are exclusive) at
15 Hz, reading the open rig slots. It is idle until a goal arrives, then
arms the way the voice `move_relative` tool does — DIAL_IN, enable wheels,
drive — and on every exit path stops, disarms and restores the previous
mode. A latched E-STOP, a lost link, stale telemetry (> 0.5 s) or a failed
arm aborts the goal; **nothing here ever clears an E-STOP**. The firmware's
500 ms operator watchdog is the backstop. Because the recorder's `--auto`
mode records whenever the robot is DIAL_IN + armed, every nav drive is
captured as an MCAP session for free.

```
GET  /healthz              liveness + state
GET  /status               state, last command + detail, goal, robot, history
GET  /clearance            latest clearance map (sectors, corridor, floor_fit)
GET  /pose                 gyro+odom pose estimate (heading_source, bias, gyro_hz)
POST /goal {bearing_deg, distance_m [, label]}   relative waypoint
POST /goal {x, y [, label]}                      odom-frame waypoint
POST /goal {reverse_m [, label]}                 straight reverse, ≤ 1 m, 0.15 m/s, blind
GET  /depth?role=near|far  latest raw uint16 depth PNG (debug; feed the preview tool)
POST /stop                 cancel (stops, disarms)
```

Reverse exists because the cone only looks forward: a dead end (clutter
closer than the 0.5 m turn clearance on both sides) is left by backing
along the path just driven, which is known free.

States: `idle`, `rotate`, `drive`, `reverse`, `blocked`, `no_depth`, `stale_depth`,
`untrusted_depth`, `reached`, `stopped`, `aborted_blocked`, `aborted_estop`,
`aborted_link`, `aborted_arm`.

## 6. Running

```bash
# in /etc/bebop/vision.env (the unit reads it):
BEBOP_VISION_EXTRA_ARGS="--no-bev --nav"
# flags: --nav-port 9097 --nav-roles near,far --nav-hz 15 --nav-v-max 0.30
#        --nav-w-max 0.9 --nav-stop-m 0.60
curl -s http://bebop.local:9097/clearance | python3 -m json.tool
curl -s -X POST -H 'Content-Type: application/json' \
  -d '{"bearing_deg": 0, "distance_m": 0.5, "label": "test"}' \
  http://bebop.local:9097/goal
curl -s http://bebop.local:9097/status
```

Offline: `.venv/bin/python tools/nav_cone_preview.py --near near.png
--far far.png --out cone.png` on uint16 depth PNGs (e.g. the bench pair in
`~/bench_e11/pair` on the robot).

## 7. Teach mode for routes

`bebop_vision/places/trail.py` (`TrailTeacher`, started by `--nav` on the
places service) records a **route** while the robot is driven, by teleop or
by nav goals. A route is an ordered list of nodes in the route frame (origin
and heading at the start of the drive), each with the distance along the
trail, far + near keyframe JPEGs and a DINOv3 descriptor of the far view. The
pose comes from `GET :9097/pose` (camera-gyro heading + odometry distance);
teaching refuses to start until that heading source is the gyro. Nodes are
taken every 0.6 m of travel or after a 25° heading change (`--teach-step-m`,
`--teach-turn-deg`), plus the first and last pose.

```
POST :9095/routes/teach/start {"from": "office room", "label": "via hallway"}
    ... drive ...
POST :9095/routes/teach/stop  {"to": "Master Bedroom"}     # or a new name
POST :9095/routes/teach/cancel
GET  :9095/routes/teach                     status (nodes, length, pose)
GET  :9095/routes[?from=&to=]               taught routes
GET  :9095/routes/{id}                      nodes (poses, keyframes)
GET  :9095/routes/{id}/nodes/{seq}/keyframe?role=far
DELETE :9095/routes/{id}
```

`from` / `to` default to place recognition (`/whereami`), but recognition
must clear its threshold or the call fails and asks for the name; a new `to`
name creates the place from the current view. Stopping stores the route
(`routes` + `route_nodes` tables, JPEGs under `routes/<id>/`) and records the
A→B edge with the trail as its guide path, so `/places/graph` edges now carry
`route_ids`. Nothing here moves the robot; following a route is the next
piece.

## 8. Route following (teach-and-repeat)

`bebop_vision/nav/follow.py` (`RouteFollower`, on the nav server) drives a
taught route node by node:

- the route frame is anchored to the live gyro pose at start (the robot is
  assumed to stand at node 0, at the route's start place);
- the target node's bearing and distance come from dead reckoning;
- the live far view is matched against the node's keyframe with DINOv3
  **patch** features (`nav/visual.py`: mutual-nearest-neighbour matches on
  the 14x14 grid, median column offset, through the far color intrinsics).
  A confident match replaces the dead-reckoned bearing and low-passes a yaw
  correction into the route frame; an unconfident one falls back to dead
  reckoning. Measured on the Thor: 53 ms per frame, bearing from a known
  pixel shift within 0.5°, and the fraction gate (≥ 30 % of patches
  matched) separates the taught scene from an unrelated view (16 %);
- the result is streamed to the driver as a short goal (≤ 0.8 m lookahead)
  at 3 Hz, so avoidance, arming and the E-STOP rules stay with `NavDrive`;
- a node is passed within 0.35 m or once its segment is overshot; arriving
  at the last node asks place recognition to confirm (`arrived` vs
  `arrived_unconfirmed`); two driver aborts in a row fail the run. Every
  outcome except an operator stop is recorded on the place graph edge.

```
POST :9097/route/follow {"route_id": "..."}            # or {"to": "Master Bedroom"[, "from": ...]}
GET  :9097/route                                         state, target node, match, history
POST :9097/route/stop
```

`to` without `from` uses place recognition for the start and takes the
newest taught route between the two places. Only forward following exists:
to come back, teach the return trip.

## 9. Live results (2026-10-21) and caveats

- `fwd 0.5 m`: reached in 2.4 s (stopped 0.36 m in, the old 0.15 m reach
  tolerance; now 0.08 m).
- `fwd 1.5 m`: reached in 5.9 s, 1.36 m travelled, two one-tick
  `corridor_blocked` holds from phantom far-camera drops (fixed above).
- `home` (odom origin, 1.7 m behind): rotated 180° in place, drove back,
  then blocked 0.4 m short with real clutter (tools, a box, a chair base)
  0.58 m from the mast = ~0.26 m from the bumper. Aborted after 4 s as
  designed. A following `about-face` goal was refused (`turn_blocked`):
  returns inside the 0.5 m corner sweep on both sides. That is the case the
  reverse goal is for.
- `reverse 0.8 m`: reached, 0.72 m, heading held. The following
  `about-face` rotated by 170° of *odometry* but ~82° physically (gyro),
  ended facing the desk 0.47 m away and was refused a further turn
  (`turn_blocked`, returns within the 0.5 m corner sweep). That run is what
  produced the gyro pose above.
- The near camera sees floor from ~0.57 m (past the chassis frame) to
  ~1.7 m; the far camera extends obstacles to the 2.5 m cap. Clearances are
  from the mast; `stop_m` accounts for the 0.32 m to the bumper.
- Sides and rear are unobserved: rotation in place is allowed when no
  return is within 0.5 m, which is a judgement, not a guarantee, and
  reverse is blind by construction.
- The rig YAML pitch values are stale by ~5-7°; the per-frame fit hides
  that for nav, but any other consumer of those extrinsics should re-measure.
- Idle cost: ~10-20 ms per tick at 15 Hz on the Thor, CPU only.
