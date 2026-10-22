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
| `bebop_vision/places/planner.py`, `bebop_vision/nav/trip.py` | Dijkstra over taught routes; `TripRunner` chains legs |
| `bebop_vision/nav/object_goal.py` | `ObjectSeeker`: VLM box -> bearing + depth range -> driver goals |
| `bebop_vision/nav/navigator.py` | `Navigator`: one command (voice/text) -> trip, seek or move |
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

## 3a. Turning in tight spaces

The old rule "refuse any in-place turn when a return is within 0.5 m" made
the robot timid beside furniture and blind to which way was clear. The
controller now uses the chassis geometry (`sweep_blocked`): each footprint
corner sweeps an arc at its own radius for the requested turn, and the turn
is refused only when a sector return sits inside one of those arcs (plus a
4 cm radius margin and half a sector of angle); anything closer than 0.28 m
anywhere refuses everything. So facing a wall at 0.45 m, a right turn of
60° (front-left corner, 0.37 m radius) is fine while a left turn of 90°
(front-right corner, 0.47 m) is not, and the controller picks the side that
clears.

When blocked (corridor, too close, drop, or a refused turn) it **escapes**:
drive an arc toward the best open sector within 45° if one exists, else
turn in place toward the most open sector whose sweep is clear, else back up
0.3 m once per goal; only then does the 4 s blocked timer count toward an
abort. Escape turns and large sector rotations are **committed** to an
absolute heading and driven to completion, and sector steering keeps its
side for 3 s, because re-choosing every tick as the view rotated made the
robot oscillate left and right live. A navigator turn command that is
refused backs up and re-issues only the *remaining* angle to the original
target (a fresh 180 on top of a partial turn over-rotated by 35° live).

Live 2026-10-21, boxed in beside a crib (returns 0.44-0.8 m all round,
corridor 0.38 m): "turn around" → one 0.3 m back-off, then a single
committed turn of 178° in 6 s; "go to the dresser" → found at 2.0 m, two
legs, arrived, no detours.

## 3b. Debugging the driver's choices

`GET :9097/trace?n=300` returns one row per driver tick while a goal is
active (60 s ring): state, mode (`direct` / `sector` / `arc`), the chosen
heading, goal bearing and distance, clearance, corridor minimum and the
nearest corridor return `(x, y, z)`, block reason, escape, commitment,
latched side, the fitted floor pitch/height, the commanded twist and the
gyro heading. Every change of decision is also a journal line
(`[nav] decide ...`). `tools/nav_trace.py` prints the trace collapsed to
changes of mind (`--all` for every tick, `--follow` to tail live):

```
cd ~/bebop/bebop-vision
.venv/bin/python tools/nav_trace.py --follow      # on the robot, or --host bebop.local
```

What the trace found on 2026-10-21 (master bedroom):

- **The front castor.** The corridor's nearest return sat at exactly the
  chassis cull edge on every heading (`corridor_m=0.38`, then 0.46 and 0.56
  as the cull was widened), at the front-right corner, 11-24 cm high: the
  front swivel wheel and fork, which swing outside the learned footprint
  after a turn (and were sticking). With the corridor "blocked", sector
  steering was never allowed to pick straight ahead, so even "forward 1 m"
  zigzagged ±17°. Fix: the front self-cull now reaches 0.56 m from the mast
  (`footprint_front_margin_m`), plus explicit low castor envelopes at both
  front corners (`castor_zones_m`); nothing within 0.56 m of the mast is
  visible, which is inside the 0.6 m stop distance anyway. With the castor
  fixed, the return disappears.
- **Flapping between direct and sector steering**, and a latched side that
  fought a goal on the other side: fixed with mode hysteresis (`commit_s`),
  dropping the side latch when the goal is clearly on the other side, and
  capping escape turns per goal (`max_escape_turns`), which were resetting
  the blocked timer indefinitely.
- **Reversing on castors** yaws the chassis 5-14° even with the heading
  hold; the escape reverse now uses 0.22 m/s and gives up after 1.5 s of no
  motion.
- Drops are only trusted within 1.5 m of the near camera and need ≥ 15
  points; mirror reflections produced 2,000 "under-floor" points at 1.8 m to
  the side.

## 3c. Path goals (pure pursuit)

A goal can be a list of pose-frame points (`POST /goal {"path": [[x, y],
...]}`) or body-frame offsets from the current pose (`{"path_body":
[[forward, left], ...]}`). The controller projects the robot onto the path
(monotonically, never jumping back to an earlier leg), aims at the point
0.6 m further along it (0.35 m when the corridor is tight), and feeds that
bearing into the same sector / avoidance / escape machinery as a point
goal. Speed is scaled by the distance to the **end** of the path, so
intermediate points are passed without stopping, and capped by the bend
ahead (full speed when the path direction at the lookahead matches the
heading, 40 % at a right angle). Re-sending a path keeps progress: the new
path is projected from the nearest segment, which is what the route
follower will do after each yaw correction. `path_progress`, `path_seg`,
`aim` and `bend_deg` appear in the status detail and the trace. Reached
when the last point is within tolerance (or the projection has passed it
and it is within 0.25 m).

Tests cover an L-shaped path driven without a stop or an in-place turn at
the corner, the corner speed cap, progress-preserving re-sends, obstacles
still blocking, and the body-frame form. Next: the follower and trip runner
sending whole trails as paths, and the seeker looking while driving.

## 3d. Near-camera depth tuning (SDK presets and filters)

`tools/orbbec_depth_tune.py` opens a camera directly (stop `bebop-vision`
first), captures a few seconds under each preset × filter set with the
robot still, and scores what the stop-cone cares about: valid pixel
fraction, achieved fps, per-pixel temporal noise on floor pixels (median,
by range band), and **flicker**: floor pixels (floor in the median frame)
that read as an obstacle (≥ 0.10 m) or a drop within 1.2 m in a given frame.

Measured 2026-10-22 on the near camera (848×480 @ 30, 2.5 s per set):

| preset / filters | valid | floor noise 0.5-1 / 1-1.7 m | phantom obstacles per frame (max) |
|---|---|---|---|
| High Accuracy, none | 87.4 % | 1.5 / 2.5 mm | 0 (0) |
| High Accuracy, temporal | 87.5 % | 0.8 / 1.3 mm | 0 (0) |
| **High Accuracy, noise + temporal** | 87.5 % | 0.7 / 1.3 mm | **0 (0)** |
| High Accuracy, noise + temporal + hole | 97.8 % | 0.8 / 1.3 mm | 3.9 (13) |
| High Accuracy, temporal + hole (old default) | 97.8 % | 0.8 / 1.3 mm | 7.1 (24) |
| High Density, noise + temporal | 92.9 % | 0.6 / 1.1 mm | 0.9 (3) |
| High Density, temporal + hole | 98.2 % | 0.6 / 1.1 mm | 3.5 (7) |
| Default, temporal + hole | 97.9 % | 0.7 / 1.2 mm | 6.3 (24) |

The raw depth is quiet; **hole filling** is what invents raised points, and
one such point inside the corridor brakes the driver. The near camera now
runs High Accuracy with `[noise, temporal]` (zero phantoms), accepting
~12 % holes, and the corridor minimum is the third-nearest blocking return
(`corridor_min_points`) so an isolated pixel can never block. The far
camera keeps `[temporal, hole]` for coverage; it is not the drop authority,
but its hole-filled phantoms would still count in the sector map, so it
deserves the same sweep (`--role far`) next.

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

First live follow (2026-10-21, office lane): taught a 1.16 m / 4-node
straight leg by nav goal, reversed back to the start (heading held), then
`POST /route/follow`: every tick matched the taught keyframes confidently
(96-99 of 196 patches, ~50 %, zero column offset), the visual correction
pulled a 34° anchoring error out of the route frame (yaw correction
converged to +26°), and the run ended `arrived` at the last node in 3.2 s
with the edge recorded (2 traversals, 2 successes). Lessons from the two
failed attempts before it: never refresh the driver's goal while it reports
`blocked` (it resets the abort timer), and the driver now steers around a
return inside the corridor edge instead of parking in front of it; a drop
in the corridor still stops it outright. Reversing can yaw the chassis by
tens of degrees as the castors swing round, so a frame anchored after a
reverse relies on the visual correction.

## 9. Multi-leg trips over the place graph

`bebop_vision/places/planner.py` runs Dijkstra over `/places/graph`. A
directed edge is usable only when it carries a taught route; its cost is
the edge's median successful duration (or the newest route's length at
0.2 m/s before any has been measured), multiplied by `1 + 2·(1 − success
rate)` so legs that failed get expensive. `GET :9095/places/plan?from=&to=`
returns the legs. `bebop_vision/nav/trip.py` (`TripRunner`, on the nav
server) drives them: one `RouteFollower.follow` per leg, the next leg
starting where the previous one ended (`arrived` or `arrived_unconfirmed`),
any failed leg failing the trip. The start place is given or recognised.

```
GET  :9095/places/plan?from=office%20room&to=Master%20Bedroom
POST :9097/trip {"to": "Master Bedroom"[, "from": "office room"]}
GET  :9097/trip            state, plan, current leg, follower status
POST :9097/trip/stop
```

Teaching tip for a chain: finish route A→B facing the way route B→C
starts; the follower re-anchors each leg at the robot's live pose and the
visual correction only absorbs tens of degrees.

First live trip (2026-10-21, master bedroom): two straight legs taught by
nav goal (1.11 m + 0.83 m, three nodes each) through a temporary mid place,
reversed back to the start, then `POST /trip {"to": "bedroom-far", "from":
"Master Bedroom"}`: the planner returned both legs (~14.5 s estimated), leg
1 ended `arrived` (recognition confirmed the mid place), leg 2 ended
`arrived_unconfirmed`, total 8.0 s, visual matches confident on every
tick (56-68 % of patches), both edges now carry measured median
durations. Note that freshly taught places inside one room (0.8 m apart)
shadow the room's own place in recognition; test places were deleted.

## 10. Object goals: "go to the dresser" without a route

`bebop_vision/nav/object_goal.py` (`ObjectSeeker`, on the nav server) is
the first use of the VLM as the decision layer, one look per leg:

1. send the far view (downscaled to 640 px) to the 27B brain and ask for a
   strict-JSON bounding box of the object; the box centre becomes a bearing
   through the far color intrinsics (replies arrive as fractions, sent-image
   pixels or original-frame pixels, sometimes mixed in one box; the parser
   normalises per axis and retries once);
2. read the far depth around that direction for the range (median window;
   none beyond ~6 m, then a default 1.5 m assumption);
3. give the driver one bounded goal (bearing, ≤ 1 m), wait for it, look
   again; `arrived` when the range is within 0.9 m + 0.25 m, failure after
   two misses, two blocked legs or 12 legs.

```
POST :9097/object/locate {"what": "the dresser"}   # bearing + range, no motion
POST :9097/object/seek   {"what": "the dresser"}   # drive to it
GET  :9097/object
POST :9097/object/stop
```

**Searching.** The far camera sees ~46° to each side, so an object out of
view is a miss. On a miss the seeker turns to the next scan heading
(45, −45, 90, −90, 135, −135, 180° from where the search began; a hint such
as "on the left" puts that side first) with a rotate-only driver goal and
looks again; the hint words are stripped from what the model is asked,
because "mirror on the left" made it answer "not visible on the left" once
the robot had turned to face it. After the object has been seen, one miss
re-looks before turning away. Live 2026-10-21: "go to the mirror on the
left" from the dresser: first look missed, one 45° left turn found "a
full-length mirror on a stand" at 4.2 m, two legs closed to 3.6 m, a miss
then re-scanned to 180° where it found a second mirror and closed to 2.6 m
before the corridor was blocked by furniture (`failed_blocked`, as
designed).

Live 2026-10-21 (master bedroom): "the dresser" was 7.0 m away through an
archway. Six 1 m legs (range 7.0 → 6.1 → 5.0 → 4.1 → 3.1 → 2.0 m, bearing
within ±6°, the VLM naming "black dresser with drawers and mirror" at
confidence 1.0, ~2 s per look) were interrupted by an operator stop during a
code fix; the resumed seek reported `arrived` at 1.1 m. The basketball and
the fireplace located at 2.3 m and 7.6 m from the same spot. Each look costs
one 27B call, so a seek is a handful of calls, never per tick.

## 11. One command surface: voice or text

`bebop_vision/nav/navigator.py` is where every "go somewhere" lands,
whether it arrives as typed text (`POST :9097/navigate {"command": "..."}`)
or through the voice brain's `navigate` tool (which just relays the user's
words). It resolves the words without a model call:

| words | mechanism | waypoints |
|---|---|---|
| a taught place (fuzzy match: "the kitchen" → `Kitchen`) with a route chain from the recognised current place | trip over the graph | one per route node, leg by leg |
| a taught place with no chain, or the current place unrecognised | object seeker with the place name as the description | one per VLM look |
| anything else ("the dresser", "the red chair by the window") | object seeker | one per VLM look |
| "forward 1 m", "back half a metre" | one bounded driver goal | one |
| "turn left 90", "turn around" | a rotate-only driver goal (`turn_deg`) | one |
| "stop" / "halt" / "wait" | stops everything | — |

A command can be a sequence: "turn around then go to the window", "back
up one metre and go to the dresser". It is split on then/and/sentence
breaks (only after a move or turn, so "the chair and the table" stays one
description), filler such as "way into the distance, there is ... navigate
to it" is trimmed, and the steps run in order; a failed step drops the
rest. A turn refused for sweep clearance backs up 0.4 m and retries once.

When a leg toward an object is blocked, the seeker **detours**: it drives
up to 1 m into the most open sector near the object's bearing (from the
live clearance map) and looks again, or backs up 0.4 m when nothing is
open; three detours, then `failed_blocked`. A blocked scan turn backs up
and retries the same heading.

A new command replaces the active one. `GET /navigate` returns one state
(`running`, `arrived`, `arrived_unconfirmed`, `stopped`, `failed_*`) plus a
spoken-style `summary` ("Heading for dresser, about 3.0 metres away.",
"On the way to Kitchen, leg 1 of 2.", "Arrived at dresser.", "Could not get
to unicorn: failed not found."). `POST /navigate/stop` and the global
`/stop` halt it.

Voice: `voice_server.py` advertises `navigate(command)`,
`navigation_status()` and `stop_navigation()` behind the same
`motion_enabled` switch as `move_relative`; the brain only relays words and
reads back the summary, so "are we there yet" is a `navigation_status`
call. The voice service must be restarted to pick the tools up.

Live 2026-10-21, by voice: "the baby crib" → five legs, arrived in 48 s.
"turn around and navigate to the window" ran only the turn (fixed: steps),
"the window curtain" found the curtain but two legs were blocked by the
bed (fixed: detours). After the fixes, "turn around then go to the baby
crib": turn refused → backed up 0.3 m → turn retried → crib found at
1.05 m → arrived.

Live 2026-10-21: `{"command": "go to the dresser"}` from 1 m away →
`arrived` on the first look; the tool path (`_run_tool("navigate",
{"command": "back half a metre"})`) drove the reverse and reported
"Done: back half a metre."

## 12. Live results (2026-10-21) and caveats

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
