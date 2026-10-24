# Data collection protocol (navd sessions)

How to turn robot drive time into training data, and how to know a session
is good. The pipeline: record on the robot → `just pull-data` on the
workstation (mirror, extract + audit, ledger) → optional SAM 3.1 × depth
labels (`bebop-vision/README.md`).

## Recording

The recorder is `bebop-vision.service`
(`main.py --record-navd /var/lib/bebop-captures --auto`). Start it from the
app (Diagnostics › Services › Vision) unless the robot starts it at boot
(`install-jetson.sh --enable-autostart`). With `--auto` a segment,
`navd_session_<stamp>.mcap`, opens when the firmware is in DialIn (the app's
drive mode) with every wheel armed and no E-STOP, and closes when any of that
ends: one file per drive. Each time a segment closes, the recorder deletes the
oldest `navd_session_*.mcap` above `--disk-budget-gb` (20 GB by default in
`main.py`; the robot's `/etc/bebop/vision.env` has set 80 since 2026-10-23).
The firmware's always-on health log (`system_*.mcap`: 1 Hz power-board and
host rows, a new file daily) shares the directory; the firmware prunes those
itself, oldest first above 2 GiB.

To stop the recorder: the app (Diagnostics › Services),
`sudo systemctl stop bebop-vision`, or Ctrl-C in a `just collect` run (a
hand-started background run: `sudo pkill -TERM -f 'record-nav[d]'` from a
separate shell). SIGTERM takes the same path as Ctrl-C, so the open segment is
flushed and `/tmp/navd_recorder.lock` released.

`just collect` runs the same recorder in the foreground over SSH, for when the
service is stopped: the cameras and the recorder lock are exclusive, so it
exits while the service runs. It does not read `vision.env`, so it prunes at
the 20 GB default.

## What a drive file contains

The per-tick messages (colour, depth, `/cmd_vel`, `/odom`) share one MCAP
`log_time`; ticks run at `--record-rate` (default 30 Hz, the cameras'
hardware-synced frame rate). IMU samples and firmware messages keep their own
times. Channels (`bebop-vision/bebop_vision/recorder_mcap.py`):

| Channel | Schema | Content |
|---|---|---|
| `/color_near`, `/color_far` | `foxglove.CompressedVideo` | H.265 from NVENC (`foxglove.CompressedImage` JPEG when no encoder) |
| `/depth_near`, `/depth_far` | `foxglove.CompressedImage` | lossless 16-bit PNG, mm |
| `/cmd_vel` | `bebop.navd.Twist` | the firmware's commanded twist (`DriveState.cmd_*`), whoever sent it |
| `/odom` | `bebop.navd.Odom` | wheel odometry pose |
| `/imu_accel_<role>`, `/imu_gyro_<role>` | `bebop.navd.ImuAccel`, `ImuGyro` | every camera IMU sample (200 Hz per camera, `config/orbbec_rig.yaml`) with its device stamp |
| `/runtime` | `bebop.runtime.v1.ServerRuntimeMessage` | the firmware's telemetry frames verbatim, as received at the 30 Hz subscription: mode, E-STOP reason, operator and watchdog flags, per-wheel targets, bus and power state |
| `/runtime_events` | same | everything else the firmware sent: acks, errors, mode and E-STOP events |
| `/intent` | `bebop.navd.Intent` | the sticky drive intent (below) |
| `/event` | `bebop.navd.Event` | operator markers |
| `/place` | `bebop.navd.PlaceSeen` | the place scout's sightings of confirmed places |
| `/calib` | JSON, once per file | intrinsics per camera serial, mounts (height, pitch, yaw), self-mask shapes |

Everything except `/calib` is Protobuf, so Rerun's MCAP decoder opens the file
directly. There is no navigation-goal concept in the pipeline: the data is
*what the robot saw* + *what you did*.

## Per-session checklist

Before:
- [ ] `bebop-vision` running (app: Diagnostics › Services), camera views live
- [ ] Jetson disk has headroom (`df -h /`); the recorder prunes at the budget, so pull before it does
- [ ] App connected, E-STOP clear, wheels arm cleanly
- [ ] Know what this session is *for*, and set it as the intent: it lands in every segment and in the ledger row

During (aim for 5–10 minutes of recorded drive time):
- [ ] Obstacle variety: drive near real clutter, around it, past it —
      the data only contains what you show it.
- [ ] Maneuver variety: straight runs, tight turns, approaches and
      stops, reversing out of tight spots.
- [ ] Lighting variety when possible (day, evening, lights off).
- [ ] Speed range: slow approaches, normal cruising, one decisive stop.
- [ ] A few near-misses (you correct late) — those ticks are gold.
- [ ] For the path-model corpus (`docs/path-model.md` section 9): mostly
      forward teleop, repeated passes through each doorway in both
      directions and each junction on both sides, planted clutter marked
      `obstacle`, deliberate off-lane starts marked `recovery`.
- [ ] Mark anything unusual (moved furniture, a new obstacle) with a `note`
      marker; the ledger row lists it.

After:
- [ ] Disarm (or switch to Idle): the segment closes and the recorder waits
      for the next drive. Mark `exclude` before you disarm.
- [ ] `just pull-data`; check the audit line (`has_color%` and
      `has_color_far%` 100, plausible `valid_near%`/`valid_far%` and
      `med_near_m`/`med_far_m` depth medians) and the ledger row (`Hz`, frame
      and gyro gaps, moving/reverse, turns, intent, places, battery)
- [ ] Optional: replay it in the Rerun navd dashboard,
      `python rerun_dashboards/open.py <session.mcap> [system_*.mcap]`
      (needs `rerun-sdk`)

## Intent, events and places in the recording

Say what a drive is for before you arm, and mark moments as you go; the
recorder writes them into the drive MCAP so the file carries its own labels
(`/intent`, `/event`, `/place`, all `bebop.navd.*`). The intent is sticky: it
is written at the start of every segment and whenever it changes, so a long
drive that rolls over several files keeps it. It can be set at any time,
armed or not.

- In the app's Drive page: the REC chip shows whether a segment is
  recording; the intent chip opens presets (keep, left, right, stop, turn
  around), free text and Clear; the markers are Obstacle, Recovery, Exclude
  and Note (hotkeys 1–4).
- By voice, on the cascade engine with "Live context and tools" on (Settings ›
  Voice & agent; on by default): "I'm driving from the living room to the
  kitchen", "mark that sock as an obstacle", "mark a recovery start",
  "exclude that bit", "clear the drive intent".
- By HTTP on the vision process:

  ```sh
  curl -X POST http://bebop.local:9092/intent \
       -d '{"text": "living room to kitchen", "from": "living room", "to": "kitchen"}'
  curl -X POST http://bebop.local:9092/event -d '{"kind": "obstacle", "note": "sock at 1 m"}'
  curl -X POST http://bebop.local:9092/intent -d '{"clear": true}'
  curl http://bebop.local:9092/notes
  ```

  `GET /notes` answers
  `{recording, intent, events_pending, places_pending, events_dropped, places_dropped}`.
  A POST needs a JSON object body (`400` otherwise; an empty body never
  clears the intent).

Event kinds are short tokens (`[a-z][a-z0-9_-]{0,31}`); the app and the voice
tool offer `obstacle` (a planted doll, sock, cable), `recovery` (an off-lane
start on purpose), `exclude` (that stretch was a mistake) and `note`. Markers
only land while a segment is recording (DialIn, every wheel armed, no
E-STOP). Outside a segment `POST /event` answers 409
(`{"error": ..., "recording": false}`), the app warns that the marker was not
saved, and the voice tool says "Not recording right now, so nothing was
marked." instead of pretending. Say "exclude that bit" before you disarm:
after a disarm or an E-STOP the segment is closed and no marker reaches it.
Today the markers are recorded and listed per session in the ledger;
extraction skips `/intent`, `/event` and `/place`, so an `exclude` mark does
not yet drop ticks from the training data.

Automatically: while a segment records, every confident scout match to a
confirmed place (the 1 Hz DINOv3 scout of the places service) writes a
`/place` row (id, name, score, margin), so a drive carries its own from/to
anchors for the place graph without you doing anything.

Turn-level intent (left / right / keep at a junction) is meant to be derived
from the hindsight path, so it needs no capture by hand; nothing derives it
yet.

## Ledger

`bebop-vision/datasets/sessions/sessions.jsonl` is rewritten by
`tools/session_ledger.py` on every `just pull-data` (or `just ledger`): one
row per `navd_session_*.mcap` with duration, per-topic counts, tick rate and
frame gaps (> 0.15 s), camera gyro rate and device-clock gaps (> 20 ms, the
recorder-stall signature), moving and reverse fractions, left/right turn
events from `/cmd_vel`, odometry path length, the `/runtime` count, the
intents, events and places, and a `health` summary of every `system_*.mcap`
whose time span overlaps the session (battery voltage and SOC, board
temperature, power-board faults, CPU/GPU load, hottest thermal zone, disk
fill). Hand edits are lost on the next run, so anything the session should
carry goes into the intent and the markers. When evaluating a navigation
approach, the ledger tells you which questions the data can answer — and
which it can't.

## Commands

```sh
just pull-data   # mirror sessions + health logs -> extract + audit -> ledger
just ledger      # ledger only (no transfer)
just collect     # foreground recorder over SSH; only while bebop-vision.service is stopped
```

`just pull-data` mirrors over the firmware's HTTP API
(`http://bebop.local:9090`: `POST /captures/finalize`, `GET /captures`,
`GET /captures/dl/<name>`) because the robot's capture dir is root-only. It
skips files the firmware marks active and files modified in the last 30 s (a
drive segment still being written), and resumes a partial download. Sessions
land in `bebop-vision/datasets/sessions/`; `tools/extract_all_navd.py`
extracts each new one to `datasets/navd-v0/<session>/` (`color/`,
`color_far/`, `depth/`, `manifest.jsonl`, `imu_*.npz`) and prints the audit
line for every session. The app's Diagnostics › Captures lists and downloads
the same files.

## What good coverage looks like

| dimension | minimum | good |
|---|---|---|
| sessions | 5 | 15+ |
| drive time | 15 min | 60+ min |
| obstacle layouts | 3 | 10+ |
| lighting conditions | 2 | 4+ |
| rooms/areas | 2 | every drivable area |

These thresholds are guidance, not gospel — the ledger's job is to make
the gaps visible, not to gate on numbers alone.
