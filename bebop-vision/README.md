# bebop-vision

The robot-side Python stack of the wheeled Bebop, run from one venv
(`bebop-vision/.venv`), plus the workstation tools that turn drive
recordings into training data. Nothing here plans or follows a path: the
operator drives from the app and the recorder captures it. The only motion
code is the voice agent's bounded `move_relative` tool, which stays off until
the operator enables motion (`docs/voice.md`). The navigation redo is a
proposal, `docs/path-model.md`.

## On the robot

Jetson AGX Thor, JetPack 7.2, Python 3.12. The units in `deploy/systemd/` run
as root from `/home/bebop/bebop/bebop-vision` with `.venv/bin/python`.

| Unit | Runs | Port | Started by |
|---|---|---|---|
| `bebop-vision.service` | `main.py --record-navd /var/lib/bebop-captures --auto --robot-url ws://127.0.0.1:9090/ws $BEBOP_VISION_EXTRA_ARGS`: owns both Orbbec cameras; the navd MCAP recorder, operator video and the drive-note routes (:9092), the places service (:9095) | 9092, 9095 | the app (Diagnostics › Services), or boot after `install-jetson.sh --enable-autostart` |
| `bebop-voice.service` | `-m bebop_vision.voice_server`: the agent gateway; the ASR -> brain -> TTS cascade (default) or Qwen3-Omni, each a TensorRT Edge-LLM server (`docs/voice.md`) | 9093 | the app, or boot after `--enable-autostart` |
| `bebop-models.service` | `-m bebop_vision.models_supervisor`: GPU budget and Edge-LLM lifecycle, registry `config/models_supervisor.yaml` (`docs/models.md`) | 9094 | enabled by `install-jetson.sh --setup-models` |
| `bebop-model-download@<id>.service` | `-m bebop_vision.download_model <id>`: one catalog download | – | the firmware, when the app asks |

One process per camera: the recorder holds both, so stop `bebop-vision`
before running a camera tool (`tools/orbbec_*`) or OrbbecViewer.

Extra `main.py` flags go in `/etc/bebop/vision.env` as
`BEBOP_VISION_EXTRA_ARGS="..."`; the robot sets `--disk-budget-gb 80` there
(since 2026-10-23). `main.py` rejects unknown flags, so a removed flag left in
that file keeps the unit in a 3 s restart loop: `--nav`, `--no-bev`,
`--live-slam*`, `--place-router`, `--router-interval`, `--router-cooldown`,
`--router-voice-url`.

| `main.py` flag | Default | |
|---|---|---|
| `--record-navd DIR` | required | MCAP directory |
| `--auto` | off | open a segment while the firmware is in DialIn with every wheel armed and no E-STOP; close it when that ends |
| `--record-rate HZ` | 30 | tick rate (the cameras are hardware-synced at 30 fps) |
| `--max-segment-mb`, `--max-segment-min` | 0 (off) | with `--auto`, roll a long drive into several files |
| `--disk-budget-gb` | 20 | when a segment closes, delete the oldest `navd_session_*.mcap` above this total |
| `--seconds` | – | stop after this long |
| `--rig YAML`, `--roles` | `config/orbbec_rig.yaml`, all | rig config; the camera roles to open (e.g. `near`) |
| `--robot-url` | `ws://bebop.local:9090/ws` | firmware runtime WebSocket |
| `--no-video-server`, `--video-port` | 9092 | operator video |
| `--video-color-kbps`, `--video-depth-kbps` | 3500, 1000 | WHEP CBR bitrates |
| `--video-max-viewers`, `--video-idr-s` | 4, 1.0 | viewers per stream (they share one encoder); IDR interval in seconds |
| `--no-places`, `--places-port`, `--places-root` | 9095, `/var/lib/bebop-places` | places service; its SQLite DB and keyframes |
| `--place-scout-interval`, `--place-pending-max`, `--place-pending-ttl-days` | 1.0 s, 50, 14 | scout period; pending-queue cap; days a pending place may go unseen before it is deleted |
| `--router-url`, `--router-model`, `--router-timeout` | `http://127.0.0.1:8012`, `RadixArk/Qwen3.8-27B-NVFP4`, 120 s | the OpenAI-compatible VLM (default: the cascade brain) for operator-requested pending-place name suggestions; the names predate the removal of the every-tick router |
| `--stall-watch FILE`, `--stall-ms`, `--stall-gc-ms` | off, 150, 50 | diagnostic: log GIL stalls with every thread's stack, and long garbage collections |
| `--no-gc-freeze` | off | skip the `gc.freeze` of the startup heap (only to compare GC pauses) |

Vision routes on :9092 (`bebop_vision/videoserver.py`):

| Route | |
|---|---|
| `POST /whep?stream=<name>` | WHEP: SDP offer in, `201` + answer out; `color_near` (default), `color_far`, `depth_near`, `depth_far`; H.264 over WebRTC; `503` past the viewer limit |
| `DELETE /whep/<id>`, `GET /whep` | drop one viewer; per-stream encoder and viewer stats |
| `GET /snapshot?stream=<name>` | the latest frame as JPEG (the voice agent's camera views, tools) |
| `POST /intent`, `POST /event`, `GET /notes` | drive notes written into the open segment (`docs/data-collection.md`) |
| `GET /healthz` | liveness |

Places on :9095 (`bebop_vision/places/`, in the same process). Every second
the scout embeds the far camera frame with DINOv3 ViT-L/16. A confident match
to a confirmed place is logged as a `/place` row in the open segment; a novel
view seen 3 times over 2 s becomes a pending place, with no LLM call. The
operator reviews pending places in the app (Memory) and imports them in
batches, optionally asking the VLM for a name first
(`POST /places/pending/{id}/suggest`). Pending places are invisible to
`/whereami` until imported. `/places/graph` holds the places and their A->B
traversal edges. Route list: the docstring of `bebop_vision/places/server.py`;
design: `docs/places.md`.

## From drive to dataset

| Step | Where | Command | Output |
|---|---|---|---|
| 1. Record | robot | `bebop-vision.service` (`main.py --record-navd /var/lib/bebop-captures --auto`) | `navd_session_<stamp>.mcap`, one per drive |
| 2. Pull | workstation | `just pull-data` -> `tools/pull_captures.py` | `datasets/sessions/`: the drive sessions and the firmware's `system_*.mcap` health logs, fetched over `http://bebop.local:9090/captures` (the robot's capture dir is root-only) |
| 3. Extract + audit | workstation | `just pull-data` -> `tools/extract_all_navd.py` (`tools/mcap_extract.py` per new session) | `datasets/navd-v0/<session>/`: `color/`, `color_far/`, `depth/`, `manifest.jsonl`, `imu_{accel,gyro}_{near,far}.npz`; one audit line per session |
| 4. Ledger | workstation | `just pull-data` -> `tools/session_ledger.py` (`just ledger` alone) | `datasets/sessions/sessions.jsonl`, one row per session, joined with the health log by time window |
| 5. Label layout | workstation | `tools/bev_student_dataset.py extract --mcap <session.mcap>` (plus every `*.mcap` in `--mcap-dir`, default `datasets/bev-student/raw/mcap/`; a session that already has `index.jsonl` is skipped without `--force`) | `datasets/bev-student/sessions/<session>/`: 10 Hz `color_{near,far}/*.jpg`, `depth_{near,far}/*.png` (uint16 mm), `index.jsonl`, `calib.json`, `meta.json` |
| 6. Labels | workstation, NVIDIA GPU | `tools/bev_student_labels.py --stride 3` | `labels/<stamp>.npz`, `labels/diagnostics.jsonl`, `preview/` per session |
| 7. QA + split | workstation | `tools/bev_student_dataset.py index`, `split`, `report` | `meta/label_stats.json`, `meta/sessions.json`, `splits/drive_split.json` |

The per-session protocol, the channels a drive file carries and the ledger
fields are in `docs/data-collection.md`. To look at a session, open it in the
Rerun navd dashboard from the repo root (needs `rerun-sdk`; see the script's
docstring): `python rerun_dashboards/open.py <session.mcap> [system_*.mcap]`.
`tools/mcap_merge_health.py` writes a copy of a session with the overlapping
health-log rows merged in (`<session dir>/merged/`).

**Labels.** SAM 3.1 is the semantic teacher and the rig's depth is the
geometry. Each sampled depth pixel is lifted into the body frame, projected
into the colour frame through the camera's depth->colour extrinsic to read the
SAM mask, classified against that camera's fitted floor plane, and
accumulated per cell; the two cameras are fused with the priority hazard >
wall > obstacle > floor. A label is a 120 x 60 grid of 5 cm cells (6 m ahead
x 3 m wide): `semantic` (0 floor, 1 wall, 2 obstacle, 3 hazard, 255 ignore),
`ego` and `coverage`; an unobserved cell is never free. `--stride 3` labels
every third 10 Hz frame. Splits are by drive, never by frame. Code:
`bebop_vision/bev_student_labels.py`; knobs: `bev_student:` in
`config/orbbec_rig.yaml`.

**Learned footprint.** `tools/bev_ego_from_data.py --out config/ego_mask.npz`
learns the chassis from the robot's own recordings: cells near the origin with
above-floor depth in nearly every frame are body, and cells beside them that
the cameras never see are what the body hides. The labeler culls depth inside
that 1 cm raster (plus `cull_ego_margin_m`, 0.02 m) and ignores below-plane
evidence within `hazard_ego_margin_m` (0.15 m). The committed raster
(2026-10-19, 633 frames) spans x -0.13..+0.32 m, y -0.34..+0.19 m; rebuild it
after a chassis or camera change.

## Layout

| Path | What |
|---|---|
| `main.py` | `bebop-vision.service` entry point: rig, recorder, video server, places |
| `bebop_vision/orbbec.py` | Gemini 335Lg rig: devices by serial, profile negotiation, presets and host-side depth filters, camera IMU and taps, self-mask pixels; writes `config/orbbec_intrinsics_<serial>.json` when it is missing |
| `bebop_vision/recorder_mcap.py` | navd MCAP writer: one file per drive; the per-tick messages share one `log_time` |
| `bebop_vision/hw_video.py` | NVENC H.265/H.264 through GStreamer `nvv4l2*enc` |
| `bebop_vision/videoserver.py`, `whep.py`, `whep_registry.py` | :9092: WHEP with one shared H.264 encoder per stream, viewer bookkeeping, `/snapshot`, the note routes |
| `bebop_vision/session_notes.py` | intent / event / place notes shared by the routes, the scout and the recorder |
| `bebop_vision/robot.py` | protobuf-over-WebSocket client of the firmware runtime |
| `bebop_vision/places/` | place memory: `scout.py`, `importer.py`, `store.py` (SQLite + keyframes), `encoder.py` (DINOv3 ViT-L/16), `capture.py`, `server.py` (:9095) |
| `bebop_vision/vlm.py` | OpenAI-compatible chat client behind the name suggestions |
| `bebop_vision/voice_server.py` | `bebop-voice` agent gateway (:9093) |
| `bebop_vision/models_supervisor.py`, `edge_serve.py` | `bebop-models` (:9094); the `tensorrt-edgellm-serve` launcher it shares with the voice gateway |
| `bebop_vision/models.py`, `download_model.py` | catalog parser; the downloader behind `bebop-model-download@<id>` |
| `bebop_vision/gc_tuning.py`, `stall_watch.py` | startup-heap freeze against full-GC stalls; the `--stall-watch` diagnostic |
| `bebop_vision/navd_proto.py`, `foxglove_proto.py`, `proto/` | `bebop.navd.*` and `foxglove.*` bindings (sources in `protos/`), the runtime pb2 |
| `bebop_vision/mcap_tools.py` | MCAP read helpers for the offline tools, footerless files included |
| `bebop_vision/sam3_concepts.py`, `segmenter.py`, `config.py`, `runtime.py` | SAM 3.1 text-prompted concept segmenter (teacher side only) |
| `bebop_vision/bev_student_labels.py`, `bev_student_data.py` | SAM 3.1 x depth labels; the `datasets/bev-student/` layout, index, splits and stats |
| `bebop_vision/master_bev.py` | geometry kept from the retired BEV: `mount_rotation`, `fit_ground_plane`, the ego raster |
| `tools/pull_captures.py`, `extract_all_navd.py`, `mcap_extract.py`, `session_ledger.py` | `just pull-data`; `mcap_extract.py` also writes `--layout bev-student` |
| `tools/mcap_merge_health.py`, `mcap_recover.py` | health log merged into a session copy; read or repair a footerless MCAP |
| `tools/bev_student_dataset.py`, `bev_student_labels.py` | label layout (`extract`, `index`, `split`, `report`, `layout`); the SAM 3.1 x depth labeler |
| `tools/export_sam31_encoder.py` | SAM 3.1 image encoder -> fp16 ONNX (`weights/sam31_encoder_fp16.onnx`, absolute workstation paths) for an optional TensorRT engine (`--trt-engine`) |
| `tools/bev_ego_from_data.py` | learned chassis footprint -> `config/ego_mask.npz` |
| `tools/set_self_mask.py` | draw the chassis self-mask polygon on the live view (workstation, HTTP only) |
| `tools/orbbec_imu_probe.py`, `orbbec_sync_test.py`, `orbbec_slot_probe.py` | camera IMU profiles and clock offset; hardware-sync bench; latest-slot skew (robot, cameras free) |
| `config/` | `models.yaml`, `models_supervisor.yaml`, `orbbec_rig.yaml`, `orbbec_intrinsics_<serial>.json`, `ego_mask.npz` (+ `.png` preview) |
| `deploy/systemd/` | the four units above |
| `tests/` | pytest suite, no hardware |
| `weights/`, `datasets/`, `runs/` | checkpoints and data (gitignored) |

Removed on 2026-10-24, for readers looking for them: the navd-v0 label fusion
and its review dashboard (`tools/fuse_navd_labels.py`,
`tools/sam_floor_label.py`, `tools/dashboard_api.py`,
`tools/dashboard_core.py`, `dashboard/`, `just review`), live SLAM
(`bebop_vision/slam/`, :9096), the every-tick place router
(`bebop_vision/router/`) and the per-place structure submaps
(`bebop_vision/places/submap.py`), `bebop_vision/download_sam3.py` (use
`download_model`), `tools/orbbec_intrinsics.py` and
`tools/dump_color_calib.py` (delete `config/orbbec_intrinsics_<serial>.json`
and restart `bebop-vision` to re-dump). History:
`git log -- bebop-vision/<path>`. Older model work: the BEV student, removed
2026-10-23 (`git log -- bebop-vision/bebop_vision/bev_student.py`), and the
navd trajectory student, on the local branch `exp/navd-ai-20260909` (not on
`origin`).

## Install

**Robot.** One venv for every unit, built by the installer from `~/bebop`:

```sh
sudo ./scripts/install-jetson.sh --setup-voice-only   # venv only; plain --setup-voice adds it to a full install
```

It creates `.venv` with `--system-site-packages` (JetPack's TensorRT
bindings must stay visible to Edge-LLM), installs `torch==2.13.0+cu132` and
`torchvision==0.28.0+cu132` from the PyTorch cu132 index (PyPI's aarch64
torch is CPU-only), registers the package with `pip install -e . --no-deps`,
installs `requirements.lock.txt` (or the `requirements.txt` floors when there
is no lock), then `pyorbbecsdk2==2.1.2` with `--no-deps`: the SDK pins
`av==12.3.0` and `tensorrt-edgellm[server]` pins `av==17.1.0`, so `pip check`
always reports that one conflict. A hand-made `python -m venv` +
`pip install -e .` gets neither the TensorRT bindings nor the camera SDK.
`requirements.lock.txt` is generated on the robot from a verified venv (recipe
in its header); until it is regenerated it still pins `cuvslam`, which only
the removed live-SLAM code imported.

**Workstation** (`just pull-data` expects this venv at `bebop-vision/.venv`):

```sh
cd bebop-vision
python3 -m venv .venv
.venv/bin/pip install -e '.[tools,server]' pytest
# SAM 3.1 teacher (labeling, NVIDIA GPU):
.venv/bin/pip install -e '.[teacher]' psutil
```

| Extra | Contents | For |
|---|---|---|
| `tools` | `av` | PyAV: the H.265 colour decode in `tools/mcap_extract.py` |
| `server` | `fastapi`, `uvicorn`, `pydantic`, `httpx` | `voice_server.py` and `models_supervisor.py` off the robot; without it their tests skip |
| `teacher` | `sam3` (GitHub), `einops`, `onnx`, `pillow`, `pycocotools`, `tensorrt>=11.0` | the SAM 3.1 teacher; `onnx` for `export_sam31_encoder.py`, `tensorrt` only for `--trt-engine` |
| `jetson` | `pyorbbecsdk2==2.1.2` | declares the camera SDK; the robot installs it `--no-deps` as above, never through this extra |

`sam3` 0.1.0 imports `einops`, `pycocotools` and `psutil` without declaring
them; the extra covers the first two, hence the explicit `psutil`. Unverified:
the `[teacher]` line has not been run on a fresh venv. `sam3` declares
`numpy<2,>=1.26` (`pip check` reports it against the workstation venv's numpy
2.4.6, 2026-10-24), and that venv has no `tensorrt`. `sam3` depends on
torchvision (through `timm`); if pip pulls a PyPI torchvision that does not
match your torch, reinstall the matching build from torch's index, e.g. for
the workstation's torch 2.11.0+cu128:

```sh
.venv/bin/pip install --force-reinstall --no-deps "torchvision==0.26.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128
```

**Tests:** `.venv/bin/python -m pytest tests/ -q` (no hardware). 268 tests;
on the workstation (2026-10-24) 267 pass and one skips (no `nvv4l2h265enc`).
On the robot, run them after every deploy.

## Model weights

The robot ships no weights. `config/models.yaml` is the catalog, read by the
firmware (`firmware/bebop-linux/src/model.rs`, which reports it to the app as
`ModelState`) and by `bebop_vision.models`. Add a model by appending an
entry; no code changes.

| Field | |
|---|---|
| `kind: hf` | Hugging Face. `download: files` (default) fetches `files` into `weights/`; `download: snapshot` fetches the whole repo into `weights/<dest or id>/` and writes `.bebop-complete` when it is done |
| `kind: local` | produced on the robot or copied from a workstation (`scp`); nothing to download. The firmware reports it present when `path` (absolute, or relative to `weights/`; default `weights/<id>`) exists |
| `gated: true` | needs a Hugging Face token whose account has accepted the model's license, e.g. <https://huggingface.co/facebook/sam3.1> |
| `purpose` | the label the app groups by: `segmentation` (`sam3.1`, `sam3`), `backbone` (`dinov3-vitl16`, the place encoder), `voice` (`qwen3-omni-30b`), `asr`, `tts`, `brain` (the cascade stages). Nothing selects a model per purpose |

**From the app:** Settings › Models shows the `bebop-models` GPU-budget card
and the catalog grouped by purpose. Save a Hugging Face token (the firmware
writes it root-only to `/etc/bebop/hf_token` and never sends it back), then
press Download on an entry; the app runs one download at a time. The firmware
starts `bebop-model-download@<id>.service`; the downloader refuses a download
that does not fit on disk, writes progress to `/run/bebop/models/<id>.json`,
and the firmware relays it in telemetry. The robot needs internet for this,
i.e. Known Network (client) mode.

**By hand:**

```sh
# Token: HF_TOKEN in the environment, /etc/bebop/hf_token, or HF_TOKEN=... in bebop-vision/.env (gitignored).
.venv/bin/python -m bebop_vision.download_model sam3.1   # one or more catalog ids
.venv/bin/python -m bebop_vision.download_model --all    # every kind: hf entry (the size hints add up to over 90 GB)
```

The downloader always writes its status file under `/run/bebop/models/`. As a
normal user that fails before anything downloads
(`[model] sam3: FAILED: [Errno 13] Permission denied: '/run/bebop'`; a gated
id with no token stops on the same error as an uncaught `PermissionError`;
workstation, 2026-10-24), so run it as root or create the directory first:
`sudo mkdir -p /run/bebop/models && sudo chown "$USER" /run/bebop/models`.

## License note

SAM 3.x code and weights (teacher tooling, `weights/sam3*.pt`) are Meta's
"SAM License" — commercial use and fine-tuning are permitted and you own
derivative works, but redistribution of SAM 3.x materials or derivatives
must carry the SAM License. Downstream artifacts you train from the
labels (e.g. distilled students) are plain torch artifacts with no such
restriction. See the header of `bebop_vision/sam3_concepts.py`.
