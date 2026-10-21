# bebop — agent working notes

## Robot access

- SSH: `ssh bebop@bebop.local`, password `bebop` (non-interactive: use
  `sshpass -p bebop ssh ...`).
- **Hardware (as of 2026-09): Jetson AGX Thor Developer Kit** — SoC tegra264
  (`cat /proc/device-tree/model` → `NVIDIA Jetson AGX Thor Developer Kit`),
  JetPack **7.2 GA / L4T R39.2.1**, CUDA **13.2**, Python **3.12**, GPU reports
  as `NVIDIA Thor` with compute cap **sm_110**. Earlier revisions of this file
  said "Orin Nano" — that is stale; anything derived for tegra234 GPIO/line
  numbering does **not** carry over (see the GPIO gotcha below). On-device
  training env for this box lives in `sim/setup/thor/`.
- Repo on the robot: `~/bebop` (same monorepo layout as here).
- Python env: ONE venv for the whole project, `~/bebop/bebop-vision/.venv`
  (created `--system-site-packages` so JetPack's TensorRT bindings are
  visible), run as `/home/bebop/bebop/bebop-vision/.venv/bin/python`.
  Versions live in `bebop-vision/requirements.txt` (torch 2.13.0+cu132,
  transformers 5.14.1, numpy 2.2.6, venv-local scipy); it pins a version only
  where one is actually required — the cu132 torch/torchvision pair, the
  pyorbbecsdk2 SDK build, onnxruntime-gpu's CUDA-13 build, tensorrt-edgellm's
  engine format — and uses floors otherwise. `pyorbbecsdk2` is deliberately
  **not** in that file: it hard-pins `av==12.3.0` while
  `tensorrt-edgellm[server]` hard-pins `av==17.1.0`, so `--setup-voice`
  installs it with `--no-deps` (capture only). That is why `pip check` always
  reports that one conflict — expected, do not chase it.
  The old "numpy pinned 1.26.4, never upgrade" note is **stale**: pyorbbecsdk2
  2.1.2 works under numpy 2.2.6 (verified). The venv-local scipy matters —
  without it, transformers' optional scipy import falls back to the system
  numpy-1-built scipy and dies on `from numpy import Inf`.
  `requirements.lock.txt` (generated; see its header) pins the exact verified
  set and `--setup-voice` prefers it over the floors — regenerate it after any
  deliberate upgrade, or the robot and the lock drift apart.
- Firmware: `bebop-linux.service` runs as root from
  `/usr/local/bin/bebop-linux`; config `/etc/bebop/bebop_wheeled.yaml`.
- Other services: `bebop-agent` owns port 9091 (its `/healthz` + setup
  WebSocket) — do NOT use 9091 for anything new. Firmware WS/HTTP is 9090,
  the bebop-vision videoserver is 9092, the places service is 9095, voice is
  9093, and the `bebop-models` supervisor is 9094. `bebop-vision.service`
  (root) runs the Python recorder; it and `bebop-voice.service` ship
  un-enabled so the operator owns the GPU and can start/stop them from the app
  (the firmware shells out to `systemctl` and mirrors the unit state in
  telemetry, `VisionState`). `install-jetson.sh --enable-autostart` opts both
  into boot, which this robot has.
- **Place memory is lofi by default.** A 1 Hz DINOv3 scout writes **pending**
  places (no LLM); the app reviews/imports them in batches, optionally asking
  the 27B for a name suggestion per place. Do not enable `--place-router` for
  normal battery runs: it asks the 27B every tick (measured ~12 W average at a
  45 s interval). Pending places are invisible to `whereami`/navigation until
  imported.
- **TensorRT Edge-LLM is the only generative runtime.** Every generative
  model (omni, or the cascade's ASR + brain + TTS, or a future VLA) is a
  `tensorrt-edgellm-serve` child — a checkpoint + an engine cache + a port.
  There is no vLLM/Docker path (removed; see `docs/models.md`).
  `bebop-models` (:9094, root, installed by `install-jetson.sh --setup-models`)
  owns the GPU budget and that lifecycle (`kind: edgellm|external|tracked`);
  `bebop-voice` (:9093, root, operator-toggled from the app) is a client that
  either supervises its own Edge-LLM child or delegates to the supervisor
  (`omni_supervisor_url`).
- `bebop-agent` is provisioning-only: Wi-Fi (nmcli) + a Hosted Network
  hotspot (`Bebop-<machine-id>`, WPA2 passphrase `bebopbebop`, gateway
  `192.168.42.1:9091`) + the setup server. It no longer does BLE,
  containers, OTA, or controller pairing.
- Network mode is a two-way switch with **no fallback**: `ap` (Hosted
  Network, boot default) or `client` (Known Network). The **physical button
  press** toggles it; the app cannot.
- **Mode button now lives on the Teensy** (`[network] button_source="serial"`,
  the default). Thor has no usable GPIO header, so the switch is wired to the
  Teensy 4.1 (`teensy_bridge`, pin 2 → GND, internal pull-up) and its
  debounced press/release events ride the **third USB CDC**
  (`USB_TRIPLE_SERIAL`, `16c0:048c` if04 → `/dev/bebop-control`, symlinked by
  `install-jetson.sh --setup-imu`). `bebop-agent` reads it in
  `control_serial.rs` and toggles via the shared `button::toggle_mode` after
  a **5 s hold** (`button_hold_secs`, default 5; set 0 to toggle on press). Set
  `button_source="gpio"` for the old Jetson-header path: Orin Nano
  `button_chip=gpiochip0`, `button_line=41`, `button_bias=none` (header
  **pin 32**, `GPIO07`, internal pull-down; switch to 3.3 V, active-high, no
  resistor). On Thor that pin is `PDD.04` on the **AON** controller
  (`tegra264-gpio-aon`, separate gpiochip) — verify with `gpioinfo` before
  trusting it.
- **USB type gotcha:** `USB_TRIPLE_SERIAL` changes the Teensy PID from
  `16c0:048b` → `16c0:048c`, so the udev rules/`/dev/bebop-imu` symlink only
  exist after re-running `install-jetson.sh --setup-imu`. Interface map:
  if00=`/dev/bebop-imu` (IMU frames, `bebop-linux`), if02=`/dev/bebop-imu-debug`,
  if04=`/dev/bebop-control` (control frames, `bebop-agent`). The IMU frame
  format is unchanged.
- AP capability on this robot is confirmed (`nmcli -f WIFI-PROPERTIES.AP
  dev show wlP1p1s0` → yes; `iw list` shows `* AP`). It is a single radio,
  so the hotspot and a client Wi-Fi connection are mutually exclusive —
  expect the SSH link to drop if you switch to Hosted over Wi-Fi. Prefer
  read-only checks (`iw list`, `nmcli ... dev show`) while connected.

## Deployment process (what actually works)

1. **Code changes on the workstation → commit → push.** CI (ci.yml)
   builds the firmware release artifact on every push to `main`; runs are
   listed via the GitHub API (no `gh` CLI on this machine):
   `curl -s https://api.github.com/repos/ballerabdude/bebop/actions/runs?per_page=3`.
2. **Python changes (bebop-vision)**: fastest path is scp the changed
   files to the exact repo paths on the Jetson, then `git pull` can land
   the same content afterward (the working-tree change matches the
   commit, so checkout/pull applies cleanly). If a `git pull` aborts on
   local changes/untracked files, they are scp leftovers —
   `git stash push -u` then pull (stash contents are redundant with the
   pushed commits).
3. **Firmware changes**: do NOT hand-build on the Jetson (`cargo` there
   is only usable inside a login shell, and `nohup` strips PATH). Use the
   installer against a green CI run:
   `sudo ./scripts/install-jetson.sh --linux-only --run-id <CI_RUN_ID>`
   (run from `~/bebop` on the Jetson after pulling). Requires the
   recorder/bebop-vision processes stopped first.
4. **Never test mid-session**: check
   `pgrep -af 'record-nav[d]|main.p[y]'` before touching cameras or
   restarting services. `pkill -f 'record-navd'` matches the invoking
   SSH shell's own command line and kills your script — use the bracket
   trick `pkill -f 'record-nav[d]'` in a SEPARATE ssh command, never in
   the same script that also starts a recorder.

## Testing on the Jetson

- Python tests: `cd ~/bebop/bebop-vision && .venv/bin/python -m pytest
  tests/ -q` (135 tests, no hardware needed). Run after every deploy.
- Rust tests are CI-side (fmt + clippy + `cargo test`: the workspace in
  `jetson-agent`, and `firmware/bebop-linux`); on-Jetson builds are for
  deploy only, never for testing.
- Camera exclusivity: ONE process per camera. The recorder
  (`--record-navd`) and goal-drive (`--goal-drive`) each open the rig;
  OrbbecViewer or a stray test process holding a camera surfaces as
  `uvc_open failed: -6` / missing device.
- Operator video: the bebop-vision process serves every stream over
  WebRTC (WHEP) on `:9092/whep?stream=color_near|color_far|depth_near|
  depth_far` — H.264 over SRTP/UDP, ~150 ms; the app is WebRTC-only (no
  MJPEG/fMP4/MSE). `/snapshot?stream=...` still returns a JPEG (tools,
  tests). Needs the `gstreamer1.0-nice` apt plugin or `webrtcbin` stays
  in NULL — `install-jetson.sh` installs it. The firmware `/video` is
  removed (OBSBOT retired, plan §9).
- Live state checks: `journalctl -u bebop-linux -n 5 --no-pager`,
  `systemctl is-active bebop-linux`, `curl http://127.0.0.1:9090/healthz`.

## Driving / navigation test loop

- **The old `--goal-drive` / `--navd-model` planner is gone** (excised from
  `main` in d99a250, 2026-09-09; archived on `exp/navd-ai-20260909`, not
  trusted). The BEV student is **disabled** on the robot since 2026-10-21
  (`/etc/bebop/vision.env` = `--no-bev --nav`); measured ~5 W of GPU for a
  grid nothing consumed.
- Navigation direction (2026-10-21): topological place-to-place over the
  places service (`:9095`), steering by appearance (DINOv3 / VLM at decision
  points), **depth only for obstacle avoidance**. No SLAM.
- The fast layer is `bebop_vision/nav/` (`docs/nav.md`): depth stop-cone +
  bearing waypoint driver, HTTP on **`:9097`**, started by `--nav` inside the
  recorder process (cameras are exclusive). Goals are a bearing + distance
  or an odom point; it arms only for the goal and disarms on every exit:
  `curl -s -X POST -H 'Content-Type: application/json' -d '{"bearing_deg":0,"distance_m":0.5}' http://127.0.0.1:9097/goal`
  then poll `/status`; `/clearance` is the live sector map. `nav` never
  clears an E-STOP — a `CAN bus can4` E-STOP at arming means the drive bus
  is down (motor power off); fix that and clear it from the app.
- Mode gate: the driver switches to DIAL_IN itself (arming requires it) and
  the recorder's `--auto` captures every nav drive as an MCAP session.

## Gotchas (learned the hard way)

- Backgrounded sessions (`nohup ... &` inside a non-interactive ssh
  shell) **inherit SIGINT=SIG_IGN** — CPython keeps an inherited
  ignore, so Ctrl-C-style shutdown never arrives. Stop them with
  `sudo pkill -TERM -f 'record-nav[d]'` (main.py translates SIGTERM
  into the graceful KeyboardInterrupt path since 2026-09-06: segments
  flush, the recorder lock is released). Use sudo — the recorder runs
  as root — and the bracket trick in a separate ssh command.
- `install-jetson.sh` prereqs: don't add flags without initializing the
  variable in the defaults block (`set -u` will kill the script).
- Hosted Network bring-up uses NetworkManager shared mode
  (`nmcli con add ... 802-11-wireless.mode ap ipv4.method shared`). The
  regulatory domain is unset (`country 00`) on this robot; AP on 2.4 GHz
  still works, but set `iw reg set <CC>` if a channel is rejected. Creating
  the profile (`con add`) does not disrupt the current Wi-Fi link; only
  `con up` does — safe to validate command syntax read-only.
- Jetson header GPIO bias lives in the **pinmux**, not in the runtime line
  request: `libgpiod`/`gpiocdev` bias flags are effectively a no-op, so an
  input with `PULL=NONE` floats. On Orin (tegra234) you can read a pin's
  pinmux register with `sudo busybox devmem <addr>` (address per pin in NVIDIA's
  `jetson-gpio` `gpio_pin_data.py`; e.g. Orin pin 29 = `0x2430068`), pull is
  bits [11:10] (0=none, 1=down, 2=up). **Those addresses are tegra234-specific
  and do not apply to Thor (tegra264)** — the Thor pin table in the same file
  exposes no pinmux addresses, so re-derive on this robot before relying on it.
  Wire a button to the **opposite** rail of the pin's built-in pull so no
  external resistor is needed.
- IMU SPI GPIO defaults are likewise Orin Nano: `firmware/bebop-linux/config/bebop_v2.yaml`
  and `src/bin/imu_probe.rs` use `gpiochip0:144` (header pin 7, INT) and
  `gpiochip0:85` (header pin 15, RST). On Thor the header lines map to tegra264
  controllers and differ — override `int_chip`/`int_line`/`rst_chip`/`rst_line`
  for the actual wiring.
- Protobuf: three binding sets (Rust prost auto via build.rs, app TS via
  `npm run gen-proto` in bebop-app/, Python pb2 checked in at
  `bebop-vision/bebop_vision/proto/bebop/runtime/v1/`). The Python pb2
  regenerates with the proto staged at
  `bebop/runtime/v1/bebop_runtime.proto` (matching the package path),
  NOT from the flat proto root.
- GIL: pyorbbecsdk capture must stay serial (§2.8 of docs/navd.md);
  thread pools are fine for numpy/cv2 work only.
- OpenBLAS/OpenMP: on the Thor, numpy's OpenBLAS worker pool spin-waits
  after even tiny matmuls. A standalone `--bev` run measured 429% CPU with
  five BLAS threads at 70-90% each; the actual BEV tick was ~35 ms. `main.py`
  and `master_bev.py` set `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_*`/
  `NUMEXPR_*`/`VECLIB_*` to 1 **before** numpy is imported. Keep that at the
  top of any new entry point; otherwise the idle pool burns cores.
- Dual cameras run matched 30 fps under hardware sync (both cams negotiate
  848x480@30 on the Thor). The old "matched 15 fps / 30+30 starves the GIL"
  rule was measured on the Orin Nano with the old BEV worker; without a BEV
  the Thor recorder sustains ~29 Hz at 30+30. The learned BEV
  (`main.py --bev-model`, `docs/bev-student.md`) is a separate 10 Hz worker over
  both depth streams; `--no-bev` disables it during `--record-navd`.
