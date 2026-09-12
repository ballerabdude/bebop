# bebop — agent working notes

## Robot access

- SSH: `ssh bebop@bebop.local`, password `bebop` (non-interactive: use
  `sshpass -p bebop ssh ...`).
- Repo on the Jetson: `~/bebop` (same monorepo layout as here).
- Python env: `~/bebop/bebop-vision/.venv` — run as
  `/home/bebop/bebop/bebop-vision/.venv/bin/python`. numpy is pinned to
  1.26.4 (numpy 2.x breaks pyorbbecsdk) — never upgrade it.
- Firmware: `bebop-linux.service` runs as root from
  `/usr/local/bin/bebop-linux`; config `/etc/bebop/bebop_wheeled.yaml`.
- Other services: `bebop-agent` owns port 9091 (its `/healthz` + setup
  WebSocket) — do NOT use 9091 for anything new. Firmware WS/HTTP is 9090,
  the bebop-vision videoserver is 9092.
- `bebop-agent` is provisioning-only: Wi-Fi (nmcli) + a Hosted Network
  hotspot (`Bebop-<machine-id>`, WPA2 passphrase `bebopbebop`, gateway
  `192.168.42.1:9091`) + the setup server. It no longer does BLE,
  containers, OTA, or controller pairing.
- Network mode is a two-way switch with **no fallback**: `ap` (Hosted
  Network, boot default) or `client` (Known Network). The **physical button
  press** toggles it; the app cannot. Button default = Orin Nano
  header **pin 32** (`gpiochip0` line 41, internal pull-down; wire the
  switch to 3.3 V, active-high, no resistor). Avoid pins 7/15 (IMU) and
  pin 29 (no internal pull).
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
  tests/ -q` (40 tests, no hardware needed). Run after every deploy.
- Firmware tests are CI-side (108 tests); on-Jetson builds are for
  deploy only.
- Camera exclusivity: ONE process per camera. The recorder
  (`--record-navd`) and goal-drive (`--goal-drive`) each open the rig;
  OrbbecViewer or a stray test process holding a camera surfaces as
  `uvc_open failed: -6` / missing device.
- Operator video: the bebop-vision process serves MJPEG on
  `:9092/video?stream=color_near|color_far|depth_near|depth_far`
  (`/snapshot?stream=...` for stills). The firmware `/video` is removed
  (OBSBOT retired, plan §9).
- Live state checks: `journalctl -u bebop-linux -n 5 --no-pager`,
  `systemctl is-active bebop-linux`, `curl http://127.0.0.1:9090/healthz`.

## Driving / navigation test loop

- Recorder + navigation in one process (the flag makes the recorder
  consume app goals and drive):
  `sudo .venv/bin/python -u main.py --record-navd /var/lib/bebop-captures --auto --goal-drive --navd-model weights/navd.onnx`
  (add `--drive-any-mode` to skip the Policy-mode gate — Dial-in works).
  Driving is model-only: `--goal-drive` refuses to start without
  `--navd-model` (a tick the model cannot serve stops the drive node;
  the reason shows in the `[navd-model]` log lines; needs
  weights/navd.onnx scp'd to the robot — it is gitignored).
- The `nav:` status lines print the drive-loop gate state
  (waiting/hold/estop/no_floor/search/rotate/hard_stop/drive) — if the
  robot isn't moving, that line says why.
- Mode gate: goals need wheels armed + Policy mode unless
  `--drive-any-mode`; the app's **Go** button (current app build) does
  the Policy+arm switch itself, **Clear** returns to Dial-in.

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
  request: `libgpiod`/`gpiocdev` bias flags are effectively a no-op here, so
  an input with `PULL=NONE` floats. Read the pin's pinmux register with
  `sudo busybox devmem <addr>` (address per pin is in NVIDIA's `jetson-gpio`
  `gpio_pin_data.py`; e.g. pin 29 = `0x2430068`); pull is bits [11:10]
  (0=none, 1=down, 2=up). On this Orin Nano: pin 29 = `PULL_NONE` (floats),
  pin 32 = pull-down. Wire a button to the **opposite** rail of the pin's
  built-in pull — then no external resistor is needed (button on pin 32 to
  3.3 V, active-high).
- Protobuf: three binding sets (Rust prost auto via build.rs, app TS via
  `npm run gen-proto` in bebop-app/, Python pb2 checked in at
  `bebop-vision/bebop_vision/proto/bebop/runtime/v1/`). The Python pb2
  regenerates with the proto staged at
  `bebop/runtime/v1/bebop_runtime.proto` (matching the package path),
  NOT from the flat proto root.
- GIL: pyorbbecsdk capture must stay serial (§2.8 of docs/navd.md);
  thread pools are fine for numpy/cv2 work only.
- Dual cameras must run matched 15 fps (hardware sync pairing); 30+30
  starves the GIL and BEV collapses to ~1 Hz.
