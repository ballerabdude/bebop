# Bebop

Bebop is a wheeled robot built on an NVIDIA Jetson AGX Thor. The current
robot (v3) is a differential-drive base: two ODrive S1 BotWheels with passive
front casters, a Robstride PowerBoard on a 13s2p Li-ion pack, two Orbbec
Gemini 335Lg depth cameras, and a Teensy 4.1 bridge for the mode button and
IMU. Everything runs on the robot, including the voice agent (TensorRT
Edge-LLM). Two Robstride arms come next; the firmware keeps its Robstride
joint path for them. The retired v2 robot was a legged humanoid; its
training and ROS 2 tooling are still in the tree.

This monorepo holds the on-robot services, the operator app and the
workstation tooling around them.

## Architecture

```
 operator app (browser: phone, tablet or desktop)
  │
  ├─ :9091 setup WS ──────────────▶ bebop-agent   Wi-Fi + Hosted Network; mode button via the Teensy
  │
  ├─ :9090 runtime WS + HTTP ─────▶ bebop-linux   100 Hz control loop, safety supervisor, system log
  │                                 ├─ SocketCAN can4 ─▶ 2× ODrive S1 (wheels)
  │                                 ├─ SocketCAN can0 ─▶ Robstride PowerBoard (battery)
  │                                 └─ systemctl ──────▶ bebop-vision, bebop-voice,
  │                                                      bebop-model-download@<id>
  ├─ :9092 video + drive notes ──┐
  ├─ :9095 places ───────────────┴▶ bebop-vision  recorder, WHEP video, place memory; owns the cameras
  ├─ :9093 voice WS ──────────────▶ bebop-voice   agent gateway: ASR → brain → TTS, or Qwen3-Omni
  └─ :9094 models HTTP ───────────▶ bebop-models  GPU budget + Edge-LLM model lifecycle
```

Every service is a systemd unit on the Thor (JetPack 7.2 / L4T R39.2.1,
CUDA 13.2, Python 3.12). The recorder and the voice agent are clients of the
runtime API too. Ports, boot order and who runs as root are in
[`docs/architecture.md`](docs/architecture.md).

## Repository layout

| Path | What it is |
|---|---|
| `firmware/bebop-linux/` | The robot runtime (Rust): SocketCAN supervisor for the ODrive wheels, the PowerBoard and the Robstride joint path; the runtime WebSocket/HTTP API on `:9090`; the always-on system log; the `sysid` single-joint actuator bench. |
| `firmware/bebop-locomotion/` | Teensy 4.1 firmware (PlatformIO). `teensy_bridge` is the production env: BNO085 IMU frames and mode-button events over USB serial. `sim` and `ros` are the legacy micro-ROS controller of an earlier wheel-legged robot (hip and knee Robstrides plus wheels); `imu_test` and `imu_spi_test` are standalone BNO085 checks (I2C, SPI). |
| `jetson-agent/` | Rust workspace: `bebop-agent` (Wi-Fi and Hosted Network provisioning, the mode button, the setup server on `:9091`), `bebop-proto` (the setup and runtime protobuf schemas), `deploy/` (agent unit and install scripts). |
| `bebop-vision/` | Python, one venv on the robot: the recorder (navd MCAP sessions), operator video (WHEP on `:9092`, `/snapshot`, drive notes), place memory (`:9095`), the voice gateway (`:9093`), the model supervisor (`:9094`) and weight downloads; plus the labeling and dataset tools (SAM 3.1 × depth labels, MCAP extraction, capture pull, session ledger). |
| `bebop-app/` | The operator app (React + TypeScript + Vite): setup wizard, driving, agent console, place memory, diagnostics, settings. `src-tauri/` is an unused Tauri 2 shell. |
| `scripts/` | `install-jetson.sh` (installs and upgrades the robot stack), `setup-rerun.sh` (on-robot MCAP → Rerun conversion), `build-gs-usb.sh`, the GPU-clocks unit, Orbbec udev rules. |
| `rerun_dashboards/` | MCAP → Rerun `.rrd` conversion and the navd and system dashboards (`open.py`). |
| `foxglove/` | Foxglove layouts: navd sessions, plus layouts for the retired v2 policy captures. |
| `sim/` | Isaac Sim / Isaac Lab training for the retired v2 humanoid (`bebop_training`), the Robstride sysid and actuator-net fitting tools, kept for the arms (`bebop_training/tools/`), and the on-device Thor training env (`setup/thor/`). |
| `ros2/` | The v2 humanoid's ROS 2 Jazzy dev workspace (URDF, `bebop_pilot`, micro-ROS agent container). Legacy. |
| `bebop-mb/` | KiCad motherboard joining the Orin Nano's 40-pin header, a Teensy 4.1, a BNO085 and a CAN transceiver. Orin Nano era. |
| `jetson-flash/` | Host-side L4T download and flash for the Orin Nano devkit. Stale for the Thor: its board config and flash commands are Orin-only. |
| `fusion-scripts/` | Fusion 360 helper scripts (hide sketches, merge bodies, rename bodies, set mass). |
| `docs/` | Cross-cutting docs; index in [`docs/README.md`](docs/README.md). |
| `docker-compose.yml` | Workstation containers `ros2_docker`, `isaac_sim`, `isaac_lab` (profiles `sim`, `lab`). |
| `Caddyfile.dev` | Dev TLS proxy for the app ([`docs/dev-https.md`](docs/dev-https.md)). |
| `justfile` | Recipes; `just` lists them. |
| `.github/workflows/ci.yml` | CI: fmt, clippy and tests for both Rust workspaces, arm64 release builds, firmware Releases on `firmware/v*` tags. |
| `AGENTS.md` | Working notes: robot access, deploy process, gotchas. |

## Getting started

[`docs/onboarding.md`](docs/onboarding.md) covers workstation setup, tests
and installing on the robot. The common recipes:

```sh
just                           # list recipes
just check                     # jetson-agent: cargo check
just test                      # jetson-agent: cargo test
just lint                      # jetson-agent: clippy, warnings are errors
just app-web                   # operator app on the Vite dev server (:1420)
just build-jetson              # bebop-agent release build (arm64 Linux only)
just deploy bebop@bebop.local  # install that bebop-agent build on the robot
just pull-data                 # mirror captures from the robot, extract, write the ledger
```

The firmware has no recipe: `cd firmware/bebop-linux && cargo test` (Linux,
`protoc` installed). On the robot, `sudo ./scripts/install-jetson.sh`
installs `bebop-agent`, `bebop-linux` and the service units.

`just fw-build` and `just fw-flash` use PlatformIO's default env (`sim`), not
the robot's `teensy_bridge`. Flash the Teensy with
`cd firmware/bebop-locomotion && pio run -e teensy_bridge --target upload`.
