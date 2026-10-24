# Onboarding

## Workstation

- Linux. The Rust crates use Linux-only APIs (SocketCAN, the GPIO character
  device); the app needs only Node.
- Rust 1.90.0 through `rustup` (`jetson-agent/rust-toolchain.toml` pins it;
  CI builds both Rust workspaces with it).
- `protoc` (`apt install protobuf-compiler`): both Rust workspaces compile
  their protos with `prost-build`.
- Node 22 (`bebop-app/.nvmrc`) for the app.
- Python 3.10 or newer for the `bebop-vision` tools and tests; the venv
  setup is in [`../bebop-vision/README.md`](../bebop-vision/README.md).
- Optional: [`just`](https://github.com/casey/just) for the repo-root
  recipes, and Docker for the Isaac Sim / Isaac Lab / ROS 2 containers of the
  retired v2 humanoid (`docker-compose.yml`).

## Build and test

| Component | From the repo root |
|---|---|
| `jetson-agent` | `just check`, `just test`, `just lint` (`cargo check`/`test`/`clippy --workspace` in `jetson-agent/`) |
| `firmware/bebop-linux` | `cd firmware/bebop-linux && cargo test` (CI also runs `cargo clippy --all-targets -- -D warnings`) |
| `bebop-app` | `cd bebop-app && npm install && npm test` |
| `bebop-vision` | `cd bebop-vision && .venv/bin/python -m pytest tests/ -q` |

CI runs only the two Rust workspaces; run the app and Python tests yourself.

## The robot

A Jetson AGX Thor Developer Kit on JetPack 7.2 / L4T R39.2.1, reachable as
`ssh bebop@bebop.local`. The repo is checked out at `~/bebop`; the
bebop-vision units hard-code `/home/bebop/bebop/bebop-vision`.

`jetson-flash/` targets the Orin Nano devkit and does not apply to the Thor;
flashing the Thor is not scripted in this repo.

### Installing the stack

From `~/bebop` on the robot:

```sh
sudo ./scripts/install-jetson.sh                    # latest firmware Release + latest green main agent (needs `gh auth login`)
sudo ./scripts/install-jetson.sh --run-id <id>      # both daemons from one CI run
sudo ./scripts/install-jetson.sh --local --build    # build and install from this checkout, no GitHub
```

It installs `bebop-agent` and `bebop-linux` (binary,
`/etc/bebop/bebop_wheeled.yaml`, unit), installs the `bebop-vision`,
`bebop-voice` and `bebop-model-download@` units without enabling them, and
enables `bebop-gpu-clocks`. Hardware and service setup is opt-in:

| Flag | Does |
|---|---|
| `--setup-can` | blacklists `mttcan`, loads `gs_usb`, brings `can*` up at 1 Mbit/s through systemd-networkd (`--build-gs-usb` builds the module if the kernel lacks it) |
| `--setup-imu` | udev rule and symlinks for the Teensy: `/dev/bebop-imu`, `/dev/bebop-imu-debug`, `/dev/bebop-control` |
| `--setup-orbbec` | Orbbec udev rules |
| `--setup-voice` | builds `bebop-vision/.venv` (recorder, places, voice, TensorRT Edge-LLM) |
| `--quantize-voice` | quantizes the downloaded Qwen3-Omni checkpoint to NVFP4 for the omni engine |
| `--setup-models` | installs and enables `bebop-models.service` |
| `--enable-autostart` | enables `bebop-vision` and `bebop-voice` at boot |
| `--config-yaml <name>.yaml` | activates another robot config from the bundle (default `bebop_wheeled.yaml`; remembered in `/etc/bebop/config-yaml-name`) |

`--setup-can`, `--setup-imu`, `--setup-orbbec`, `--setup-voice` and
`--quantize-voice` have `-only` variants that skip the binaries;
`--help` lists every flag. `sudo ./scripts/setup-rerun.sh` adds the
MCAP → Rerun converter behind `GET /captures/rrd/<stem>.rrd`.
[`../AGENTS.md`](../AGENTS.md) has the deploy process used on the bench and
the checks to run before restarting services.

### Running the agent on a Jetson

To iterate on `bebop-agent` alone:

```sh
just build-jetson               # on an arm64 Linux host; or take bebop-agent-aarch64 from CI
just deploy bebop@bebop.local   # scp the binary + rsync jetson-agent/deploy/, then run install.sh
```

`jetson-agent/deploy/scripts/install.sh` puts the binary at
`/usr/local/bin/bebop-agent`, installs `bebop-agent.service`, and seeds
`/etc/bebop/agent.toml` from `jetson-agent/deploy/examples/agent.toml` when
it does not exist yet. It runs `systemctl enable --now`, which leaves an
agent that is already running on the old binary:
`sudo systemctl restart bebop-agent` picks up the new one.

### The Teensy

The robot's Teensy 4.1 runs the `teensy_bridge` env:

```sh
cd firmware/bebop-locomotion && pio run -e teensy_bridge --target upload
```

Then run `sudo ./scripts/install-jetson.sh --setup-imu-only` on the robot
if `/dev/bebop-control` is missing (the udev rule matches the
`USB_TRIPLE_SERIAL` id `16c0:048c`). `just fw-build` and `just fw-flash` use
the default env (`sim`), whose USB setup lacks `/dev/bebop-control`, so
flashing it disables the mode button.

## Running the app

```sh
just app-web        # = cd bebop-app && npm run dev: Vite on 0.0.0.0:1420
```

Open `http://localhost:1420/#/` and add the robot (`bebop.local` or its IP;
`192.168.42.1` over its hotspot). The microphone (push-to-talk) needs a
secure context: see [`dev-https.md`](dev-https.md). To serve the app from
the robot itself, run `npm install` in `bebop-app/` there, then
`sudo ./bebop-app/deploy/install-app-dev.sh` installs `bebop-app.service`
(the same dev server at `http://bebop.local:1420`).

## Next

- [`architecture.md`](architecture.md): how the pieces connect.
- [`setup-protocol.md`](setup-protocol.md): provisioning and the mode
  button.
