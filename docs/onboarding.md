# Onboarding

## Prerequisites

### Your dev machine (macOS or Linux)

- Rust toolchain (`rustup`, current stable — pinned via
  `jetson-agent/rust-toolchain.toml`)
- `protoc` (the protobuf compiler) — `brew install protobuf` or
  `apt install protobuf-compiler`
- Docker Desktop / Docker Engine (used by `buildx` for the robot-app image
  and by the Isaac Sim / ROS 2 dev containers)
- (Optional) [`just`](https://github.com/casey/just) for the canned
  recipes in the top-level `justfile` (`just check`, `just build-jetson`,
  `just deploy user@robot.local`, ...)
- For working on the mobile app: Node 20+ (managed via `nvm`,
  `.nvmrc` in `bebop-app`)

### Your Jetson (target)

- JetPack ≥ 6.x (L4T r36.x) recommended
- `bluez` (BlueZ 5)
- NetworkManager (usually default)
- Docker + `nvidia-container-toolkit` with the `nvidia` runtime configured
  as a **named** runtime in `/etc/docker/daemon.json`:

  ```json
  {
    "runtimes": {
      "nvidia": {
        "path": "nvidia-container-runtime",
        "runtimeArgs": []
      }
    }
  }
  ```

## First build

The Rust workspace lives in `jetson-agent/`:

```sh
cd jetson-agent
cargo build --workspace
```

This works on macOS as well as Linux: the `bluer` crate is target-gated to
Linux in `jetson-agent/bebop-agent/Cargo.toml`, and
`bebop-agent/src/ble/server_stub.rs` provides a no-op BLE server on other
platforms. On macOS the agent will run, but the BLE subsystem just logs a
warning and sleeps — useful for iterating on the container manager, OTA
poller, dispatcher, etc.

If you have `just` installed (recipes run from the repo root):

```sh
just check     # cd jetson-agent && cargo check --workspace --all-targets
just test      # cd jetson-agent && cargo test --workspace
just lint      # cd jetson-agent && cargo clippy --workspace --all-targets -- -D warnings
```

## Running the agent on a Jetson

> First time on a brand-new device? You need to flash Jetson Linux onto
> the Orin Nano first — see [`../jetson-flash/README.md`](../jetson-flash/README.md).
> Once L4T is installed and the Jetson boots, come back here.

1. Build for arm64. The agent is built natively now — no `cross` / QEMU /
   Docker. Pick a build host:
   - **On the robot (or any arm64 Ubuntu box):**
     ```sh
     sudo apt install -y libdbus-1-dev pkg-config protobuf-compiler
     just build-jetson
     # equivalent to:
     # cd jetson-agent && cargo build --release -p bebop-agent
     ```
   - **From an x86 dev machine:** grab the `bebop-agent-aarch64` artifact
     from the latest green CI run on `main` (built on `ubuntu-22.04-arm`,
     glibc 2.35 → JetPack 6 compatible).
2. Push it to the robot and install:
   ```sh
   just deploy bebop@robot.local
   # equivalent to:
   # scp jetson-agent/target/release/bebop-agent bebop@robot.local:/tmp/
   # rsync -a jetson-agent/deploy/ bebop@robot.local:/tmp/deploy/
   # ssh bebop@robot.local 'sudo /tmp/deploy/scripts/install.sh /tmp/bebop-agent'
   ```

The install script drops the binary at `/usr/local/bin/bebop-agent`,
installs the systemd unit from
`jetson-agent/deploy/systemd/bebop-agent.service`, and seeds
`/etc/bebop/agent.toml` from `jetson-agent/deploy/examples/agent.toml` if
it doesn't already exist.

## Running the mobile app

The companion app lives at [`bebop-app/`](../bebop-app/). Quick start
(from `bebop-app/`):

```sh
nvm use
npm install
npm run tauri dev                                  # desktop dev build
npm run tauri android init && npm run tauri android dev
npm run tauri ios init     && npm run tauri ios dev
```

You can also run the React UI directly in a Web Bluetooth-capable
browser (Chrome / Edge) with `npm run dev` — the app auto-detects which
transport to use. The repo-root recipes `just app-dev` and `just app-web`
wrap these.

## Next steps

- Read [`architecture.md`](architecture.md) to see how the pieces connect.
- Read [`ble-protocol.md`](ble-protocol.md) to understand the wire format
  shared by the agent and the mobile app.
- Read [`ota-flow.md`](ota-flow.md) for the update lifecycle.
