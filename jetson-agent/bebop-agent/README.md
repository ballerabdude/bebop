# `bebop-agent`

On-device daemon that runs on every Bebop robot. Responsibilities:

- **BLE GATT server** (`src/ble/`) — surface used by the Bebop mobile app to
  provision Wi-Fi, read device info, and control the robot application.
- **Wi-Fi provisioning** (`src/wifi/`) — wraps `nmcli` / NetworkManager.
- **Container manager** (`src/containers/`) — keeps the robot application
  container running via the local Docker daemon (NVIDIA container runtime).
- **OTA updater** (`src/ota/`) — polls a signed manifest and rolls the robot
  application forward when a new image is published.
- **Shared state** (`src/state.rs`) — cheap-to-clone handle passed between
  subsystems.

## Building

On a Jetson (or an `arm64` Ubuntu dev box):

```sh
sudo apt install -y libdbus-1-dev pkg-config protobuf-compiler
cargo build --release -p bebop-agent
```

For cross-compiling from an x86 host you probably want
[`cross`](https://github.com/cross-rs/cross):

```sh
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu -p bebop-agent
```

## Running (dev)

```sh
BEBOP_AGENT_CONFIG=./deploy/examples/agent.toml \
RUST_LOG=info,bebop_agent=debug \
cargo run -p bebop-agent
```

The agent expects:

- BlueZ (`bluetoothd`) running and the adapter available via D-Bus.
- NetworkManager available for Wi-Fi control.
- Docker with the `nvidia` runtime configured (on-robot only).

## Configuration

See [`../deploy/examples/agent.toml`](../deploy/examples/agent.toml).

## Packaging / Install

See [`../deploy/`](../deploy/) for the systemd unit and install script.
