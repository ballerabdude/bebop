# `bebop-agent`

On-device provisioning daemon that runs on every Bebop robot. Responsibilities:

- **Wi-Fi provisioning** (`src/wifi/`) — wraps `nmcli` / NetworkManager to
  scan, join, and report link status.
- **SoftAP fallback** (`src/ap.rs`) — raises a `Bebop-XXXX` WPA2 hotspot when
  no known network is available (`auto` mode) or always (`ap` mode), and
  tears it down once a client network connects.
- **Setup server** (`src/server.rs`) — protobuf-over-WebSocket on `:9091`
  (plus a status page) so the companion app can provision over the SoftAP or
  the LAN.
- **Shared state** (`src/state.rs`) — cheap-to-clone handle passed between
  subsystems.

The agent no longer manages containers, OTA, or Bluetooth controllers; the
robot firmware (`firmware/bebop-linux`) runs natively and the app drives it
directly over the runtime WS.

## Building

```sh
cargo build --release -p bebop-agent
```

`protoc` must be on `PATH` (protobuf codegen). The binary is built natively
on arm64 — on the robot, an arm64 dev box, or in CI on `ubuntu-22.04-arm`
(artifact `bebop-agent-aarch64`).

## Running (dev)

```sh
BEBOP_AGENT_CONFIG=./deploy/examples/agent.toml \
RUST_LOG=info,bebop_agent=debug \
cargo run -p bebop-agent
```

The agent expects NetworkManager (`nmcli`) to be available. Raising the
SoftAP requires root.

## Configuration

See [`../deploy/examples/agent.toml`](../deploy/examples/agent.toml). The
key knob is `[network] mode` (`auto` / `client` / `ap`) and
`ap_password` (WPA2, 8..=63 chars).

## Packaging / Install

See [`../deploy/`](../deploy/) for the systemd unit and install script.
