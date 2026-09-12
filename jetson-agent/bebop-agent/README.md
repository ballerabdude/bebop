# `bebop-agent`

On-device provisioning daemon that runs on every Bebop robot. Responsibilities:

- **Wi-Fi provisioning** (`src/wifi/`) — wraps `nmcli` / NetworkManager to
  scan, join, and report link status.
- **Hosted Network** (`src/ap.rs`) — hosts the `Bebop-XXXX` WPA2 hotspot
  while `mode = "ap"` (boot default) and tears it down for `mode = "client"`.
  No automatic fallback.
- **Mode button** (`src/button.rs`) — a GPIO long-press (default header
  pin 29 / `gpiochip0` line 105) toggles the mode
  between Hosted (`ap`) and Known (`client`) and persists it. Edit the
  Hosted SSID/password/band from the app.
- **Setup server** (`src/server.rs`) — protobuf-over-WebSocket on `:9091`
  (plus a status page) so the companion app can provision over the Hosted
  Network or the LAN.
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
Hosting the network requires root.

## Configuration

See [`../deploy/examples/agent.toml`](../deploy/examples/agent.toml). The
key knobs are `[network] mode` (`ap` / `client`, button-owned) and
`ap_ssid` / `ap_password` / `ap_band` for the Hosted Network. The mode
button is under `[network] button_*` (default pin 29 / `gpiochip0` line 105).

## Packaging / Install

See [`../deploy/`](../deploy/) for the systemd unit and install script.
