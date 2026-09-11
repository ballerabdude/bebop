# Architecture

## Components

### `jetson-agent/bebop-agent` (Rust, systemd service)

The provisioning daemon. It owns first-time Wi-Fi setup and the SoftAP
fallback; everything else on the robot runs as native binaries/venvs.

Subsystems (see `jetson-agent/bebop-agent/src/`):

- **`wifi/`** — wraps `nmcli` to scan/join Wi-Fi, and polls link status.
- **`ap.rs`** — SoftAP supervisor. Raises the `Bebop-XXXX` hotspot when no
  known network is available (`auto` mode), or always (`ap` mode), and tears
  it down once a client network connects.
- **`server.rs`** — setup server: protobuf-over-WebSocket (`:9091`) plus a
  minimal status page. Reachable on the LAN or over the SoftAP.
- **`dispatcher.rs`** — maps `ClientRequest`s to Wi-Fi / config actions.
- **`state.rs`** — shared in-memory state (`Arc<Inner>` with async `RwLock`s).

See [`setup-protocol.md`](setup-protocol.md) for the wire protocol.

### `jetson-agent/bebop-proto` (Rust, shared crate)

Protobuf schema for the setup protocol, compiled with `prost-build` into
Rust types used by the agent. The companion app generates TypeScript
bindings from the same `.proto` via `npm run gen-proto`.

### `firmware/bebop-linux` (Rust, systemd service)

The robot firmware: SocketCAN + ODRIVE/Robstride control loop, safety
supervisor, IMU, and the runtime WebSocket API (`:9090`) the operator app
drives through. Runs as a native binary (`bebop-linux.service`). The vision
stack runs from a Python venv (`bebop-vision/`) and serves MJPEG on `:9092`.

### `bebop-app` (Tauri 2 + React + TypeScript)

The customer-facing companion app:

- **Provisioning** — talks to `bebop-agent`'s setup server over WebSocket
  (`SetupTransport`, `:9091`) to join Wi-Fi and name the robot. Over the
  robot's SoftAP during first-time setup, or over the LAN afterwards.
- **Operating** — talks to `bebop-linux`'s runtime server (`RuntimeTransport`,
  `:9090`) for telemetry, modes, motor bench, and driving; and to the
  bebop-vision server (`:9092`) for video. Client-side gamepad input uses the
  Web Gamepad API and streams twists over the runtime WS — no robot-side
  Bluetooth pairing is involved.

## Boot sequence on a Jetson

```
 power on
   │
   ▼
 systemd
   ├── NetworkManager.service
   ├── bebop-agent.service ──┐
   │                         ▼
   │                   bebop-agent main
   │                   ├── Wi-Fi status poller
   │                   ├── SoftAP supervisor  (Bebop-XXXX, fallback)
   │                   └── setup server :9091
   └── bebop-linux.service
       └── runtime WS :9090 ─── SocketCAN ── motors
```

On first boot (no saved Wi-Fi) the SoftAP comes up after
`network.ap_auto_after_secs`; the user joins `Bebop-XXXX`, provisions Wi-Fi
through the app, and the robot rejoins its saved network on later boots.

## Process / trust boundaries

| Component     | Runs as | Why                                              |
|---------------|---------|--------------------------------------------------|
| `bebop-agent` | root    | NetworkManager write access (`nmcli` scan/join/AP) |
| `bebop-linux` | root    | SocketCAN, GPIO/SPI                             |
| mobile app    | off-device | untrusted; reachable only on the LAN/SoftAP  |

## SoftAP notes

- The Jetson Orin Nano Wi-Fi module (Realtek RTL8822CE) supports AP mode.
  Verify with `iw list` (`* AP`) or
  `nmcli -f WIFI-PROPERTIES.AP dev show <iface>`.
- It is a single radio: the robot cannot be a Wi-Fi client and a hotspot at
  the same time. The SoftAP is therefore a fallback state, not the steady
  state.
- The setup passphrase is fixed (`bebopbebop`) for now; move to a per-device
  derived code before customer shipments.

## Cross-platform builds

The agent has no Linux-specific dependencies beyond shelling out to `nmcli`
at runtime, so `cargo check` / `cargo test` work on any host. Producing a
real Jetson binary is a native arm64 `cargo build`, run on the robot, an
arm64 dev box, or CI (`ubuntu-22.04-arm`), whose artifact is uploaded as
`bebop-agent-aarch64`.
