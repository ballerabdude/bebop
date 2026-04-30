# Architecture

## Components

### `jetson-agent/bebop-agent` (Rust, systemd service)

The only first-party process running on the host OS. Everything else lives
in containers.

Subsystems (see `jetson-agent/bebop-agent/src/`):

- **`ble/`** — GATT server on top of BlueZ, handles mobile-app requests.
- **`wifi/`** — wraps `nmcli` to scan and join Wi-Fi networks.
- **`containers/`** — supervises the robot app container via the Docker
  API (Bollard). Uses the NVIDIA container runtime.
- **`ota/`** — polls a signed JSON manifest and rolls the app container
  forward when it changes.
- **`state.rs`** — shared in-memory state handle (`Arc<Inner>` with async
  `RwLock`s) passed to every subsystem.

### `jetson-agent/bebop-proto` (Rust, shared crate)

Protobuf schema for the BLE wire protocol. Compiled with `prost-build` into
plain Rust types used by the agent. The mobile app re-implements the same
framing + UUIDs natively (no path-dependency on this crate) — see
`ble-protocol.md`.

### `jetson-agent/robot-app` (container)

The Bebop robot application itself. NVIDIA L4T base image, pulled by the
agent, run with `--runtime=nvidia` and `--network=host`. Built and pushed
independently of the agent; the agent simply tracks the latest tag for its
configured channel via the OTA manifest.

### `bebop-app` (Tauri 2 + React + TypeScript)

The customer-facing companion app. Single app for first-time setup, ongoing
configuration, and OTA dashboard. Talks to `bebop-agent` over BLE through a
`BebopTransport` abstraction with two implementations:

- **`TauriTransport`** — desktop / iOS / Android, backed by the `btleplug`
  crate inside the Tauri shell (`bebop-app/src-tauri/src/ble/`).
- **`WebBluetoothTransport`** — plain browsers that expose Web Bluetooth
  (Chrome, Edge), used when running outside Tauri.

Both transports speak the same length-prefixed framing and protobuf
messages as the agent. The framing implementation is mirrored in three
places — see `ble-protocol.md` for the canonical wire format.

## Boot sequence on a Jetson

```
 power on
   │
   ▼
 systemd
   ├── bluetooth.service    (BlueZ)
   ├── NetworkManager.service
   ├── docker.service
   └── bebop-agent.service ──┐
                             ▼
                       bebop-agent main
                       ├── BLE GATT server (advertising "Bebop-XXXXXX")
                       ├── Wi-Fi status poller
                       ├── Container supervisor
                       │     └── docker run nvcr.io/your-org/bebop-app:<tag>
                       └── OTA poller  ◀──── https://.../channels/stable.json
```

## Process / trust boundaries

| Component          | Runs as | Why                                 |
|--------------------|---------|-------------------------------------|
| `bebop-agent`      | root    | BlueZ, NetworkManager, Docker socket |
| robot app          | inside Docker (nvidia runtime) | isolation, simple OTA |
| mobile app         | off-device | untrusted; must authenticate over BLE |

See [`ble-protocol.md`](ble-protocol.md) for the authentication scheme.

## Why this split?

- Native agent (not containerised) because it needs host BlueZ +
  NetworkManager + the Docker socket. Containerising it would require a
  privileged container with host networking and D-Bus — same blast radius,
  more complexity.
- Robot application containerised so deploys are just new image tags and
  so CUDA/TensorRT versions travel with the app.

## Cross-platform builds

The agent depends on `bluer`, which only links on Linux (it uses BlueZ
over D-Bus). Two things keep development workable elsewhere:

- The `bluer` dep is target-gated in `jetson-agent/bebop-agent/Cargo.toml`
  to `cfg(target_os = "linux")`.
- `jetson-agent/bebop-agent/src/ble/server_stub.rs` provides a no-op BLE
  server on non-Linux hosts, so `cargo check` / `cargo test` /
  `cargo build` all work on macOS dev boxes (BLE just logs a warning and
  sleeps).

Producing a real Jetson binary is done with `cross` (configured via
`jetson-agent/Cross.toml`) — see `onboarding.md` and the `build-jetson`
recipe in the top-level `justfile`.
