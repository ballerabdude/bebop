# `bebop-agent`

The on-robot provisioning daemon (`bebop-agent.service`, root): Wi-Fi, the
Hosted Network hotspot, the mode button and the setup server on `:9091`.
Each subsystem runs as its own tokio task and shares one `AppState`
(`src/state.rs`).

| Module | Does |
|---|---|
| `src/wifi/` | wraps `nmcli` to scan and join networks and polls the client link |
| `src/ap.rs` | the Hosted Network supervisor: hosts the `Bebop-<first 6 chars of /etc/machine-id>` WPA2 hotspot while `network.mode = "ap"` (the boot default) and lowers it for `"client"`; no automatic fallback |
| `src/button.rs`, `src/control_serial.rs` | the mode button. `button_source = "serial"` (default): press/release frames from the Teensy on `/dev/bebop-control`. `"gpio"`: a Jetson header line polled through `gpiocdev` (Orin Nano pin 32). Holding it `button_hold_secs` (default 5) toggles `ap` ⇄ `client` and saves the mode to the config file |
| `src/server.rs`, `src/dispatcher.rs` | the setup server: `GET /healthz`, `GET /ws` (`bebop.proto`, one message per binary frame), the status page `GET /` |
| `src/config/` | the `agent.toml` schema and its defaults |

The agent does no containers, OTA or Bluetooth; the app talks to the robot
runtime (`firmware/bebop-linux`) directly. Protocol, network modes and the
button: [`../../docs/setup-protocol.md`](../../docs/setup-protocol.md).

## Building

```sh
cargo build --release -p bebop-agent
```

`protoc` must be on `PATH` (`bebop-proto` compiles its schemas at build
time). Build natively on arm64 (the robot or an arm64 dev box), or take the
CI artifact `bebop-agent-aarch64` (built on `ubuntu-22.04-arm`).

## Running (dev)

From `jetson-agent/`:

```sh
BEBOP_AGENT_CONFIG=./deploy/examples/agent.toml \
RUST_LOG=info,bebop_agent=debug \
cargo run -p bebop-agent
```

It needs NetworkManager (`nmcli`); raising the hotspot needs root.

## Configuration

`/etc/bebop/agent.toml`, or the file named by `BEBOP_AGENT_CONFIG`. Every key
is optional; the defaults are in `src/config/mod.rs`, and
[`../deploy/examples/agent.toml`](../deploy/examples/agent.toml) lists each
one, commented out. Top level: `robot_name` (default: `/etc/hostname`) and
`state_dir` (`/var/lib/bebop`). Under `[network]`: `mode` (`ap` / `client`,
owned by the button), `ap_ssid`, `ap_password`, `ap_band` (`"2.4"` / `"5"`),
`setup_bind_addr` (`0.0.0.0:9091`), and the button keys `button_enabled`,
`button_source`, `control_device`, `button_hold_secs`, plus `button_chip`,
`button_line`, `button_active_low` and `button_bias` for the `gpio` source.

## Packaging and install

[`../deploy/`](../deploy/): the systemd unit, `install.sh` / `uninstall.sh`
and the example config. `just deploy <host>` from the repo root copies a
build to a robot and runs `install.sh`; `scripts/install-jetson.sh` installs
it from CI (or, with `--local`, from the checkout) together with
`bebop-linux`.
