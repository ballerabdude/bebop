# Setup protocol

How the operator app (`bebop-app`) provisions a robot through
`bebop-agent`'s setup server, and how the mode button switches networks.
Code: `jetson-agent/bebop-agent/src/`; schema:
`jetson-agent/bebop-proto/proto/bebop.proto`.

## Network modes

The robot has exactly two network modes (`network.mode` in
`/etc/bebop/agent.toml`), switched **only** by the physical mode button.
There is no automatic fallback, and the app cannot change the mode.

| Mode | Name | Behaviour |
|---|---|---|
| `ap` | Hosted Network | Hosts the setup hotspot continuously (boot default). |
| `client` | Known Network | Joins a saved Wi-Fi network; never hosts. |

The hotspot SSID defaults to `Bebop-<first 6 chars of /etc/machine-id>`
and its WPA2 passphrase to `bebopbebop`. There is one Wi-Fi radio, so
switching modes drops the previous link. A fresh robot boots into Hosted
Network and is reachable at `192.168.42.1:9091`.

## Mode button

Holding the button for `network.button_hold_secs` (default 5) toggles
`ap` ⇄ `client`, saves the mode to `/etc/bebop/agent.toml` and reconciles
the hotspot right away; `button_hold_secs = 0` toggles on the press instead.
`network.button_source` picks where the button lives:

- **`serial`** (default; the Thor has no usable GPIO header). The switch
  runs from Teensy 4.1 pin 2 to GND (internal pull-up, no resistor). The
  `teensy_bridge` firmware debounces it (30 ms) and sends press and release
  frames on its third USB CDC port, which udev links to `/dev/bebop-control`
  (`install-jetson.sh --setup-imu`; `network.control_device`). Frame
  (`firmware/bebop-locomotion/include/ControlProtocol.h`, 14 bytes,
  little-endian): magic `0xBE 0xC0`, type (`0x01` = button), seq, `t_us`
  (u32), payload `(state << 8) | button_id` (u32; state 1 pressed /
  0 released, button 0 = network mode), CRC-16/CCITT-FALSE over bytes 0-11.
  The agent applies the hold time; it retries opening the port with backoff
  (0.5 s up to 5 s) while the Teensy is absent.
- **`gpio`**, the Orin Nano path: a switch on a Jetson header line
  (`button_chip`, `button_line`, `button_bias`, `button_active_low`), polled
  every 100 ms with a two-sample debounce through the `gpiocdev` crate. The
  defaults are Orin Nano header pin 32 (`gpiochip0` line 41), whose pinmux
  has a pull-down: switch to 3.3 V, active-high, no resistor. Jetson header
  bias comes from the pinmux, not from the line request, so check a line's
  idle level with `gpioinfo`/`gpioget` before relying on it.

### Bench check (serial)

```sh
cat /dev/bebop-imu-debug            # Teensy diagnostics: the 1 Hz [BRIDGE] line's btn= counts presses
journalctl -u bebop-agent -f        # then hold the button for 5 s
#   button pressed (serial); hold to toggle
#   button long-press (serial) detected
#   mode button toggled network mode mode=ap
#   Hosted Network raised ssid=Bebop-...
```

If the port is missing the agent logs `control: open failed; retrying after
backoff`, then a single hint to check that the Teensy runs `USB_TRIPLE_SERIAL`
firmware and that udev created `/dev/bebop-control`. Once the hotspot is up,
join it and open `http://192.168.42.1:9091` for the status page; hold the
button again to go back to Known Network.

## Transport

The setup server binds `network.setup_bind_addr` (default `0.0.0.0:9091`)
and is reachable two ways:

- **Over the Hosted Network**: join the hotspot (WPA2) and connect to the
  gateway, `192.168.42.1:9091`.
- **Over the LAN**: in Known Network mode, use the robot's address
  (`bebop.local` through mDNS on most networks).

| Path | Purpose |
|---|---|
| `GET /healthz` | Returns `ok`. The app probes it before setup and on the Robots page. |
| `GET /ws` | WebSocket upgrade; one `ClientRequest` per binary frame. |
| `GET /` | Status page: robot name, mode, Wi-Fi and hotspot state. |

All three allow any origin (CORS `*`); there is no authentication.

## Framing

Each binary WebSocket frame carries exactly one `bebop.v1.ClientRequest`
(to the agent) or `bebop.v1.AgentResponse` (to the app), encoded with
`prost` on the agent and `@bufbuild/protobuf` in the app. There is no length
prefix: the frame boundary is the message boundary. The exchange is strictly
request/response, nothing is pushed, and text frames are ignored. A frame
that does not decode gets an `AgentResponse` with `request_id = 0`, status
`RESPONSE_STATUS_ERROR` and the decode error in `message`. The agent answers
only `OK` or `ERROR`.

## Messages

| Request | Response | Notes |
|---|---|---|
| `GetDeviceInfo` | `DeviceInfo` | serial (`/proc/device-tree/serial-number`), model (`bebop-v1`), agent version, JetPack (first line of `/etc/nv_tegra_release`), `hostname` (carries the robot name) |
| `ScanWifi` | `WifiScanResult` | nearby networks with signal, security and whether a profile is saved |
| `SetWifiCredentials` | `WifiStatus` | see below |
| `GetWifiStatus` | `WifiStatus` | the current client link |
| `GetRobotConfig` / `SetRobotConfig` | `RobotConfig` | only `robot_name` is stored; `owner_id`, `timezone` and `extra` are not |
| `GetNetworkConfig` / `SetNetworkConfig` | `NetworkConfig` | Hosted Network settings, see below |

## Provisioning Wi-Fi

`SetWifiCredentials` depends on the mode:

- **Known Network (`client`)**: the robot joins at once.
- **Hosted Network (`ap`)**: the profile is **saved, not applied**, so the
  hotspot (and the app's session) stays up. The reply says to long-press the
  button; the robot joins the saved network when the button switches it to
  Known Network.

The app's setup wizard (connect → Wi-Fi → identity) shows a "Wi-Fi saved"
panel in that case.

## Editing the Hosted Network

Settings › Network in the app edits `ap_ssid`, `ap_password` and `ap_band`
(`"2.4"` or `"5"`) through `SetNetworkConfig`. `mode` is ignored on write
(the button owns it) and `ap_address` is read-only (`host:port` while
hosting). An empty field keeps the current value, and responses never include
the password. Changing any of them while hosting re-raises the hotspot, which
briefly drops connected devices; reconnect with the new credentials.

## Regenerating bindings

`bebop-proto/build.rs` generates the Rust types at build time from every
`.proto` in `jetson-agent/bebop-proto/proto/`. The app's TypeScript is
generated with buf and committed (`src/proto/bebop_pb.ts`,
`src/proto/bebop_runtime_pb.ts`):

```sh
cd bebop-app && npm run gen-proto
```

The `.proto` is the single source of truth for both sides.
