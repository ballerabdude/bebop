# Setup Protocol

How the companion app (`bebop-app`) provisions a robot via `bebop-agent`'s
setup server, and how the physical mode button works.

## Network modes

The robot has exactly two network modes, switched **only** by the physical
button long-press. There is no automatic fallback.

| Mode     | Name           | Behaviour                                                           |
|----------|----------------|---------------------------------------------------------------------|
| `ap`     | Hosted Network | Hosts the `Bebop-XXXX` hotspot continuously (boot default).         |
| `client` | Known Network  | Joins a saved Wi-Fi network; never hosts.                           |

Because the Wi-Fi radio is single-ended, switching modes drops the previous
link. A fresh robot boots into Hosted Network so it is reachable at
`192.168.42.1:9091` out of the box.

## Mode button

A momentary switch from a GPIO header pin to GND. A **long press** (default
5 s, fired on release) toggles the mode and persists it to
`/etc/bebop/agent.toml`. Short taps are ignored.

- Default pin: Orin Nano header **pin 32** = `gpiochip0` line **41**
  (`GPIO07`). This pin idles high, so a switch to GND reads active-low.
  Avoid pins 7 and 15 (IMU INT/RST). (Pin 29 idles low on this board.)
- The agent requests the line via the pure-Rust `gpiocdev` crate (GPIO
  uAPI v2). Note the Jetson pinmux fixes each pin's bias, so the runtime
  bias request is effectively cosmetic — pick a pin that idles the right way.
- Wire the switch between the pin and any GND pin.

## Transport

The agent serves a binary WebSocket on the configured setup address
(`network.setup_bind_addr`, default `0.0.0.0:9091`). The same server is
reachable two ways:

- **Over the Hosted Network** — join the `Bebop-XXXX` hotspot (WPA2), then
  connect to the gateway (`192.168.42.1:9091` by default).
- **Over the LAN** — once the robot is on Wi-Fi in Known Network mode, use
  its normal address (`bebop.local` resolves via mDNS/avahi on most
  networks).

Endpoints:

| Path        | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `/healthz`  | Liveness probe; returns `ok`. Used by the app's pre-flight.    |
| `/ws`       | WebSocket upgrade; one `ClientRequest` per binary frame.       |
| `/`         | Status page (robot name, mode, Wi-Fi/Hosted state).            |

## Framing

Each WebSocket binary frame carries exactly one `bebop.v1.ClientRequest`
(agent-bound) or `bebop.v1.AgentResponse` (app-bound), encoded with `prost`
on the agent side and `@bufbuild/protobuf` on the app side. There is no
extra length prefix — the WebSocket frame boundary is the message boundary.

## Messages

`bebop.proto` defines the provisioning surface:

- `GetDeviceInfo` / `DeviceInfo` — serial, model, agent version, JetPack,
  hostname.
- `ScanWifi` / `WifiScanResult` — nearby networks.
- `SetWifiCredentials` / `WifiStatus` — join a network.
- `GetWifiStatus` / `WifiStatus` — current link.
- `GetRobotConfig` / `SetRobotConfig` / `RobotConfig` — robot name.
- `GetNetworkConfig` / `SetNetworkConfig` / `NetworkConfig` — the Hosted
  Network settings (`ap_ssid`, `ap_password`, `ap_band`) plus the read-only
  `mode` and `ap_address`.

## Provisioning Wi-Fi

`SetWifiCredentials` behaves differently per mode:

- **Known Network (`client`)** — the robot joins immediately.
- **Hosted Network (`ap`)** — the profile is **saved but not applied**; the
  hotspot stays up so the current session isn't killed. The robot joins the
  saved network only when the button switches it to Known Network, at which
  point the profile autoconnects.

The app's Wi-Fi screen shows a "saved — long-press to switch" confirmation
when provisioning while Hosted.

## Editing the Hosted Network

The app's dashboard exposes the Hosted Network settings (SSID, password,
band 2.4/5 GHz) via `SetNetworkConfig`. `mode` is **read-only** in the API.
Changing the SSID/password/band while hosting recreates and re-raises the
hotspot, which briefly drops connected devices — reconnect with the new
credentials. An empty `ap_password` keeps the existing passphrase, and
responses never include it.

## Regenerating bindings

The Rust types are generated at build time by `bebop-proto/build.rs` from
`jetson-agent/bebop-proto/proto/bebop.proto`. The TypeScript bindings are
generated (and committed) with:

```sh
cd bebop-app && npm run gen-proto
```

Keep the `.proto` as the single source of truth for both sides.
