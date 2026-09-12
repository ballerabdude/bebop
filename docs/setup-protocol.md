# Setup Protocol

How the companion app (`bebop-app`) provisions a robot via `bebop-agent`'s
setup server, and how the physical mode button works.

## Network modes

The robot has exactly two network modes, switched **only** by the physical
button press. There is no automatic fallback.

| Mode     | Name           | Behaviour                                                           |
|----------|----------------|---------------------------------------------------------------------|
| `ap`     | Hosted Network | Hosts the `Bebop-XXXX` hotspot continuously (boot default).         |
| `client` | Known Network  | Joins a saved Wi-Fi network; never hosts.                           |

Because the Wi-Fi radio is single-ended, switching modes drops the previous
link. A fresh robot boots into Hosted Network so it is reachable at
`192.168.42.1:9091` out of the box.

## Mode button

A momentary switch from a GPIO header pin to 3.3 V. A **press** (debounced)
toggles the mode and persists it to `/etc/bebop/agent.toml`. Set
`button_hold_secs > 0` to require a hold instead of a press.

- Default pin: Orin Nano header **pin 32** = `gpiochip0` line **41**
  (`GPIO07`). It has an internal pull-down in the pinmux, so wire the
  switch from the pin to **3.3 V** (header pins 1/17) — no resistor needed
  — and leave `button_active_low = false`. Avoid pins 7 and 15 (IMU
  INT/RST), and pin 29 (no internal pull; it floats).
- The agent requests the line via the pure-Rust `gpiocdev` crate (GPIO
  uAPI v2). The Jetson pinmux fixes each pin's idle bias (a runtime bias
  request is effectively cosmetic), so pick the wiring to match the pin's
  idle level. (For a button-to-GND, add an external pull-up and set
  `button_active_low = true`.)

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

The app's Wi-Fi screen shows a "saved — press the button to switch" confirmation
when provisioning while Hosted.

## Editing the Hosted Network

The app's dashboard exposes the Hosted Network settings (SSID, password,
band 2.4/5 GHz) via `SetNetworkConfig`. `mode` is **read-only** in the API.
Changing the SSID/password/band while hosting recreates and re-raises the
hotspot, which briefly drops connected devices — reconnect with the new
credentials. An empty `ap_password` keeps the existing passphrase, and
responses never include it.

## Bench-verifying the button

The Jetson pinmux fixes each pin's idle level, and the runtime bias request
is effectively cosmetic — so confirm the wiring matches the idle level
before relying on the button.

```sh
# Idle level on header pin 32 (gpiochip0 line 41). With the default
# active-high wiring (switch to 3.3 V) this should read 0.
gpioget gpiochip0 105

# Watch edges while pressing the switch. Stop the agent first so it
# releases the line; expect a rising edge on press for active-high.
sudo systemctl stop bebop-agent
gpiomon gpiochip0 105          # Ctrl-C to exit; press the button
sudo systemctl start bebop-agent

# End-to-end: press the button, then watch the toggle + hotspot come up.
journalctl -u bebop-agent -f
# ... press the button ...
# expect: "button pressed" -> "mode button toggled network mode mode=ap"
#         -> "Hosted Network raised ssid=Bebop-XXXX"
```

To go back to Known Network, press the button again. If the hotspot is up,
join `Bebop-XXXX` on a phone/PC and open `http://192.168.42.1:9091`.

## Regenerating bindings

The Rust types are generated at build time by `bebop-proto/build.rs` from
`jetson-agent/bebop-proto/proto/bebop.proto`. The TypeScript bindings are
generated (and committed) with:

```sh
cd bebop-app && npm run gen-proto
```

Keep the `.proto` as the single source of truth for both sides.
