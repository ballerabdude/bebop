# Setup Protocol

How the companion app (`bebop-app`) provisions a robot via `bebop-agent`'s
setup server.

## Transport

The agent serves a binary WebSocket on the configured setup address
(`network.setup_bind_addr`, default `0.0.0.0:9091`). The same server is
reachable two ways:

- **Over the robot's setup SoftAP** during first-time provisioning. Join the
  `Bebop-XXXX` Wi-Fi network (WPA2, default passphrase `bebopbebop`), then
  connect to the gateway (`192.168.42.1:9091` by default).
- **Over the LAN** once the robot is on Wi-Fi, using its normal address.

Endpoints:

| Path        | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `/healthz`  | Liveness probe; returns `ok`. Used by the app's pre-flight.    |
| `/ws`       | WebSocket upgrade; one `ClientRequest` per binary frame.       |
| `/`         | Minimal status page (robot name, Wi-Fi/hotspot state).         |

There is no BLE/GATT surface anymore — tests confirmed the Jetson's Wi-Fi
module supports AP mode, and the SoftAP + setup server replaces Bluetooth.

## Framing

Each WebSocket binary frame carries exactly one `bebop.v1.ClientRequest`
(agent-bound) or `bebop.v1.AgentResponse` (app-bound), encoded with `prost`
on the agent side and `@bufbuild/protobuf` on the app side. There is no
additional length prefix — the WebSocket frame boundary is the message
boundary — so the ATT-MTU fragmentation scheme from the old BLE transport is
gone.

## Messages

`bebop.proto` defines the provisioning surface:

- `GetDeviceInfo` / `DeviceInfo` — serial, model, agent version, JetPack,
  hostname.
- `ScanWifi` / `WifiScanResult` — nearby networks.
- `SetWifiCredentials` / `WifiStatus` — join a network.
- `GetWifiStatus` / `WifiStatus` — current link.
- `GetRobotConfig` / `SetRobotConfig` / `RobotConfig` — robot name.
- `GetNetworkConfig` / `SetNetworkConfig` / `NetworkConfig` — provisioning
  mode (`auto` / `client` / `ap`) plus the SoftAP SSID + setup address.

## Joining Wi-Fi

`SetWifiCredentials` is fire-and-forget by design. The agent replies
immediately, then drops the SoftAP and switches the Wi-Fi radio to client
mode. The app is reachable over that SoftAP, so the reply must be sent
before the radio switches. After the robot joins, the app loses the setup
link and the user reconnects to their own network, then points the app at
the robot's new address.

While a join is in flight the agent suppresses its SoftAP supervisor so the
hotspot isn't re-raised mid-connection.

## Networking modes

`[network] mode` in `/etc/bebop/agent.toml`:

| Mode     | Behaviour                                                        |
|----------|------------------------------------------------------------------|
| `auto`   | Join a known network; raise the SoftAP if none connects within `ap_auto_after_secs` (default 25 s). |
| `client` | Only ever act as a Wi-Fi client (never raise the SoftAP).        |
| `ap`     | Always host the setup SoftAP.                                    |

## Regenerating bindings

The Rust types are generated at build time by `bebop-proto/build.rs` from
`jetson-agent/bebop-proto/proto/bebop.proto`. The TypeScript bindings are
generated (and committed) with:

```sh
cd bebop-app && npm run gen-proto
```

Keep the `.proto` as the single source of truth for both sides.
