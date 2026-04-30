# BLE Protocol

The mobile app talks to `bebop-agent` over a single custom GATT service.
This document specifies what that looks like on the wire.

The same protocol is implemented in three places — keep them in sync:

| Side                    | UUIDs                                          | Framing                                          |
|-------------------------|------------------------------------------------|--------------------------------------------------|
| Agent (Rust, BlueZ)     | `jetson-agent/bebop-agent/src/ble/uuids.rs`    | `jetson-agent/bebop-agent/src/ble/framing.rs`    |
| Mobile, native (Tauri)  | `bebop-app/src-tauri/src/ble/`                 | `bebop-app/src-tauri/src/ble/framing.rs`         |
| Mobile, Web Bluetooth   | `bebop-app/src/ble/protocol.ts`                | `bebop-app/src/ble/protocol.ts`                  |

## Service

| Role           | UUID                                     |
|----------------|------------------------------------------|
| Primary service | `b3b0b000-0b3b-4f9b-9b3b-b3b0b3b0b3b0`   |

## Characteristics

| Name       | UUID                                      | Properties              | Direction            |
|------------|-------------------------------------------|-------------------------|----------------------|
| `request`  | `b3b0b001-0b3b-4f9b-9b3b-b3b0b3b0b3b0`    | Write, WriteWithoutResp | Mobile → Agent       |
| `response` | `b3b0b002-0b3b-4f9b-9b3b-b3b0b3b0b3b0`    | Notify                  | Agent → Mobile       |
| `status`   | `b3b0b003-0b3b-4f9b-9b3b-b3b0b3b0b3b0`    | Read, Notify            | Agent → Mobile       |

## Framing

Every write or notification carries a 4-byte header:

```
 0        1        2-3                4..N
 version  flags    payload_len (BE)   payload
```

- `version` = `0x01` (current protocol)
- `flags`:
  - bit 0 (`FRAGMENT`): more fragments follow
  - bit 1 (`FINAL`): last (or only) fragment of this logical message
- `payload_len`: number of payload bytes in *this* frame

Large `ClientRequest`/`AgentResponse` payloads (e.g. Wi-Fi scan results) are
split into frames that fit within the negotiated ATT MTU, with `FRAGMENT`
set on all but the last frame.

## Logical payload

After reassembly, each payload is a `prost`-encoded
[`bebop.v1.ClientRequest`](../jetson-agent/bebop-proto/proto/bebop.proto)
(mobile → agent) or
[`bebop.v1.AgentResponse`](../jetson-agent/bebop-proto/proto/bebop.proto)
(agent → mobile). See the `.proto` for the full message list.

## Authentication (planned)

The wire format already reserves the door for this — there is no
`AuthRequest` in `bebop.proto` yet, and `BleConfig::require_pairing`
defaults to `true` in the agent config but is currently a no-op. The
intended flow:

1. Mobile connects and enables notifications on `response`.
2. First write on `request` must be `GetDeviceInfoRequest` — allowed
   without auth so the phone can show "you're talking to robot X".
3. The agent's `DeviceInfo` response will include a per-connection
   `challenge` (to be added to the proto).
4. Mobile sends back an `AuthRequest { hmac_sha256(pairing_code, challenge) }`
   (also to be added).
5. On success, the agent unlocks the remaining RPCs. On failure, the link
   is force-disconnected.

Until that lands, BLE should be considered trusted-proximity only — treat
it like "your phone is in my hand" and avoid exposing anything sensitive.

## Request ID

Every `ClientRequest` has a 32-bit `request_id`. The agent echoes it in the
matching `AgentResponse`, so the mobile side can correlate async replies.
