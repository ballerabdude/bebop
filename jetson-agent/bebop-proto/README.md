# `bebop-proto`

The protobuf wire types for both robot APIs, in one crate. Rust types are
generated with prost-build at build time; the crate depends on `prost` only.

| Proto | Package → Rust module | Served by | Transport | Clients |
|---|---|---|---|---|
| `proto/bebop.proto` | `bebop.v1` → `bebop_proto::v1` | `bebop-agent` setup server | binary WebSocket on `:9091` | `bebop-app` (setup wizard, Settings › Network): device info, Wi-Fi scan / credentials / status, robot and network config |
| `proto/bebop_runtime.proto` | `bebop.runtime.v1` → `bebop_proto::runtime::v1` | `bebop-linux` | binary WebSocket, `ws://<robot>:9090/ws` | `bebop-app`; bebop-vision's `RobotClient` (the recorder and the voice agent) |

`bebop_proto::Message` re-exports `prost::Message`. The setup protocol is in
[`docs/setup-protocol.md`](../../docs/setup-protocol.md), the runtime API in
[`firmware/bebop-linux/README.md`](../../firmware/bebop-linux/README.md#runtime-api).
`bebop-linux` uses the crate as a path dependency.

## Bindings

| Consumer | Generated file | How |
|---|---|---|
| Rust: `bebop-agent`, `bebop-linux` | `$OUT_DIR/bebop.v1.rs`, `$OUT_DIR/bebop.runtime.v1.rs` (not committed) | `build.rs` runs prost-build over every `proto/*.proto` at build time; needs `protoc` (apt `protobuf-compiler`) |
| TypeScript: `bebop-app` | `bebop-app/src/proto/bebop_pb.ts`, `bebop_runtime_pb.ts` (committed) | `cd bebop-app && npm run gen-proto`: `buf generate ../jetson-agent/bebop-proto/proto` with `protoc-gen-es` (`bebop-app/buf.gen.yaml`) |
| Python: `bebop-vision` | `bebop-vision/bebop_vision/proto/bebop/runtime/v1/bebop_runtime_pb2.py` (committed; the runtime proto only) | `protoc --python_out` on a copy staged at `bebop/runtime/v1/` |

The Python file, from the repo root, with libprotoc 3.21.12 (on 2026-10-24
this reproduced the committed file byte for byte; the TypeScript generation
did the same for both files):

```sh
stage=$(mktemp -d)
mkdir -p "$stage/bebop/runtime/v1"
cp jetson-agent/bebop-proto/proto/bebop_runtime.proto "$stage/bebop/runtime/v1/"
protoc -I "$stage" --python_out=bebop-vision/bebop_vision/proto \
    bebop/runtime/v1/bebop_runtime.proto
```

Staging names the descriptor `bebop/runtime/v1/bebop_runtime.proto`, matching
its package path, as the committed file records; compiling from the flat
`proto/` dir would not. `sim/bebop_training/proto/bebop_runtime_pb2.py` is an
older copy (2026-07-13) that is not regenerated.

## Changing a proto

- Never renumber or reuse a field number. Append new messages at the end of
  the file: a message inserted mid-file shifts the descriptor index of every
  later one and churns the generated TypeScript.
- A removed field, oneof case or enum value keeps its number and name as
  `reserved`, with a dated comment. That keeps a future field off those
  numbers, so old clients and old recordings never decode a removed field as a
  new one: bebop-vision's recorder stores the raw `ServerRuntimeMessage`
  frames on `/runtime` and `/runtime_events` in every `navd_session_*.mcap`.
- Then rebuild (Rust regenerates itself), run `npm run gen-proto`, regenerate
  the Python file, and commit the regenerated TypeScript and Python files.

### Reserved in `bebop_runtime.proto`

`bebop.proto` has none.

| Message | Numbers | Names | Removed |
|---|---|---|---|
| `ClientRuntimeMessage` | 11, 12 | `set_policy_dry_run`, `set_policy_capture` | 11 on 2026-10-24 with the legged policy runner; 12 on 2026-06-12, when its capture became always-on |
| `ClientRuntimeMessage` | 19–22 | `set_camera_pose`, `subscribe_nav`, `unsubscribe_nav`, `set_navigation_goal` | 2026-10-23 |
| `ClientRuntimeMessage` | 27 | `set_model_purpose` | 2026-10-24 |
| `ServerRuntimeMessage` | 7 | `bus_status` | 2026-10-24 |
| `ServerRuntimeMessage` | 8, 9 | `nav_mask`, `nav_goal` | 2026-10-23 |
| `Mode` (enum) | 3 | `MODE_RUN_POLICY` | 2026-10-24 |
| `TelemetryFrame`, `Snapshot` | 14 | `policy_io` | 2026-10-24 |
| `TelemetryFrame`, `Snapshot` | 17, 18 | `camera`, `nav` | 2026-10-23 |
| `ImuStats` | 24 | `heading_accuracy_rad` | 2026-10-24 |
| `VisionState` | 6 | `mode` | 2026-10-24 |
| `VoiceState` | 6 | `model` | 2026-10-24 |
| `ModelState` | 6 | `selection` | 2026-10-24 |
| `ModelCatalogEntry` | 6, 7, 10 | `files`, `revision`, `path` | 2026-10-24 |

The messages `BusStatusUpdate`, `PolicyIoStats`, `SetPolicyDryRun`,
`SetModelPurpose` and `PurposeSelection` went with them on 2026-10-24; `Mode`
is now `MODE_UNSPECIFIED = 0`, `MODE_IDLE = 1`, `MODE_DIAL_IN = 2`. On the
wire, a client that still sends tag 11 or 27 gets
`Error: empty or unsupported client message`, and `SetMode(3)` gets
`unknown mode value 3`.
