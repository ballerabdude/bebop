# Bebop Docs

Cross-cutting documentation for the Bebop platform. Per-component READMEs
(Cargo crate docs, mobile-app dev notes, deploy scripts, etc.) live next
to the code they describe.

| Doc                                  | What it covers                                                           |
|--------------------------------------|--------------------------------------------------------------------------|
| [`onboarding.md`](onboarding.md)     | Dev-machine setup, first build, deploying to a Jetson, mobile app dev.   |
| [`architecture.md`](architecture.md) | The components (`jetson-agent/`, `bebop-app/`, robot-app container), boot sequence, and trust boundaries. |
| [`ble-protocol.md`](ble-protocol.md) | GATT service / characteristic UUIDs, length-prefixed framing, the protobuf envelope, and the (planned) auth handshake. |
| [`ota-flow.md`](ota-flow.md)         | How the agent polls the manifest server, swaps the robot-app container, and how to roll back / channel-promote. |

If you're new, read them in the order above.
