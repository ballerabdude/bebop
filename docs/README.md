# Bebop docs

Cross-cutting docs for the Bebop robot. Each component's README sits next to
its code (listed below).

| Doc | What it covers |
|---|---|
| [`onboarding.md`](onboarding.md) | Workstation setup, builds and tests, installing the stack on the robot, running the app. |
| [`architecture.md`](architecture.md) | Hardware, the on-robot services and their ports, boot sequence, trust boundaries, the Hosted Network, builds. |
| [`setup-protocol.md`](setup-protocol.md) | `bebop-agent`'s setup WebSocket (`:9091`), Wi-Fi provisioning, the two network modes and the mode button. |
| [`dev-https.md`](dev-https.md) | A secure context for the app's microphone during development: localhost, Chrome's insecure-origin flag, the `Caddyfile.dev` TLS proxy. |
| [`data-collection.md`](data-collection.md) | Recording navd drive sessions, pulling and extracting them (`just pull-data`), the session ledger. |
| [`depth-tuning.md`](depth-tuning.md) | Gemini 335Lg depth presets and host-side filters, measured 2026-10-22. |
| [`path-model.md`](path-model.md) | Proposal (2026-10-22/23): a learned path model and guarded follower to replace the stripped navigation stack. The model, the follower and the safety envelope are not implemented. |
| [`places.md`](places.md) | Place memory (`:9095`): the 1 Hz DINOv3 scout, the pending queue and operator import, `/whereami`, the place graph. |
| [`voice.md`](voice.md) | `bebop-voice` (`:9093`): the cascade and omni engines, the agent WebSocket protocol, tools and motion limits. |
| [`models.md`](models.md) | The model catalog, weight downloads (`bebop-model-download@<id>`), TensorRT Edge-LLM as the only generative runtime, and `bebop-models` (`:9094`): the GPU budget and its registry. |
| [`hardware/battery-wiring.html`](hardware/battery-wiring.html) | The 13s2p 21700 battery pack: cell layout, BMS and charger wiring, thermistor placement. |

New here: read `onboarding.md`, then `architecture.md`.

Component READMEs:

| README | Component |
|---|---|
| [`../firmware/bebop-linux/README.md`](../firmware/bebop-linux/README.md) | `bebop-linux`, the robot runtime (`:9090`), and the `sysid` actuator bench. |
| [`../firmware/bebop-locomotion/README.md`](../firmware/bebop-locomotion/README.md) | The Teensy 4.1 firmware: `teensy_bridge` (BNO085 frames and the mode button over USB serial) and the legacy envs. |
| [`../jetson-agent/README.md`](../jetson-agent/README.md) | `bebop-agent`, `bebop-proto` and the agent's deploy scripts. |
| [`../jetson-agent/bebop-proto/README.md`](../jetson-agent/bebop-proto/README.md) | Both protobuf schemas, their generated bindings and the reserved tags. |
| [`../bebop-vision/README.md`](../bebop-vision/README.md) | The recorder, operator video, place memory, voice and model services, and the labeling and dataset tools. |
| [`../bebop-app/README.md`](../bebop-app/README.md) | The operator app: pages, services and ports, development. |
| [`../AGENTS.md`](../AGENTS.md) | Working notes: robot access, the deploy process used on the bench, gotchas. |

The retired design docs (navd, the BEV student experiments, live SLAM, the
legacy place router and internal map) were removed on 2026-10-24; they live
in git history (`git log --diff-filter=D --name-only -- docs/`).
