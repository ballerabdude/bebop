# Bebop Docs

Cross-cutting documentation for the Bebop platform. Per-component READMEs
(Cargo crate docs, mobile-app dev notes, deploy scripts, etc.) live next
to the code they describe.

| Doc                                  | What it covers                                                           |
|--------------------------------------|--------------------------------------------------------------------------|
| [`onboarding.md`](onboarding.md)     | Dev-machine setup, first build, deploying to a Jetson, mobile app dev.   |
| [`architecture.md`](architecture.md) | The components (`jetson-agent/`, `bebop-app/`, `firmware/`), boot sequence, and trust boundaries. |
| [`setup-protocol.md`](setup-protocol.md) | The `bebop-agent` setup WebSocket, Hosted Network provisioning flow, and network modes. |
| [`navd.md`](navd.md)                 | Depth-camera obstacle avoidance: dual Gemini 335Lg → BEV occupancy → goal-conditioned planner → learned student (design + phased spec). |
| [`place-router.md`](place-router.md) | The System 1 visual router and its first skill (place memory): decide when a view is worth remembering, then execute it. |

If you're new, read them in the order above.
