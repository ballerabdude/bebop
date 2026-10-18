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
| [`place-router.md`](place-router.md) | Place memory: lofi scout capture into a pending queue, then operator batch import with optional model name suggestions (the legacy every-tick router is documented there too). |
| [`slam.md`](slam.md)                 | Live RGB-D visual localization POC, tracking status, and visual-heading turn diagnostic. |
| [`internal-map.md`](internal-map.md) | Per-place local structure submaps, the anchor frame, and the place graph of remembered routes, with honest status. |

If you're new, read them in the order above.
