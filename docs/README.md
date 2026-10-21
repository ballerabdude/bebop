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
| [`bev-student.md`](bev-student.md)   | Learned semantic BEV: SAM 3.1 teacher, DINOv3 TensorRT student, training and the model-only live service. |
| [`bev-experiments.md`](bev-experiments.md) | 2026-10-19 label audit and model-improvement experiments: clean labels, hazard metrics, leaderboard, next steps. |
| [`bev-matched-grid.md`](bev-matched-grid.md) | Deployed BEV: 160×320 grid over 4 m × 4 m, learned chassis footprint cull, rail-edge hazard margin, TensorRT deploy numbers. |
| [`bev-segformer.md`](bev-segformer.md) | SegFormer-B5 / higher-resolution BEV queue (stopped in favour of the matched-grid run). |
| [`place-router.md`](place-router.md) | Place memory: lofi scout capture into a pending queue, then operator batch import with optional model name suggestions (the legacy every-tick router is documented there too). |
| [`slam.md`](slam.md)                 | Live RGB-D visual localization POC, tracking status, and visual-heading turn diagnostic. |
| [`internal-map.md`](internal-map.md) | Per-place local structure submaps, the anchor frame, and the place graph of remembered routes, with honest status. |
| [`nav.md`](nav.md)                   | Depth stop-cone + bearing waypoint driver (`--nav`, `:9097`): the fast obstacle-avoiding layer under vision-first navigation. |

If you're new, read them in the order above.
