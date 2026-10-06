# Models: the on-robot supervisor (`bebop-models`)

The Thor has **one GPU** (~122 GB unified) and several model consumers — voice
(Qwen3-Omni, the cascade), vision (navd / SAM / DINOv3), navigation, and a future
VLA model. The big models can't co-reside, so we run a **supervisor** that owns
the GPU budget and lifecycle and lets consumers time-share it.

## Principle: a control plane, not one inference process

The models are heterogeneous and one class is latency/hardware-coupled, so a
single process serving everything would be worse:

| Backend | Runtime | Managed by |
|---|---|---|
| Edge-LLM `tensorrt-edgellm-serve` | C++ / TensorRT | supervisor (child procs) |
| vLLM-Omni (Qwen3-Omni) | Python | supervisor (Docker container) |
| future VLA (pi0.5 / Alpamayo / Cosmos3-Edge) | Edge-LLM action, or a container | supervisor |
| navd / SAM / DINOv3 | onnxruntime / torch, **camera-coupled** | **stays inside `bebop-vision`**, registered as `tracked` |

Camera-coupled real-time vision inference stays in-process with the frames (no
IPC hop); the supervisor only counts it against the budget.

## Service

`bebop-vision/bebop_vision/models_supervisor.py` (unit:
`deploy/systemd/bebop-models.service`, port **:9094**).

```
GET  /healthz            budget, resident, MemAvailable
GET  /models             registry + live state
POST /models/{id}/load   start (evict ready residents until it fits the budget)
POST /models/{id}/unload stop
POST /models/{id}/sleep  ask an omni server to release the GPU (level 2)
POST /models/{id}/wake   restore a slept backend
```

`sleep`/`wake` proxy vLLM-Omni's `/v1/omni/sleep|wakeup`. Measured on Thor:
level 2 frees **~84 GB in ~1 s**; wake reloads and returns `WARM` in **~4 min**.
Requires the server started with `--enable-sleep-mode` (the container paths add
it). Use level 2 — level 1 only freed ~18 GB on unified memory and its wake
hung. See [`vllm-omni.md`](vllm-omni.md).

Registry: `bebop-vision/config/models_supervisor.yaml` (falls back to built-in
defaults). A backend is `container` (Docker image), `external` (already-running
OpenAI-compatible URL), or `tracked` (runs elsewhere; counted only). Each has a
`footprint_gb` used by the budget (`GPU_BUDGET_GB`, default 108 GB), so `load`
evicts residents when needed.

Install / enable:

```bash
sudo ./scripts/install-jetson.sh --setup-models      # installs + enables the unit
sudo ./scripts/install-jetson.sh --pull-vllm-omni    # pre-pull the vLLM-Omni image
```

## GPU sharing policy

1. **One big model resident at a time.** Loading a backend evicts ready backends
   until `resident + new ≤ budget`.
2. **Sleep, don't stop.** An idle backend can be `sleep`-ed (frees the GPU in
   ~1 s, wake ~4 min) instead of `unload`-ed (full cold start). Idle-sleep in the
   supervisor is the next step; today it's driven by the API/app.
3. **Small models may co-reside** (e.g. navd + a small VLA) when footprints fit.

## Migration path

- **Today:** the supervisor owns the vLLM-Omni container, and `bebop-voice` can
  delegate to it — set **Omni source (supervisor)** on the Voice page
  (`omni_supervisor_url`, e.g. `http://127.0.0.1:9094`) and the voice service
  `load`s `omni-vllm` through the supervisor (`SupervisedServer`) instead of
  managing lifecycle itself. The app's Models page (**GPU budget** card) shows
  what is resident and loads/unloads backends.
- **Next:** have `bebop-vision` (for VLA) call the supervisor too, and add
  idle-stop (unload after N minutes unused).
- **VLA:** registered as just another backend; the robot loads it when it needs
  to *act*, and omni when it needs to *talk* — time-shared on the one GPU.
