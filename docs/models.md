# Models: the on-robot supervisor (`bebop-models`)

The Thor has **one GPU** (~122 GB unified) and several model consumers — voice
(omni / cascade), vision (navd / SAM / DINOv3), navigation, and a future VLA.
This service owns the **GPU budget and model lifecycle** so consumers do not
each invent it.

## One runtime: TensorRT Edge-LLM

Every **generative** model on the robot runs under
[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) — a
`tensorrt-edgellm-serve` process. That gives the robot a single runtime shape
for every generative model:

```
checkpoint + engine cache dir + port
```

Edge-LLM compiles a checkpoint to **ahead-of-time TensorRT engines** and serves
them, so starting a backend is deserialization (~40 s) rather than a compile.
The first start *builds* the engines (tens of minutes on the Thor); after that
they are cached and reused.

There is deliberately no second generative runtime. Earlier the voice stack
also supported a vLLM-Omni Docker container; it was removed because its
lifecycle (multi-minute cold start, sleep/wake to free the GPU, six bind-mounted
JIT caches, an sm_110 autotuner that could crash fatally) was the single largest
source of complexity here — for one model. See
[`voice.md`](voice.md) for the cascade (ASR → VLM brain → TTS), which is the
Edge-LLM-native way to get a high-quality voice without an end-to-end omni
model.

## Exception: camera-coupled perception

`navd` (ONNX Runtime) and `DINOv3` (torch) run **inside `bebop-vision`**, not
here. They are coupled to the camera frames and the process must hold the rig
(one owner per camera), so they are registered as `tracked` — counted against
the budget, not managed. Anything that needs a *frame* stays with the frames;
anything that is *shareable and generative* is owned here.

## Service

`bebop-vision/bebop_vision/models_supervisor.py` (unit:
`deploy/systemd/bebop-models.service`, port **:9094**).

```
GET  /healthz            budget, resident, MemAvailable
GET  /models             registry + live state (state, pid, base_url)
POST /models/{id}/load   start (evicts ready residents until it fits the budget)
POST /models/{id}/unload stop
```

Registry: `bebop-vision/config/models_supervisor.yaml` (falls back to built-in
defaults). A backend is `edgellm` (a `tensorrt-edgellm-serve` child this service
starts/stops), `external` (an already-running OpenAI-compatible URL; debug
escape hatch), or `tracked` (runs elsewhere; counted only). Each has a
`footprint_gb` used by the budget (`GPU_BUDGET_GB`, default 108 GB), so `load`
evicts residents when needed.

Install / enable:

```bash
sudo ./scripts/install-jetson.sh --setup-models      # installs + enables the unit
```

## GPU sharing policy

1. **Evict to fit.** Loading a backend evicts ready backends until
   `resident + new ≤ budget`. With Edge-LLM footprints the common set
   (omni / brain + ASR + TTS + navd) fits comfortably, so this rarely fires —
   keep it for a future large VLA that genuinely cannot co-reside.
2. **Small models co-reside** when their footprints fit.
3. **Camera-coupled vision is counted, never managed** (see above).

## Migration path

- **Today:** `bebop-models` owns `edgellm` backends; `bebop-voice` can either
  supervise its own Edge-LLM child (default) or delegate lifecycle to the
  supervisor — set **Omni source (supervisor)** on the Voice page
  (`omni_supervisor_url`, e.g. `http://127.0.0.1:9094`) and the voice service
  `load`s the backend through it (`SupervisedServer`). The app's Models page
  (**GPU budget** card) shows what is resident and loads/unloads backends.
- **Next:** register the cascade stages (ASR / brain / TTS) in the registry so
  the supervisor owns them too, and add idle-stop (unload after N minutes
  unused).
- **VLA:** just another `edgellm` backend; the robot loads it when it needs to
  *act*.
