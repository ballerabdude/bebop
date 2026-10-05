# Voice: on-robot speech-to-speech (Qwen3-Omni)

Status: **in progress** (branch `feat/voice-omni`). Provisioning foundation
landed; firmware control path, voice service, and app page are next.

## Goal (v1)

A new app page where the operator talks to the robot and it talks back.
**No actions, no tool calls** — pure speech-to-speech. Fully local: the model
runs on the robot, audio never leaves the device (except between the phone and
the robot on the LAN).

## Why Qwen3-Omni + TensorRT Edge-LLM

- **One model, native speech-out.** `Qwen/Qwen3-Omni-30B-A3B-Instruct`
  (Apache-2.0) takes audio (or text/image) in and emits text **and 24 kHz
  speech** out. No ASR → LLM → TTS glue, so prosody and turn feel are preserved.
- **NVIDIA supports it on Thor.** TensorRT Edge-LLM (TRT-Edge-LLM) lists Qwen3
  Omni 30B-A3B as a supported model with a six-engine layout (Thinker, Talker,
  CodePredictor, audio encoder, visual encoder, Code2Wav) and a streaming
  Thinker/Talker path.
- **Prebuilt aarch64 wheel — no source build.** The wheel matrix includes
  *Jetson Thor: JetPack 7.0/7.1/7.2, CUDA 13, SM110, platform TensorRT 10*.
  So the robot runs `pip install "tensorrt-edgellm[server]==0.11.0"` and
  `tensorrt-edgellm-serve`, not a CMake/CuTeDSL build.
- **HF provisioning already exists.** The model is a normal HF checkpoint; we
  extend the existing catalog/downloader to fetch a full-repo snapshot.

Hardware context: Thor dev kit, 122 GB unified RAM, ~273 GB/s bandwidth,
JetPack 7.2 / CUDA 13.2 / sm_110. Qwen3-30B-A3B-class MoE runs ~60-80 tok/s at
C=1 on Thor, which is enough for conversational latency.

## Architecture

```
   phone / PC (Tauri app)
   ┌──────────────────────────────────────────────┐
   │  VoiceScreen.tsx                             │
   │   • getUserMedia → 16 kHz mono PCM16         │
   │   • AudioContext playback of 24 kHz PCM16    │
   │   • control over runtime WS (:9090)          │
   │   • audio over voice WS (:9093)              │
   └───────────────┬───────────────────┬──────────┘
                   │ control           │ audio (turn-based)
        ┌──────────▼─────────┐   ┌─────▼──────────────────────────┐
        │ bebop-linux (:9090)│   │ bebop-voice.service (:9093)    │
        │  SetVoiceEnabled   │   │  Python WS gateway             │
        │  VoiceState        │   │   ├─ spawns tensorrt-edgellm-  │
        │  systemctl start/  │   │   │   serve on 127.0.0.1:8000  │
        │  stop voice unit   │   │   ├─ WAV/PCM → OpenAI audio in  │
        └────────────────────┘   │   └─ text + PCM chunks → app   │
                                 └─────┬──────────────────────────┘
                                       │ OpenAI-compatible HTTP
                                 ┌─────▼──────────────────────────┐
                                 │ tensorrt-edgellm-serve         │
                                 │ Qwen3-Omni-30B-A3B (6 engines) │
                                 │ cache: /var/lib/bebop-voice/   │
                                 └────────────────────────────────┘
```

### Why a separate port for audio

The runtime WS (`:9090`) is protobuf, one envelope per frame, and the firmware
would have to relay raw audio to the Python service. The video stack already
establishes the pattern of a separate Python server on its own port (`:9092`).
Voice audio gets `:9093`; the runtime WS stays the **control plane** (start/stop
+ status), which is what "reuse the websocket session" means here.

## Model provisioning

`bebop-vision/config/models.yaml` gains a snapshot entry:

```yaml
  - id: qwen3-omni-30b
    name: Qwen3-Omni 30B-A3B
    description: End-to-end speech-to-speech (audio in, natural speech out).
    kind: hf
    purpose: voice
    repo: Qwen/Qwen3-Omni-30B-A3B-Instruct
    download: snapshot
    dest: qwen3-omni-30b
    revision: main
    gated: false
    bytes: 63000000000
```

The catalog schema gains `download: files|snapshot` and `dest`:

- `files` (default) — the existing per-file `hf_hub_download` behavior.
- `snapshot` — `snapshot_download` the whole repo into
  `weights/<dest or id>/`, then write a `.bebop-complete` marker. The firmware
  treats that marker as `ready` (a 60 GB tree must not be walked every poll).

The app's Models page renders it automatically (grouped under "voice") and the
existing download button drives `bebop-model-download@qwen3-omni-30b.service`.
The HF token flow is unchanged.

**Quantization (open):** the 30B checkpoint is ~60 GB in BF16. TRT-Edge-LLM can
serve it directly (checkpoint-direct builder), or we first run
`tensorrt-edgellm-quantize` to NVFP4 (~20 GB, faster) using the `[tools]` extra
on the robot. v1 can serve BF16 to reduce moving parts; NVFP4 is the tuning
step. Engine build is cached in `/var/lib/bebop-voice/edgellm` and happens on
first voice-service start (surfaced as `VoiceState.detail = "building engines"`).

## Wire protocol

### Control (runtime WS `:9090`)

- `ClientRuntimeMessage.SetVoiceEnabled { bool enabled }` (tag 28) — firmware
  shells out to `systemctl start/stop bebop-voice.service`, mirroring
  `SetVisionEnabled`.
- `TelemetryFrame.voice` / `Snapshot.voice` (tag 21) — new `VoiceState`:

```proto
message VoiceState {
  bool   present = 1;   // unit installed
  bool   running = 2;   // systemd active
  string state   = 3;   // raw ActiveState
  string detail  = 4;   // SubState / last error / "building engines"
  string service = 5;
  string model   = 6;   // catalog id being served
  bool   ready   = 7;   // engines built and server answering /health
}
```

### Audio (voice WS `:9093`)

Turn-based v1 (push-to-talk or client VAD), JSON control frames + binary audio:

- app → robot: `{"type":"utterance_start","sample_rate":16000,"format":"pcm16"}`
  then binary PCM16 frames, then `{"type":"utterance_end"}`.
- robot → app: `{"type":"text","delta":"..."}` (transcript/response), then
  `{"type":"audio_start","sample_rate":24000,"format":"pcm16"}`, binary PCM16
  frames, `{"type":"audio_end"}`, `{"type":"done"}`.
- robot → app: `{"type":"tool","name":"..."}` when the model invokes a tool
  (the app shows "checking…"); the model's spoken answer follows.
- `{"type":"error","message":"..."}` on failure.
- `GET /healthz` for readiness.

Full-duplex (barge-in) is a later step (needs a full-duplex model such as
NemotronLabs VoiceChat-11B, or careful VAD + cancellation).

### Tool calling (agentic, read-only)

The model server runs with `--enable-auto-tool-choice --tool-call-parser
qwen3_xml`, so the `qwen3_xml` parser turns the model's XML tool calls into
OpenAI `tool_calls`.

> **Constraint:** `/v1/chat/completions` rejects `tools` combined with audio
> output (`400 unsupported_feature: tools cannot be combined with audio
> output`). The gateway therefore runs **two phases** per turn when tools are
> enabled (`_run_turn`):
>
> 1. **Plan (text-only, `modalities:["text"]`, tools offered).** Loop the model
>    up to `MAX_TOOL_ROUNDS`: execute any tool calls against the robot's
>    services, append `assistant(tool_calls)` + `tool` messages, repeat.
> 2. **Speak (`modalities:["text","audio"]`, no tools).** Make one final pass
>    over the same conversation to produce the spoken answer streamed to the app.
>
> The speaking pass must **not** replay the `tool_calls` / `tool` messages: with
> no tool schemas in the request the model re-emits the tool-call XML
> (`<tool_call>…</tool_call>`) instead of answering. Phase B rebuilds a clean
> message list and injects the results as text into the system prompt
> ("You just used your sensors. Here is what you found: …").
>
> This costs an extra text pass per turn vs. the tools-off path. The Voice
> page's "Tool use" switch (`VoiceConfig.tools_enabled`, `--no-tools`) turns it
> off for pure chit-chat.

The allowlist (`TOOL_SCHEMAS` / `execute_tool`) is deliberately read-only in
v1 — no motion:

- `get_robot_state` — opens `ws://127.0.0.1:9090/ws`, sends `GetSnapshot`,
  formats mode / E-STOP / armed wheels / battery / odom as JSON.
- `describe_scene` — fetches **both** cameras (`color_near` and `color_far`)
  from `:9092/snapshot`, sends them as two labelled images to the model, and
  returns the description as the tool result. If both cameras are down, it
  asks the firmware to enable vision (`SetVisionEnabled(true)`) and retries for
  up to ~25 s, so "what do you see?" turns its own eyes on.

Unknown tool names are refused, never executed. Motion tools (heat/e-stop-aware)
are a later phase and must go through a deterministic safety layer.

## Components and changes

| Area | Files | State |
|---|---|---|
| Catalog + downloader | `bebop-vision/config/models.yaml`, `bebop_vision/models.py`, `bebop_vision/download_model.py`, `firmware/bebop-linux/src/model.rs`, `tests/test_models.py` | ✅ done |
| Design | `docs/voice.md` | ✅ |
| Firmware control | `jetson-agent/bebop-proto/proto/bebop_runtime.proto`, `firmware/bebop-linux/src/voice.rs`, `src/server/{ws,handlers,telemetry}.rs`, `src/main.rs` | ✅ deployed to robot |
| App control + page | `bebop-app/src/proto/*`, `runtime/{types,wsTransport,index}.ts`, `screens/VoiceScreen.tsx`, `voice/useVoiceSession.ts`, `screens/DashboardScreen.tsx`, `screens/MotorBenchScreen.tsx`, `App.tsx` | ✅ done (push-to-talk) |
| Voice service | `bebop-vision/bebop_vision/voice_server.py`, `deploy/systemd/bebop-voice.service`, `tests/test_voice_server.py` | ✅ deployed to robot |
| Agentic tools (read-only) | `bebop-vision/bebop_vision/voice_server.py` (`TOOL_SCHEMAS`, `execute_tool`, `_run_turn` loop), `bebop-app/src/{screens/VoiceScreen.tsx,voice/useVoiceSession.ts}` | ✅ done (`get_robot_state`, `describe_scene` w/ both cameras) |
| Install | `scripts/install-jetson.sh` (`--setup-voice`), `requirements-voice.txt` | ✅ venv bootstrapped on robot |
| Bring-up | engine build, latency tuning, NVFP4 | ⏳ model downloading; engine build next |

### Bring-up status (2026-10-04)

- Firmware (branch `feat/voice-omni`) installed on the Thor via
  `install-jetson.sh --local --build --linux-only`; the catalog serves 6
  models including `qwen3-omni-30b`, and the app shows it on the Models page.
- `tensorrt-edgellm[server]==0.11.0` (aarch64 wheel, 447 MB) installed into
  `bebop-vision/.venv-voice`; `runtime.load()` validated.
- `voice_server --stub` protocol verified on the Thor (text + 24 kHz PCM).
- Remaining: finish the Qwen3-Omni snapshot download, start
  `bebop-voice.service` (first start builds the six TensorRT engines), then
  test push-to-talk from the app.

## Latency budget (target)

| Stage | Budget |
|---|---|
| Client capture + VAD end-of-utterance | ~0.2-0.3 s |
| Audio encode + transport (LAN) | ~0.05 s |
| Thinker prefill + first Talker frame | ~0.3-0.5 s |
| Code2Wav + transport + playback | ~0.1 s |
| **End-to-end** | **~0.7-1.0 s** |

Knobs: `codec_chunk_frames` (smaller = sooner but rougher), `talker_top_k`,
`max_audio_length`, and a short system prompt.

## Mobile

The Voice page is touch-first:

- **Tap-to-talk toggle**, not press-and-hold. Hold fights the microphone
  permission prompt (the prompt steals the pointer, so `pointerup` never
  fires and the mic stays open) and touch `pointerleave` jitter. Tap starts
  capture, tap again sends. `useVoiceSession` uses `AudioWorklet` with a
  `ScriptProcessorNode` fallback for pre-AudioWorklet WebViews, reads back
  `ctx.sampleRate` (mobile often ignores the requested 16 kHz), and maps
  `getUserMedia` failures to actionable messages.
- **No zoom / callout / pull-to-refresh** (`index.html` viewport +
  `App.css` `touch-action`, `-webkit-touch-callout`, `overscroll-behavior`),
  plus `env(safe-area-inset-bottom)` padding.

### The secure-context problem (important)

`getUserMedia` only works in a **secure context** — HTTPS or a native app.
But the app talks to the robot over **cleartext** `ws://<ip>:9090`,
`ws://<ip>:9093`, and `http://<ip>:9093/healthz`. So:

- **Mobile browser over plain HTTP** (`http://<workstation>:1420`): mic is
  blocked by the secure-context requirement.
- **Served over HTTPS**: the cleartext robot sockets become *mixed content*
  and are blocked.

The clean path is **Tauri mobile**, whose webview is a secure custom scheme
and can be configured to allow cleartext to the LAN. When adding mobile
targets (`tauri android init` / `tauri ios init`):

- **iOS** `Info.plist`: `NSMicrophoneUsageDescription`, and an ATS exception
  (`NSAllowsLocalNetworking` or a per-host exception) for the robot.
- **Android** `AndroidManifest.xml`: `RECORD_AUDIO`, and
  `android:usesCleartextTraffic="true"` (or a network-security-config scoped
  to the LAN).
- The alternative, browser-only deployment would require terminating TLS on
  the robot (self-signed) and using `wss://` / `https://`.

For **dev**, see [`dev-https.md`](dev-https.md): the Chrome
`unsafely-treat-insecure-origin-as-secure` flag (desktop), or a self-signed
Caddy proxy (`Caddyfile.dev`). The app picks `wss`/`https` automatically when
the page is HTTPS (`bebop-app/src/runtime/urls.ts`).

## Risks / open questions

1. **Engine build time & disk.** First serve builds six engines; budget tens of
   GB and several minutes. Confirm the wheel builds Qwen3-Omni without the
   `[tools]` extra (checkpoint-direct builder) or whether NVFP4 quantization is
   required first.
2. **Talker `text_projection`.** The C++ build needs `-DENABLE_CUTE_DSL=gemm`;
   the prebuilt wheel should already handle this — verify audio is not garbled.
3. **GPU/bandwidth contention** with `bebop-vision` (camera + navd ONNX). The
   LLM is bandwidth-bound; schedule voice when not training/recording heavily.
4. **Audio transport choice.** v1 uses a dedicated `:9093` WS. If a single
   connection is required, audio can move onto `:9090` as a protobuf `bytes`
   envelope (precedent: `NavMaskFrame.grid`), at the cost of a firmware relay.
5. **Model license/gating.** Qwen3-Omni is Apache-2.0; verify the HF repo is
   not gated (if it is, flip `gated: true` and the token flow already works).
