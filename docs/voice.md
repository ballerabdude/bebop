# Voice + agent (`bebop-voice`, `:9093`)

`bebop-voice` is the robot's agent server: the operator talks or types to
the robot, and it answers with streamed text, optional speech and tool calls
(place memory, robot status, drive notes, bounded motion). Every model runs
on the robot as a TensorRT Edge-LLM server ([`models.md`](models.md)).

| | |
|---|---|
| Code | `bebop-vision/bebop_vision/voice_server.py` (FastAPI + uvicorn) |
| Unit | `bebop-vision/deploy/systemd/bebop-voice.service`: root, `/home/bebop/bebop/bebop-vision/.venv/bin/python -u -m bebop_vision.voice_server`, `Restart=on-failure`, `TimeoutStopSec=60` |
| Start/stop | the firmware (`SetVoiceEnabled`), from the app's Autonomy console or Diagnostics › Services; installed but not enabled (`install-jetson.sh --enable-autostart` opts it into boot) |
| Ports | `9093` (gateway, `0.0.0.0`); cascade stages on `127.0.0.1`: `8011` ASR, `8012` brain, `8013` TTS; omni child `8000` |
| State | `/var/lib/bebop-voice/config.json` (operator settings); `/var/lib/bebop-voice/edgellm/` (engine caches: `asr/`, `tts/`, `brain/` for the cascade, the directory itself for the omni child) |

Audio never passes through the firmware. The runtime WebSocket (`:9090`)
carries only control and status; the voice stack is its own unit because it
owns large GPU models under the repo checkout while `bebop-linux` runs
sandboxed (`ProtectHome=read-only`), and its engine builds stay out of the
safety-critical process (`firmware/bebop-linux/src/voice.rs`).

## Engines

`backend` in the config picks one engine; only that one is loaded. A saved
change (Settings › Voice & agent › Engine, or `POST /config`) applies on the
next service start; `--backend` overrides the saved choice for that run
and is written back only if a `POST /config` then saves the config.

| Engine | Pipeline | Tools | |
|---|---|---|---|
| `cascade` | ASR -> VLM brain -> TTS, three Edge-LLM servers | yes | default |
| `omni` | one end-to-end Qwen3-Omni-30B-A3B model | none (chat only) | selectable |

Config files written before `config_version: 2` saved `omni` on every write,
so a pre-v2 `omni` is migrated to `cascade` on load; a v2 file keeps the
operator's choice.

### Cascade backend (default)

| Stage | Model (HF repo id) | Port | Endpoint | Ready when `/health` reports |
|---|---|---|---|---|
| ASR | `Qwen/Qwen3-ASR-1.7B` | 8011 | `POST /v1/audio/transcriptions` | `transcription` |
| Brain | `RadixArk/Qwen3.8-27B-NVFP4` (multimodal) | 8012 | `POST /v1/chat/completions`, streamed, with tools | `chat` |
| TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 8013 | `POST /v1/audio/speech` (`response_format: pcm`, 24 kHz) | `speech` |

- `CascadePipeline` starts the stages one at a time (ASR, TTS, brain) to
  bound GPU memory and reuses a stage that already answers. Each runs
  `tensorrt-edgellm-serve <repo id> --cache-dir <cache>/<stage> --port <port>
  --host 127.0.0.1` (TTS through the `edge_serve` launcher, below), given the
  operator's token as `HF_TOKEN` (`models.load_token()`: the `HF_TOKEN`
  environment variable, `/etc/bebop/hf_token`, then `bebop-vision/.env`). The
  catalog's `qwen3-asr-1.7b`, `qwen3-tts-1.7b` and `qwen3.8-27b-nvfp4`
  snapshots under `weights/` are not read by `voice_server.py`.
- The first start builds each stage's engines; a stage gets up to 90 min
  (`BUILD_TIMEOUT_S`). A failed stage sets `phase: error`
  (`<stage> failed to start`).
- Brain flags: `--enable-auto-tool-choice --tool-call-parser qwen3_xml
  --max-input-len 8192 --max-kv-cache-capacity 16384 --speculative-config
  '{"method": "mtp", "num_speculative_tokens": 3}'`. MTP uses the
  checkpoint's own layers (no draft model); the mode is part of the engine
  bundle name, so toggling it means one rebuild.
- TTS runs through `python -m bebop_vision.edge_serve`, a drop-in launcher
  that strips the VLM-only `max_image_tokens*` kwargs Edge-LLM 0.11.0
  forwards to its TTS runtime.
- A turn: transcribe (voice turns) -> brain with the system prompt and,
  with live context on, the robot status, both camera frames and the tools
  (`max_tokens` 512, temperature 0.7, at most 3 brain calls: tool rounds,
  then the answer) -> TTS per finished sentence while the reply streams.
- Voices are the TTS server's speaker ids (`/v1/voices`, e.g. `aiden`,
  `dylan`, `eric`, `serena`). The configured voice is used when the TTS lists
  it (or the list is unavailable), else `aiden` (or the first listed one).

### Omni backend (chat only)

The catalog's `voice` entry, `qwen3-omni-30b`
(`Qwen/Qwen3-Omni-30B-A3B-Instruct`, snapshot, not gated), from one of three
sources; the first configured one wins:

| Source | Config | Lifecycle owner |
|---|---|---|
| Supervisor | `omni_supervisor_url` (+ `omni_backend_id`, default `omni-edgellm`) | `bebop-models` (`:9094`): `POST /models/{id}/load` blocks until ready and returns `base_url`; on service stop it unloads only with `omni_unload_on_stop` |
| External | `omni_url` | someone else (debug): an OpenAI-compatible server; the service polls `GET /v1/models` for up to 10 min |
| Local Edge-LLM child | neither set (default) | `bebop-voice` (`EdgeLLMServer`, port 8000, cache `/var/lib/bebop-voice/edgellm`) |

Provisioning the local child:

1. Settings › Models: download `qwen3-omni-30b` (into
   `weights/qwen3-omni-30b/`, done when `.bebop-complete` exists).
2. `sudo ./scripts/install-jetson.sh --quantize-voice` (after
   `--setup-voice`; `--num-samples`, default 512): `tensorrt-edgellm-quantize
   llm --quantization nvfp4` into `weights/qwen3-omni-30b-nvfp4/`. The
   installer notes that Edge-LLM supports Qwen3-Omni only at NVFP4 (or INT4
   AWQ); the FP16 checkpoint builds engines but fails runtime init.
3. The service serves `weights/qwen3-omni-30b-nvfp4/` when it has a
   `config.json`, else the snapshot; with neither it reports `phase: error`,
   `omni checkpoint not provisioned (see docs/voice.md)`.

The child runs `tensorrt-edgellm-serve <dir> --cache-dir ... --port 8000
--enable-auto-tool-choice --tool-call-parser qwen3_xml` plus any
`--edgellm-arg`; no tools are ever sent, so the parser flags are inert. The
first start builds six TensorRT engines, reported through `/healthz`
(`component`, `components_done`, `log_tail`). After its `/health` answers,
the service runs one text and one short silent-audio warm-up turn before
`ready`: the first generation after engine load came out garbled on the
Thor.

An omni turn is one model call: the user's audio (or text) and, with live
context on, both camera frames and the robot status; text and base64 PCM16
audio stream back (`modalities: [text, audio]`, text only for
`speak: false`). Voice turns get no `transcript` event (there is no ASR).
Default voice `chelsie`; speaker ids from the server's `/v1/voices`.
Measured against the live omni server (2026-10-04): each 1280×800 snapshot
costs ~480 prompt tokens (~960 for both) and the audio ~80; that engine was
built `i4096-b1-kv8192` (~4,096 input tokens), so keep history short.

## Control plane: runtime WS `:9090`

- `SetVoiceEnabled { bool enabled }` (`ClientRuntimeMessage` tag 28): the
  firmware (`src/voice.rs`) queues `systemctl start|stop bebop-voice.service`
  on a worker thread and acks at once (`voice start requested`); an
  uninstalled unit gets an error reply.
- The worker re-polls `systemctl show` every 1.5 s; telemetry carries
  `VoiceState` (`TelemetryFrame.voice` / `Snapshot.voice`, tag 21):

```proto
message VoiceState {
  bool   present = 1;  // LoadState=loaded
  bool   running = 2;  // ActiveState=active
  string state   = 3;  // raw ActiveState
  string detail  = 4;  // SubState (+ " (Result)" unless success), or the last systemctl error
  string service = 5;  // "bebop-voice.service"
  reserved 6;          // `model`, removed 2026-10-24
  reserved "model";
}
```

`running` means the Python gateway is up, not that the models are ready;
readiness is `/healthz` `phase`.

## HTTP (`:9093`)

CORS `*`. `/healthz` and `/config` answer at once, also while engines build
or after a failure.

`GET /healthz`:

| Key | Meaning |
|---|---|
| `ok` | `phase` is `ready` or `stub` |
| `phase` | `idle`, `starting`, `building`, `ready`, `error`, `stub` |
| `detail` | last meaningful status line, e.g. `starting brain (RadixArk/Qwen3.8-27B-NVFP4)`, `cascade ready` |
| `model` | `cascade`; for omni the catalog id, `qwen3-omni (supervised:<id>)` or `qwen3-omni (external)`; `stub` |
| `heartbeat_ms`, `heartbeat_age_s`, `elapsed_s`, `component`, `components_done`, `log_tail` | build progress of the local omni child (`log_tail`: its last 20 log lines); the other engines only stamp `heartbeat_ms` once ready |
| `backend` | the configured engine, `cascade` or `omni` (it takes effect on the next start) |
| `tools_supported` | turns go to the cascade |
| `tools_enabled`, `motion_enabled` | the two permission switches |
| `turns_active`, `turn_active` | running turns server-wide, and whether there is one |

`GET /config` returns the settings plus `voices` (speaker ids of the running
engine), `backends` (`["omni", "cascade"]`) and `tools_supported`.
`POST /config` takes any subset of the fields below, saves
`/var/lib/bebop-voice/config.json` and returns the same view; an unknown
voice is 400 (when the engine lists voices), a failed save 500.

| Field | Default | Applies |
|---|---|---|
| `voice` | `chelsie` | next turn |
| `system_prompt` | a short Bebop persona | next turn |
| `history_turns` | `4` (clamped 0-20) | next turn |
| `keep_audio_history` | `false` | omni only: keep the user's audio in history |
| `tools_enabled` | `true` | next turn: live context and the non-motion tools |
| `motion_enabled` | `false` | next turn: adds `move_relative` |
| `backend` | `cascade` | next service start |
| `omni_supervisor_url`, `omni_backend_id` (`omni-edgellm`), `omni_unload_on_stop` (`false`), `omni_url` | empty | next service start |

A saved `omni_backend_id` of `omni-vllm` (the removed vLLM container) is
rewritten to `omni-edgellm` on load.

CLI (`python -m bebop_vision.voice_server`): `--host` (`0.0.0.0`), `--port`
(`9093`), `--cache-dir` (`/var/lib/bebop-voice/edgellm`); `--backend`,
`--omni-url`, `--omni-supervisor-url`, `--omni-backend-id` override the saved
config; `--voice`, `--system-prompt`, `--history-turns`,
`--keep-audio-history`, `--no-tools` apply only when no config file exists;
`--edgellm-arg` (repeatable) goes to the omni child; `--stub` replaces the
models with a fixed text reply and a 440 Hz tone; `--check` prints the resolved
config and checkpoint state, then exits.

## Agent protocol (voice WS `:9093`)

`ws://<robot>:9093/voice?session=<id>` carries voice and typed turns. JSON
text frames are control and events; binary frames are mono PCM16 (up: at the
`sample_rate` given in `utterance_start`, default 16000; down: 24 kHz).
`session` keys the conversation history, kept across reconnects (LRU of 32
sessions); without it each connection gets a fresh history. There is no
voice override on the socket: the speaker is `POST /config` `voice`. A
connection opened before the engine is ready gets
`{"type": "error", "message": "voice service not ready (<phase>): <detail>"}`
and is closed.

**Turn model.** Each turn runs as its own server task, so the socket keeps
reading while the brain thinks, speaks or drives. At most one turn per
session runs; a new one is rejected (`turn_rejected`), so send `cancel`
first. Rejecting rather than auto-cancelling keeps a stray second message
from silently aborting a motion. A cancelled turn that is still winding down
(stopping the wheels) does not block: the next turn waits for it (up to
10 s), then starts. Events go to the socket that started the turn.

App -> robot:

| Message | Meaning |
|---|---|
| `{"type":"utterance_start","sample_rate":48000,"format":"pcm16","speak"?:bool,"turn_id"?:str}` | start a voice turn; binary PCM16 frames follow |
| `{"type":"utterance_end"}` | end of the utterance: the turn starts (no audio: just `done`) |
| `{"type":"user_text","text":"...","speak"?:bool,"turn_id"?:str}` | a typed turn; the text goes straight to the model (no ASR) |
| `{"type":"cancel"}` | cancel the session's running turn (no-op when idle) |
| `{"type":"reset"}` | cancel any running turn, then clear the session history (answers `{"type":"reset"}`) |

`speak` defaults to `true`; `false`, `0` and the strings `"false"`, `"0"`,
`"no"`, `"off"`, `""` skip TTS for that turn (no `audio_*` events, no binary
frames). `turn_id` (1-64 characters) lets the client pick the id; otherwise
the server assigns one. A voice turn takes both from `utterance_start`.

Robot -> app, per turn (each carries `turn_id`):

| Event | Fields |
|---|---|
| `turn_start` | `source: "voice"\|"text"`, `speak` |
| `transcript` | `text` (the ASR result or the typed text), `final: true`, `source`; none for an omni voice turn |
| `assistant_delta` | `text`: streamed reply text (concatenate) |
| `tool_call` | `call_id`, `name`, `args` (parsed JSON object) |
| `tool_result` | `call_id`, `ok`, `text` (what the model was told), `ms` |
| `motion_progress` | `call_id`, `done_m`, `total_m`, `done_deg`, `total_deg`: ~5 Hz during `move_relative`, plus a final one |
| `audio_start` / `audio_end` | `sample_rate: 24000`, `format: "pcm16"`; binary PCM16 frames in between |
| `turn_end` | `status: "done"\|"cancelled"\|"error"`, `error?`: always the turn's last event |

Outside a turn (no `turn_id`):

| Frame | When |
|---|---|
| `turn_rejected` | `reason: "busy"`, `active_turn_id`, `source`; no turn starts, the running one is unaffected |
| `done` | an `utterance_end` with no audio |
| `error` | `message`: not ready (then the socket closes), a bad JSON frame, an empty `user_text`, a socket error |
| `reset` | the reply to `reset` |

Clients key off these typed events; nothing else is sent (no `text`/`tool`
mirrors, no `done`/`error` next to `turn_end`).

Ordering within a turn: `turn_start` -> `transcript` -> `audio_start`? ->
(`assistant_delta` | `tool_call` -> `motion_progress`* -> `tool_result`)* ->
`audio_end`? -> `turn_end`. `ok: false` marks a refusal or failure (a tool
that declined to move, an unreachable service); the text says why. A tool
cut off by a cancel reports `tool_result{ok: false, text: "cancelled"}`.

**Cancel.** `cancel`, `reset` and closing the socket cancel the running
turn: brain streaming and TTS stop at once; `move_relative` checks the
turn's cancel flag every 50 ms control tick (task cancellation also lands at
each tick's sleep). Its `finally` always sends a zero twist and undoes what
it changed (disarm, restore the mode) before the turn ends `cancelled`.
There is no confirmation gate for motion (operator decision): motion is
gated by `motion_enabled` and the drive-seat rule below.

**History** keeps whole turns, bounded by `history_turns` turns (not
messages): the user's words (transcript or typed text), each tool-calling
assistant message with its tool results, and the reply. Camera frames are
never kept. A cancelled turn is still committed (cut-off tool calls get a
placeholder result, the reply ends "(interrupted)") so the model knows what
already happened. Omni keeps a spoken user turn only with
`keep_audio_history`; the cascade keeps its transcript.

## Live context

With `tools_enabled`, every turn (either engine) carries:

- the robot status: `GetSnapshot` over `ws://127.0.0.1:9090/ws`, as JSON
  (`mode`, `estop`, `estop_reason`, `wheels_armed`, `drive`, `odom`,
  `battery_pct`, `battery_v`, `vision_running`) in the system prompt. A
  fetch measured a flat ~2 s (2026-10-16), so it is cached (15 s TTL) and
  refreshed every 5 s in the background;
- both color frames, `color_near` and `color_far`, from
  `http://127.0.0.1:9092/snapshot?stream=...`, in the user message. When
  neither arrives the service sends `SetVisionEnabled(true)` and retries for
  up to ~15 s.

The switch is Settings › Voice & agent › Live context and tools
(`tools_enabled`; `--no-tools` when no config file exists).

## Tools (cascade only)

The brain gets OpenAI tools; the service runs each call, returns the
result, and the answer streams (and is spoken per sentence) as usual.

| Tool | Does | Switch |
|---|---|---|
| `where_am_i`, `list_places`, `place_facts` | read place memory over HTTP `:9095` ([`places.md`](places.md)) | `tools_enabled` |
| `remember_place`, `add_place_fact`, `rename_place`, `forget_place` | write it (`remember_place` is `POST /places`, the current view of every camera) | `tools_enabled` |
| `robot_status` | the cached status line: mode, E-STOP, armed wheels, odometry, battery | `tools_enabled` |
| `note_drive_intent` `{text, from_place?, to_place?, task?, target?, mode?}` | the drive intent and its run (`data-collection.md` › Runs), `POST :9092/intent`; `mode: continue` sends only the spoken fields; all fields empty clears it | `tools_enabled` |
| `mark_drive_event` `{kind: obstacle\|recovery\|exclude\|note, note?}` | a marker, `POST :9092/event`; 409 (no segment recording) is reported as not marked | `tools_enabled` |
| `move_relative` `{forward_m?, turn_deg?}` | bounded odom-relative motion | `motion_enabled` |

There are no navigation tools: `move_relative` is the only motion tool
since the navigation stack was stripped on 2026-10-22. The app disables
"Allow moves" while "Live context and tools" is off.

### Motion is deliberately narrow

`move_relative` is odom-relative only; there is no map, so it cannot take
"the door" or a room name. It:

- clamps one call to ≤ 2 m and ≤ 180°, at ≤ 0.2 m/s and ≤ 0.6 rad/s
  (tolerance 3 cm / 0.06 rad); with both given it turns on the spot first,
  then drives, holding the heading;
- refuses while the E-STOP is latched, while another client holds the drive
  seat (`DriveState.has_active_operator && !you_are_active_operator`), when
  the wheels are armed in a mode other than `DIAL_IN`, when the runtime does
  not answer within 5 s, or when dial-in or arming does not happen;
- if the wheels are already armed in `DIAL_IN` (a person armed them), drives
  as-is and leaves them armed and the mode untouched; otherwise arms for the
  duration only: mode -> `DIAL_IN` (arming requires it), arm all wheels,
  each step waited up to 4 s;
- **yields to any person at once.** Its runtime connection announces
  `ClientHello { name: "voice", autonomous: true }`. The firmware hands an
  autonomous holder's seat to the first drive command, zero or non-zero,
  from a non-autonomous connection and applies that twist; a preempting stop
  claims a fresh seat for `server.operator_grace_ms` (2000 ms), so voice
  cannot take it back. Voice sees the seat taken in the next telemetry
  frame, stops within one 50 ms tick without another non-zero twist, ends
  the tool `ok: false` with `stopped: <name> took over after <distance>
  (<pose>). I let them drive.`, and leaves the wheels and mode to the new
  holder. The app writes a release stop only when it owes one, so an app
  control that stops voice must call `twistStream.forceStop()`, not
  `stop()`.
  `DriveState.active_operator_autonomous` tells viewers voice is driving;
- stops within one tick on `cancel` / disconnect, E-STOP, a lost runtime
  link or a person taking the seat, and reports how far it actually got;
- closes the loop on wheel odometry and reports what telemetry showed, not
  what was asked; wheel-odometry heading over-read in-place turns ~2x on
  carpet live on 2026-10-21 (`AGENTS.md`), so turns can come up short there;
- on every exit path stops, then undoes only what it changed (disarm if it
  armed, restore the mode if it switched it), unless another client now
  holds the seat;
- relies on the firmware as the backstop: the operator watchdog
  (`server.operator_timeout_ms`, 500 ms) zeroes a stale twist (a crashed
  tool coasts to a stop), every wheel command is slewed and clamped to
  `vel_max`, and one WS client holds the drive seat.

### E-STOP is operator-only

No tool latches, clears or resets an E-STOP. A latched one means a person
has to go and look at the robot, so the brain may only report it
(`robot_status`) and refuse to move: `move_relative` bails before it arms
anything. `test_no_model_tool_can_control_the_estop`
(`tests/test_voice_server.py`) asserts that no place, status or motion tool
name contains `estop`, `e_stop`, `reset`, `clear` or `arm`. `RobotClient`
(`bebop_vision/robot.py`) has no E-STOP reset; only the app clears a latch
(`ResetEStop` over its own runtime connection).

## Measured latency (cascade)

Thor, 2026-10-16 (`git show 714ad2d`): a 4.5 s utterance over `/voice`,
mean time from the last spoken word to the first audio byte.

| Pipeline | First audio |
|---|---|
| brain waits for the whole reply, robot status fetched per turn | 6.22 s |
| + streamed brain, TTS per sentence | 5.49 s |
| + cached robot status | 3.50 s |

The remaining ~3.5 s was ASR ~0.4 s, first token ~0.5 s (two images = 977
prompt tokens, 0.41 s prefill), the first sentence (~1.3 s of decode at
22 tok/s) and TTS's first chunk (~0.4 s): it is set by the length of the
model's first sentence. TTS runs ~2x faster than playback, so it stays
ahead.

## App

| Piece | File (`bebop-app/src/`) |
|---|---|
| Agent console, `#/r/:host/autonomy` (Stop task, transcript, camera, robot status) | `screens/autonomy/AutonomyScreen.tsx`, `Composer.tsx`, `Transcript.tsx`, `RobotStatus.tsx` |
| Socket and turn model | `voice/useAgentSession.ts`, `voice/agentTurns.ts` |
| Mic capture and playback | `voice/useVoiceIO.ts` |
| `/healthz` and `/config` | `voice/voiceService.ts` |
| Settings › Voice & agent | `screens/autonomy/VoiceSettings.tsx`, mounted by `app/routes/SettingsRoute.tsx` |
| Start/stop (`SetVoiceEnabled`) | the console's service card; `screens/diagnostics/ServicesTab.tsx` |

## Mobile

- The mic is hold-to-talk; a tap (< 300 ms) latches it and a second tap
  sends, which also survives the permission prompt stealing the pointer.
  Capture is an AudioWorklet with a ScriptProcessorNode fallback; it asks
  for 16 kHz, reads back `ctx.sampleRate` (mobile often ignores the request)
  and reports that rate in `utterance_start`. `getUserMedia` failures map to
  actionable messages.
- No double-tap zoom, long-press callout or pull-to-refresh (`App.css`
  `touch-action: manipulation`, `-webkit-touch-callout: none`,
  `overscroll-behavior: none`); pinch zoom stays allowed (the `index.html`
  viewport no longer disables it). `viewport-fit=cover` plus
  `env(safe-area-inset-*)` padding keep the shell clear of notches.

### The secure-context problem

`getUserMedia` works only in a secure context (HTTPS, localhost or a native
app), but the robot's services are cleartext (`ws://<robot>:9090`,
`ws://<robot>:9093`, `http://<robot>:9093/healthz`). A plain-HTTP page gets
no mic; an HTTPS page has its cleartext sockets blocked as mixed content.
The app picks `wss`/`https` itself when the page is HTTPS
(`bebop-app/src/runtime/urls.ts`), so a TLS proxy in front of the robot
works; for development see [`dev-https.md`](dev-https.md) (Chrome's
insecure-origin flag, the `Caddyfile.dev` proxy, Tailscale).

A native build of the unused `bebop-app/src-tauri/` shell would avoid both
(a secure custom scheme that can allow LAN cleartext). It would need, on
iOS, `NSMicrophoneUsageDescription` and an ATS exception
(`NSAllowsLocalNetworking` or a per-host one), and on Android
`RECORD_AUDIO` and `android:usesCleartextTraffic="true"` (or a
network-security-config scoped to the LAN).

## Install

- `sudo ./scripts/install-jetson.sh --setup-voice` (`--setup-voice-only`
  skips the binary install) creates or refreshes the single
  `bebop-vision/.venv` (`--system-site-packages`, so JetPack's TensorRT
  bindings are visible), installs `requirements.lock.txt` (else the
  `requirements.txt` floors), among them Edge-LLM 0.11.0
  (`tensorrt-edgellm[server,server-tools,tools]==0.11.0`, a prebuilt aarch64
  wheel for JetPack 7.x, CUDA 13, SM110: nothing compiles on the robot), and
  checks that `tensorrt_edgellm.runtime.load()` works.
- The normal install path installs `bebop-voice.service`, not enabled.
- Omni only: download the checkpoint, then `--quantize-voice` (above).
- The cascade stages are not in the `bebop-models` registry, so the
  supervisor's GPU budget does not count them ([`models.md`](models.md)).
