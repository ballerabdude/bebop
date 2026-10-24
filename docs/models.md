# Models: catalog, downloads and the GPU supervisor

The robot ships no model weights. Three pieces handle models:

| Piece | Code | Does |
|---|---|---|
| Catalog | `bebop-vision/config/models.yaml` | what can be provisioned; read by the firmware and the Python downloader |
| Downloads | `firmware/bebop-linux/src/model.rs`, `bebop-model-download@<id>.service`, `bebop_vision/download_model.py` | the operator's HF token, one systemd unit per download, progress in telemetry |
| Supervisor | `bebop-models.service`, `bebop_vision/models_supervisor.py` (`:9094`) | GPU budget and lifecycle of registered Edge-LLM backends |

TensorRT Edge-LLM is the only generative runtime (below). There is no
per-purpose model selection: `purpose` is a label the app groups by, and
`voice_server.py` takes the first `purpose: voice` entry as the omni
checkpoint. The old selector (`SetModelPurpose`,
`/etc/bebop/model_selection.json`) was removed on 2026-10-24: no service
ever used the choice (the firmware only echoed it back in telemetry), and a
robot may still carry a stale copy of the file.

## Catalog (`config/models.yaml`)

One list, `models:`; adding an entry needs no code change. The firmware
reads `/home/bebop/bebop/bebop-vision/config/models.yaml` once at start
(`model.rs`); `bebop_vision/models.py` parses the same file.

| Key | Meaning |
|---|---|
| `id` | stable id (`DownloadModel.model_id`, the unit instance); `[A-Za-z0-9._-]` to be downloadable |
| `name`, `description` | shown in the app |
| `kind` | `hf` (download from Hugging Face, the default) or `local` (produced on the robot or copied over; presence only) |
| `purpose` | label: `segmentation`, `backbone`, `voice`, `asr`, `tts`, `brain` |
| `repo`, `revision` | Hugging Face repo (required for `hf`) and revision (default `main`) |
| `download` | `files` (default: the named `files`) or `snapshot` (the whole repo) |
| `files` | repo-relative files for `download: files` (required there) |
| `dest` | snapshot directory under `weights/` (default: `id`) |
| `gated` | needs an HF token whose account accepted the license |
| `bytes` | size hint: the downloader's fallback when Hugging Face metadata has no size, and the size reported for a ready snapshot |
| `path` | `local` only: absolute, or relative to `weights/` (default `weights/<id>`) |

Ready means a non-empty file at every `weights/<file>` (`files`), at
`weights/<dest or id>/.bebop-complete` (`snapshot`), or at `path` (`local`).
On the robot `weights/` is `/home/bebop/bebop/bebop-vision/weights`.

| `id` | `purpose` | Repo | Download | Gated | Used by |
|---|---|---|---|---|---|
| `sam3.1` | segmentation | `facebook/sam3.1` | `sam3.1_multiplex.pt` | yes | the SAM teacher (`bebop_vision/sam3_concepts.py`, `tools/bev_student_labels.py`) |
| `sam3` | segmentation | `facebook/sam3` | `sam3.pt` | yes | the same, `--sam-version sam3` |
| `dinov3-vitl16` | backbone | `facebook/dinov3-vitl16-pretrain-lvd1689m` | `model.safetensors` | yes | the place-memory encoder ([`places.md`](places.md)) |
| `qwen3-omni-30b` | voice | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | snapshot, ~63 GB | no | the voice `omni` engine ([`voice.md`](voice.md)) |
| `qwen3-asr-1.7b` | asr | `Qwen/Qwen3-ASR-1.7B` | snapshot, ~4 GB | no | not read by code (see below) |
| `qwen3-tts-1.7b` | tts | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | snapshot, ~4 GB | no | not read by code |
| `qwen3.8-27b-nvfp4` | brain | `RadixArk/Qwen3.8-27B-NVFP4` | snapshot, ~18 GB | no | not read by code |

The voice cascade launches its stages with the HF repo ids
(`voice_server.py` `CASCADE_*_MODEL`), passing `HF_TOKEN`; nothing reads the
three cascade snapshots under `weights/`.

## Downloads

1. The app (Settings › Models, `bebop-app/src/components/AIModelCard.tsx`)
   sends `SetHfToken` (tag 24), `ClearHfToken` (25) and
   `DownloadModel{model_id}` (26) over the runtime WebSocket.
2. The firmware writes the token to `/etc/bebop/hf_token` (root, `0600`,
   atomic; it must be ≥ 8 characters without whitespace) and never puts it
   in telemetry; `ModelState.token_set` is the only trace.
3. `DownloadModel` is validated at once (known id, `kind: hf`, a valid unit
   instance name, the template unit installed, a token for a gated entry),
   then queued: the worker deletes `/run/bebop/models/<id>.json` and runs
   `systemctl start --no-block bebop-model-download@<id>.service`.
4. The unit (`bebop-vision/deploy/systemd/bebop-model-download@.service`,
   `Type=exec`, root, after `network-online.target`) runs
   `.venv/bin/python -u -m bebop_vision.download_model %i`. It is installed
   by the normal install path and never enabled.
5. The downloader reads the token from `HF_TOKEN`, then
   `/etc/bebop/hf_token`, then `bebop-vision/.env`; sizes the download from
   Hugging Face metadata (falling back to `bytes`); refuses when free space
   is below that plus 256 MiB; fetches `files` into `weights/` or the
   snapshot into `weights/<dest or id>/` and then writes `.bebop-complete`.
   Every 0.5 s it writes `{state, detail, bytes_downloaded, bytes_total}` to
   `/run/bebop/models/<id>.json`; a gated repo or HTTP 401/403 ends as
   `unauthorized`, anything else as `failed`.
6. The firmware re-polls every 1.5 s (`systemctl show` per unit, the status
   files, free space on `weights/`) and publishes `ModelState` in telemetry
   (`TelemetryFrame.model` / `Snapshot.model`, tag 20).

```proto
message ModelState {
  bool   token_set       = 1;   // /etc/bebop/hf_token exists and is non-empty
  bool   present         = 2;   // the download template unit is installed
  uint64 disk_free_bytes = 3;
  string detail          = 4;   // last provisioning error
  repeated ModelCatalogEntry models = 5;
  reserved 6; reserved "selection";   // per-purpose selection, removed 2026-10-24
}
// ModelCatalogEntry: id, name, description, kind, repo, gated, bytes_total,
// purpose; live: ready, state (idle | downloading | ready | failed |
// unauthorized), detail, bytes_downloaded. Tags 6, 7, 10 (files, revision,
// path) are reserved, removed 2026-10-24.
```

From the shell (in `bebop-vision/`, root for the token):
`python -m bebop_vision.download_model <id> [<id> ...]` or `--all`.

## Edge-LLM, the only generative runtime

Every generative model on the robot is a `tensorrt-edgellm-serve` process: a
checkpoint, an engine cache directory and a port. The first start builds
ahead-of-time TensorRT engines (tens of minutes on the Thor); later starts
deserialize them (~40 s) (`models_supervisor.py`).

- Version: `tensorrt-edgellm[server,server-tools,tools]==0.11.0`
  (`bebop-vision/requirements.txt`), pinned exactly: the engines and the
  `edge_serve.py` patch are specific to this build. `install-jetson.sh
  --setup-voice` installs it into the single `bebop-vision/.venv`.
- `bebop_vision/edge_serve.py`: `resolve_edgellm_bin()` (the venv's console
  script next to the interpreter, then `PATH`), shared by `bebop-voice` and
  `bebop-models`, and a drop-in launcher with the 0.11.0 TTS kwargs patch.
- Who starts them: `bebop-voice` (cascade stages on 8011/8012/8013, or the
  omni child on 8000; [`voice.md`](voice.md)) and `bebop-models` (registry
  `edgellm` backends).
- The vLLM-Omni Docker runtime was removed on 2026-10-16 (`58fd661`): its
  lifecycle (multi-minute cold start, sleep/wake to free the GPU, bind-mounted
  JIT caches, an sm_110 autotuner that could crash) was the largest source of
  complexity, for one model.

DINOv3 (torch, the place encoder) is not generative and runs inside
`bebop-vision`, which owns the cameras; the registry counts it as `tracked`.

## Supervisor: `bebop-models` (`:9094`)

`bebop-vision/bebop_vision/models_supervisor.py`; unit
`bebop-vision/deploy/systemd/bebop-models.service` (root, the shared venv).
`sudo ./scripts/install-jetson.sh --setup-models` installs it and runs
`systemctl enable --now bebop-models.service`. Measured on the Thor
(2026-10-16, `58fd661`): `POST /models/omni-edgellm/load` reached `ready` in
40 s with a warm engine cache.

Registry: `bebop-vision/config/models_supervisor.yaml`, or the file named by
`BEBOP_MODELS_CONFIG`. It is the only source of backends; an unreadable
registry leaves the supervisor with none.

| Kind | Meaning |
|---|---|
| `edgellm` | a child the supervisor starts and stops: `tensorrt-edgellm-serve <model> --cache-dir <cache_dir> --host <host> --port <port> <serve_args...>` (`host` default `127.0.0.1`), given `HF_TOKEN`; ready when `GET <base_url>/v1/models` is 200 (polled every 5 s, up to 90 min); its log goes to the `bebop-models` journal |
| `external` | an already-running OpenAI-compatible server at `base_url` (debug); `load` only checks it. The default when `kind` is omitted |
| `tracked` | runs in another process; counted against the budget whether or not it runs, never started or stopped |

Entry keys: `id`, `purpose`, `kind`, `footprint_gb`, `model`, `cache_dir`,
`host`, `port`, `base_url` (default `http://127.0.0.1:<port>`),
`serve_args`. The shipped registry:

| `id` | `kind` | `footprint_gb` | Notes |
|---|---|---|---|
| `omni-edgellm` | edgellm | 20 | `weights/qwen3-omni-30b-nvfp4` (from `install-jetson.sh --quantize-voice`), cache `/var/lib/bebop-voice/edgellm`, port 8000, `--enable-auto-tool-choice --tool-call-parser qwen3_xml` |
| `vision` | tracked | 6 | DINOv3 place memory inside `bebop-vision` |

API (CORS `*`):

```
GET  /healthz              {ok, budget_gb, resident_gb, mem: {MemTotal, MemAvailable}}   (GB)
GET  /models               {backends: [{id, purpose, kind, state, detail, footprint_gb, base_url, pid}]}
POST /models/{id}/load     start; blocks until ready or failed; returns the backend
POST /models/{id}/unload   stop; returns the backend
```

States: `stopped`, `starting`, `ready`, `error`, `tracked`. An unknown id is
400 `{error}`. Environment: `BEBOP_GPU_BUDGET_GB` (default 108),
`BEBOP_MODELS_HOST` (`0.0.0.0`), `BEBOP_MODELS_PORT` (`9094`).

GPU budget (the Thor has one GPU and ~122 GB of unified memory):

- `resident_gb` is the summed `footprint_gb` of `ready` and `tracked`
  backends.
- Loading a backend that is not already `ready`, `starting` or `tracked`
  first walks the other non-tracked `ready`/`starting` backends in registry
  order and stops each one until `resident_gb + footprint ≤ budget` and
  `MemAvailable ≥ footprint`; then it starts the backend, even when it still
  does not fit. The shipped registry has one `edgellm` backend, so nothing is
  evicted.
- The cascade stages are supervised by `bebop-voice` (`StageServer`), not
  registered here, so the budget does not count them. The registry comment
  shows the shape an entry for the brain would take (`kind: edgellm`,
  `model: RadixArk/Qwen3.8-27B-NVFP4`, `port: 8012`).

Clients:

- `bebop-voice` with `omni_supervisor_url` set (Settings › Voice & agent ›
  Omni source, e.g. `http://127.0.0.1:9094`): `SupervisedServer` loads
  `omni_backend_id` (default `omni-edgellm`) and uses the returned
  `base_url`; it unloads on stop only with `omni_unload_on_stop`.
- The app's Settings › Models › GPU budget card
  (`bebop-app/src/screens/settings/ModelSupervisorPanel.tsx`): resident
  backends, load and unload; before a load it names the backends that would
  be evicted (`bebop-app/src/screens/settings/modelEvictions.ts` mirrors
  `Supervisor.load()`).
