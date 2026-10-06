# vLLM-Omni on Thor: startup cost and how we cut it

The voice "Omni" backend runs `Qwen3-Omni-30B-A3B` under **vLLM-Omni** in the
`bebop-vllm-omni-thor` container (`bebop-vision/deploy/docker/vllm-omni-thor/`).
It sounds markedly better than TensorRT Edge-LLM (higher-precision Talker), but
its cold start was **~29 min** on the Thor. Timed experiments (2026-10-06) cut
that to **~6 min, reproducibly**, with no quality change.

## The result

| Config | Cold start |
|---|---|
| Baseline (single-thread load, autotune on, no persistent caches) | **1725 s (28.7 min)** |
| + parallel safetensors load | 579 s |
| + disable flashinfer autotune | 588 s |
| + persistent flashinfer/torch/**Triton** caches | **371 s (6.2 min)** |

The last row is the shipped configuration (reproduced twice). It is what
`ContainerServer` and the `bebop-models` supervisor now start:

```
--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 14}'
--kernel-config              '{"enable_flashinfer_autotune": false}'
--enable-sleep-mode
# bind-mounts: .cache/{vllm,huggingface,flashinfer,torch_extensions,torch} and .triton
```

## What the 29 min was made of (baseline, 04:22:33 → 04:51:18)

| Stage | Work | Time |
|---|---|---|
| 0 `thinker` | weight load (46 GiB NVFP4 compressed-tensors) | **504 s** |
| | profile + KV cache + warmup + flashinfer autotune | 307 s |
| 1 `talker` | weight load | 24 s (page cache) |
| | flashinfer autotune | **4.2 min** |
| 2 `code2wav` | load + CUDA-graph warmup (11 sizes) | ~1.1 min |
| server | orchestration + API bring-up | ~4.7 min |

## What each lever actually did

1. **Parallel weight load — the big one (507 s → ~50 s).**
   vLLM's `DefaultModelLoader` reads safetensors **single-threaded** by default
   (`enable_multithread_load` is off), so the 46 GiB checkpoint was parsed on
   one of the Thor's 14 cores. `--model-loader-extra-config` with
   `num_threads: 14` is a ~10x cut on the dominant line item. No quality cost.

2. **Persist the Triton cache (also big, and non-obvious).**
   `code2wav` captures 11 CUDA graphs whose SnakeBeta activation needs a Triton
   kernel; that JIT happened inside the container's writable layer and was wiped
   by `docker rm -f`. Warm, `code2wav` dropped from **130–590 s → ~19 s**. Its
   compile also *fails* for bf16 (`snake_activation.py … Expected dtype
   ['fp32','fp64'] but got bf16`) and falls back to a slow path, which is why the
   cold cost was so variable. Mounting `.cache/triton` stabilises it.
   - `.cache/flashinfer`, `.cache/torch_extensions`, `.cache/torch` are mounted
     too (flashinfer/inductor JIT).

3. **Disable the flashinfer/TRT-LLM MoE autotuner (stability, ~neutral time).**
   On sm_110 the autotuner intermittently throws a **fatal**
   `dispatchMoeGemmSelectClusterShapeTmaWarpSpecialized` while profiling an
   unsupported tactic and kills the stage (reproduced: one run died at
   "Autotuning process ends"). `--kernel-config
   '{"enable_flashinfer_autotune": false}'` removes it; time is unchanged
   because its ~1–2 min is small next to the JIT savings.

## Things that did NOT help
- **`-O0`** (`--optimization-level 0`): it also disables the norm/act quant
  fusions, and `code2wav`'s graph warmup then stalled ~5–10 min. Net **worse**
  (~14 min). Use `--kernel-config` instead if you want only the autotuner off.
- **Persisting the flashinfer cache *with autotune on***: the second start loaded
  a cached tactic and crashed the thinker stage. With autotune off this cannot
  happen (and the cache still speeds kernel JIT).

## Why Edge-LLM started in ~40 s

| | Edge-LLM | vLLM-Omni (now) |
|---|---|---|
| process start → ready | **~40 s** | **~6 min** |
| what start does | deserialize TRT engines + load external weights | Python engine init ×3 + parallel weight load + KV profile + cached JIT |

Edge-LLM's six engines are **ahead-of-time compiled, serialized TensorRT
artifacts** cached at `/var/lib/bebop-voice/edgellm/engines/…` (~1.9 GB), so
serve time is deserialization. vLLM does the equivalent work at serve time;
caching the JIT/autotune artifacts (levers 1–3) is the closest analog. The
trade-off: Edge-LLM's first start *builds* those engines (tens of minutes) and
its Talker is NVFP4 (worse audio), which is why vLLM-Omni is preferred.

## Speech lags the text (`codec_chunk_frames`)

With the stock config, speech started ~2 s after the text and arrived in ~2 s
blocks, so it *felt* like the text streamed first and the voice came after —
unlike Edge-LLM, whose talker pairs audio frames with the thinker's tokens.

Cause: the Talker->Code2Wav stage (`qwen3_omni.talker2code2wav_async_chunk`)
buffers `codec_chunk_frames` codec frames before emitting **any** waveform:

```python
chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
chunk_length = length % chunk_size_config
if chunk_length != 0 and not is_finished:
    return None       # nothing emitted until a full chunk is ready
```

Qwen3-Omni's codec is ~12.5 Hz, so 25 frames ≈ 2 s of audio. Measured on Thor
(same prompt):

| `codec_chunk_frames` | first audio | audio chunks |
|---|---|---|
| 25 (stock) | **1.96 s** | 13 |
| **8 (shipped)** | **0.91 s** | 37 |

`codec_left_context_frames: 25` still gives the decoder 25 frames of context, so
quality is preserved; smaller values trade a little chunk-boundary quality for
lower latency (try 5 for ~0.5 s). Set in
`deploy/docker/vllm-omni-thor/qwen3_omni_1gpu.yaml` (`connectors →
connector_of_shared_memory → extra`); it is read at startup.

Note this does not stop text from finishing first — the MoE thinker is simply
much faster than speech (~4 s of text vs ~23 s of audio here). It only removes
the *initial* delay before the voice starts.

## Remaining levers (not yet done)
1. **Keep it resident / sleep mode — validated (2026-10-06).**
   `POST /v1/omni/sleep {"stage_ids":[0,1,2],"level":2}` returns in **1 s** and
   frees **~84 GB** (unified `Mem used` 100 → 16 GB); `POST /v1/omni/wakeup`
   restores state `WARM` in **242 s (~4 min)**. Requires `--enable-sleep-mode`.
   This is how to hand the GPU to vision/VLA without a cold start: the
   supervisor starts `omni-vllm` at boot (or on app open) and sleeps it when
   idle. Caveats measured:
   - **Use level 2.** Level 1 (offload to CPU) only freed ~18 GB on Thor's
     *unified* memory — it parked 67 GB in shared memory, effectively doubling
     the footprint, and its wake did not complete.
   - Wake (~4 min) is shorter than a cold start but not instant; pre-wake on app
     open so it's warm by the time the operator talks.
2. **`talker` init is now the biggest item (~90–150 s)** — the code_predictor
   warmup over buckets `[1,2,4,8,16,32,64]`. Worth seeing if its `torch.compile`
   artifact can be cached.
3. **Shrink the profile/warmup**: `max_num_batched_tokens: 32768` in
   `qwen3_omni_1gpu.yaml` drives the `thinker` profile. Voice turns are short;
   lowering it trades long-prompt prefill latency for startup.

## Measuring a change
```bash
sudo docker logs --since 1h <container> \
  | grep -aE "Loading weights took|init engine .* took|Autotuning process (starts|ends)"
```
Watch `Loading weights took` and `init engine (profile, create kv cache, warmup
model) took`.
