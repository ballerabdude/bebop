# vLLM-Omni on Thor: startup cost and how to cut it

The voice "Omni" backend runs `Qwen3-Omni-30B-A3B` under **vLLM-Omni** in the
`bebop-vllm-omni-thor` container (`bebop-vision/deploy/docker/vllm-omni-thor/`).
It is the best-sounding backend but its **cold start is ~25-30 min** on the
Thor. This is the measured breakdown and the levers that actually move it.

## Measured cold start (2026-10-06, one container)

Container start `04:22:33` → first `GET /v1/models` 200 at `04:51:18`:
**~28.75 min**. Per stage (the model is three vLLM engines):

| Stage | Work | Time |
|---|---|---|
| 0 `thinker` | engine init | **~13.7 min** |
| | └ weight load (46 GiB, NVFP4 compressed-tensors) | **504 s (8.4 min)** |
| | └ profile + KV cache + warmup + flashinfer autotune | 307 s |
| 1 `talker` | engine init | **~8.6 min** |
| | └ weight load | **24 s** (file was in page cache) |
| | └ flashinfer autotune ("Autotuning process starts/ends") | **4.2 min** |
| 2 `code2wav` | engine init + CUDA graph warmup (11 sizes) | ~1.1 min |
| server | orchestration + API server bring-up | ~4.7 min |

Two conclusions:
1. **Weight load is I/O + CPU-dequant bound.** The talker loaded the *same*
   46 GiB checkpoint in 24 s once it was in the OS page cache; the thinker paid
   504 s. With only ~12 GB RAM free (the model itself takes most of the 122 GB
   unified), a restart **will re-read and re-dequant** most of the checkpoint.
2. **The flashinfer JIT + autotune runs every start.** Its cache lives in
   `/root/.cache/flashinfer` *inside the container*. `ContainerServer` does
   `docker rm -f` on every start, so the writable-layer cache is discarded and
   the JIT/autotune is redone (~4-8 min, counted across all stages).

## Why Edge-LLM started so much faster

Measured warm start of the *same* Qwen3-Omni model under TensorRT Edge-LLM
(`bebop-voice` journal, 2026-10-05 23:48):

| | Edge-LLM | vLLM-Omni |
|---|---|---|
| process start → ready | **~40 s** (23:48:01 → 23:48:40) | ~29 min |
| what start does | deserialize TRT engines + load external weights | Python engine init ×3 + weight load + KV profile + JIT + autotune |

Edge-LLM is fast because its six engines are **ahead-of-time compiled,
serialized TensorRT artifacts** cached at
`/var/lib/bebop-voice/edgellm/engines/…/{llm,audio,kv8192…}` (thinker, talker,
code_predictor, audio, code2wav, visual; ~1.9 GB). Start-up is just:

```
23:48:02  Loading runtime bundle …
23:48:28  thinker engine loaded (787 I/O tensors)
23:48:32  talker engine loaded
23:48:36  code_predictor engine loaded
23:48:38  Engine loaded and ready
23:48:40  warm-up complete; ready
```

No graph tracing, no kernel JIT, no tactic autotune — **all of that happened
once, at engine-build time**. vLLM-Omni instead does a full Python engine init
per stage *every* cold start: `profile_run` to size the KV cache, flashinfer
CUDA-kernel JIT, and MoE tactic autotuning (the very work TensorRT baked into
the `.engine` files).

The trade-offs are why we still prefer vLLM-Omni:
- Edge-LLM's **first** start builds those engines (tens of minutes), and the
  build is host-venv, not containerised.
- To fit/quantize, Edge-LLM ran the Talker in **NVFP4**, which is why its audio
  was worse; vLLM-Omni's talker-safe checkpoint keeps the Talker at higher
  precision.

The startup levers below are the closest vLLM analog to Edge-LLM's cached
engines: cache the JIT/autotune results (lever 1) or skip autotune (lever 3),
and keep the model resident so the cost never lands in front of the operator.

## What is already in our favour
- `enforce_eager: true` on every stage → **torch.compile and CUDA-graph capture
  are off**, so there is no compile step (the one exception is `code2wav`,
  which captures 11 small CUDA graphs for its decoder — ~0.7 min).
- The flashinfer CUTLASS `.cu` instantiations are baked into the image.

## Levers, best first

1. **Parallelise weight loading (done, needs a timed restart to confirm).**
   vLLM's `DefaultModelLoader` reads safetensors **single-threaded** by default
   (`enable_multithread_load` is off), so the 46 GiB checkpoint is parsed on one
   of the Thor's 14 cores. Both container paths now pass
   `--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 14}'`.
   This targets the **504 s** thinker weight load (the single biggest item).

2. **Keep it resident and use vLLM sleep mode rather than stop/start.**
   The real fix: don't cold-start on demand. `ModelConfig.enable_sleep_mode`
   is available on CUDA, so the process can stay up and `/sleep` (offload
   weights to CPU RAM) to hand the GPU to vision/VLA, then `/wake_up` in
   seconds. That removes the ~29 min from the operator's path entirely and
   addresses the "don't waste the GPU" concern at the same time. Have the
   supervisor own this (start at boot / on app open, sleep when idle).

3. **Persist the kernel/JIT caches on the host (done).**
   `ContainerServer` and `bebop-models` bind-mount
   `/home/bebop/.cache/flashinfer` → `/root/.cache/flashinfer` and
   `…/.cache/torch_extensions`. The second start reuses the compiled kernels
   and tuning results instead of redoing them. No inference cost.
   - Verify the host dir is populated after a run:
     `sudo du -sh /home/bebop/.cache/flashinfer`.

4. **Skip flashinfer autotune** when startup matters more than peak throughput:
   pass `-O0` (`--optimization-level 0`) to the server. vLLM maps O0 to
   `kernel_config.enable_flashinfer_autotune: false` (see
   `vllm/config/vllm.py`), which removes the ~4 min autotune. It also turns off
   the norm/act quant fusions, so expect a modest decode slowdown — measure
   with a voice turn before committing.
   - In `config/models_supervisor.yaml` add `-O0` to the backend's `serve_args`.

5. **Shrink the warmup window.** `thinker` uses `max_num_batched_tokens: 32768`
   in `qwen3_omni_1gpu.yaml`; the profile/warmup scales with it. Voice turns
   are short, so dropping it (e.g. 8192-16384) cuts the 307 s profile phase at
   the cost of long-prompt prefill latency. Similarly `gpu_memory_utilization`
   drives the KV-cache profiling.

6. **Warm the page cache before start.** `vmtouch`/`cat` the checkpoint after
   boot so the first stage's read is warm. Bounded by free RAM (~12 GB here),
   so only partial.

7. **Do not enable torch.compile/CUDA graphs** to "speed up" startup — they add
   a first-run compile. They help steady-state inference, not bring-up; if you
   ever enable them, keep `~/.cache/vllm` mounted (it already is) so the compile
   cache survives.

## Measuring a change
Timed restart (do it when no voice session is active):

```bash
sudo docker logs --since 1h <container> | grep -aE "Loading weights took|init engine .* took|Autotuning process (starts|ends)"
```
`init engine (profile, create kv cache, warmup model) took N s` and
`Loading weights took N s` are the two numbers to watch.
