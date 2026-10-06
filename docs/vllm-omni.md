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

## What is already in our favour
- `enforce_eager: true` on every stage → **torch.compile and CUDA-graph capture
  are off**, so there is no compile step (the one exception is `code2wav`,
  which captures 11 small CUDA graphs for its decoder — ~0.7 min).
- The flashinfer CUTLASS `.cu` instantiations are baked into the image.

## Levers, best first

1. **Persist the kernel/JIT caches on the host (done).**
   `ContainerServer` and `bebop-models` now bind-mount
   `/home/bebop/.cache/flashinfer` → `/root/.cache/flashinfer` and
   `…/.cache/torch_extensions`. The second start reuses the compiled kernels
   and tuning results instead of redoing them. No inference cost.
   - Verify the host dir is populated after a run:
     `sudo du -sh /home/bebop/.cache/flashinfer`.

2. **Keep it resident; don't cold-start on demand.** The real UX fix. Have the
   supervisor start `omni-vllm` at boot (or when the app opens) and idle-stop
   only after a long timeout — a ~28 min start is invisible if it happens
   before the operator needs it. See `docs/models.md`.

3. **Skip flashinfer autotune** when startup matters more than peak throughput:
   pass `-O0` (`--optimization-level 0`) to the server. vLLM maps O0 to
   `kernel_config.enable_flashinfer_autotune: false` (see
   `vllm/config/vllm.py`), which removes the ~4 min autotune. It also turns off
   the norm/act quant fusions, so expect a modest decode slowdown — measure
   with a voice turn before committing.
   - In `config/models_supervisor.yaml` add `-O0` to the backend's `serve_args`.

4. **Shrink the warmup window.** `thinker` uses `max_num_batched_tokens: 32768`
   in `qwen3_omni_1gpu.yaml`; the profile/warmup scales with it. Voice turns
   are short, so dropping it (e.g. 8192-16384) cuts the 307 s profile phase at
   the cost of long-prompt prefill latency. Similarly `gpu_memory_utilization`
   drives the KV-cache profiling.

5. **Warm the page cache before start.** `vmtouch`/`cat` the checkpoint after
   boot so the first stage's read is warm. Bounded by free RAM (~12 GB here),
   so only partial.

6. **Do not enable torch.compile/CUDA graphs** to "speed up" startup — they add
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
