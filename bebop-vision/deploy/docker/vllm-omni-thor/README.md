# vLLM-Omni on Jetson AGX Thor (sm_110)

This directory builds a container that runs **vLLM 0.20 + vLLM-Omni 0.20** on the
Thor and serves **Qwen3-Omni-30B-A3B** (text + speech out). It is a **spike /
future option**, not the shipping stack — see the notes at the bottom.

Verified on the Thor: `vllm serve … --omni` reaches `ready`, and a chat request
returns text *and* a 24 kHz audio payload.

## Build

```bash
docker build -t bebop-vllm-omni-thor bebop-vision/deploy/docker/vllm-omni-thor
```

The Dockerfile starts from NVIDIA's Thor vLLM image (toolchain + torch) and:
- installs the Thor `vllm 0.20.0+cu130` wheel (+ `torch 2.11`) and `vllm-omni==0.20.0`
  from the Jetson index, with real PyPI as an extra index;
- patches `vllm.platforms.cuda.get_device_capability` to fall back to torch on
  the Tegra NVML (`NVMLError_InvalidArgument`);
- patches vllm-omni's `-1e9` FP16 logit mask (overflows `c10::Half`) to `-65504.0`.

## Model

Use a **vLLM/ModelOpt-compatible NVFP4** checkpoint — **not** the Edge-LLM
TensorRT NVFP4 (it produces a `BFloat16 vs Half` dtype mismatch). Tested:

```
hf download catplusplus/Qwen3-Omni-30B-A3B-Instruct-NVFP4-talker-safe \
  --local-dir /home/bebop/qwen3-omni-talker-safe
```

## Run (single GPU)

The default vllm-omni deploy layout puts the talker + code2wav on a **second
GPU**; on one Thor use `qwen3_omni_1gpu.yaml` (all stages on device 0,
`enforce_eager`, tightened memory). Persisting `~/.cache/vllm` keeps the
`torch.compile`/autotune artifacts across restarts.

```bash
docker run -d --name vllm-omni-serve --runtime nvidia --network host \
  -v /home/bebop/qwen3-omni-talker-safe:/models/qwen3-omni:ro \
  -v $PWD/qwen3_omni_1gpu.yaml:/cfg/qwen3_omni_1gpu.yaml:ro \
  -v ~/.cache/vllm:/root/.cache/vllm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  bebop-vllm-omni-thor \
  /models/qwen3-omni --omni --deploy-config /cfg/qwen3_omni_1gpu.yaml \
  --host 0.0.0.0 --port 8101 --max-model-len 16384 \
  --init-timeout 3600 --stage-init-timeout 3600
```

Then: `curl localhost:8101/v1/models`, and `POST /v1/chat/completions` (the
response carries an `audio` object) or the `/v1/realtime` WebSocket.

## Cost / caveats vs TensorRT Edge-LLM

- **Startup is slow**: ~20–40 min cold (model load + `torch.compile` + FlashInfer
  autotune + CUDA-graph capture across 3 stages), a few minutes warm. Edge-LLM
  uses prebuilt TensorRT engines and is ready in ~1–2 min.
- **Heavier**: the image is ~54 GB; a single-GPU stage layout is required.
- **Checkpoint compatibility matters** (see above).
- Use it for the **broader model coverage / duplex roadmap**; keep
  **Edge-LLM (Omni + Cascade)** as the shipping stack.
