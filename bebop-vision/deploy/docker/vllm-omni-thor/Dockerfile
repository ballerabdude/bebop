# vLLM-Omni for Jetson AGX Thor (sm_110)
#
# The prebuilt NVIDIA Thor vLLM image ships vLLM 0.19.0, which sits in a dead
# zone: vllm-omni releases need vLLM <=0.18 (0.18) or >=0.20 (0.20+). But the
# Jetson index (pypi.jetson-ai-lab.io/sbsa/cu130) also publishes a **prebuilt
# vLLM 0.20.0+cu130 wheel and torch 2.11.0**, so we upgrade in place — no
# source build. vllm-omni 0.20 pairs with vLLM 0.20 and has no transformers
# pin (it works with the image's transformers too).
#
# Build (on the Thor):
#   docker build -t bebop-vllm-omni-thor \
#     bebop-vision/deploy/docker/vllm-omni-thor
#
# Run:
#   docker run --rm -it --runtime nvidia --network host \
#     -v ~/bebop/bebop-vision/weights/qwen3-omni-30b:/models/qwen3-omni:ro \
#     bebop-vllm-omni-thor \
#     /models/qwen3-omni --omni --port 8101

FROM ghcr.io/nvidia-ai-iot/vllm:latest-jetson-thor

ARG VLLM_VERSION=0.20.0+cu130
ARG VLLM_OMNI_VERSION=0.20.0

# Use real PyPI for packages the Jetson index lacks (diffusers, …) and drop the
# image's pip constraint so torch can move 2.10 -> 2.11 (needed by vLLM 0.20).
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
    PIP_APPEND_CONSTRAINT= \
    PIP_CONSTRAINT= \
    UV_CONSTRAINT=

# Upgrade to the Thor vLLM 0.20 wheel (+ matching torch) and add vllm-omni.
RUN pip install --no-cache-dir "vllm==${VLLM_VERSION}" "vllm-omni==${VLLM_OMNI_VERSION}"

# Thor NVML fix: `get_device_capability` resolves a physical device id that the
# Tegra NVML rejects (NVMLError_InvalidArgument), and only RuntimeError is
# caught. Fall back to torch's CUDA capability so import-time probes (e.g.
# cutlass_fp8_supported) work.
RUN python - <<'PY'
p = "/opt/venv/lib/python3.12/site-packages/vllm/platforms/cuda.py"
s = open(p).read()
old = """        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except RuntimeError:
            return None"""
new = """        try:
            physical_device_id = cls.device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return DeviceCapability(major=major, minor=minor)
        except Exception:  # bebop: Thor NVML raises NVMLError, not RuntimeError
            try:
                import torch as _torch
                major, minor = _torch.cuda.get_device_capability(device_id)
                return DeviceCapability(major=major, minor=minor)
            except Exception:
                return None"""
assert old in s, "cuda.py patch anchor not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched cuda.py get_device_capability for Thor")
PY

# vllm-omni writes -1e9 into FP16 logits (`masked_fill_`), which overflows
# c10::Half. Use the FP16 minimum instead (a strong, representable suppression).
RUN F=/opt/venv/lib/python3.12/site-packages/vllm_omni/model_executor/models/qwen3_omni/qwen3_omni.py; \
    if [ -f "$F" ]; then sed -i 's/self\.suppressed_tokens\.unsqueeze(0), -1e9/self.suppressed_tokens.unsqueeze(0), -65504.0/' "$F"; fi

RUN python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__)
try:
    import vllm_omni
    print("vllm_omni", getattr(vllm_omni, "__version__", "?"))
except Exception as e:  # noqa: BLE001
    print("vllm_omni import failed:", type(e).__name__, e)
PY

# Serve with `--omni` for audio output; pass the model path plus flags:
#   docker run ... bebop-vllm-omni-thor /models/qwen3-omni --omni --port 8101
ENTRYPOINT ["vllm", "serve"]
