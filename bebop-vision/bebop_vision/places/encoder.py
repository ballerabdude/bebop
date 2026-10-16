"""DINOv3 place encoder: a camera view -> an L2-normalised descriptor.

Backed by the on-robot DINOv3 ViT-L/16 weights
(`bebop-vision/weights/model.safetensors`). The Hugging Face repo is gated,
so the architecture config is constructed in code — the values were read
once from the repo's `config.json` and verified by loading the checkpoint
(303.1 M params) — and the weights load straight from the weights directory.
No network, no HF token, no `config.json` on disk.

Preprocessing is numpy/cv2 rather than the transformers image processor,
which would drag in torchvision + Pillow (absent from the recorder venv):
resize to 224x224, RGB, ImageNet mean/std. The CLS token is the descriptor.

Measured on the Thor: ~46 ms/embedding (fp32, CUDA). That is already well
under the agreed cadence (embed every few seconds), so this is the shipped
backend; an ONNX/TensorRT backend is a later optimisation behind the same
`encode_jpeg`/`encode_bgr` interface.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights"

# DINOv3 ViT-L/16 (facebook/dinov3-vitl16-pretrain-lvd1689m). These are the
# repo's own config.json values; keeping them in code avoids a gated fetch.
DINOV3_CONFIG = dict(
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    patch_size=16,
    image_size=224,
    num_register_tokens=4,
    rope_theta=100.0,
    use_gated_mlp=False,
    mlp_bias=True,
    key_bias=False,
    proj_bias=True,
    query_bias=True,
    value_bias=True,
    layerscale_value=1.0,
    pos_embed_rescale=2.0,
)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@runtime_checkable
class PlaceEncoder(Protocol):
    """Anything that can turn a view into a fixed-length descriptor."""

    dim: int

    def encode_bgr(self, bgr: np.ndarray) -> np.ndarray: ...

    def encode_jpeg(self, jpeg: bytes) -> np.ndarray: ...


def find_weights_dir(weights_dir: Path | str | None = None) -> Path:
    """Locate a directory holding the DINOv3 weights.

    Prefers `weights/dinov3-vitl16/`, then the shared `weights/` (where the
    catalog drops the bare `model.safetensors`).
    """
    base = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR
    for candidate in (base / "dinov3-vitl16", base):
        if not candidate.is_dir():
            continue
        if any(
            (candidate / n).exists()
            for n in ("model.safetensors", "pytorch_model.bin")
        ):
            return candidate
    raise FileNotFoundError(
        f"DINOv3 weights not found under {base} (expected model.safetensors); "
        "download them from the app's Models screen (DINOv3 ViT-L/16)."
    )


class DinoV3Encoder:
    """Lazy-loading DINOv3 descriptor extractor.

    The model is loaded on first use (a few seconds + ~1.2 GB) so importing
    this module — or starting the service before the camera is up — stays
    cheap. Inference is serialised by a lock: the recognizer thread and any
    HTTP handler share one model.
    """

    def __init__(
        self,
        weights_dir: Path | str | None = None,
        *,
        device: str | None = None,
        image_size: int = 224,
    ):
        self.weights_dir = weights_dir
        self.image_size = int(image_size)
        self._device = device
        self._model = None
        self._torch = None
        self._lock = threading.Lock()
        self.dim = int(DINOV3_CONFIG["hidden_size"])

    # -- lifecycle --------------------------------------------------------

    def _load(self):
        import torch
        from transformers import AutoModel, DINOv3ViTConfig

        weights = find_weights_dir(self.weights_dir)
        cfg = DINOv3ViTConfig(**DINOV3_CONFIG)
        model = AutoModel.from_pretrained(str(weights), config=cfg)
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)
        self._torch = torch
        self._model = model
        self._device = device
        return model

    def warmup(self) -> None:
        """Force the load now (call from a background thread at startup)."""
        with self._lock:
            if self._model is None:
                self._load()

    @property
    def device(self) -> str:
        return self._device or "unloaded"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- inference --------------------------------------------------------

    def _preprocess(self, bgr: np.ndarray):
        import cv2

        x = cv2.resize(bgr, (self.image_size, self.image_size),
                       interpolation=cv2.INTER_LINEAR)
        x = x[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, [0,1]
        x = (x - _MEAN) / _STD
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def encode_bgr(self, bgr: np.ndarray) -> np.ndarray:
        import torch

        if bgr is None:
            raise ValueError("encode_bgr got None")
        with self._lock:
            if self._model is None:
                self._load()
            px = torch.from_numpy(self._preprocess(bgr)).to(self._device)
            with torch.inference_mode():
                out = self._model(pixel_values=px)
            cls = out.last_hidden_state[:, 0].float().cpu().numpy().reshape(-1)
        return cls / (np.linalg.norm(cls) + 1e-12)

    def encode_jpeg(self, jpeg: bytes) -> np.ndarray:
        import cv2

        buf = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("could not decode JPEG")
        return self.encode_bgr(bgr)
