"""Architecture registry: timm, Hugging Face and custom backbones behind one interface.

A :class:`Backbone` hides where a network came from so the heads, the feature
cache and the benchmarks see one shape of thing: a normalisation recipe, a
``forward`` that returns named features and a ``pooled`` global vector.

Sources (the prefix picks the loader; a bare name means timm):

* ``timm:<model>`` wraps a plain ``timm.create_model(num_classes=0)`` and
  reads the stages through ``forward_intermediates`` (every curated model has
  it in timm 1.0.29). Kind ``"pyramid"``: ``forward`` returns the CNN stages
  keyed by stride (``c2`` = stride 4 ... ``c5`` = stride 32; a stride-2 stem
  output, which ResNet and MobileNetV4 expose, is dropped so every pyramid
  has the same keys). ``pooled`` is global average pooling over the last
  stage; for ConvNeXt (v1 and v2) it is followed by the checkpoint's own
  ``head.norm`` LayerNorm, so the vector equals timm's pre-logits embedding
  (``forward_head(pre_logits=True)``) and a zero-shot cosine comparison with
  the robot or with timm output is meaningful. A ``features_only`` wrapper
  would have dropped that norm with the head, which is why it is not used.
* ``hf:<repo-or-local-dir>`` loads a DINOv3 ViT through ``transformers``.
  Kind ``"tokens"``: ``forward`` returns the patch tokens without the CLS and
  the four register tokens (as ``bebop_vision/places/encoder.py`` does), the
  CLS vector and the token grid; ``pooled`` is the CLS vector.
* ``custom:<module>:<attr>`` imports a factory with :mod:`importlib`;
  :func:`register` binds a factory to a short name in-process.

:func:`get` returns every backbone in eval mode whatever the source (timm
builds in train mode, ``transformers`` ``from_pretrained`` in eval); a
trainer flips it with ``train()`` like any module.

``torch`` is imported at module level here on purpose (CONTRACT.md allows it
for this module); ``timm``, ``transformers`` and ``cv2`` are imported lazily
so listing :func:`available` costs nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

#: ImageNet statistics; what every curated checkpoint was trained with and
#: the fallback when a source does not state its own.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: The two DINOv3 ViTs are gated on the Hub, so their architecture configs
#: are kept in code (as the place encoder does) and used when neither the
#: weights directory nor the Hub can provide ``config.json``. The ViT-L
#: values were read from the repo and verified against the checkpoint
#: (303.1 M params); ViT-S is the same recipe at the published ViT-S width
#: (21.6 M params), not yet verified against its checkpoint.
_DINOV3_VITL16 = dict(
    hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
    patch_size=16, image_size=224, num_register_tokens=4, rope_theta=100.0,
    use_gated_mlp=False, mlp_bias=True, key_bias=False, proj_bias=True, query_bias=True,
    value_bias=True, layerscale_value=1.0, pos_embed_rescale=2.0,
)
_DINOV3_VITS16 = dict(_DINOV3_VITL16, hidden_size=384, num_hidden_layers=12,
                      num_attention_heads=6, intermediate_size=1536)
KNOWN_HF_CONFIGS: Dict[str, dict] = {
    "facebook/dinov3-vitl16-pretrain-lvd1689m": _DINOV3_VITL16,
    "facebook/dinov3-vits16-pretrain-lvd1689m": _DINOV3_VITS16,
}

_DINOV3_LICENCE = ("DINOv3 licence (non-commercial research terms, gated download: "
                   "accept on huggingface.co/facebook before pulling weights)")
_GRN_NOTE = ("ConvNeXt-V2 GRN divides by a spatial L2 norm with eps 1e-6; the norm "
             "overflows fp16 on large activations, so benchmark in fp16 before trusting "
             "an fp16 TensorRT engine")

#: Curated starting points. ``params_m`` is the backbone-only count measured
#: on this venv (timm ``features_only``, classifier head excluded; DINOv3 the
#: whole ViT) so the column compares what would ship on the Thor.
CURATED: List[dict] = [
    dict(name="timm:convnext_tiny.dinov3_lvd1689m", source="timm", params_m=27.8, dinov3=True,
         url="https://huggingface.co/timm/convnext_tiny.dinov3_lvd1689m",
         notes="ConvNeXt-T distilled from DINOv3 ViT-7B; strongest small pyramid. " + _DINOV3_LICENCE),
    dict(name="timm:convnext_small.dinov3_lvd1689m", source="timm", params_m=49.5, dinov3=True,
         url="https://huggingface.co/timm/convnext_small.dinov3_lvd1689m",
         notes="ConvNeXt-S DINOv3 student; the larger sibling when latency allows. " + _DINOV3_LICENCE),
    dict(name="timm:convnextv2_atto.fcmae_ft_in1k", source="timm", params_m=3.4, dinov3=False,
         url="https://huggingface.co/timm/convnextv2_atto.fcmae_ft_in1k",
         notes="Smallest ConvNeXt-V2 (FCMAE pretrain, IN-1k finetune). " + _GRN_NOTE),
    dict(name="timm:convnextv2_nano.fcmae_ft_in22k_in1k", source="timm", params_m=15.0, dinov3=False,
         url="https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k",
         notes="ConvNeXt-V2 nano with IN-22k pretraining. " + _GRN_NOTE),
    dict(name="timm:efficientvit_b1.r224_in1k", source="timm", params_m=4.6, dinov3=False,
         url="https://huggingface.co/timm/efficientvit_b1.r224_in1k",
         notes="MIT EfficientViT (linear attention, 4-stage pyramid), built for edge TensorRT."),
    dict(name="timm:mobilenetv4_conv_medium.e500_r256_in1k", source="timm", params_m=7.2, dinov3=False,
         url="https://huggingface.co/timm/mobilenetv4_conv_medium.e500_r256_in1k",
         notes="Conv-only MobileNetV4; last stage is 960 channels wide, trained at 256 px."),
    dict(name="timm:resnet18.a1_in1k", source="timm", params_m=11.2, dinov3=False,
         url="https://huggingface.co/timm/resnet18.a1_in1k",
         notes="The baseline everyone can run; 5-stage pyramid (stem stride 2 dropped)."),
    dict(name="timm:resnet50.a1_in1k", source="timm", params_m=23.5, dinov3=False,
         url="https://huggingface.co/timm/resnet50.a1_in1k",
         notes="ResNet-50 with the timm A1 recipe; c5 is 2048 channels."),
    dict(name="hf:facebook/dinov3-vits16-pretrain-lvd1689m", source="hf", params_m=21.6, dinov3=True,
         url="https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m",
         notes="DINOv3 ViT-S/16 (4 register tokens), dense tokens at stride 16. " + _DINOV3_LICENCE),
    dict(name="hf:facebook/dinov3-vitl16-pretrain-lvd1689m", source="hf", params_m=303.1, dinov3=True,
         url="https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
         notes="DINOv3 ViT-L/16, the place encoder's model (~46 ms fp32 on the Thor at 224). "
               + _DINOV3_LICENCE),
]


@dataclass
class FeatureInfo:
    """One named feature map: its stride w.r.t. the input and its channels."""

    name: str
    stride: int
    channels: int


# --------------------------------------------------------------------------
# Base class
# --------------------------------------------------------------------------

class Backbone(nn.Module):
    """A feature extractor with a known normalisation and a global vector.

    Subclasses set the public attributes in ``__init__`` and implement
    :meth:`forward` and :meth:`pooled`. ``id`` is the stable string that keys
    feature caches and run manifests: the source-qualified name plus
    ``@<sha8>`` of the weights file when one was loaded from disk.
    """

    id: str
    kind: str
    feature_info: List[FeatureInfo]
    embed_dim: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    input_range: str = "01"

    def __init__(self, id: str, kind: str, feature_info: Sequence[FeatureInfo], embed_dim: int,
                 mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
        super().__init__()
        if kind not in ("pyramid", "tokens"):
            raise ValueError(f"kind must be 'pyramid' or 'tokens', got {kind!r}")
        self.id = str(id)
        self.kind = kind
        self.feature_info = list(feature_info)
        self.embed_dim = int(embed_dim)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        self.input_range = "01"

    # -- to implement ------------------------------------------------------
    def forward(self, x: Tensor) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def pooled(self, feats: Dict[str, Any]) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- shared ------------------------------------------------------------
    def preprocess(self, bgr: np.ndarray, size: Optional[Tuple[int, int]] = None) -> Tensor:
        """A BGR uint8 ``[H, W, 3]`` frame -> normalised ``[1, 3, H, W]`` float32.

        ``size`` is ``(W, H)`` as cv2 takes it. Bilinear resize, like the
        place encoder, so lab features match what the robot computes.
        """
        import cv2

        if bgr is None:
            raise ValueError("preprocess got None (missing frame)")
        img = np.asarray(bgr)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected [H, W, 3] BGR, got {img.shape}")
        if size is not None and (int(size[0]), int(size[1])) != (img.shape[1], img.shape[0]):
            img = cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_LINEAR)
        x = img[:, :, ::-1].astype(np.float32)
        if img.dtype == np.uint8:
            x /= 255.0
        x = (x - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]

    def stage_modules(self) -> List[nn.Module]:
        """Trainable units from shallow to deep, for LP-FT style partial
        unfreezing (``unfreeze_blocks`` counts from the end of this list).
        Default: every direct child, in registration order."""
        return [m for _, m in self.named_children()]

    @property
    def params_m(self) -> float:
        return sum(p.numel() for p in self.parameters()) / 1e6

    def extra_repr(self) -> str:
        return f"id={self.id!r}, kind={self.kind!r}, embed_dim={self.embed_dim}"


# --------------------------------------------------------------------------
# timm
# --------------------------------------------------------------------------

def _stride_key(stride: int) -> str:
    """Stride 4 -> ``c2``, 8 -> ``c3``, 16 -> ``c4``, 32 -> ``c5``."""
    return f"c{int(round(math.log2(stride)))}"


class TimmBackbone(Backbone):
    """A plain timm model (``num_classes=0``) as a pyramid Backbone.

    ``weights`` is a local checkpoint file (``.pth``/``.pt``/``.safetensors``)
    that replaces the Hub download: a plain timm state dict (what
    ``bb.net.state_dict()`` saves), optionally wrapped under ``state_dict`` /
    ``model`` / ``state_dict_ema``, with ``module.`` (DataParallel) or
    ``net.`` (``Backbone.state_dict()``) prefixes and FeatureListNet's
    flattened names (``stages_0.`` for ``stages.0.``) accepted. Classifier
    weights the headless model has no slot for are dropped; anything else
    missing or unexpected is an error. A ``HeadModel`` checkpoint is not a
    backbone checkpoint.
    """

    def __init__(self, model_name: str, pretrained: bool = True,
                 weights: Optional[str | Path] = None, **kw: Any):
        import timm
        from timm.models import FeatureInfo as TimmFeatureInfo

        if weights is not None:
            # The file replaces the Hub download: build untrained, then load.
            pretrained = False
        net = timm.create_model(model_name, pretrained=pretrained, num_classes=0, **kw)
        if not hasattr(net, "forward_intermediates"):
            raise ValueError(f"{model_name}: this timm model has no forward_intermediates(); "
                             "the lab needs it to read the pyramid stages")
        if weights is not None:
            _load_timm_weights(net, weights, model_name)

        raw_info = list(net.feature_info)
        reductions = [int(d["reduction"]) for d in raw_info]
        channels = [int(d["num_chs"]) for d in raw_info]
        # Keep strides >= 4 so every pyramid has exactly c2..c5; when two
        # stages share a stride the deeper one wins (it is the richer map).
        keep: Dict[str, int] = {}
        for i, r in enumerate(reductions):
            if r >= 4:
                keep[_stride_key(r)] = i
        if not keep:
            raise ValueError(f"{model_name}: no feature stage with stride >= 4")
        info = [FeatureInfo(name=k, stride=reductions[i], channels=channels[i])
                for k, i in keep.items()]
        cfg = getattr(net, "pretrained_cfg", {}) or {}
        ident = f"timm:{model_name}"
        if weights is not None:
            ident += "@" + _sha8(weights)
        super().__init__(ident, "pyramid", info, embed_dim=info[-1].channels,
                         mean=cfg.get("mean", IMAGENET_MEAN), std=cfg.get("std", IMAGENET_STD))
        self.model_name = model_name
        self.net = net
        self._keep = keep
        self._indices = list(keep.values())
        #: timm's own view of the kept stages (``.reduction()``, ``.channels()``).
        self.timm_feature_info = TimmFeatureInfo(raw_info, tuple(self._indices))
        self.input_size = tuple(cfg.get("input_size", (3, 224, 224))[1:][::-1])  # (W, H)

    @property
    def pool_norm(self) -> Optional[nn.LayerNorm]:
        """The checkpoint's post-pool LayerNorm (ConvNeXt ``head.norm``), or
        None for plain-GAP models. A property, not a second registration: the
        module already lives under ``net`` and must appear once in the state
        dict."""
        norm = getattr(getattr(self.net, "head", None), "norm", None)
        return norm if isinstance(norm, nn.LayerNorm) else None

    def forward(self, x: Tensor) -> Dict[str, Any]:
        outs = self.net.forward_intermediates(x, indices=self._indices, intermediates_only=True)
        return dict(zip(self._keep, outs))

    def pooled(self, feats: Dict[str, Any]) -> Tensor:
        last = feats[self.feature_info[-1].name]
        v = last.mean(dim=(2, 3))
        norm = self.pool_norm
        if norm is not None:
            # timm's NormMlpClassifierHead: pool, then LayerNorm2d over the
            # channels of the [B, C, 1, 1] map, which on [B, C] is layer_norm.
            v = nn.functional.layer_norm(v.float(), (v.shape[1],), norm.weight, norm.bias,
                                         norm.eps).to(v.dtype)
        return v

    def stage_modules(self) -> List[nn.Module]:
        # Direct children in forward order (stem, stages, head...), with the
        # stage container opened up so "unfreeze the last block" means one
        # stage and not the whole trunk. ResNet keeps layer1..4 as children.
        mods: List[nn.Module] = []
        for name, m in self.net.named_children():
            if name in ("stages", "blocks", "layers") and isinstance(m, (nn.Sequential, nn.ModuleList)):
                mods.extend(list(m))
            else:
                mods.append(m)
        return mods


# --------------------------------------------------------------------------
# Hugging Face (DINOv3 ViTs)
# --------------------------------------------------------------------------

class HFBackbone(Backbone):
    """A ``transformers`` ViT (DINOv3) as a tokens Backbone.

    ``name`` is a Hub repo id or a local directory. ``weights`` may point at
    a directory holding ``model.safetensors`` (with or without
    ``config.json``) as the place encoder's weights dir does; the config then
    comes from that directory, the Hub, or :data:`KNOWN_HF_CONFIGS`, in that
    order.
    """

    def __init__(self, name: str, pretrained: bool = True,
                 weights: Optional[str | Path] = None, **kw: Any):
        from transformers import AutoConfig, AutoModel

        weights_dir = Path(weights) if weights is not None else None
        config = self._resolve_config(name, weights_dir, AutoConfig)
        if pretrained or weights_dir is not None:
            src = str(weights_dir) if weights_dir is not None else name
            net = AutoModel.from_pretrained(src, config=config, **kw)
        else:
            net = AutoModel.from_config(config, **kw)
        # No eval() here: nn.Module.__init__ has not run yet, so it would
        # leave the wrapper's own flag at True while the ViT's is False and a
        # later train() / eval() round trip would flip the ViT. get() sets
        # the mode on the whole module once it is built.

        patch = int(getattr(config, "patch_size", 16))
        hidden = int(config.hidden_size)
        ident = f"hf:{name}"
        wfile = _weights_file(weights_dir) if weights_dir is not None else None
        if wfile is not None:
            ident += "@" + _sha8(wfile)
        super().__init__(ident, "tokens", [FeatureInfo("tokens", patch, hidden)], embed_dim=hidden)
        self.net = net
        self.patch_size = patch
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.input_size = (int(getattr(config, "image_size", 224)),) * 2

    @staticmethod
    def _resolve_config(name: str, weights_dir: Optional[Path], AutoConfig):
        candidates = []
        if weights_dir is not None and (weights_dir / "config.json").exists():
            candidates.append(str(weights_dir))
        candidates.append(name)
        last_err: Optional[Exception] = None
        for src in candidates:
            try:
                return AutoConfig.from_pretrained(src)
            except Exception as e:  # gated repo, offline, not a dir
                last_err = e
        if name in KNOWN_HF_CONFIGS:
            from transformers import DINOv3ViTConfig
            return DINOv3ViTConfig(**KNOWN_HF_CONFIGS[name])
        raise RuntimeError(f"could not resolve a config for hf:{name}: {last_err}")

    def forward(self, x: Tensor) -> Dict[str, Any]:
        p = self.patch_size
        if x.shape[2] % p or x.shape[3] % p:
            # The patch conv would floor the remainder away, so the token grid
            # would no longer cover the image edge; make the caller resize.
            raise ValueError(f"input {tuple(x.shape[2:])} is not a multiple of the patch size {p}")
        out = self.net(pixel_values=x)
        h = out.last_hidden_state
        skip = 1 + self.num_register_tokens
        grid = (x.shape[2] // p, x.shape[3] // p)
        tokens = h[:, skip:]
        if tokens.shape[1] != grid[0] * grid[1]:
            raise RuntimeError(f"token count {tokens.shape[1]} != grid {grid}: unexpected "
                               f"special-token layout (registers={self.num_register_tokens})")
        return {"tokens": tokens, "cls": h[:, 0], "grid": grid}

    def pooled(self, feats: Dict[str, Any]) -> Tensor:
        return feats["cls"]

    def stage_modules(self) -> List[nn.Module]:
        # Embeddings first, then each transformer layer; the final norm rides
        # with the last layer so unfreezing "one block" includes it. The layer
        # list lives at ``model.layer`` (transformers 5.16 DINOv3) or
        # ``encoder.layer`` (other HF ViTs): take the first ModuleList found.
        layers: List[nn.Module] = []
        for m in self.net.modules():
            if isinstance(m, nn.ModuleList) and len(m) > 0:
                layers = list(m)
                break
        if not layers:
            return super().stage_modules()
        mods: List[nn.Module] = []
        emb = getattr(self.net, "embeddings", None)
        if emb is not None:
            mods.append(emb)
        norm = getattr(self.net, "norm", None) or getattr(self.net, "layernorm", None)
        if isinstance(norm, nn.Module):
            layers[-1] = nn.ModuleList([layers[-1], norm])
        mods.extend(layers)
        return mods


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

Factory = Callable[..., Backbone]
_REGISTRY: Dict[str, Factory] = {}


def register(name: str, factory: Factory) -> None:
    """Bind ``factory(pretrained, weights, **kw) -> Backbone`` to ``name``."""
    if not callable(factory):
        raise TypeError("factory must be callable")
    _REGISTRY[str(name)] = factory


def available() -> List[dict]:
    """The curated list (copies) plus any in-process registered factories."""
    out = [dict(e) for e in CURATED]
    for name in sorted(_REGISTRY):
        out.append(dict(name=name, source="custom", params_m=None, dinov3=False, url="",
                        notes="registered in-process via backbones.register"))
    return out


def get(name: str, pretrained: bool = True, weights: Optional[str | Path] = None, **kw: Any) -> Backbone:
    """Build a Backbone from ``timm:x`` / bare timm name, ``hf:x``,
    ``custom:<module>:<attr>`` or a :func:`register`-ed name. The result is
    in eval mode (a frozen feature extractor until a trainer says otherwise)."""
    name = str(name)
    return _build(name, pretrained, weights, kw).eval()


def _build(name: str, pretrained: bool, weights: Optional[str | Path], kw: Dict[str, Any]) -> Backbone:
    if name in _REGISTRY:
        bb = _REGISTRY[name](pretrained, weights, **kw)
        return _check_custom(bb, name)
    source, sep, rest = name.partition(":")
    if sep and source in ("timm", "hf", "custom") and not rest:
        raise ValueError(f"empty backbone spec after {source}: in {name!r}")
    if source == "custom" and rest:
        mod_name, _, attr = rest.rpartition(":")
        if not mod_name or not attr:
            raise ValueError(f"custom source must be custom:<module>:<attr>, got {name!r}")
        factory = getattr(importlib.import_module(mod_name), attr)
        bb = factory(pretrained, weights, **kw)
        return _check_custom(bb, name)
    if source == "hf" and rest:
        return HFBackbone(rest, pretrained=pretrained, weights=weights, **kw)
    if source == "timm" and rest:
        return TimmBackbone(rest, pretrained=pretrained, weights=weights, **kw)
    # No known prefix: a bare timm name.
    return TimmBackbone(name, pretrained=pretrained, weights=weights, **kw)


def _check_custom(bb: Any, name: str) -> Backbone:
    if not isinstance(bb, Backbone):
        raise TypeError(f"{name}: factory returned {type(bb).__name__}, not a Backbone")
    return bb


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _sha8(path: str | Path) -> str:
    """First 8 hex digits of the file's sha1 (whole file: a checkpoint's
    identity must change when any weight does)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


_FLAT_KEY = re.compile(r"^([A-Za-z]+)_(\d+)\.")


def _load_timm_weights(net: nn.Module, weights: str | Path, model_name: str) -> None:
    """Load a local checkpoint into a headless timm model with clear errors
    (see :class:`TimmBackbone` for the accepted layouts)."""
    p = Path(weights)
    if not p.is_file():
        raise FileNotFoundError(f"{model_name}: weights {p} is not a checkpoint file "
                                "(.pth/.pt/.safetensors with a timm state dict)")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(p))
    else:
        sd = torch.load(p, map_location="cpu", weights_only=True)
    for wrapper in ("state_dict_ema", "model_ema", "state_dict", "model"):
        if isinstance(sd, dict) and isinstance(sd.get(wrapper), dict):
            sd = sd[wrapper]
            break
    if not isinstance(sd, dict) or not sd:
        raise ValueError(f"{model_name}: {p} holds no state dict")
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    if all(k.startswith("net.") for k in sd):
        sd = {k[len("net."):]: v for k, v in sd.items()}

    model_keys = set(net.state_dict())
    classifier = (getattr(net, "pretrained_cfg", {}) or {}).get("classifier")
    cls_prefixes = tuple(f"{c}." for c in ((classifier,) if isinstance(classifier, str) else (classifier or ())))
    fixed: Dict[str, Tensor] = {}
    for k, v in sd.items():
        if k not in model_keys:
            flat = _FLAT_KEY.sub(lambda m: f"{m.group(1)}.{m.group(2)}.", k)
            if flat in model_keys:
                k = flat
            elif cls_prefixes and k.startswith(cls_prefixes):
                continue  # classifier weights; num_classes=0 has no slot for them
        fixed[k] = v
    result = net.load_state_dict(fixed, strict=False)
    # A features_only (FeatureListNet) checkpoint never had the head, so its
    # post-pool norm stays at init; the trunk is complete, which is what the
    # file promised. Anything missing in the trunk is a mismatch.
    head_only = all(k.startswith("head.") for k in result.missing_keys)
    if result.unexpected_keys or not head_only:
        def few(keys: Sequence[str]) -> str:
            return ", ".join(list(keys)[:5]) + (" ..." if len(keys) > 5 else "")
        raise ValueError(f"{model_name}: {p} does not match the model; missing "
                         f"[{few(result.missing_keys)}], unexpected [{few(result.unexpected_keys)}]")
    if result.missing_keys:
        logging.getLogger(__name__).warning(
            "%s: %s has no %s (a features_only checkpoint); pooled() uses the init norm",
            model_name, p, ", ".join(result.missing_keys))


def _weights_file(d: Path) -> Optional[Path]:
    if d.is_file():
        return d
    for n in ("model.safetensors", "pytorch_model.bin"):
        if (d / n).exists():
            return d / n
    return None
