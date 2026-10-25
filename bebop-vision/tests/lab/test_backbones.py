"""backbones.py: registry, pyramids, pooling, preprocessing, ids.

Everything runs with ``pretrained=False`` on tiny inputs; no weights, no
network. The HF test builds a randomly initialised DINOv3-like model from a
small config saved to a temp dir and skips when transformers lacks the class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from bebop_vision.lab import backbones
from bebop_vision.lab.backbones import Backbone, FeatureInfo

PYRAMIDS = ["convnext_tiny", "resnet18", "efficientvit_b1", "mobilenetv4_conv_medium"]


@pytest.fixture(scope="module")
def pyramids():
    return {n: backbones.get(n, pretrained=False).eval() for n in PYRAMIDS}


# -- curated list ----------------------------------------------------------

def test_available_is_the_curated_list():
    names = [e["name"] for e in backbones.available()]
    for required in ("timm:convnext_tiny.dinov3_lvd1689m", "timm:convnext_small.dinov3_lvd1689m",
                     "timm:convnextv2_atto.fcmae_ft_in1k", "timm:convnextv2_nano.fcmae_ft_in22k_in1k",
                     "timm:efficientvit_b1.r224_in1k", "timm:mobilenetv4_conv_medium.e500_r256_in1k",
                     "timm:resnet18.a1_in1k", "timm:resnet50.a1_in1k",
                     "hf:facebook/dinov3-vits16-pretrain-lvd1689m",
                     "hf:facebook/dinov3-vitl16-pretrain-lvd1689m"):
        assert required in names
    for e in backbones.available():
        assert set(e) >= {"name", "source", "params_m", "notes", "dinov3", "url"}
        assert e["source"] in ("timm", "hf", "custom")
        if e["dinov3"]:
            assert "licence" in e["notes"].lower()
        if "convnextv2" in e["name"]:
            assert "GRN" in e["notes"] and "fp16" in e["notes"]
        if e["source"] != "custom":
            assert isinstance(e["params_m"], float) and e["params_m"] > 0
    # the list is a copy: callers cannot corrupt the registry
    backbones.available()[0]["name"] = "x"
    assert backbones.available()[0]["name"] != "x"


# -- timm pyramids ---------------------------------------------------------

@pytest.mark.parametrize("name", PYRAMIDS)
def test_pyramid_contract_keys_match_feature_info(name, pyramids):
    bb = pyramids[name]
    assert isinstance(bb, Backbone) and bb.kind == "pyramid"
    assert [f.name for f in bb.feature_info] == ["c2", "c3", "c4", "c5"]
    assert [f.stride for f in bb.feature_info] == [4, 8, 16, 32]
    x = torch.zeros(1, 3, 64, 96)
    with torch.inference_mode():
        feats = bb(x)
    assert set(feats) == {"c2", "c3", "c4", "c5"}
    for fi in bb.feature_info:
        t = feats[fi.name]
        assert t.shape[0] == 1 and t.shape[1] == fi.channels
        assert t.shape[2] == 64 // fi.stride and t.shape[3] == 96 // fi.stride
    # consistent with timm's own feature_info (same channels, same reductions)
    timm_info = {int(r): int(c) for r, c in zip(bb.timm_feature_info.reduction(),
                                                 bb.timm_feature_info.channels())}
    assert timm_info == {fi.stride: fi.channels for fi in bb.feature_info}
    # and with what timm's forward_intermediates hands out for every stage
    with torch.inference_mode():
        allfeats = bb.net.forward_intermediates(x, intermediates_only=True)
    for fi in bb.feature_info:
        assert any(torch.equal(t, feats[fi.name]) for t in allfeats)


@pytest.mark.parametrize("name", PYRAMIDS)
def test_pooled_dim_is_embed_dim(name, pyramids):
    bb = pyramids[name]
    assert bb.embed_dim == bb.feature_info[-1].channels
    with torch.inference_mode():
        v = bb.pooled(bb(torch.randn(2, 3, 64, 96)))
    assert v.shape == (2, bb.embed_dim)
    assert torch.isfinite(v).all()


def test_convnext_pooled_is_gap_then_layernorm(pyramids):
    bb = pyramids["convnext_tiny"]
    assert isinstance(bb.pool_norm, torch.nn.LayerNorm)
    with torch.inference_mode():
        feats = bb(torch.randn(1, 3, 64, 96))
        v = bb.pooled(feats)
    gap = feats["c5"].mean(dim=(2, 3))
    expect = torch.nn.functional.layer_norm(gap, (bb.embed_dim,), eps=1e-6)
    assert torch.allclose(v, expect, atol=1e-5)
    # plain GAP everywhere else
    r = pyramids["resnet18"]
    assert r.pool_norm is None
    with torch.inference_mode():
        f = r(torch.randn(1, 3, 64, 96))
        assert torch.allclose(r.pooled(f), f["c5"].mean(dim=(2, 3)))


def test_convnext_pooled_uses_the_checkpoints_head_norm():
    """The LayerNorm after GAP is the model's own ``head.norm``, so pooled()
    reproduces timm's pre-logits embedding for any (pretrained) affine, not
    only the identity init."""
    bb = backbones.get("convnext_tiny", pretrained=False)
    assert bb.pool_norm is bb.net.head.norm
    assert "pool_norm.weight" not in bb.state_dict()     # registered once, under net
    with torch.no_grad():
        bb.net.head.norm.weight.mul_(2.0)
        bb.net.head.norm.bias.add_(0.1)
    x = torch.randn(2, 3, 64, 96)
    with torch.inference_mode():
        feats = bb(x)
        v = bb.pooled(feats)
        final, _ = bb.net.forward_intermediates(x, intermediates_only=False)
        ref = bb.net.forward_head(final, pre_logits=True)
    assert torch.allclose(v, ref, atol=1e-6)
    identity = torch.nn.functional.layer_norm(feats["c5"].mean(dim=(2, 3)), (bb.embed_dim,), eps=1e-6)
    assert not torch.allclose(v, identity, atol=1e-2)   # the affine really is applied
    # same story for ConvNeXt-V2
    v2 = backbones.get("convnextv2_atto", pretrained=False)
    assert v2.pool_norm is v2.net.head.norm and v2.embed_dim == 320


def test_get_returns_eval_mode_for_every_source():
    for name in ("resnet18", "convnext_tiny", f"custom:{__name__}:make_toy"):
        bb = backbones.get(name, pretrained=False)
        assert not bb.training and all(not m.training for m in bb.modules())
        bb.train()
        assert all(m.training for m in bb.modules())


def test_get_accepts_prefixed_and_bare_timm_names():
    a = backbones.get("timm:resnet18", pretrained=False)
    b = backbones.get("resnet18", pretrained=False)
    assert a.id == b.id == "timm:resnet18"
    c = backbones.get("timm:resnet18.a1_in1k", pretrained=False)
    assert c.id == "timm:resnet18.a1_in1k"
    assert c.mean == (0.485, 0.456, 0.406) and c.std == (0.229, 0.224, 0.225)
    assert c.input_range == "01"
    with pytest.raises(ValueError):
        backbones.get("timm:", pretrained=False)


def test_id_strings_are_stable(pyramids):
    for name in PYRAMIDS:
        again = backbones.get(name, pretrained=False)
        assert again.id == pyramids[name].id == f"timm:{name}"
    assert "@" not in pyramids["resnet18"].id   # no weights file, no hash


def test_timm_local_weights_file_adds_sha8_and_loads(tmp_path):
    src = backbones.get("resnet18", pretrained=False)
    with torch.no_grad():
        for p in src.net.parameters():
            p.fill_(0.5)
    ckpt = tmp_path / "resnet18.pth"
    torch.save(src.net.state_dict(), ckpt)
    bb = backbones.get("resnet18", weights=ckpt)
    assert bb.id.startswith("timm:resnet18@") and len(bb.id.split("@")[1]) == 8
    assert all(torch.all(p == 0.5) for p in bb.net.parameters())
    # same bytes -> same id; different bytes -> different id
    assert backbones.get("resnet18", weights=ckpt).id == bb.id
    with torch.no_grad():
        for p in src.net.parameters():
            p.fill_(0.25)
    torch.save(src.net.state_dict(), tmp_path / "other.pth")
    assert backbones.get("resnet18", weights=tmp_path / "other.pth").id != bb.id


def test_timm_weights_with_dotted_names_flattened_names_and_prefixes(tmp_path):
    """ConvNeXt's top-level names are dotted (stages.0.blocks...), which is
    where a FeatureListNet state dict (stages_0...) or a Backbone.state_dict()
    (net....) differ from the plain timm layout; all three must load."""
    import timm

    src = backbones.get("convnext_tiny", pretrained=False)
    with torch.no_grad():
        for p in src.net.parameters():
            p.fill_(0.5)
    plain = tmp_path / "plain.pth"
    torch.save(src.net.state_dict(), plain)
    bb = backbones.get("convnext_tiny", weights=plain)
    assert bb.id.startswith("timm:convnext_tiny@")
    assert all(torch.all(p == 0.5) for p in bb.net.parameters())

    wrapped = tmp_path / "wrapped.pth"            # Backbone.state_dict() under a trainer wrapper
    torch.save({"state_dict": src.state_dict(), "epoch": 3}, wrapped)
    assert all(torch.all(p == 0.5) for p in backbones.get("convnext_tiny", weights=wrapped).net.parameters())

    flat_net = timm.create_model("convnext_tiny", pretrained=False, features_only=True)
    with torch.no_grad():
        for p in flat_net.parameters():
            p.fill_(0.25)
    flat = tmp_path / "features_only.pth"
    torch.save(flat_net.state_dict(), flat)
    bf = backbones.get("convnext_tiny", weights=flat)
    trunk = [p for n, p in bf.net.named_parameters() if not n.startswith("head.")]
    assert trunk and all(torch.all(p == 0.25) for p in trunk)
    assert torch.all(bf.net.head.norm.weight == 1.0)   # FeatureListNet never had the head norm

    # a classifier the headless model has no slot for is dropped, not an error
    with_fc = timm.create_model("convnext_tiny", pretrained=False, num_classes=10).state_dict()
    torch.save(with_fc, tmp_path / "with_fc.pth")
    backbones.get("convnext_tiny", weights=tmp_path / "with_fc.pth")

    # clear errors: a directory, a checkpoint for another architecture
    with pytest.raises(FileNotFoundError):
        backbones.get("convnext_tiny", weights=tmp_path)
    torch.save(backbones.get("resnet18", pretrained=False).net.state_dict(), tmp_path / "r18.pth")
    with pytest.raises(ValueError, match="does not match"):
        backbones.get("convnext_tiny", weights=tmp_path / "r18.pth")
    partial = {k: v for k, v in src.net.state_dict().items() if not k.startswith("stages.3.")}
    torch.save(partial, tmp_path / "partial.pth")
    with pytest.raises(ValueError, match="missing"):
        backbones.get("convnext_tiny", weights=tmp_path / "partial.pth")


def test_stage_modules_shallow_to_deep(pyramids):
    for name in PYRAMIDS:
        bb = pyramids[name]
        mods = bb.stage_modules()
        assert len(mods) >= 4 and all(isinstance(m, torch.nn.Module) for m in mods)
        params_total = sum(p.numel() for m in mods for p in m.parameters())
        assert params_total == sum(p.numel() for p in bb.net.parameters())
    # the stage container is opened up: four ConvNeXt stages are four units
    stages = [m for m in pyramids["convnext_tiny"].stage_modules()
              if m in list(pyramids["convnext_tiny"].net.stages)]
    assert len(stages) == 4


# -- preprocess ------------------------------------------------------------

def test_preprocess_bgr_to_normalised_rgb(pyramids):
    bb = pyramids["resnet18"]
    img = np.zeros((20, 30, 3), np.uint8)
    img[..., 0] = 255          # pure blue in BGR
    x = bb.preprocess(img)
    assert x.shape == (1, 3, 20, 30) and x.dtype == torch.float32
    mean, std = bb.mean, bb.std
    # channel 0 is R (0), channel 2 is B (1.0) after BGR -> RGB
    assert math.isclose(float(x[0, 0, 0, 0]), (0.0 - mean[0]) / std[0], rel_tol=1e-5)
    assert math.isclose(float(x[0, 1, 0, 0]), (0.0 - mean[1]) / std[1], rel_tol=1e-5)
    assert math.isclose(float(x[0, 2, 0, 0]), (1.0 - mean[2]) / std[2], rel_tol=1e-5)
    # size is (W, H) as cv2 takes it
    y = bb.preprocess(img, size=(64, 32))
    assert y.shape == (1, 3, 32, 64)
    with pytest.raises(ValueError):
        bb.preprocess(None)


# -- custom source ---------------------------------------------------------

class ToyBackbone(Backbone):
    """A 1-layer 'pyramid' for the registry tests."""

    def __init__(self, pretrained, weights):
        super().__init__("custom:toy", "pyramid", [FeatureInfo("c5", 32, 8)], embed_dim=8)
        self.conv = torch.nn.Conv2d(3, 8, 1)
        self.pretrained = pretrained
        self.weights = weights

    def forward(self, x):
        return {"c5": torch.nn.functional.avg_pool2d(self.conv(x), 32)}

    def pooled(self, feats):
        return feats["c5"].mean(dim=(2, 3))


def make_toy(pretrained, weights, **kw):
    return ToyBackbone(pretrained, weights)


def not_a_backbone(pretrained, weights, **kw):
    return torch.nn.Identity()


def test_custom_factory_registers_and_resolves():
    backbones.register("toy", make_toy)
    bb = backbones.get("toy", pretrained=False, weights="w.bin")
    assert isinstance(bb, ToyBackbone) and bb.pretrained is False and bb.weights == "w.bin"
    assert bb.id == "custom:toy"
    assert "toy" in [e["name"] for e in backbones.available()]
    with torch.inference_mode():
        assert bb.pooled(bb(torch.zeros(1, 3, 64, 96))).shape == (1, 8)


def test_custom_module_path_resolves_with_importlib():
    bb = backbones.get(f"custom:{__name__}:make_toy", pretrained=False)
    assert isinstance(bb, ToyBackbone)
    with pytest.raises(TypeError):
        backbones.get(f"custom:{__name__}:not_a_backbone", pretrained=False)
    with pytest.raises(ValueError):
        backbones.get("custom:nomodule", pretrained=False)
    with pytest.raises(TypeError):
        backbones.register("bad", 3)


# -- hf source (DINOv3) ------------------------------------------------------

def _dinov3_available():
    try:
        from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _dinov3_available(), reason="transformers has no DINOv3 model class")
def test_hf_tokens_backbone_from_tiny_local_config(tmp_path):
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    cfg = DINOv3ViTConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                          intermediate_size=64, patch_size=16, image_size=224,
                          num_register_tokens=4)
    d = tmp_path / "tiny-dinov3"
    torch.manual_seed(0)
    DINOv3ViTModel(cfg).save_pretrained(d)

    # pretrained=True on a local dir: weights come from disk, no network
    bb = backbones.get(f"hf:{d}", pretrained=True)
    assert bb.kind == "tokens" and bb.embed_dim == 32
    assert bb.feature_info == [FeatureInfo("tokens", 16, 32)]
    assert bb.id == f"hf:{d}"
    assert bb.mean == (0.485, 0.456, 0.406)
    with torch.inference_mode():
        feats = bb(torch.randn(2, 3, 64, 96))
    assert feats["grid"] == (4, 6)
    assert feats["tokens"].shape == (2, 24, 32)       # CLS + 4 registers dropped
    assert feats["cls"].shape == (2, 32)
    assert bb.pooled(feats).shape == (2, bb.embed_dim)
    assert torch.equal(bb.pooled(feats), feats["cls"])
    # the wrapper and the ViT agree on the mode, both ways
    assert not bb.training and not bb.net.training
    bb.train()
    assert bb.training and bb.net.training
    bb.eval()

    # pretrained=False: same architecture from the config, random init
    rnd = backbones.get(f"hf:{d}", pretrained=False)
    assert rnd.embed_dim == 32 and rnd.id == bb.id
    assert not rnd.training and not rnd.net.training

    # weights dir given separately: id carries the sha8 of model.safetensors
    bw = backbones.get(f"hf:{d}", weights=d)
    assert bw.id.startswith(f"hf:{d}@") and len(bw.id.rsplit("@", 1)[1]) == 8
    assert backbones.get(f"hf:{d}", weights=d).id == bw.id

    # input not a multiple of the patch size is an error, not a silent crop
    with pytest.raises(ValueError):
        with torch.inference_mode():
            bb(torch.randn(1, 3, 40, 96))

    assert len(bb.stage_modules()) == 1 + cfg.num_hidden_layers


@pytest.mark.skipif(not _dinov3_available(), reason="transformers has no DINOv3 model class")
def test_hf_known_config_fallback_for_gated_repo():
    """Without network or a local config, the curated DINOv3 ids still
    resolve from the in-code configs (random init here)."""
    from bebop_vision.places.encoder import DINOV3_CONFIG

    cfg = backbones.HFBackbone._resolve_config("facebook/dinov3-vitl16-pretrain-lvd1689m",
                                               None, _FailingAutoConfig)
    assert cfg.hidden_size == DINOV3_CONFIG["hidden_size"]
    assert cfg.num_register_tokens == 4
    with pytest.raises(RuntimeError):
        backbones.HFBackbone._resolve_config("nobody/nothing", None, _FailingAutoConfig)


class _FailingAutoConfig:
    @staticmethod
    def from_pretrained(src):
        raise OSError(f"offline: {src}")
