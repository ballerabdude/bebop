"""features.py: FeatureCache over a fake Ticks and a pretrained=False backbone.

The frame source is mocked as the contract asks: a Ticks-like object whose
``select(modality)`` returns itself and whose items are ``types.Sample``
records carrying small random BGR frames (``None`` for a dropped frame).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bebop_vision.lab import backbones
from bebop_vision.lab.features import PRODUCER, FeatureCache
from bebop_vision.lab.types import Sample, file_identity

T0 = 1_792_800_000_000_000_000
DT = 50_000_000


class FakeSession:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.stem


class FakeTicks:
    """Enough of the Ticks surface for FeatureCache.compute."""

    def __init__(self, session: FakeSession, frames, modality="color_near"):
        self.session = session
        self._frames = list(frames)
        self._modality = modality
        self.t_ns = np.array([T0 + i * DT for i in range(len(self._frames))], np.int64)
        self.selected = []

    def select(self, *modalities, history_s=None, tol_ns=None):
        self.selected.append(modalities)
        return self

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        f = self._frames[i]
        return Sample(t_ns=int(self.t_ns[i]), session=self.session.name, index=i,
                      data={self._modality: f}, flags={"has_color_near": f is not None})


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    return backbones.get("convnext_tiny", pretrained=False).eval()


@pytest.fixture
def session(tmp_path):
    p = tmp_path / "navd_session_20260101_000000.mcap"
    p.write_bytes(np.random.default_rng(1).bytes(4096))
    return FakeSession(p)


def _frames(n, rng, drop=()):
    out = []
    for i in range(n):
        out.append(None if i in drop else rng.integers(0, 256, (40, 64, 3), dtype=np.uint8))
    return out


def test_compute_writes_fp16_pooled_and_meta(session, lab_cache, backbone):
    rng = np.random.default_rng(0)
    ticks = FakeTicks(session, _frames(10, rng))
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32), "crop": None}
    key = fc.key(backbone, geometry)
    assert not fc.has(session, key)

    out = fc.compute(session, ticks, backbone, geometry, what=("pooled", "c5"),
                     device="cpu", batch=4, fp16=True)
    assert out == lab_cache / session.name / "features" / key
    assert ticks.selected == [("color_near",)]

    pooled = np.load(out / "pooled.npy")
    t_ns = np.load(out / "t_ns.npy")
    assert pooled.dtype == np.float16 and pooled.shape == (10, backbone.embed_dim)
    assert np.isfinite(pooled.astype(np.float32)).all()
    assert t_ns.dtype == np.int64 and np.array_equal(t_ns, ticks.t_ns)
    c5 = np.load(out / "c5.npy", mmap_mode="r")
    assert c5.dtype == np.float16 and c5.shape == (10, 768, 1, 2)   # 64x32 at stride 32

    meta = json.loads((out / "meta.json").read_text())
    assert meta["source"] == str(session.path)
    assert meta["identity"] == file_identity(session.path)
    assert meta["producer"] == PRODUCER == "features@1"
    assert meta["backbone"] == backbone.id == "timm:convnext_tiny"
    assert meta["geometry"] == {"role": "near", "size": [64, 32], "crop": None}
    assert meta["what"] == ["pooled", "c5"] and meta["dtype"] == "float16"
    assert meta["n"] == 10 and meta["n_ticks"] == 10 and meta["n_missing"] == 0
    assert meta["shapes"] == {"pooled": [10, 768], "c5": [10, 768, 1, 2]}

    # the stored pooled vector is the backbone's pooled() of preprocess()
    with torch.inference_mode():
        ref = backbone.pooled(backbone(backbone.preprocess(ticks[3]["color_near"], size=(64, 32))))
    assert np.allclose(pooled[3].astype(np.float32), ref[0].numpy(), atol=2e-2)


def test_load_has_round_trip(session, lab_cache, backbone):
    rng = np.random.default_rng(0)
    ticks = FakeTicks(session, _frames(6, rng))
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32), "crop": None}
    key = fc.key(backbone, geometry)
    fc.compute(session, ticks, backbone, geometry, device="cpu", batch=16)
    assert fc.has(session, key)

    t_ns, arr = fc.load(session, key)
    assert np.array_equal(t_ns, ticks.t_ns) and arr.shape == (6, backbone.embed_dim)
    with pytest.raises(KeyError):
        fc.load(session, key, what="c4")          # not computed
    assert fc.list(session)[0]["key"] == key and fc.list(session)[0]["current"]

    # a producer bump or a different MCAP invalidates
    meta_path = fc.dir(session, key) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps(dict(meta, producer="features@0")))
    assert not fc.has(session, key)
    with pytest.raises(FileNotFoundError):
        fc.load(session, key)
    meta_path.write_text(json.dumps(meta))
    assert fc.has(session, key)
    session.path.write_bytes(b"different recording" * 300)
    assert not fc.has(session, key)


def test_missing_frames_are_dropped_not_padded(session, lab_cache, backbone):
    rng = np.random.default_rng(0)
    ticks = FakeTicks(session, _frames(9, rng, drop={0, 4, 8}))
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32)}
    out = fc.compute(session, ticks, backbone, geometry, what=("pooled", "c4"),
                     device="cpu", batch=4, fp16=False)
    t_ns = np.load(out / "t_ns.npy")
    pooled = np.load(out / "pooled.npy")
    c4 = np.load(out / "c4.npy", mmap_mode="r")
    assert np.array_equal(t_ns, ticks.t_ns[[1, 2, 3, 5, 6, 7]])
    assert pooled.shape == (6, backbone.embed_dim) and pooled.dtype == np.float32
    assert c4.shape == (6, 384, 2, 4) and c4.dtype == np.float32
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n"] == 6 and meta["n_ticks"] == 9 and meta["n_missing"] == 3
    t2, arr = fc.load(session, fc.key(backbone, geometry), what="c4")
    assert arr.shape == (6, 384, 2, 4) and len(t2) == 6


def test_trim_leaves_a_clean_dir_and_the_right_rows(session, lab_cache, backbone):
    rng = np.random.default_rng(3)
    frames = _frames(7, rng, drop={2, 6})
    ticks = FakeTicks(session, frames)
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32)}
    out = fc.compute(session, ticks, backbone, geometry, what=("pooled", "c5"), device="cpu",
                     batch=3, fp16=False)
    assert sorted(p.name for p in out.iterdir()) == ["c5.npy", "meta.json", "pooled.npy", "t_ns.npy"]
    assert not out.with_name(out.name + ".tmp").exists()
    _, c5 = fc.load(session, fc.key(backbone, geometry), what="c5")
    assert c5.shape[0] == 5
    with torch.inference_mode():
        ref = backbone(backbone.preprocess(frames[5], size=(64, 32)))["c5"]
    assert np.allclose(np.asarray(c5[-1]), ref[0].numpy(), atol=1e-4)   # last row is frame 5


class InterruptingTicks(FakeTicks):
    """Raises at tick ``fail_at`` (a Ctrl-C or a decode error mid-run)."""

    def __init__(self, session, frames, fail_at):
        super().__init__(session, frames)
        self.fail_at = fail_at

    def __getitem__(self, i):
        if i == self.fail_at:
            raise KeyboardInterrupt
        return super().__getitem__(i)


def test_interrupted_recompute_keeps_the_previous_cache_intact(session, lab_cache, backbone):
    rng = np.random.default_rng(0)
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32)}
    key = fc.key(backbone, geometry)
    out = fc.compute(session, FakeTicks(session, _frames(8, rng)), backbone, geometry,
                     what=("pooled", "c5"), device="cpu", batch=2)
    t0, c5_before = fc.load(session, key, what="c5")
    c5_before = np.array(c5_before)
    pooled_before = np.load(out / "pooled.npy")

    with pytest.raises(KeyboardInterrupt):
        fc.compute(session, InterruptingTicks(session, _frames(8, rng), fail_at=6), backbone,
                   geometry, what=("pooled", "c5"), device="cpu", batch=2)
    assert not out.with_name(out.name + ".tmp").exists()
    assert fc.has(session, key)
    t1, c5_after = fc.load(session, key, what="c5")
    assert np.array_equal(t0, t1) and np.array_equal(np.array(c5_after), c5_before)
    assert np.array_equal(np.load(out / "pooled.npy"), pooled_before)
    assert np.all(np.any(c5_before.astype(np.float32) != 0, axis=(1, 2, 3)))   # no zero-filled rows

    # a first compute that dies leaves nothing behind at all
    other = {"role": "near", "size": (32, 32)}
    with pytest.raises(KeyboardInterrupt):
        fc.compute(session, InterruptingTicks(session, _frames(8, rng), fail_at=3), backbone, other,
                   device="cpu", batch=2)
    assert not fc.has(session, fc.key(backbone, other))
    assert not fc.dir(session, fc.key(backbone, other)).exists()
    assert fc.list(session) and all(e["key"] == key for e in fc.list(session))


def test_recompute_replaces_stale_stage_files(session, lab_cache, backbone):
    """The key ignores the tick selection, so a stage file from an earlier
    run over other ticks must not be served against the new t_ns."""
    rng = np.random.default_rng(0)
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32)}
    key = fc.key(backbone, geometry)
    ticks_a = FakeTicks(session, _frames(8, rng))
    fc.compute(session, ticks_a, backbone, geometry, what=("pooled", "c5"), device="cpu")
    ticks_b = FakeTicks(session, _frames(8, rng))
    ticks_b.t_ns = ticks_b.t_ns + 4_000_000_000
    out = fc.compute(session, ticks_b, backbone, geometry, what=("pooled",), device="cpu")
    assert not (out / "c5.npy").exists()
    t_ns, _ = fc.load(session, key)
    assert np.array_equal(t_ns, ticks_b.t_ns)
    with pytest.raises(KeyError):
        fc.load(session, key, what="c5")
    # even a planted file is not served when meta does not list it
    np.save(out / "c5.npy", np.zeros((8, 768, 1, 2), np.float16))
    with pytest.raises(KeyError):
        fc.load(session, key, what="c5")


def test_compute_restores_mode_and_device(session, lab_cache, backbone):
    rng = np.random.default_rng(0)
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32)}
    backbone.train()
    try:
        fc.compute(session, FakeTicks(session, _frames(3, rng)), backbone, geometry, device="cpu")
        assert backbone.training and all(m.training for m in backbone.modules())
    finally:
        backbone.eval()
    assert next(backbone.parameters()).device.type == "cpu"
    if not torch.cuda.is_available():
        pytest.skip("device round trip needs a second device")
    fc.compute(session, FakeTicks(session, _frames(3, rng)), backbone, geometry, device="cuda")
    assert next(backbone.parameters()).device.type == "cpu"


def test_crop_then_resize(session, lab_cache, backbone):
    rng = np.random.default_rng(0)
    frames = _frames(3, rng)
    ticks = FakeTicks(session, frames)
    fc = FeatureCache(lab_cache)
    geometry = {"role": "near", "size": (64, 32), "crop": (8, 40, 0, 64)}
    out = fc.compute(session, ticks, backbone, geometry, device="cpu")
    pooled = np.load(out / "pooled.npy")
    with torch.inference_mode():
        ref = backbone.pooled(backbone(backbone.preprocess(frames[1][8:40, 0:64], size=(64, 32))))
    assert np.allclose(pooled[1].astype(np.float32), ref[0].numpy(), atol=2e-2)
    assert "crop8-40-0-64" in out.name


def test_unknown_stage_is_an_error(session, lab_cache, backbone):
    ticks = FakeTicks(session, _frames(2, np.random.default_rng(0)))
    with pytest.raises(KeyError):
        FeatureCache(lab_cache).compute(session, ticks, backbone, {"role": "near", "size": (64, 32)},
                                        what=("tokens",), device="cpu")


def test_key_is_stable_and_geometry_sensitive(backbone, lab_cache):
    fc = FeatureCache(lab_cache)
    g = {"role": "near", "size": (224, 128), "crop": None}
    k1 = fc.key(backbone, g)
    k2 = fc.key(backbone, {"role": "near", "size": [224, 128]})   # list vs tuple, crop omitted
    assert k1 == k2
    assert k1 == fc.key(backbones.get("convnext_tiny", pretrained=False), g)   # another instance
    assert k1 != fc.key(backbone, {"role": "far", "size": (224, 128)})
    assert k1 != fc.key(backbone, {"role": "near", "size": (224, 96)})
    assert k1 != fc.key(backbone, {"role": "near", "size": (224, 128), "crop": (0, 64, 0, 224)})
    assert k1 != fc.key(backbones.get("resnet18", pretrained=False), g)
    assert "/" not in k1 and ":" not in k1 and " " not in k1
    assert k1.startswith("timm_convnext_tiny__near__224x128__")
    with pytest.raises(ValueError):
        fc.key(backbone, {"role": "near", "crop": (1, 2)})
