"""ConstantVelocity, EmbeddingModel, PathHead, SegHead and HeadModel on CPU
with fake backbones."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import synth
from bebop_vision.lab.models import (
    ConstantVelocity,
    EmbeddingModel,
    HeadModel,
    PathHead,
    SegHead,
    block_groups,
    constant_velocity_path,
    tail_parameters,
)
from bebop_vision.lab.types import Embedding, InputSpec, Mask, Model, Path2D, Sample
from fakes_models import FakePyramid, FakeTicks, FakeTokens


def _to_body(xi, yi, thi, xs, ys):
    c, s = math.cos(thi), math.sin(thi)
    return c * (xs - xi) + s * (ys - yi), -s * (xs - xi) + c * (ys - yi)


# -- constant velocity --------------------------------------------------------

def test_constant_velocity_reproduces_arc():
    v, w, hz = 0.3, 0.4, 20.0
    traj = synth.expected_trajectory(duration_s=4.0, hz=hz, v=v, w=w)
    path = constant_velocity_path(v, w, horizon_s=3.0, dt_s=0.2)
    assert path.pts.shape == (1, 15, 2) and path.rel_t.shape == (15,)
    np.testing.assert_allclose(path.rel_t, np.arange(1, 16) * 0.2, atol=1e-6)
    # body-frame future of the synthetic odometry from tick 0 (Euler-integrated
    # at 20 Hz, hence the 2 cm tolerance over 3 s)
    i = 0
    for k, tau in enumerate(path.rel_t):
        j = i + int(round(tau * hz))
        dx, dy = _to_body(traj["x"][i], traj["y"][i], traj["theta"][i], traj["x"][j], traj["y"][j])
        assert abs(path.pts[0, k, 0] - dx) < 0.02 and abs(path.pts[0, k, 1] - dy) < 0.02
    # analytic check: an arc of radius v / w
    r = v / w
    np.testing.assert_allclose(np.hypot(path.pts[0, :, 0], path.pts[0, :, 1] - r), r, atol=1e-6)
    np.testing.assert_allclose(path.psi[0], w * path.rel_t, atol=1e-6)
    straight = constant_velocity_path(0.5, 0.0, 2.0, 0.5)
    np.testing.assert_allclose(straight.pts[0], [[0.25, 0], [0.5, 0], [0.75, 0], [1.0, 0]], atol=1e-6)


def test_constant_velocity_model_protocol_and_sources():
    cv = ConstantVelocity(horizon_s=3.0, dt_s=0.2)
    assert isinstance(cv, Model)
    assert cv.spec.modalities == ("odom", "cmd_vel") and cv.outputs == {"path": Path2D}
    ticks = FakeTicks("d", 10, vx=0.3, wz=0.4).select(*cv.spec.modalities)
    out = cv.step(ticks[2])
    np.testing.assert_allclose(out["path"].pts, constant_velocity_path(0.3, 0.4, 3.0, 0.2).pts, atol=1e-6)
    # odom source: differences the odom window
    cvo = ConstantVelocity(source="odom")
    assert cvo.spec.history("odom") > 0
    ticks_o = FakeTicks("d", 20, vx=0.3, wz=0.4).select(*cvo.spec.modalities, history_s=cvo.spec.history_s)
    vx, wz = cvo.velocity(ticks_o[10])
    assert abs(vx - 0.3) < 0.01 and abs(wz - 0.4) < 1e-6
    # nothing at all -> standstill
    s = Sample(t_ns=0, session="d", index=0, data={"odom": None, "cmd_vel": None})
    assert np.all(cv.step(s)["path"].pts == 0)


# -- embedding -----------------------------------------------------------------

def test_embedding_model_pooled_vector_and_missing_frame():
    torch.manual_seed(0)
    bb = FakePyramid()
    m = EmbeddingModel(bb, modality="color_far", size=(96, 64))
    assert m.spec.modalities == ("color_far",) and m.outputs == {"embedding": Embedding}
    assert m.name == "emb-fake_pyramid"
    m.prepare(None)
    ticks = FakeTicks("d", 6, drop_far_every=3).select("color_far")
    outs = [m.step(ticks[i]) for i in range(3)]
    e0 = outs[0]["embedding"]
    assert isinstance(e0, Embedding) and e0.vector.shape == (32,) and np.all(np.isnan(e0.vector))
    assert e0.source == "fake:pyramid"
    assert np.all(np.isfinite(outs[1]["embedding"].vector))
    batched = m.batch_step([ticks[i] for i in range(3)])
    np.testing.assert_allclose(batched[1]["embedding"].vector, outs[1]["embedding"].vector, atol=1e-5)


# -- heads ----------------------------------------------------------------------

def test_path_head_shapes_and_weights():
    head = PathHead(in_dim=32, n_scalars=2, n_points=15, k=3, hidden=16)
    pts, w = head(torch.randn(4, 32), torch.randn(4, 2))
    assert pts.shape == (4, 3, 15, 2) and w.shape == (4, 3)
    torch.testing.assert_close(w.sum(1), torch.ones(4))
    # scalars default to zeros when omitted
    pts2, _ = head(torch.randn(2, 32))
    assert pts2.shape == (2, 3, 15, 2)


@pytest.mark.parametrize("bb_cls", [FakePyramid, FakeTokens])
def test_seg_head_output_stride_4(bb_cls):
    bb = bb_cls()
    head = SegHead(bb.feature_info, n_classes=3, channels=32, classes=("floor", "wall", "thing"))
    x = torch.randn(2, 3, 64, 96)
    logits = head(bb(x))
    assert logits.shape == (2, 3, 16, 24)
    assert sum(p.numel() for p in head.parameters()) < 50_000


def test_seg_head_rejects_wide_channels():
    with pytest.raises(ValueError):
        SegHead(FakePyramid().feature_info, n_classes=2, channels=256)


# -- HeadModel -------------------------------------------------------------------

def test_head_model_path_step_and_spec():
    torch.manual_seed(0)
    bb = FakePyramid()
    head = PathHead(bb.embed_dim, n_scalars=2, n_points=15, k=2, hidden=16)
    m = HeadModel(bb, head, modality="color_near", scalar_modalities=("cmd_vel",), horizon_s=3.0, dt_s=0.2)
    assert isinstance(m, Model)
    assert m.kind == "path" and m.outputs == {"path": Path2D}
    assert m.spec == InputSpec(("color_near", "cmd_vel"))
    assert m.name == "path-fake_pyramid"
    ticks = FakeTicks("d", 4).select(*m.spec.modalities)
    m.prepare(None, {"device": "cpu"})
    out = m.step(ticks[1])
    p = out["path"]
    assert isinstance(p, Path2D) and p.pts.shape == (2, 15, 2) and p.weights.shape == (2,)
    np.testing.assert_allclose(p.rel_t, np.arange(1, 16) * 0.2, atol=1e-6)
    assert np.all(np.isfinite(p.pts))
    # a horizon the head cannot produce is refused up front
    with pytest.raises(ValueError):
        HeadModel(bb, head, scalar_modalities=("cmd_vel",), horizon_s=2.0, dt_s=0.2)
    with pytest.raises(ValueError):
        HeadModel(bb, head, scalar_modalities=(), horizon_s=3.0, dt_s=0.2)


def test_head_model_seg_step_returns_mask():
    bb = FakePyramid()
    head = SegHead(bb.feature_info, n_classes=2, channels=16, classes=("floor", "not"))
    m = HeadModel(bb, head, modality="color_far")
    assert m.kind == "seg" and m.outputs == {"mask": Mask}
    ticks = FakeTicks("d", 3).select("color_far")
    out = m.step(ticks[0])["mask"]
    assert isinstance(out, Mask)
    assert out.camera == "far" and out.stride == 4 and out.classes == ("floor", "not")
    assert out.values.shape == (2, 16, 24)
    # resized input: the stride reported is relative to the recorded frame
    m2 = HeadModel(bb, head, modality="color_far", size=(48, 32))
    assert m2.step(ticks[0])["mask"].stride == 8


def test_head_model_uses_cached_features_when_frozen():
    torch.manual_seed(0)
    bb = FakePyramid()
    head = PathHead(bb.embed_dim, n_scalars=0, n_points=15, hidden=16)
    m = HeadModel(bb, head, mode="frozen")
    s = Sample(t_ns=0, session="d", index=0, data={"color_near": None, "features": np.ones(32, np.float32)})
    assert m.uses_cached_features([s])
    pts, _ = m.forward_batch([s])
    assert pts.shape == (1, 1, 15, 2)
    m_full = HeadModel(bb, head, mode="full")
    assert not m_full.uses_cached_features([s])
    # a batch mixing cached and missing features is refused, not run through
    # the backbone on blank frames
    bare = Sample(t_ns=1, session="d", index=1, data={"color_near": None})
    with pytest.raises(ValueError, match="no cached features"):
        m.forward_batch([s, bare])


def test_lpft_staging_pyramid_and_tokens():
    bb = FakePyramid()
    head = PathHead(bb.embed_dim, n_points=15, hidden=16)
    m = HeadModel(bb, head, unfreeze_blocks=1)
    groups = [g for g, _ in block_groups(bb)]
    assert groups == ["stem_0", "stages_0", "stages_1", "stages_2", "stages_3"]
    s0 = {id(p) for p in m.trainable_parameters(0)}
    assert s0 == {id(p) for p in head.parameters()}
    s1 = {id(p) for p in m.trainable_parameters(1)}
    # last stage + the pooling norm registered after it, nothing earlier
    expect = s0 | {id(p) for p in bb.stages_3.parameters()} | {id(p) for p in bb.norm.parameters()}
    assert s1 == expect
    assert {id(p) for p in m.trainable_parameters("all")} == {id(p) for p in m.parameters()}
    m2 = HeadModel(bb, head, unfreeze_blocks=2)
    assert {id(p) for p in bb.stages_2.parameters()} <= {id(p) for p in m2.trainable_parameters(1)}
    assert not {id(p) for p in bb.stages_1.parameters()} & {id(p) for p in m2.trainable_parameters(1)}

    tb = FakeTokens(depth=3)
    th = PathHead(tb.embed_dim, n_points=15, hidden=16)
    tm = HeadModel(tb, th, unfreeze_blocks=2)
    t1 = {id(p) for p in tm.trainable_parameters(1)}
    assert {id(p) for p in tb.blocks[1].parameters()} <= t1
    assert {id(p) for p in tb.blocks[2].parameters()} <= t1
    assert {id(p) for p in tb.norm.parameters()} <= t1
    assert not {id(p) for p in tb.blocks[0].parameters()} & t1
    assert id(tb.cls_token) not in t1 and id(tb.patch_embed.weight) not in t1
    assert tail_parameters(tb, 0) == []

    # set_stage freezes the rest and keeps a frozen backbone in eval
    m.train()
    m.set_stage(0)
    assert not bb.training and head.training
    assert all(not p.requires_grad for p in bb.parameters())
    assert all(p.requires_grad for p in head.parameters())
    m.set_stage(1)
    assert all(p.requires_grad for p in bb.stages_3.parameters())
    assert all(not p.requires_grad for p in bb.stages_0.parameters())


def test_set_stage_keeps_frozen_blocks_in_eval():
    """Freezing parameters is not enough: BatchNorm statistics and Dropout in
    a frozen stage must not change under the fine-tune batches."""
    torch.manual_seed(0)
    bb = FakePyramid()
    for i in range(4):
        stage = getattr(bb, f"stages_{i}")
        stage.append(torch.nn.BatchNorm2d(stage[-1].conv.out_channels))
        stage.append(torch.nn.Dropout(0.5))
    head = PathHead(bb.embed_dim, n_points=15, hidden=16)
    m = HeadModel(bb, head, mode="lpft", unfreeze_blocks=1)
    m.train()
    m.set_stage(1)
    assert bb.training and head.training
    frozen_bn, trained_bn = bb.stages_0[-2], bb.stages_3[-2]
    assert not frozen_bn.training and trained_bn.training
    assert not bb.stages_0[-1].training and bb.stages_3[-1].training       # Dropout follows its block
    assert not bb.stages_0[0].act.training and bb.stages_3[0].act.training  # parameterless leaves too
    before = frozen_bn.running_mean.clone(), trained_bn.running_mean.clone()
    with torch.no_grad():
        bb(torch.randn(4, 3, 64, 96))
    assert torch.equal(frozen_bn.running_mean, before[0])
    assert not torch.equal(trained_bn.running_mean, before[1])
    # stage 0 freezes the whole backbone; eval mode is left alone
    m.set_stage(0)
    assert not bb.training and all(not sub.training for sub in bb.modules())
    m.eval()
    m.set_stage(1)
    assert all(not sub.training for sub in m.net.modules())


def test_residual_path_head_starts_at_the_constant_velocity_arc():
    from bebop_vision.lab.models import constant_velocity_path
    torch.manual_seed(0)
    head = PathHead(in_dim=768, n_scalars=2, n_points=15, residual="cv", dt_s=0.2)
    feats = torch.randn(3, 768) * 50.0                  # DINOv3 ConvNeXt-sized features
    tw = torch.tensor([[0.3, 0.4], [0.2, -0.6], [0.25, 0.0]])
    pts, w = head(feats, tw)
    for b, (vx, wz) in enumerate(tw.tolist()):
        want = constant_velocity_path(vx, wz, 3.0, 0.2).best
        np.testing.assert_allclose(pts[b, 0].detach().numpy(), want, atol=1e-5)
    # and it learns: one step moves it off the arc
    loss = pts.pow(2).sum()
    loss.backward()
    assert head.delta.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        PathHead(in_dim=8, n_scalars=0, residual="cv")
