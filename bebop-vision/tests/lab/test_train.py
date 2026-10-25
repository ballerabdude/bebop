"""fit(), RunDir and the ready-made losses on CPU with fake backbones."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from bebop_vision.lab.dataset import LabDataset, make_splits
from bebop_vision.lab.models import HeadModel, PathHead, SegHead
from bebop_vision.lab.train import RunDir, TrainConfig, evaluate, fit, path_loss, seg_loss
from bebop_vision.lab.types import InputSpec
from fakes_models import FakeLabelSet, FakePyramid, FakeTicks, arc


def _path_setup(n_train=24, n_val=8, k=1, scalars=("cmd_vel",), seed=0):
    torch.manual_seed(seed)
    bb = FakePyramid()
    head = PathHead(bb.embed_dim, n_scalars=HeadModel.scalar_dim(scalars), n_points=15, k=k, hidden=32)
    model = HeadModel(bb, head, modality="color_near", scalar_modalities=scalars, horizon_s=3.0, dt_s=0.2)
    spec = model.spec
    ta, tb = FakeTicks("drive_a", n_train), FakeTicks("drive_b", n_val)
    label = arc(0.3, 0.4)
    train_ds = LabDataset([ta], spec, labels={"drive_a": FakeLabelSet(ta.t_ns, label, invalid_last=2)})
    val_ds = LabDataset([tb], spec, labels={"drive_b": FakeLabelSet(tb.t_ns, label)})
    return model, train_ds, val_ds


def test_fit_frozen_path_head_learns_constant_arc(tmp_path):
    model, train_ds, val_ds = _path_setup()
    cfg = TrainConfig(epochs=25, lr=1e-2, batch=8, device="cpu", mode="frozen", seed=0)
    run = fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / "run")
    rows = run.metrics()
    assert len(rows) == 25 and rows[0]["stage"] == 0 and rows[-1]["stage"] == 0
    # mean L1 in metres over 15 points: starts around 0.13-0.16, ends at
    # 0.01-0.02 (three seeds measured); the head only has random features
    # to fit, so the residual is the noise floor of 24 samples, not the arc
    assert rows[-1]["val_loss"] < 0.03 < 0.1 < rows[0]["val_loss"]
    assert run.manifest["best"]["metric"] == "val_loss"
    assert (run.path / "best.pt").exists() and (run.path / "last.pt").exists()
    # the backbone never moved
    assert all(not p.requires_grad for p in model.backbone.parameters())
    # the trained model predicts the arc through the Model protocol
    sample = val_ds[0][0]
    out = model.step(sample)["path"]
    assert np.abs(out.best - arc(0.3, 0.4).best).mean() < 0.05


def test_fit_is_deterministic(tmp_path):
    runs = []
    for i in range(2):
        model, train_ds, val_ds = _path_setup(seed=3)
        cfg = TrainConfig(epochs=3, lr=1e-2, batch=8, device="cpu", seed=7)
        runs.append(fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / f"run{i}").metrics())
    assert [r["train_loss"] for r in runs[0]] == [r["train_loss"] for r in runs[1]]
    assert [r["val_loss"] for r in runs[0]] == [r["val_loss"] for r in runs[1]]


def test_fit_lpft_stages_and_full(tmp_path):
    model, train_ds, val_ds = _path_setup(n_train=8, n_val=4)
    model.unfreeze_blocks = 1
    before = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    cfg = TrainConfig(epochs=3, lp_epochs=1, lr=1e-2, batch=4, device="cpu", mode="lpft", unfreeze_blocks=1)
    run = fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / "lpft")
    rows = run.metrics()
    assert [r["stage"] for r in rows] == [0, 1, 1]
    assert rows[1]["lr"] == pytest.approx(1e-3) and rows[0]["lr"] == pytest.approx(1e-2)
    after = dict(model.backbone.named_parameters())
    moved = {n for n, p in before.items() if not torch.equal(p, after[n].detach())}
    assert moved and all(n.startswith(("stages_3", "norm")) for n in moved)

    model_f, train_ds, val_ds = _path_setup(n_train=8, n_val=4)
    before = {n: p.detach().clone() for n, p in model_f.backbone.named_parameters()}
    cfg = TrainConfig(epochs=1, lr=1e-2, batch=4, device="cpu", mode="full")
    run = fit(model_f, train_ds, val_ds, cfg, path_loss, tmp_path / "full")
    assert run.metrics()[0]["stage"] == "all"
    after = dict(model_f.backbone.named_parameters())
    assert any(not torch.equal(before[n], after[n].detach()) for n in before if n.startswith("stem_0"))


def test_manifest_contents_and_checkpoint_roundtrip(tmp_path):
    model, train_ds, val_ds = _path_setup(n_train=8, n_val=4)
    cfg = TrainConfig(epochs=2, lr=1e-2, batch=4, device="cpu", notes="unit test")

    def metrics_fn(out, labels):
        pts, _ = out
        return {"fde": float((pts[:, 0, -1] - labels["path"][:, -1]).norm(dim=-1).mean())}

    run = fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / "run", metrics_fn=metrics_fn)
    man = json.loads((run.path / "manifest.json").read_text())
    assert man["cfg"]["notes"] == "unit test" and man["cfg"]["mode"] == "frozen"
    assert man["backbone"] == "fake:pyramid" and man["model"] == "path-fake_pyramid"
    assert man["spec"] == {"modalities": ["color_near", "cmd_vel"], "history_s": {}, "tol_ns": 50_000_000}
    assert man["labels"] == [{"variant": "hindsight", "version": 1}]
    assert man["split"] == {"train": ["drive_a"], "val": ["drive_b"]}
    assert man["sizes"] == {"train": 8, "val": 4}
    assert man["started"] and man["finished"] and man["epochs_done"] == 2
    assert "git" in man
    rows = run.metrics()
    assert "fde" in rows[0] and rows[0]["n_train"] == 8
    ckpt = run.load_checkpoint("best")
    assert ckpt["manifest"]["backbone"] == "fake:pyramid" and "epoch" in ckpt
    fresh, _, _ = _path_setup(n_train=8, n_val=4, seed=99)
    info = run.load_into(fresh, "best")
    for (n, a), (_, b) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert torch.equal(a, b), n
    assert info["manifest"]["best"]["epoch"] == ckpt["epoch"]


def test_run_dir_roundtrip(tmp_path):
    run = RunDir(tmp_path / "r")
    assert run.manifest == {} and run.metrics() == []
    run.write_manifest({"cfg": TrainConfig(device="cpu"), "path": tmp_path, "n": np.int64(3)})
    man = run.manifest
    assert man["cfg"]["lr"] == 1e-3 and man["path"] == str(tmp_path) and man["n"] == 3
    run.append_metrics({"epoch": 0, "loss": float("nan")})
    run.append_metrics({"epoch": 1, "loss": 0.5})
    assert run.metrics() == [{"epoch": 0, "loss": None}, {"epoch": 1, "loss": 0.5}]
    lin = torch.nn.Linear(3, 2)
    p = run.save_checkpoint("best", lin.state_dict(), man, epoch=1)
    assert p == run.path / "best.pt"
    ckpt = run.load_checkpoint("best")
    assert torch.equal(ckpt["state_dict"]["weight"], lin.weight.detach())
    assert ckpt["manifest"] == man and ckpt["epoch"] == 1


def test_fit_without_val_selects_on_train_loss(tmp_path):
    model, train_ds, _ = _path_setup(n_train=8)
    cfg = TrainConfig(epochs=2, lr=1e-2, batch=4, device="cpu")
    run = fit(model, train_ds, None, cfg, path_loss, tmp_path / "run")
    assert run.manifest["best"]["metric"] == "train_loss"
    assert "val_loss" not in run.metrics()[0]


def test_path_loss_multi_hypothesis_and_mask():
    pts = torch.zeros(2, 2, 15, 2)
    pts[:, 1] = 1.0
    weights = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    labels = {"path": torch.ones(2, 15, 2), "valid": torch.tensor([True, False])}
    loss = path_loss((pts, weights), labels)
    # hypothesis 1 matches exactly; only the weight CE of the valid row remains
    assert loss.item() == pytest.approx(-np.log(0.1), abs=1e-5)


def test_seg_loss_and_seg_fit_smoke(tmp_path):
    torch.manual_seed(0)
    bb = FakePyramid()
    head = SegHead(bb.feature_info, n_classes=2, channels=16)
    model = HeadModel(bb, head, modality="color_near")
    t = FakeTicks("d", 6)

    def transform(sample, labels):
        labels["mask"] = torch.zeros(16, 24, dtype=torch.long)
        return sample, labels

    ds = LabDataset([t], model.spec, transform=transform)
    cfg = TrainConfig(epochs=2, lr=1e-2, batch=3, device="cpu")
    run = fit(model, ds, ds, cfg, seg_loss, tmp_path / "seg")
    rows = run.metrics()
    assert rows[-1]["val_loss"] < rows[0]["val_loss"]
    res = evaluate(model, ds, cfg, seg_loss)
    assert res["val_loss"] == pytest.approx(rows[-1]["val_loss"], rel=1e-5)


def test_make_splits_feeds_dataset():
    names = ["a", "b", "c", "d"]
    split = make_splits(names, holdout=1, seed=0)
    ticks = {n: FakeTicks(n, 3) for n in names}
    spec = InputSpec(("odom",))
    train = LabDataset([ticks[n] for n in split["train"]], spec)
    val = LabDataset([ticks[n] for n in split["val"]], spec)
    assert len(train) == 9 and len(val) == 3
    assert not set(train.drives) & set(val.drives)


def test_fit_rejects_unknown_select_after_first_epoch(tmp_path):
    model, train_ds, val_ds = _path_setup(n_train=8, n_val=4)
    cfg = TrainConfig(epochs=3, lr=1e-2, batch=4, device="cpu", select="fde")
    with pytest.raises(ValueError, match="fde"):
        fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / "run")
    run = RunDir(tmp_path / "run")
    assert len(run.metrics()) == 1 and not (run.path / "best.pt").exists()
    # a metrics_fn that provides the column makes the same config valid
    run = fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / "ok",
              metrics_fn=lambda out, labels: {"fde": float(out[0][:, 0, -1].norm(dim=-1).mean())})
    assert run.manifest["best"]["metric"] == "fde" and (run.path / "best.pt").exists()
    # a non-default select without a validation set cannot be honoured
    with pytest.raises(ValueError, match="validation"):
        fit(model, train_ds, None, cfg, path_loss, tmp_path / "noval")
    with pytest.raises(ValueError):
        TrainConfig(select="")


def test_empty_val_ds_counts_as_none(tmp_path):
    model, train_ds, _ = _path_setup(n_train=8)
    split = make_splits(["drive_a"], holdout=0)
    val_ds = LabDataset([], model.spec)
    assert split["val"] == [] and len(val_ds) == 0
    cfg = TrainConfig(epochs=2, lr=1e-2, batch=4, device="cpu")
    run = fit(model, train_ds, val_ds, cfg, path_loss, tmp_path / "run")
    assert run.manifest["best"]["metric"] == "train_loss" and run.manifest["sizes"]["val"] == 0
    assert all("val_loss" not in r for r in run.metrics())
    with pytest.raises(ValueError):
        evaluate(model, val_ds, cfg, path_loss)


def test_manifest_numpy_nan_becomes_null(tmp_path):
    run = RunDir(tmp_path / "r")
    run.write_manifest({"ade": np.float64("nan"), "arr": np.array([1.0, np.inf]),
                        "t": torch.tensor([float("nan"), 2.0]), "s": np.float32("nan")})
    text = (run.path / "manifest.json").read_text()
    assert "NaN" not in text and "Infinity" not in text

    def strict(_):
        raise ValueError("bare constant")

    assert json.loads(text, parse_constant=strict) == {"ade": None, "arr": [1.0, None], "t": [None, 2.0], "s": None}


def test_dataset_without_collate_is_refused(tmp_path):
    model, train_ds, val_ds = _path_setup(n_train=8, n_val=4)
    plain = [train_ds[i] for i in range(len(train_ds))]   # a list has no collate_fn
    with pytest.raises(TypeError, match="collate_fn"):
        fit(model, plain, val_ds, TrainConfig(epochs=1, batch=4, device="cpu"), path_loss, tmp_path / "run")
