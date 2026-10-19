"""CPU-only tests for the DINOv3 BEV student and its dataset/training glue."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bebop_vision.bev_student import (BevStudent, build_lift_lut,
                                      dummy_lift_lut)
from bebop_vision.bev_student_labels import GridSpec, save_label_npz

ROOT = Path(__file__).resolve().parents[1]


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_bev_student_under_test", ROOT / "train_bev_student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- LUT / model -----------------------------------------------------------


def test_lift_lut_shapes_and_weights():
    lut = build_lift_lut()
    t = lut["tensors"]
    assert tuple(t["feat_idx"].shape) == (8, 4, 1800)
    assert tuple(t["depth_idx"].shape) == (8, 4, 1800)
    assert tuple(t["feat_valid"].shape) == (8, 1800)
    assert lut["meta"]["bev_hw"] == [60, 30]
    assert lut["meta"]["feat_hw"] == [60, 106]
    # every valid cell has normalised bilinear weights
    sums = t["feat_w"].sum(dim=1)
    valid = t["feat_valid"].reshape(-1)
    flat = sums.reshape(-1)
    assert np.allclose(flat[valid].numpy(), 1.0, atol=1e-5)
    assert np.allclose(flat[~valid].numpy(), 0.0)
    # both cameras contribute; near-only heights can be empty
    assert int(t["feat_valid"][4:].sum()) > 1000


def test_tiny_model_forward_shapes():
    model = BevStudent(tiny=True, lut=dummy_lift_lut()).eval()
    assert model.num_parameters() < 2_000_000
    for b in (1, 2):
        color = torch.randn(b, 2, 3, 240, 424)
        depth = torch.rand(b, 2, 1, 240, 424)
        with torch.inference_mode():
            out = model(color, depth)
        assert tuple(out["semantic"].shape) == (b, 4, 120, 60)
        assert tuple(out["ego"].shape) == (b, 1, 120, 60)
        assert tuple(out["coverage"].shape) == (b, 1, 120, 60)


def test_tiny_model_rejects_wrong_camera_axis():
    model = BevStudent(tiny=True, lut=dummy_lift_lut())
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 240, 424), torch.rand(1, 1, 240, 424))


# --- dataset ---------------------------------------------------------------


def _fake_session(root):
    sd = root / "sessions" / "drive_a"
    for sub in ("color_near", "color_far", "depth_near", "depth_far",
                "labels"):
        (sd / sub).mkdir(parents=True)
    stamp = 123456
    name = f"{stamp:020d}"
    bgr = np.zeros((64, 64, 3), np.uint8)
    bgr[..., 1] = 120
    for role in ("near", "far"):
        cv2.imwrite(str(sd / f"color_{role}" / f"{name}.jpg"), bgr)
        cv2.imwrite(str(sd / f"depth_{role}" / f"{name}.png"),
                    np.full((64, 64), 1200, np.uint16))
    sem = np.zeros(GridSpec().shape, np.uint8)
    ego = np.zeros(GridSpec().shape, bool)
    cov = np.ones(GridSpec().shape, bool)
    save_label_npz(sd / "labels" / f"{name}.npz",
                   type("L", (), {"semantic": sem, "ego": ego,
                                  "coverage": cov})())
    row = {"stamp_ns": stamp, "session": sd.name, "frame": 0,
           "color_near": f"color_near/{name}.jpg",
           "color_far": f"color_far/{name}.jpg",
           "depth_near": f"depth_near/{name}.png",
           "depth_far": f"depth_far/{name}.png",
           "label": None, "split": "train"}
    with open(sd / "index.jsonl", "w") as f:
        f.write(json.dumps(row) + "\n")
    (root / "splits").mkdir(exist_ok=True)
    with open(root / "splits" / "drive_split.json", "w") as f:
        json.dump({"sessions": {"drive_a": "train"}}, f)
    return sd


def test_dataset_sample_shapes(tmp_path):
    _fake_session(tmp_path)
    trainer = _load_trainer()
    ds = trainer.BevStudentDataset(tmp_path, "train", train=False)
    assert len(ds) == 1
    s = ds[0]
    assert tuple(s["color"].shape) == (2, 3, 240, 424)
    assert tuple(s["depth"].shape) == (2, 1, 240, 424)
    assert tuple(s["semantic"].shape) == (120, 60)
    assert s["semantic"].dtype == torch.int64
    assert s["ego"].dtype == torch.float32
    w, counts = trainer.class_weights_for(ds)
    assert w.shape == (4,)
    assert counts[0] > 0


def test_one_batch_overfit_smoke(tmp_path):
    trainer = _load_trainer()
    model = BevStudent(tiny=True, lut=dummy_lift_lut()).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    b = 2
    color = torch.randn(b, 2, 3, 240, 424)
    depth = torch.rand(b, 2, 1, 240, 424)
    batch = {
        "color": color, "depth": depth,
        "semantic": torch.zeros(b, 120, 60, dtype=torch.long),
        "ego": torch.zeros(b, 120, 60),
        "coverage": torch.ones(b, 120, 60),
    }
    cw = torch.ones(4)
    losses = []
    for _ in range(20):
        opt.zero_grad(set_to_none=True)
        out = model(color, depth)
        loss, _ = trainer.compute_loss(out, batch, cw, "cpu")
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.9
