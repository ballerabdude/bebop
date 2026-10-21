"""CPU-only tests for the DINOv3 BEV student and its dataset/training glue."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bebop_vision.bev_student import (BevStudent, build_lift_lut,
                                      dummy_lift_lut, student_from_checkpoint,
                                      student_from_args, grid_from_model)
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


@pytest.mark.parametrize("encoder", ["dinov3", "segformer-b5"])
def test_resolution_and_checkpoint_roundtrip(encoder):
    # Non-default camera size and full-resolution lift must survive reload.
    lut = build_lift_lut(depth_hw=(96, 160), feat_hw=(24, 40), bev_hw=(120, 60))
    model = BevStudent(tiny=True, encoder=encoder, lut=lut).eval()
    color = torch.randn(1, 2, 3, 96, 160)
    depth = torch.rand(1, 2, 1, 96, 160)
    out = model(color, depth)
    assert out["semantic"].shape == (1, 4, 120, 60)
    sum(v.square().mean() for v in out.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.encoder_parameters())
    ck = {"model": model.state_dict(), "model_config": model.model_config()}
    restored = student_from_checkpoint(ck).eval()
    with torch.inference_mode():
        actual = restored(color, depth)
    for name in out:
        torch.testing.assert_close(actual[name], out[name])
    with pytest.raises(ValueError, match="match LUT"):
        restored(color[..., :-1], depth[..., :-1])


def test_segformer_export_parity(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from tools.export_bev_student import ExportWrapper
    model = BevStudent(tiny=True, encoder="segformer-b5", lut=build_lift_lut(
        depth_hw=(64, 96), feat_hw=(16, 24), bev_hw=(64, 96),
        grid_override=GridSpec(rows_override=64, cols_override=96))).eval()
    x = torch.randn(2, 4, 64, 96)
    with torch.inference_mode():
        original = model(x[:, :3].unsqueeze(0), x[:, 3:4].unsqueeze(0))
    wrapper = ExportWrapper(model).eval()
    path = tmp_path / "student.onnx"
    with torch.inference_mode():
        ref = wrapper(x)
        assert ref[0].shape == (4, 64, 96)
        for name, value in zip(("semantic", "ego", "coverage"), ref):
            torch.testing.assert_close(value, original[name][0], atol=1e-5, rtol=1e-5)
        torch.onnx.export(wrapper, (x,), str(path), input_names=["input"],
                          output_names=["semantic", "ego", "coverage"],
                          opset_version=18, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for expected, actual in zip(ref, sess.run(None, {"input": x.numpy()})):
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-4, rtol=1e-4)


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


def test_high_resolution_preprocessing_matches_runtime(tmp_path):
    from types import SimpleNamespace
    from bebop_vision.bev_student_runtime import BevStudentRuntime
    sd = _fake_session(tmp_path)
    trainer = _load_trainer()
    hw = (480, 848)
    ds = trainer.BevStudentDataset(tmp_path, "train", image_hw=hw)
    sample = ds[0]
    frames = {}
    for role in ("near", "far"):
        frames[role] = SimpleNamespace(
            color_jpeg=next((sd / f"color_{role}").glob("*.jpg")).read_bytes(),
            depth=cv2.imread(str(next((sd / f"depth_{role}").glob("*.png"))),
                             cv2.IMREAD_UNCHANGED), depth_scale_mm=1.0)
    rt = BevStudentRuntime.__new__(BevStudentRuntime)
    rt.image_hw = hw
    actual = rt.preprocess(frames)
    expected = torch.cat([sample["color"], sample["depth"]], dim=1).numpy()
    assert actual.shape == (2, 4, 480, 848)
    np.testing.assert_allclose(actual, expected)


def test_matching_input_output_grid_training_and_reload(tmp_path):
    trainer = _load_trainer()
    model = student_from_args(dict(tiny=True, encoder="segformer-b5",
                                   height=96, width=160, bev_rows=96,
                                   bev_cols=160, full_res_lift=True))
    grid = grid_from_model(model)
    sd = _fake_session(tmp_path)
    sem = np.zeros(grid.shape, np.uint8)
    ego = np.zeros(grid.shape, bool)
    cov = np.ones(grid.shape, bool)
    np.savez_compressed(next((sd / "labels").glob("*.npz")),
                        semantic=sem, ego=ego, coverage=cov)
    (sd / "labels" / "grid.json").write_text(json.dumps({"grid": grid.metadata()}))
    with pytest.raises(ValueError, match="label grid"):
        trainer.BevStudentDataset(tmp_path, "train")
    ds = trainer.BevStudentDataset(tmp_path, "train", image_hw=(96, 160), grid=grid)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    batch = next(iter(loader))
    out = model(batch["color"], batch["depth"])
    assert out["semantic"].shape == (1, 4, 96, 160)
    loss, _ = trainer.compute_loss(out, batch, torch.ones(4), "cpu")
    loss.backward()
    assert torch.isfinite(loss)
    result = trainer.evaluate(model, loader, "cpu", torch.ones(4))
    assert np.isfinite(result["loss"])
    restored = student_from_checkpoint({"model": model.state_dict(),
                                        "model_config": model.model_config()})
    assert grid_from_model(restored).metadata() == grid.metadata()
    bands = trainer._bands(360)
    assert int(bands["0_1m"].sum()) == 60


def test_wall_boundary_metric_is_a_ring():
    trainer = _load_trainer()
    m = torch.zeros(1, 4, 4)
    m[..., 1:3, 1:3] = 1.0
    edge = trainer._edge(m)
    assert int(edge.sum()) == 4


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
