"""Single-head model: one 6-class output (floor, wall, obstacle, hazard,
ego, unknown) replaces the ego and coverage sigmoid heads."""

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from bebop_vision.bev_student import (EGO_CLASS, SINGLE_HEAD_CLASSES,
                                      UNKNOWN_CLASS, BevStudent, dummy_lift_lut,
                                      student_from_args, student_from_checkpoint)
from bebop_vision.bev_student_labels import IGNORE
from bebop_vision.bev_student_runtime import BevStudentRuntime
from bebop_vision.master_bev import FREE, HAZARD, OCCUPIED

SEM_HAZARD = 3   # semantic class; master_bev.HAZARD is the occupancy code

ROOT = Path(__file__).resolve().parents[1]


def _trainer():
    spec = importlib.util.spec_from_file_location(
        "train_bev_student", ROOT / "train_bev_student.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_head_forward_has_one_output():
    model = BevStudent(tiny=True, lut=dummy_lift_lut(), single_head=True,
                       num_classes=6).eval()
    assert not hasattr(model, "ego_head") and not hasattr(model, "cov_head")
    color = torch.randn(2, 2, 3, 240, 424)
    depth = torch.rand(2, 2, 1, 240, 424)
    with torch.inference_mode():
        out = model(color, depth)
    assert set(out) == {"semantic"}
    assert tuple(out["semantic"].shape) == (2, 6, 120, 60)
    with pytest.raises(ValueError):
        BevStudent(tiny=True, lut=dummy_lift_lut(), single_head=True, num_classes=4)


def test_single_head_survives_checkpoint_and_args():
    model = BevStudent(tiny=True, lut=dummy_lift_lut(), single_head=True,
                       num_classes=6)
    ck = {"model": model.state_dict(), "model_config": model.model_config()}
    again = student_from_checkpoint(ck)
    assert again.single_head and again.num_classes == 6
    args = {"tiny": True, "single_head": True}
    assert student_from_args(args).num_classes == len(SINGLE_HEAD_CLASSES)


def test_export_wrapper_emits_only_semantic(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "export_bev_student", ROOT / "tools" / "export_bev_student.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    model = BevStudent(tiny=True, lut=dummy_lift_lut(), single_head=True,
                       num_classes=6).eval()
    wrapper = exp.ExportWrapper(model).eval()
    x = torch.randn(2, 4, 240, 424)
    with torch.inference_mode():
        ref = wrapper(x)
        assert len(ref) == 1 and ref[0].shape == (6, 120, 60)
        path = tmp_path / "single.onnx"
        torch.onnx.export(wrapper, (x,), str(path), input_names=["input"],
                          output_names=["semantic"], opset_version=18,
                          dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"input": x.numpy()})
    assert len(outs) == 1
    np.testing.assert_allclose(outs[0], ref[0].numpy(), atol=1e-4, rtol=1e-4)


def test_map_semantic_and_class_weights():
    t = _trainer()
    sem = np.array([[0, 1, 2, 3], [255, 255, 0, 255]], np.uint8)
    ego = np.zeros((2, 4), bool)
    ego[1, 1] = True
    assert np.array_equal(t.map_semantic(sem, ego, 4), sem)
    five = t.map_semantic(sem, ego, 5)
    assert five.tolist() == [[0, 1, 2, 3], [4, 4, 0, 4]]
    six = t.map_semantic(sem, ego, 6)
    assert six.tolist() == [[0, 1, 2, 3], [5, EGO_CLASS, 0, 5]]
    assert t.unknown_index(4) == IGNORE and t.unknown_index(6) == UNKNOWN_CLASS
    assert t.num_classes_for(single_head=True) == 6


def test_single_head_loss_and_eval_without_extra_heads():
    t = _trainer()
    model = BevStudent(tiny=True, lut=dummy_lift_lut(), single_head=True,
                       num_classes=6).train()
    b = 2
    color = torch.randn(b, 2, 3, 240, 424)
    depth = torch.rand(b, 2, 1, 240, 424)
    label = torch.full((b, 120, 60), UNKNOWN_CLASS, dtype=torch.long)
    label[:, 60:, :] = 0                 # floor near
    label[:, 110:, 25:35] = EGO_CLASS    # chassis
    label[:, 30:40, 10:20] = SEM_HAZARD
    ego = (label == EGO_CLASS).float()
    cov = ((label != UNKNOWN_CLASS) & (label != EGO_CLASS)).float()
    batch = {"color": color, "depth": depth, "semantic": label,
             "ego": ego, "coverage": cov}
    cw = torch.ones(6)
    out = model(color, depth)
    loss, parts = t.compute_loss(out, batch, cw, "cpu", num_classes=6,
                                 hazard_tversky_weight=0.5)
    assert torch.isfinite(loss) and parts["ego"] == 0.0 and parts["coverage"] == 0.0
    loss.backward()
    # evaluate() derives ego/coverage from the predicted classes
    model.eval()
    ld = [batch]
    with torch.no_grad():
        m = t.evaluate(model, ld, "cpu", cw, num_classes=6)
    for key in ("miou", "ego_fp_rate", "coverage_precision", "unknown_iou",
                "ego_iou"):
        assert key in m
    assert len(m["confusion"]) == 6


def test_runtime_postprocess_single_head_classes():
    rt = BevStudentRuntime.__new__(BevStudentRuntime)
    rt.rows, rt.cols = 120, 60
    rt.cell_m, rt.range_m, rt.width_m = 0.05, 6.0, 3.0
    rt.cell_x_m = rt.cell_y_m = 0.05
    rt.ego_static = np.zeros((120, 60), bool)
    rt.inflate_kernel = None
    rt.hazard_source = "model"
    rt.hazard_threshold = 0.5
    rt.engine_path = "fake.engine"
    rt.single_head = True
    semantic = np.full((120, 60), UNKNOWN_CLASS, np.uint8)
    semantic[60:, :] = 0
    semantic[110:, 25:35] = EGO_CLASS
    semantic[70:75, 10:15] = 2
    semantic[65:68, 40:45] = SEM_HAZARD
    logits = np.full((6, 120, 60), -8.0, np.float32)
    for c in range(6):
        logits[c][semantic == c] = 8.0

    class _F:
        stamp_us = 1
    grid, stats = rt.postprocess(logits, None, None, {"near": _F(), "far": _F()},
                                 ages_s={})
    assert grid.ego[110:, 25:35].all() and int(grid.ego.sum()) == 100
    assert not grid.coverage[:60].any() and not grid.coverage[grid.ego].any()
    assert grid.coverage[60:110].all()
    assert np.all(grid.semantic[:60] == 255) and np.all(grid.semantic[grid.ego] == 255)
    assert np.all(grid.raw[70:75, 10:15] == OCCUPIED)
    assert np.all(grid.raw[65:68, 40:45] == HAZARD)
    assert np.all(grid.raw[grid.ego] == FREE)
    assert stats["semantic_counts"]["unknown"] == int((grid.semantic == 255).sum())
