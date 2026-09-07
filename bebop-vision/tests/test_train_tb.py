"""Hand-label preference + TensorBoard plumbing for the navd trainer.

Synthetic only: builds tiny fake navd-v0 session dirs (manifest.jsonl +
labels/depth npz + color jpg following the recorder's on-disk shapes) so
NavdDataset, class_weights_from and train_navd's TB helpers are exercised
without a real dataset or GPU.

train_navd.py is a top-level script rather than a bebop_vision package
module, so it is loaded by file path below instead of relying on sys.path.
"""

import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from bebop_vision.navd import NavdDataset, class_weights_from
from bebop_vision.navd_pre import IMAGENET_MEAN, IMAGENET_STD

BEBOP_VISION_ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    """Load train_navd.py under a unique module name (it is a script, not
    an installed package member, so `import train_navd` is not reliable)."""
    spec = importlib.util.spec_from_file_location(
        "train_navd_under_test", BEBOP_VISION_ROOT / "train_navd.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train_navd = _load_train_module()


# --------------------------------------------------------------------------
# fake navd-v0 session fixtures
# --------------------------------------------------------------------------

def _label_map(cls):
    """(60, 60) uint8 label filled with one class — matches the fused/hand
    key shape and value range the recorder/dashboard write."""
    return np.full((60, 60), cls, np.uint8)


def _depth_map():
    """(480, 848) uint16 depth in mm: a near floor band, a far wall, and an
    invalid (0) patch — the recorder's raw depth convention."""
    d = np.full((480, 848), 2500, np.uint16)
    d[:160] = 1200
    d[160:200] = 4200
    d[400:440, 100:200] = 0
    return d


def _color_img():
    """Small deterministic RGB uint8 image; jpg-encoded on disk like the
    recorder does."""
    img = np.zeros((48, 64, 3), np.uint8)
    img[:, :32] = (30, 120, 60)
    img[:, 32:] = (200, 180, 40)
    return img


def make_session(tmp_path, ticks):
    """Fake session dir from `ticks`: one dict per manifest row with keys
    stamp (int), fused (array) and optional hand (array or None)."""
    sd = tmp_path / "navd_session_999999_000000"
    (sd / "labels").mkdir(parents=True)
    (sd / "depth").mkdir()
    (sd / "color").mkdir()
    depth, color = _depth_map(), _color_img()
    rows = []
    for t in ticks:
        stamp = t["stamp"]
        extra = {"hand": t["hand"]} if t.get("hand") is not None else {}
        np.savez(sd / "labels" / f"{stamp:020d}.npz",
                 fused=t["fused"], **extra)
        np.savez(sd / "depth" / f"{stamp:020d}.npz", near=depth, far=depth)
        cv2.imwrite(str(sd / "color" / f"{stamp:020d}.jpg"), color)
        rows.append({"stamp_ns": stamp, "goal": {"type": "none"},
                     "odom": {"x": 0.0, "y": 0.0, "theta": 0.0},
                     "cmd_vel": {"vx": 0.3, "wz": 0.1}})
    with open(sd / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return sd


# --------------------------------------------------------------------------
# NavdDataset hand-label preference
# --------------------------------------------------------------------------

def test_dataset_prefers_hand_and_falls_back(tmp_path):
    """A tick with a `hand` key trains on the human correction; a tick
    without one keeps using `fused` exactly as before."""
    sd = make_session(tmp_path, [
        {"stamp": 1000, "fused": _label_map(0), "hand": _label_map(1)},
        {"stamp": 2000, "fused": _label_map(2)},
    ])
    ds = NavdDataset([sd], augment=False)
    assert len(ds) == 2
    np.testing.assert_array_equal(ds[0]["label"], _label_map(1))
    np.testing.assert_array_equal(ds[1]["label"], _label_map(2))


def test_n_hand_counts_only_hand_ticks(tmp_path):
    """n_hand is the tick census of `hand`-carrying label files (here 2 of
    3), not the file count or the sample count."""
    sd = make_session(tmp_path, [
        {"stamp": 1000, "fused": _label_map(0), "hand": _label_map(1)},
        {"stamp": 2000, "fused": _label_map(2)},
        {"stamp": 3000, "fused": _label_map(0), "hand": _label_map(2)},
    ])
    ds = NavdDataset([sd])
    assert ds.n_hand == 2


def test_class_weights_prefer_hand(tmp_path):
    """Inverse-frequency weighting must track what __getitem__ feeds the
    model: file B's hand label (all navigable) counts, not its fused one
    (all blocked), so navigable is no longer treated as near-absent."""
    sd = make_session(tmp_path, [
        {"stamp": 1000, "fused": _label_map(0)},
        {"stamp": 2000, "fused": _label_map(0), "hand": _label_map(1)},
    ])
    w = class_weights_from([sd]).numpy()
    # hand-aware counts: blocked 3600 (file A fused) + navigable 3600
    # (file B hand) -> identical low weights for both, caution pinned at
    # the inverse-of-one-pixel ceiling.
    counts = np.array([3600, 3600, 0])
    raw = counts.sum() / np.maximum(counts, 1) * 3.0
    np.testing.assert_allclose(w, raw / raw.mean(), rtol=1e-6)
    # and it is genuinely different from the fused-only weighting, where
    # navigable would be vanishingly rare (w >> 1).
    assert w[1] < 0.01


# --------------------------------------------------------------------------
# train_navd TensorBoard helpers
# --------------------------------------------------------------------------

def test_tb_run_tag_and_dir_resolution(tmp_path):
    """Tag = --out basename (+ comment); --tb is treated as a parent dir
    unless it already looks like a run dir (tag-named or holding events)."""
    assert train_navd.tb_run_tag("weights/navd_v3") == "navd_v3"
    assert train_navd.tb_run_tag("weights/navd_v3", "lr1e-4") == \
        "navd_v3_lr1e-4"
    # parent-dir convention: events land one level down, one run per tag
    assert train_navd.tb_resolve_dir(tmp_path / "runs", "navd_v3") == \
        tmp_path / "runs" / "navd_v3"
    # explicit run dir: named for the tag -> used as-is
    explicit = tmp_path / "runs" / "navd_v3"
    assert train_navd.tb_resolve_dir(explicit, "navd_v3") == explicit
    # existing event files -> already a run dir, even under another name
    prior = tmp_path / "manual_run"
    prior.mkdir()
    (prior / "events.out.tfevents.1.host").touch()
    assert train_navd.tb_resolve_dir(prior, "navd_v3") == prior


def test_tb_setup_off_by_default():
    """No --tb -> no writer, no import, no side effects."""
    assert train_navd.tb_setup(None, "weights/navd_v1") is None
    assert train_navd.tb_setup("", "weights/navd_v1") is None


def test_tb_setup_writes_event_file(tmp_path):
    """With tensorboard installed, tb_setup returns a working writer whose
    events land in <tb_root>/<run_tag>."""
    pytest.importorskip("tensorboard")
    w = train_navd.tb_setup(str(tmp_path / "runs"), "weights/navd_x",
                            comment="c1")
    assert w is not None
    w.add_scalar("smoke/loss", 1.25, 0)
    w.close()
    run_dir = tmp_path / "runs" / "navd_x_c1"
    assert list(run_dir.glob("events.out.tfevents.*"))


def test_tb_setup_missing_dep_is_one_line(tmp_path, monkeypatch):
    """Missing tensorboard must cost one actionable line (SystemExit with
    a pip hint), not a traceback — simulated by hiding the module even
    where it is installed."""
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    with pytest.raises(SystemExit) as e:
        train_navd.tb_setup(str(tmp_path / "runs"), "weights/navd_x")
    assert "pip install tensorboard" in str(e.value)


def test_label_panel_layout():
    """teacher|prediction panel: 6x nearest-upscaled with a 2px black
    separator between a palette-green left half and charcoal right half."""
    panel = train_navd.label_panel(_label_map(1), _label_map(0))
    assert panel.shape == (60 * 6, (60 + 2 + 60) * 6, 3)
    assert panel.dtype == np.uint8
    np.testing.assert_array_equal(panel[10, 10],
                                  train_navd.CLASS_PALETTE[1])
    np.testing.assert_array_equal(panel[10, 720],
                                  train_navd.CLASS_PALETTE[0])
    # separator: cols 60..61 pre-upscale -> cols 360..371 post-upscale
    assert not panel[:, 360:372].any()


def test_denorm_color_roundtrip():
    """denorm_color inverts the ImageNet normalization within one gray
    level, and zero-normalized input maps to the mean color."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    c = ((img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    back = train_navd.denorm_color(c.astype(np.float32))
    assert back.shape == img.shape and back.dtype == np.uint8
    assert np.abs(back.astype(int) - img.astype(int)).max() <= 1
    np.testing.assert_allclose(
        train_navd.denorm_color(np.zeros((3, 4, 4), np.float32)),
        np.round(IMAGENET_MEAN * 255).astype(np.uint8)[:, None, None]
        .repeat(4, 1).repeat(4, 2).transpose(1, 2, 0))
