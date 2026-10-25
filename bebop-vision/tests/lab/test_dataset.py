"""LabDataset, lab_collate and make_splits against fake Ticks / LabelSets."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bebop_vision.lab.dataset import LabDataset, lab_collate, make_splits
from bebop_vision.lab.types import InputSpec, Sample
from fakes_models import FakeLabelSet, FakeTicks, arc


def _ticks(name="drive_a", n=10, **kw):
    return FakeTicks(name, n=n, **kw)


def test_dataset_items_and_shapes():
    spec = InputSpec(("color_near", "cmd_vel"))
    ta, tb = _ticks("drive_a", 10), _ticks("drive_b", 6, vx=0.5, wz=0.0)
    labels = {"drive_a": FakeLabelSet(ta.t_ns, arc(0.3, 0.4)),
              "drive_b": FakeLabelSet(tb.t_ns, arc(0.5, 0.0))}
    ds = LabDataset([ta, tb], spec, labels=labels)
    assert len(ds) == 16
    assert ds.drives == ["drive_a", "drive_b"]
    assert ds.drive_of(0) == "drive_a" and ds.drive_of(12) == "drive_b"
    sample, lab = ds[0]
    assert isinstance(sample, Sample)
    assert set(sample.data) == {"color_near", "cmd_vel"}          # the spec was applied via select()
    assert sample["color_near"].shape == (64, 96, 3)
    assert set(lab) == {"path", "rel_t", "valid", "psi", "v"}
    assert lab["path"].shape == (15, 2) and lab["rel_t"].shape == (15,)
    assert bool(lab["valid"]) is True
    np.testing.assert_allclose(lab["path"].numpy(), arc(0.3, 0.4).best, atol=1e-6)
    sample_b, lab_b = ds[12]
    assert sample_b.session == "drive_b"
    np.testing.assert_allclose(lab_b["path"].numpy(), arc(0.5, 0.0).best, atol=1e-6)
    assert ds.label_variants() == [{"variant": "hindsight", "version": 1}]
    assert ds.t_ns.shape == (16,)


def test_invalid_label_keeps_shape_and_flags_false():
    spec = InputSpec(("color_near",))
    t = _ticks("d", 8)
    ds = LabDataset([t], spec, labels={"d": FakeLabelSet(t.t_ns, arc(0.3, 0.4), invalid_last=3)})
    _, lab_ok = ds[0]
    _, lab_bad = ds[7]
    assert bool(lab_ok["valid"]) and not bool(lab_bad["valid"])
    assert lab_bad["path"].shape == lab_ok["path"].shape
    assert torch.all(lab_bad["path"] == 0)


def test_collate_stacks_labels_and_keeps_samples():
    spec = InputSpec(("color_near", "color_far"))
    t = _ticks("d", 8, drop_far_every=4)
    ds = LabDataset([t], spec, labels={"d": FakeLabelSet(t.t_ns, arc(0.3, 0.4), invalid_last=2)})
    samples, labels = lab_collate([ds[i] for i in range(8)])
    assert len(samples) == 8 and all(isinstance(s, Sample) for s in samples)
    assert samples[0]["color_far"] is None and samples[1]["color_far"].shape == (64, 96, 3)
    assert labels["path"].shape == (8, 15, 2)
    assert labels["valid"].dtype == torch.bool and labels["valid"].sum() == 6
    # the torch DataLoader path, as fit() uses it
    dl = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=ds.collate_fn)
    sizes = [len(s) for s, _ in dl]
    assert sizes == [3, 3, 2]


def test_multiple_label_variants_are_prefixed():
    spec = InputSpec(("color_near",))
    t = _ticks("d", 4)
    labels = {"d": [FakeLabelSet(t.t_ns, arc(0.3, 0.4), variant="hindsight"),
                    FakeLabelSet(t.t_ns, arc(0.3, 0.0), variant="cv", version=2)]}
    ds = LabDataset([t], spec, labels=labels)
    _, lab = ds[0]
    assert "hindsight.path" in lab and "cv.path" in lab and "cv.valid" in lab
    assert {v["variant"] for v in ds.label_variants()} == {"hindsight", "cv"}
    # single-session shorthand: variant-keyed labels
    ds2 = LabDataset([t], spec, labels={"hindsight": labels["d"][0]})
    assert "path" in ds2[0][1]
    with pytest.raises(KeyError):
        LabDataset([t, _ticks("e", 2)], spec, labels={"hindsight": labels["d"][0]})


def test_features_attach_to_samples_and_transform_runs():
    spec = InputSpec(("color_near",))
    t = _ticks("d", 5)
    pooled = np.arange(5 * 4, dtype=np.float32).reshape(5, 4)
    seen = []

    def transform(sample, labels):
        seen.append(sample.index)
        labels["extra"] = torch.tensor(1.0)
        return sample, labels

    ds = LabDataset([t], spec, features={"d": (t.t_ns, pooled)}, transform=transform)
    s, lab = ds[3]
    np.testing.assert_array_equal(s["features"], pooled[3])
    assert lab["extra"] == 1.0 and seen == [3]
    # a stamp outside the tolerance has no feature: the tick is dropped
    ds2 = LabDataset([t], spec, features={"d": (t.t_ns + 10 ** 9, pooled)})
    assert len(ds2) == 0 and ds2.dropped_no_features == {"d": 5}


def test_ticks_without_features_are_dropped_and_t_ns_follows_items():
    # the far camera drops colour, so its feature cache misses ticks; one
    # missing tick used to send the whole batch through the backbone. The
    # fakes tick every 50 ms, so match tighter than that (real 10 Hz ticks
    # are 100 ms apart, twice the default tolerance).
    spec = InputSpec(("color_near",), tol_ns=10_000_000)
    t = _ticks("d", 6)
    keep = np.array([0, 1, 2, 4, 5])                    # tick 3 had no frame
    pooled = np.arange(6 * 4, dtype=np.float32).reshape(6, 4)[keep]
    ds = LabDataset([t], spec, features={"d": (t.t_ns[keep], pooled)})
    assert len(ds) == 5 and ds.dropped_no_features == {"d": 1}
    np.testing.assert_array_equal(ds.t_ns, t.t_ns[keep])
    assert all(ds[i][0].get("features") is not None for i in range(len(ds)))
    # a drive without a features entry keeps every tick
    u = _ticks("e", 3)
    both = LabDataset([t, u], spec, features={"d": (t.t_ns[keep], pooled)})
    assert len(both) == 8 and both.dropped_no_features == {"d": 1}


def test_no_labels_gives_empty_dict():
    ds = LabDataset([_ticks("d", 3)], InputSpec(("odom",)))
    s, lab = ds[1]
    assert lab == {} and s["odom"]["x"] > 0
    _, labels = lab_collate([ds[0], ds[1]])
    assert labels == {}


def test_make_splits_deterministic_and_by_drive():
    names = [f"navd_session_2026101{i}_000000" for i in range(8)]
    a = make_splits(names, holdout=2, seed=0)
    b = make_splits(list(reversed(names)), holdout=2, seed=0)
    assert a == b                                   # order of the input does not matter
    assert len(a["val"]) == 2 and len(a["train"]) == 6
    assert set(a["train"]) | set(a["val"]) == set(names)
    assert not set(a["train"]) & set(a["val"])
    assert make_splits(names, holdout=2, seed=1) != a
    frac = make_splits(names, holdout=0.25, seed=0)
    assert len(frac["val"]) == 2 and frac == a
    explicit = make_splits(names, holdout=[names[3], names[5]])
    assert explicit["val"] == sorted([names[3], names[5]])
    assert make_splits(names, holdout=0)["val"] == []
    with pytest.raises(ValueError):
        make_splits(names, holdout=8)
    with pytest.raises(ValueError):
        make_splits(names, holdout=["nope"])
    with pytest.raises(ValueError):
        make_splits(names + [names[0]], holdout=1)


def test_labels_must_cover_every_drive():
    spec = InputSpec(("color_near", "cmd_vel"))
    ta, tb = _ticks("drive_a", 4), _ticks("drive_b", 4)
    with pytest.raises(ValueError, match="drive_b"):
        LabDataset([ta, tb], spec, labels={"drive_a": FakeLabelSet(ta.t_ns, arc(0.3, 0.4))})
    # no labels at all is still fine
    assert LabDataset([ta, tb], spec)[5][1] == {}


def test_all_invalid_set_borrows_sibling_template():
    spec = InputSpec(("color_near",))
    ta, tc = _ticks("drive_a", 4), _ticks("drive_c", 3)
    ds = LabDataset([ta, tc], spec, labels={"drive_a": FakeLabelSet(ta.t_ns, arc(0.3, 0.4)),
                                            "drive_c": FakeLabelSet(tc.t_ns, arc(0.3, 0.4), invalid_last=3)})
    _, lab_c = ds[4]
    assert set(lab_c) == {"path", "rel_t", "psi", "v", "valid"} == set(ds.label_keys())
    assert not bool(lab_c["valid"]) and lab_c["path"].shape == (15, 2) and torch.all(lab_c["path"] == 0)
    np.testing.assert_allclose(lab_c["rel_t"].numpy(), np.arange(1, 16) * 0.2, atol=1e-6)  # the grid, not zeros
    _, labels = lab_collate([ds[0], ds[4], ds[6]])
    assert labels["path"].shape == (3, 15, 2) and labels["valid"].tolist() == [True, False, False]


def test_all_invalid_set_shapes_from_stacked_arrays():
    """A labels.LabelSet carries rel_t / pts[T, N, 2] / psi / v whatever
    ``valid`` says; the template comes from those when no tick is valid."""

    class ArrayLabelSet:
        variant, version = "hindsight", 1

        def __init__(self, t_ns):
            self.t_ns = np.asarray(t_ns, dtype=np.int64)
            self.rel_t = np.arange(1, 16) * 0.2
            self.pts = np.full((len(t_ns), 15, 2), np.nan)
            self.psi = np.full((len(t_ns), 15), np.nan)
            self.v = np.full((len(t_ns), 15), np.nan)

        def get(self, t_ns):
            return {"valid": False, "path": None} if t_ns in self.t_ns else None

    t = _ticks("d", 3)
    ds = LabDataset([t], InputSpec(("odom",)), labels={"d": ArrayLabelSet(t.t_ns)})
    _, lab = ds[1]
    assert set(lab) == {"path", "rel_t", "psi", "v", "valid"} and lab["path"].shape == (15, 2)
    assert not bool(lab["valid"]) and torch.all(lab["path"] == 0) and torch.all(lab["psi"] == 0)
    assert lab_collate([ds[0], ds[2]])[1]["path"].shape == (2, 15, 2)
