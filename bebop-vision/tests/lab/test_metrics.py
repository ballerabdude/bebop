"""metrics: ADE / FDE against arc labels with known error, IoU, leaderboard, worst."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from _runner_fakes import ARC, FakeCV, FakeLabelSet, FakeSession, arc_body
from bebop_vision.lab.metrics import (
    leaderboard,
    metric_name,
    parse_metric,
    path_errors,
    per_tick_errors,
    score_masks,
    score_paths,
    worst,
)
from bebop_vision.lab.runner import Run, run
from bebop_vision.lab.types import Mask, Path2D

N_TICKS = int(round(ARC["duration_s"] * ARC["hz"]))
HZ = ARC["hz"]
# ticks whose remaining recording reaches h seconds: remaining = (N-1-i)/hz >= h
N_AT = {h: sum(1 for i in range(N_TICKS) if (N_TICKS - 1 - i) / HZ >= h - 1e-9) for h in (1.0, 2.0, 3.0)}


@pytest.fixture
def session():
    return FakeSession()


@pytest.fixture
def cv_run(session):
    return run(session, FakeCV(dt_s=0.1), save=False)


def _stat(df: pl.DataFrame, metric: str, h: float, col: str):
    row = df.filter((pl.col("metric") == metric) & (pl.col("horizon") == h))
    assert row.height == 1
    return row[col][0]


# --------------------------------------------------------------------------
# score_paths
# --------------------------------------------------------------------------

def test_cv_matches_exact_arc_labels(session, cv_run):
    labels = FakeLabelSet(session.ticks())
    df = score_paths(cv_run, labels)
    assert df.columns == ["metric", "horizon", "p50", "mean", "p90", "n"]
    assert df.height == 6
    assert sorted(df["metric"].unique().to_list()) == ["ade", "fde"]
    for m in ("ade", "fde"):
        for h in (1.0, 2.0, 3.0):
            for s in ("p50", "mean", "p90"):
                assert abs(_stat(df, m, h, s)) < 1e-5, (m, h, s)
            # a label truncated by the end of the recording does not have
            # the horizon; n counts the ticks that do
            assert _stat(df, m, h, "n") == N_AT[h]
    assert N_AT == {1.0: 60, 2.0: 40, 3.0: 20}


def test_euler_odom_labels_are_close_to_the_arc(session, cv_run):
    """Labels from the synth's Euler-integrated odom: within the
    discretisation error (v w dt / 2 per second ~ 3 mm/s)."""
    labels = FakeLabelSet(session.ticks(), source="odom")
    df = score_paths(cv_run, labels)
    assert 0.0 < _stat(df, "fde", 3.0, "p50") < 0.02
    assert _stat(df, "ade", 3.0, "mean") < 0.01
    assert _stat(df, "ade", 1.0, "p90") < 0.005


def test_offset_labels_have_the_known_error(session, cv_run):
    labels = FakeLabelSet(session.ticks(), offset=(0.0, 0.1))
    df = score_paths(cv_run, labels)
    for m in ("ade", "fde"):
        for h in (1.0, 2.0, 3.0):
            for s in ("p50", "mean", "p90"):
                assert _stat(df, m, h, s) == pytest.approx(0.1, abs=1e-5)
    labels = FakeLabelSet(session.ticks(), offset=(0.03, 0.04))
    df = score_paths(cv_run, labels)
    assert _stat(df, "fde", 2.0, "mean") == pytest.approx(0.05, abs=1e-5)


def test_prediction_is_interpolated_onto_the_label_grid(session):
    """A 0.3 s prediction grid against 0.2 s labels: linear interpolation of
    the arc chord, error well under the chord sag (v w dt^2 / 8 ~ 1.4 mm)."""
    r = run(session, FakeCV(dt_s=0.3), save=False)
    labels = FakeLabelSet(session.ticks())
    df = score_paths(r, labels)
    assert _stat(df, "ade", 3.0, "p90") < 2e-3
    assert _stat(df, "ade", 3.0, "p90") > 0.0
    # 3.0 = 10 x 0.3 is on the prediction grid: exact there
    assert _stat(df, "fde", 3.0, "p90") < 1e-5
    assert _stat(df, "fde", 3.0, "n") == N_AT[3.0]


def test_short_prediction_leaves_long_horizons_unscored(session):
    r = run(session, FakeCV(horizon_s=2.0), save=False)
    labels = FakeLabelSet(session.ticks())
    df = score_paths(r, labels)
    assert _stat(df, "fde", 2.0, "n") == N_AT[2.0]
    assert _stat(df, "ade", 2.0, "p50") == pytest.approx(0.0, abs=1e-5)
    assert _stat(df, "fde", 3.0, "n") == 0 and _stat(df, "ade", 3.0, "n") == 0
    assert _stat(df, "fde", 3.0, "p50") is None and _stat(df, "ade", 3.0, "mean") is None


def test_score_paths_accepts_mappings_and_custom_horizons(session, cv_run):
    labels = FakeLabelSet(session.ticks(), offset=(0.2, 0.0))
    preds = {int(t): p for t, p in zip(cv_run.t_ns, cv_run.outputs["path"])}
    df = score_paths(preds, labels.paths(), horizons=(0.5, 1.5))
    assert df["horizon"].to_list() == [0.5, 1.5, 0.5, 1.5]
    assert _stat(df, "ade", 1.5, "p50") == pytest.approx(0.2, abs=1e-5)
    # a list of pairs works too
    df2 = score_paths(list(preds.items()), labels, horizons=(1.0,))
    assert _stat(df2, "fde", 1.0, "n") == N_AT[1.0]


def test_score_paths_with_baseline_adds_a_source_column(session, cv_run):
    labels = FakeLabelSet(session.ticks())
    bad = run(session, FakeCV(name="bad"), save=False)
    for i, p in enumerate(bad.outputs["path"]):
        bad.outputs["path"][i] = Path2D(p.rel_t, p.pts + np.array([0.0, 0.25], np.float32))
    df = score_paths(bad, labels, baseline=cv_run)
    assert df.columns == ["metric", "horizon", "p50", "mean", "p90", "n", "source"]
    assert df.height == 12
    model = df.filter(pl.col("source") == "model")
    base = df.filter(pl.col("source") == "baseline")
    assert _stat(model, "fde", 3.0, "p50") == pytest.approx(0.25, abs=1e-5)
    assert _stat(base, "fde", 3.0, "p50") == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------
# per-tick errors, worst
# --------------------------------------------------------------------------

def test_per_tick_errors_skips_unlabelled_and_unpredicted_ticks(session):
    r = run(session, FakeCV(skip_ticks=(1, 2)), save=False)
    labels = FakeLabelSet(session.ticks())
    df = per_tick_errors(r, labels)
    assert df.columns == ["t_ns", "motion_m", "ade@1.0", "fde@1.0", "ade@2.0", "fde@2.0", "ade@3.0", "fde@3.0"]
    # the last ticks have no label (less than one 0.2 s grid step ahead),
    # ticks 1 and 2 no prediction
    unlabelled = sum(1 for i in range(N_TICKS) if (N_TICKS - 1 - i) / HZ < 0.2 - 1e-9)
    assert unlabelled == 4
    assert df.height == N_TICKS - unlabelled - 2
    assert int(r.t_ns[1]) not in df["t_ns"].to_list()
    assert df["fde@3.0"].null_count() == df.height - N_AT[3.0] + 2
    assert df["fde@1.0"].drop_nulls().len() == N_AT[1.0] - 2


def test_worst_returns_the_perturbed_ticks(session, cv_run):
    ticks = session.ticks()
    labels = FakeLabelSet(ticks, per_tick_offset={5: (0.0, 0.5), 10: (0.4, 0.0), 15: (0.0, -0.3)})
    w = worst(cv_run, labels, n=3)
    assert [t for t, _ in w] == [int(ticks.t_ns[5]), int(ticks.t_ns[10]), int(ticks.t_ns[15])]
    assert [round(v, 5) for _, v in w] == [0.5, 0.4, 0.3]
    assert all(isinstance(t, int) and isinstance(v, float) for t, v in w)
    # a horizon the perturbed tick lacks is not a candidate: tick 70 has
    # 0.45 s of recording left, so it is the worst at fde@0.4 and absent
    # from fde@3.0
    labels = FakeLabelSet(ticks, per_tick_offset={70: (0.0, 0.9)})
    assert worst(cv_run, labels, n=1, metric="fde@0.4")[0][0] == int(ticks.t_ns[70])
    assert worst(cv_run, labels, n=1, metric="fde@3.0")[0][1] < 1e-5
    assert worst(cv_run, labels, n=0) == []


def test_path_errors_anchors_the_prediction_at_the_origin():
    """A label point before the first predicted point is interpolated from
    the robot (0, 0), not clamped to the first prediction."""
    pred = Path2D(np.array([0.5, 1.0]), np.array([[1.0, 0.0], [2.0, 0.0]]))
    label = Path2D(np.array([0.25, 0.5, 1.0]), np.array([[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    e = path_errors(pred, label, horizons=(0.5, 1.0))
    assert e == {"ade@0.5": 0.0, "fde@0.5": 0.0, "ade@1.0": 0.0, "fde@1.0": 0.0}
    # a label point past the prediction's end is not extrapolated
    label = Path2D(np.array([0.5, 1.0, 1.5]), np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    e = path_errors(pred, label, horizons=(1.0, 1.5))
    assert e["fde@1.0"] == 0.0 and math.isnan(e["ade@1.5"]) and math.isnan(e["fde@1.5"])
    # the horizon must sit on a label point (within half a grid step): with
    # a 0.5 s grid, 1.2 snaps to 1.0 and 2.0 is 0.5 past the last point
    e = path_errors(pred, label, horizons=(1.2, 2.0))
    assert e["fde@1.2"] == 0.0 and math.isnan(e["fde@2.0"]) and math.isnan(e["ade@2.0"])
    # ADE averages the points inside the horizon only
    pred = Path2D(np.array([1.0, 2.0]), np.array([[1.0, 0.0], [2.0, 0.0]]))
    label = Path2D(np.array([1.0, 2.0]), np.array([[1.0, 0.1], [2.0, 0.3]]))
    e = path_errors(pred, label, horizons=(1.0, 2.0))
    assert e["ade@1.0"] == pytest.approx(0.1) and e["fde@1.0"] == pytest.approx(0.1)
    assert e["ade@2.0"] == pytest.approx(0.2) and e["fde@2.0"] == pytest.approx(0.3)


def _held_label(trunc_s: float, horizon_s: float = 3.0, dt_s: float = 0.2, **flags) -> dict:
    """A label the way labels.py writes a truncated one: the full rel_t grid,
    the stop pose held from trunc_s on with v = 0, valid, truncated_at_s set."""
    rel_t = np.round(np.arange(1, int(round(horizon_s / dt_s)) + 1) * dt_s, 6)
    pts = arc_body(ARC["v"], ARC["w"], rel_t)
    real = rel_t <= trunc_s + 1e-9
    stop = pts[real][-1] if real.any() else np.zeros(2)
    pts = np.where(real[:, None], pts, stop)
    v = np.where(real, ARC["v"], 0.0)
    return {"path": Path2D(rel_t, pts[None], v=v[None]), "valid": True,
            "truncated_at_s": trunc_s, **flags}


def test_truncated_labels_are_scored_up_to_the_truncation_only(session, cv_run):
    """A real-style truncated label (held stop pose) does not score the
    stop pose: the horizons past truncated_at_s are NaN for that tick."""
    pred = cv_run.outputs["path"][0]
    e = path_errors(pred, _held_label(1.0)["path"])            # the raw label: held pose scored
    assert e["fde@3.0"] == pytest.approx(0.584, abs=1e-3)
    t0 = int(cv_run.t_ns[0])
    labels = {t0: _held_label(1.0)}
    df = per_tick_errors(cv_run, labels)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["ade@1.0"] == pytest.approx(0.0, abs=1e-6) and row["fde@1.0"] == pytest.approx(0.0, abs=1e-6)
    assert row["ade@2.0"] is None and row["fde@2.0"] is None
    assert row["ade@3.0"] is None and row["fde@3.0"] is None
    # off-grid truncation keeps the real points before it
    df = per_tick_errors(cv_run, {t0: _held_label(1.9)})
    assert df.row(0, named=True)["fde@2.0"] is None        # 2.0 is 0.1 s past the last real point
    assert df.row(0, named=True)["fde@1.0"] == pytest.approx(0.0, abs=1e-6)
    # the horizon reported for a dataset follows: n@3.0 counts untruncated ticks only
    labels = {int(t): _held_label(3.0 if i % 2 else 1.0) for i, t in enumerate(cv_run.t_ns)}
    df = score_paths(cv_run, labels)
    assert _stat(df, "fde", 1.0, "n") == N_TICKS
    assert _stat(df, "fde", 3.0, "n") == N_TICKS // 2
    # no truncation: None (FakeLabelSet) or NaN (the real flags array) both mean untruncated
    for none in (None, math.nan):
        rec = _held_label(3.0)
        rec["truncated_at_s"] = none
        df = per_tick_errors(cv_run, {t0: rec})
        assert df.row(0, named=True)["fde@3.0"] == pytest.approx(0.0, abs=1e-6)
    # the flag may sit under "flags" (FakeLabelSet's layout)
    rec = _held_label(1.0)
    rec["flags"] = {"truncated_at_s": rec.pop("truncated_at_s")}
    assert per_tick_errors(cv_run, {t0: rec}).row(0, named=True)["fde@2.0"] is None
    # truncated before the first grid point: nothing real to score, tick skipped
    assert per_tick_errors(cv_run, {t0: _held_label(0.1)}).height == 0


def test_path_errors_uses_the_best_hypothesis():
    rel_t = np.array([1.0, 2.0])
    pts = np.array([[[1.0, 0.0], [2.0, 0.0]], [[1.0, 1.0], [2.0, 1.0]]])
    label = Path2D(rel_t, np.array([[1.0, 1.0], [2.0, 1.0]]))
    assert path_errors(Path2D(rel_t, pts, weights=np.array([0.2, 0.8])), label, (2.0,))["fde@2.0"] == 0.0
    assert path_errors(Path2D(rel_t, pts, weights=np.array([0.8, 0.2])), label, (2.0,))["fde@2.0"] == 1.0


def test_metric_names():
    assert metric_name("fde", 3) == "fde@3.0"
    assert parse_metric("fde@3.0") == ("fde", 3.0)
    assert parse_metric("ADE@1") == ("ade", 1.0)
    with pytest.raises(ValueError):
        parse_metric("iou@1.0")
    with pytest.raises(ValueError):
        parse_metric("fde")


# --------------------------------------------------------------------------
# score_masks
# --------------------------------------------------------------------------

def _mask_run(name, masks, t_ns, classes=("floor", "wall")):
    out = [Mask("near", m, classes=classes) for m in masks]
    return Run(np.asarray(t_ns), {"seg": out}, {"model": name})


def test_score_masks_known_iou():
    # 4x4: pred floor on the left half, ref floor on the top half
    pred = np.zeros((4, 4), np.int64)
    pred[:, 2:] = 1
    ref = np.zeros((4, 4), np.int64)
    ref[2:, :] = 1
    df = score_masks([pred], [ref], ("floor", "wall"))
    assert df.columns == ["class_id", "class", "iou", "intersection", "union", "n"]
    assert df["class"].to_list() == ["floor", "wall", "mean"]
    # floor: inter 4 (top-left quadrant), union 12; wall the same by symmetry
    assert df["intersection"].to_list()[:2] == [4, 4]
    assert df["union"].to_list()[:2] == [12, 12]
    assert df["iou"].to_list() == pytest.approx([4 / 12, 4 / 12, 4 / 12])
    assert df["n"].to_list() == [1, 1, 1]
    # identical masks: IoU 1; a class absent on both sides has no IoU and is
    # left out of the mean
    df = score_masks([ref, pred], [ref, pred], {0: "floor", 1: "wall", 7: "ghost"})
    assert df.filter(pl.col("class") == "ghost")["iou"][0] is None
    assert df.filter(pl.col("class") == "mean")["iou"][0] == 1.0
    assert df["n"][0] == 2


def test_score_masks_is_dataset_level_and_matches_by_t_ns():
    t = [100, 200, 300]
    a = np.zeros((2, 2), np.int64)
    b = np.ones((2, 2), np.int64)
    pred = _mask_run("p", [a, b, a], t)
    # the reference misses tick 200 and is given out of order; only the two
    # shared ticks are scored, by stamp
    ref = {300: b, 100: a}
    df = score_masks(pred, ref, ("floor", "wall"))
    assert df["n"][0] == 2
    # tick 100 agrees (floor 4/4); tick 300 is pred floor vs ref wall, which
    # adds 4 pred-only pixels to floor's union and 4 ref-only to wall's
    floor = df.filter(pl.col("class") == "floor").row(0, named=True)
    wall = df.filter(pl.col("class") == "wall").row(0, named=True)
    assert (floor["intersection"], floor["union"], floor["iou"]) == (4, 8, 0.5)
    assert (wall["intersection"], wall["union"], wall["iou"]) == (0, 4, 0.0)
    # summed over ticks, not averaged per tick: one empty tick does not
    # pull the class IoU down to a mean of per-image IoUs
    pred2 = _mask_run("p", [a, a], [1, 2])
    ref2 = _mask_run("r", [a, b], [1, 2])
    df = score_masks(pred2, ref2, ("floor", "wall"))
    assert df.filter(pl.col("class") == "floor")["iou"][0] == pytest.approx(4 / 8)


def test_score_masks_argmax_resize_and_ignore():
    ref = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], np.int64)
    # float [C, H, W] prediction at half the resolution, argmaxed and nearest-resized
    prob = np.zeros((2, 1, 2), np.float32)
    prob[0, 0, 0] = 0.9       # left: floor
    prob[1, 0, 1] = 0.9       # right: wall
    df = score_masks([prob], [ref], ("floor", "wall"))
    assert df["iou"].to_list()[:2] == [1.0, 1.0]
    # ignore index excludes reference pixels (right column marked 255)
    ref2 = ref.copy()
    ref2[:, 3] = 255
    pred = np.zeros((2, 4), np.int64)
    df = score_masks([pred], [ref2], ("floor", "wall"), ignore=255)
    assert df.filter(pl.col("class") == "floor")["union"][0] == 6     # 4 ref + 2 pred-only
    assert df.filter(pl.col("class") == "floor")["iou"][0] == pytest.approx(4 / 6)
    # a Run paired with a LabelSet-like object (t_ns + get)
    class Ref:
        t_ns = np.array([1, 2])
        def get(self, t):
            return {"mask": ref} if t == 1 else None
    df = score_masks(_mask_run("p", [ref, ref], [1, 2]), Ref(), ("floor", "wall"))
    assert df["n"][0] == 1 and df["iou"][0] == 1.0


def test_score_masks_honours_the_prediction_crop():
    """A head over a crop is scored inside its crop, not stretched over the
    whole reference image."""
    ref = np.zeros((80, 128), np.int64)
    ref[40:, :] = 1                                   # floor (1) in the bottom half
    pred = Mask("near", np.zeros((8, 12), np.int64), stride=4, crop=(0, 32, 0, 48))
    df = score_masks([pred], [ref], ("wall", "floor"))
    wall = df.filter(pl.col("class") == "wall").row(0, named=True)
    assert (wall["intersection"], wall["union"], wall["iou"]) == (32 * 48, 32 * 48, 1.0)
    assert df.filter(pl.col("class") == "floor")["union"][0] == 0      # not covered: not scored
    assert df.filter(pl.col("class") == "mean")["iou"][0] == 1.0
    # a crop straddling the class boundary at a coarse stride: rows 32..48
    # of the crop are wall above 40 and floor below, 8 crop pixels per 4 rows
    pred = Mask("near", np.zeros((4, 12), np.int64), stride=4, crop=(32, 48, 0, 48))
    df = score_masks([pred], [ref], ("wall", "floor"))
    wall = df.filter(pl.col("class") == "wall").row(0, named=True)
    assert (wall["intersection"], wall["union"]) == (8 * 48, 16 * 48)
    # both sides cropped: scored over the overlap at the reference's density
    ref_m = Mask("near", ref[:, 64:], crop=(0, 80, 64, 128))
    pred = Mask("near", np.ones((80, 96), np.int64), crop=(0, 80, 32, 128))   # all floor
    df = score_masks([pred], [ref_m], ("wall", "floor"))
    floor = df.filter(pl.col("class") == "floor").row(0, named=True)
    assert (floor["intersection"], floor["union"]) == (40 * 64, 80 * 64)
    # disjoint extents score nothing
    pred = Mask("near", np.ones((8, 8), np.int64), crop=(0, 8, 0, 8))
    assert score_masks([pred], [ref_m], ("wall", "floor"))["n"][0] == 0
    # a crop-less Mask covers the whole image at its stride once a crop is in play
    pred = Mask("near", np.zeros((8, 12), np.int64), stride=4, crop=(0, 32, 0, 48))
    ref_m = Mask("near", ref[::2, ::2], stride=2)
    df = score_masks([pred], [ref_m], ("wall", "floor"))
    assert df.filter(pl.col("class") == "wall")["union"][0] == 16 * 24


def test_score_masks_positional_pairing_keeps_none_ticks_in_place():
    a = np.zeros((2, 2), np.int64)
    b = np.ones((2, 2), np.int64)
    r = Run(np.array([1, 2, 3]), {"seg": [Mask("near", a), None, Mask("near", b)]}, {"model": "p"})
    df = score_masks(r, np.stack([a, a, b]), ("floor", "wall"))
    assert df["n"][0] == 2
    assert df["iou"].to_list() == [1.0, 1.0, 1.0]
    # the same against a LabelSet-like reference with a gap of its own, by stamp
    class Ref:
        t_ns = np.array([1, 2, 3])
        def get(self, t):
            return None if t == 1 else {"mask": a if t == 2 else b}
    df = score_masks(r, Ref(), ("floor", "wall"))
    assert df["n"][0] == 1 and df["iou"].to_list() == [None, 1.0, 1.0]
    # an unstamped prediction list against that reference pairs by tick position
    df = score_masks([a, None, b], Ref(), ("floor", "wall"))
    assert df["n"][0] == 1 and df["iou"][1] == 1.0


# --------------------------------------------------------------------------
# leaderboard
# --------------------------------------------------------------------------

def test_leaderboard_is_wide_and_sorted(session, cv_run):
    labels = FakeLabelSet(session.ticks())
    bad = run(session, FakeCV(name="bad"), save=False)
    bad.meta.update(backbone="timm:fake", mode="full")
    for i, p in enumerate(bad.outputs["path"]):
        bad.outputs["path"][i] = Path2D(p.rel_t, p.pts + np.array([0.0, 0.25], np.float32))
    seg = score_masks([np.zeros((2, 2), int)], [np.zeros((2, 2), int)], ("floor", "wall"))
    board = leaderboard([
        (bad.meta, score_paths(bad, labels, baseline=cv_run)),
        (cv_run.meta, score_paths(cv_run, labels)),
        ({"model": "seg", "backbone": "timm:fake", "spec": {"modalities": ["color_near"]}}, seg),
    ])
    assert board["model"].to_list() == ["cv", "bad", "seg"]          # by fde@3.0 p50, nulls last
    assert board.columns[:6] == ["model", "backbone", "modalities", "mode", "session", "n_ticks"]
    assert "fde@3.0 p50" in board.columns and "ade@1.0 p90" in board.columns and "n@3.0" in board.columns
    assert "iou/floor" in board.columns and "iou/mean" in board.columns
    row = board.filter(pl.col("model") == "bad").row(0, named=True)
    assert row["fde@3.0 p50"] == pytest.approx(0.25, abs=1e-5)       # the baseline rows were skipped
    assert row["modalities"] == "odom,cmd_vel" and row["backbone"] == "timm:fake" and row["mode"] == "full"
    assert row["session"] == session.name and row["n@3.0"] == N_AT[3.0]
    assert row["iou/floor"] is None
    seg_row = board.filter(pl.col("model") == "seg").row(0, named=True)
    assert seg_row["iou/floor"] == 1.0 and seg_row["iou/wall"] is None and seg_row["fde@3.0 p50"] is None
    assert seg_row["modalities"] == "color_near"


def test_leaderboard_empty_and_custom_sort(cv_run, session):
    assert leaderboard([]).height == 0
    labels = FakeLabelSet(session.ticks())
    board = leaderboard([(cv_run.meta, score_paths(cv_run, labels))], sort_by="nope")
    assert board.height == 1 and board["model"][0] == "cv"


def test_min_motion_gate_drops_standing_ticks_for_model_and_baseline():
    rel_t = np.arange(1, 16, dtype=np.float32) * 0.2
    still = Path2D(rel_t=rel_t, pts=np.zeros((15, 2), np.float32))
    moving = Path2D(rel_t=rel_t, pts=np.stack([rel_t * 0.3, np.zeros(15)], axis=1))
    labels = {1: still, 2: moving, 3: still}
    good = {1: still, 2: moving, 3: still}            # perfect everywhere
    bad = {1: still, 2: still, 3: still}              # "never moves"
    all_ticks = score_paths(bad, labels)
    gated = score_paths(bad, labels, baseline=good, min_motion_m=0.05)
    fde = lambda df, src=None: df.filter((pl.col("metric") == "fde") & (pl.col("horizon") == 3.0)
                                         & (True if src is None else pl.col("source") == src))
    assert fde(all_ticks)["n"][0] == 3 and fde(all_ticks)["p50"][0] == 0.0
    # only the moving tick is scored, on both sides: the stay-put model is 0.9 m off
    assert fde(gated, "model")["n"][0] == 1
    assert math.isclose(fde(gated, "model")["p50"][0], 0.9, abs_tol=1e-6)
    assert fde(gated, "baseline")["n"][0] == 1 and fde(gated, "baseline")["p50"][0] == 0.0
    assert per_tick_errors(bad, labels)["motion_m"].to_list()[1] == pytest.approx(0.9)
