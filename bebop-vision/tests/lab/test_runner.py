"""runner.run / Run: iteration, timing, stacking per output kind, save / load."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from _runner_fakes import (
    ARC,
    FakeBatchedCV,
    FakeCV,
    FakeMultiModel,
    FakeSession,
    FakeTicks,
    arc_body,
)
from bebop_vision.lab import runner
from bebop_vision.lab.runner import Run, run
from bebop_vision.lab.types import Classes, Embedding, Grid, InputSpec, Mask, Path2D, Scalars


N_TICKS = int(round(ARC["duration_s"] * ARC["hz"]))


@pytest.fixture
def session(lab_cache):
    return FakeSession(cache=lab_cache)


# --------------------------------------------------------------------------
# run()
# --------------------------------------------------------------------------

def test_run_prepares_once_and_selects_the_spec(session):
    model = FakeCV()
    ticks = session.ticks()
    r = run(session, model, ticks=ticks, save=False, device="cpu")
    assert len(model.prepared) == 1
    calib, cfg = model.prepared[0]
    assert calib is session.calib
    assert cfg == {"device": "cpu"}
    assert model.steps == N_TICKS == len(r)
    # the Ticks were narrowed to exactly the model's modalities (the fake
    # records every select() call it saw)
    assert ticks.selected == []           # select returns a copy, the input is untouched
    assert r.spec == InputSpec(("odom", "cmd_vel"))
    assert r.meta["spec"] == {"modalities": ["odom", "cmd_vel"], "history_s": {}, "tol_ns": 50_000_000}
    assert r.meta["model"] == "cv"
    assert r.meta["session"]["name"] == session.name
    assert r.meta["session"]["identity"] is None      # the fake has no file on disk
    assert r.path is None


def test_run_without_ticks_uses_session_ticks(session):
    r = run(session, FakeCV(), save=False)
    np.testing.assert_array_equal(r.t_ns, FakeTicks(session).t_ns)


def test_run_stacks_paths_and_matches_the_arc(session):
    model = FakeCV(dt_s=0.1)
    r = run(session, model, save=False)
    rel_t, pts, weights = r.paths()
    assert rel_t.shape == (30,)
    assert pts.shape == (N_TICKS, 1, 30, 2)
    assert weights.shape == (N_TICKS, 1)
    assert pts.dtype == np.float32 and rel_t.dtype == np.float32
    np.testing.assert_allclose(weights, 1.0)
    expect = arc_body(ARC["v"], ARC["w"], rel_t)
    for i in range(N_TICKS):
        np.testing.assert_allclose(pts[i, 0], expect, atol=1e-5)
    assert r.kinds() == {"path": Path2D}
    assert r.names_of(Path2D) == ["path"]
    assert r.valid("path").all()


def test_run_times_every_tick_for_an_unbatched_model(session):
    r = run(session, FakeCV(), save=False)
    assert r.step_ms.shape == (N_TICKS,) and np.all(np.isfinite(r.step_ms))
    assert r.sample_ms.shape == (N_TICKS,) and np.all(np.isfinite(r.sample_ms))
    assert np.all(r.step_ms >= 0.0)
    assert r.meta["batch"] == 1
    lat = r.latency_ms()
    assert set(lat) == {"p50", "p90", "p99", "mean", "max", "n"}
    assert lat["n"] == N_TICKS
    assert lat["p50"] <= lat["p90"] <= lat["p99"] <= lat["max"]


def test_run_uses_batch_step_in_chunks(session):
    model = FakeBatchedCV()
    r = run(session, model, save=False, batch=16)
    assert model.batches == [16] * (N_TICKS // 16) + ([N_TICKS % 16] if N_TICKS % 16 else [])
    assert model.steps == N_TICKS
    assert r.meta["batch"] == 16
    # the chunk time is spread evenly over its ticks
    assert np.allclose(r.step_ms[:16], r.step_ms[0])
    assert np.all(np.isfinite(r.step_ms))


def test_run_keeps_missing_outputs_as_none(session):
    model = FakeCV(skip_ticks=(3, 7))
    r = run(session, model, save=False)
    valid = r.valid("path")
    assert not valid[3] and not valid[7] and valid.sum() == N_TICKS - 2
    assert r.outputs["path"][3] is None
    _, pts, weights = r.paths()
    assert np.all(np.isnan(pts[3])) and np.all(np.isnan(weights[3]))
    assert np.all(np.isfinite(pts[4]))


def test_run_keeps_undeclared_outputs(session):
    class Chatty(FakeCV):
        def step(self, sample):
            out = super().step(sample)
            out["extra"] = Scalars({"k": 1.0})
            return out

    r = run(session, Chatty(), save=False)
    assert set(r.outputs) == {"path", "extra"}
    assert r.meta["declared_outputs"] == {"path": "Path2D"}


def test_run_rejects_a_batch_step_of_the_wrong_length(session):
    class Broken(FakeBatchedCV):
        def batch_step(self, samples):
            return []

    with pytest.raises(RuntimeError, match="batch_step returned 0"):
        run(session, Broken(), save=False)


def test_run_progress_writes_to_stderr(session, capsys):
    run(session, FakeCV(), save=False, progress=True)
    err = capsys.readouterr().err
    assert "[lab run] cv:" in err and f"{N_TICKS}/{N_TICKS} ticks" in err


def test_run_records_backbone_mode_and_model_meta(session):
    r = run(session, FakeMultiModel(), save=False)
    assert r.meta["backbone"] == "timm:fake_tiny"
    assert r.meta["mode"] == "frozen"
    assert r.meta["model_meta"] == {"note": "fake"}
    assert r.meta["model_class"].endswith("FakeMultiModel")
    assert r.meta["spec"]["history_s"] == {"odom": 0.5}


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------

def test_run_saves_to_the_session_cache_by_default(session, lab_cache):
    r = run(session, FakeCV())
    assert r.path is not None
    assert r.path.parent == lab_cache / session.name / "runs"
    assert r.path.name.startswith("cv-")
    assert (r.path / "outputs.npz").is_file() and (r.path / "run.json").is_file()
    meta = json.loads((r.path / "meta.json").read_text())
    assert meta["producer"] == runner.PRODUCER and meta["model"] == "cv"
    assert runner.list_runs(session) == [r.path]
    # a second run in the same second gets a distinct directory
    r2 = run(session, FakeCV())
    assert r2.path != r.path
    assert len(runner.list_runs(session)) == 2


def test_default_run_dir_falls_back_to_env_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("BEBOP_LAB_CACHE", str(tmp_path / "root"))
    s = FakeSession(cache=None)       # no cache_dir: a duck-typed session
    d = runner.default_run_dir(s, "cv")
    assert d.parent == tmp_path / "root" / s.name / "runs"


def test_run_dirs_use_cache_dir_without_building_the_index(session, lab_cache):
    """list_runs / default_run_dir read Session.cache_dir; they never call
    index(), which on a real Session builds the whole MCAP index."""
    assert runner.list_runs(session) == []
    d = runner.default_run_dir(session, "cv")
    assert d.parent == lab_cache / session.name / "runs"
    assert session.index_calls == 0
    assert not (lab_cache / session.name).exists()      # nothing created by a read
    # an explicit cache_dir wins over the environment root
    session.cache_dir = lab_cache / "elsewhere"
    assert runner.default_run_dir(session, "cv").parent == lab_cache / "elsewhere" / "runs"


def test_model_meta_never_costs_the_run(session, tmp_path):
    class NumpyMeta(FakeCV):
        def meta(self):
            return {"amp": np.bool_(True), "n": np.int64(3), "arr": np.arange(2)}

    r = run(session, NumpyMeta(), out_dir=tmp_path / "np")
    assert r.path is not None
    assert r.meta["model_meta"] == {"amp": True, "n": 3, "arr": [0, 1]}
    assert json.loads((r.path / "run.json").read_text())["model_meta"]["amp"] is True

    class Unserialisable(FakeCV):
        def meta(self):
            return {"obj": object()}

    r = run(session, Unserialisable(), out_dir=tmp_path / "obj")
    assert r.path is not None and "model_meta" not in r.meta

    class Raising(FakeCV):
        def meta(self):
            raise RuntimeError("no")

    r = run(session, Raising(), out_dir=tmp_path / "raise")
    assert r.path is not None and "model_meta" not in r.meta


def test_save_load_round_trip_paths(session, tmp_path):
    r = run(session, FakeCV(with_psi=True), save=False)
    d = r.save(tmp_path / "run")
    assert d == tmp_path / "run" and r.path == d
    back = Run.load(d)
    np.testing.assert_array_equal(back.t_ns, r.t_ns)
    np.testing.assert_allclose(back.step_ms, r.step_ms)
    np.testing.assert_allclose(back.sample_ms, r.sample_ms)
    assert back.spec == r.spec
    assert back.meta["model"] == "cv" and back.meta["producer"] == runner.PRODUCER
    assert back.meta["outputs"]["path"] == {"kind": "Path2D", "K": 1, "N": 30}
    assert back.meta["timing"]["latency"]["n"] == N_TICKS
    for a, b in zip(r.outputs["path"], back.outputs["path"]):
        np.testing.assert_array_equal(a.rel_t, b.rel_t)
        np.testing.assert_array_equal(a.pts, b.pts)
        np.testing.assert_array_equal(a.weights, b.weights)
        # Path2D leaves psi in the model's dtype; the store is float32
        np.testing.assert_allclose(a.psi, b.psi, rtol=1e-6)
        assert b.psi.dtype == np.float32 and b.v is None
    with np.load(d / "outputs.npz") as z:
        assert set(z.files) == {"t_ns", "step_ms", "sample_ms", "path.valid", "path.rel_t",
                                "path.pts", "path.weights", "path.psi"}
        assert z["path.pts"].shape == (N_TICKS, 1, 30, 2)


def test_save_load_round_trip_every_kind(session, tmp_path):
    model = FakeMultiModel(none_at=5)
    r = run(session, model, save=False)
    d = r.save(tmp_path / "multi")
    back = Run.load(d)
    assert back.kinds() == {"path": Path2D, "seg": Mask, "prob": Mask, "emb": Embedding,
                            "cls": Classes, "grid": Grid, "stats": Scalars}
    assert back.outputs["path"][5] is None and back.outputs["emb"][5] is None
    assert back.valid("path").sum() == N_TICKS - 1

    with np.load(d / "outputs.npz") as z:
        assert z["path.pts"].shape == (N_TICKS, 3, 6, 2)
        assert z["path.weights"].shape == (N_TICKS, 3)
        assert z["seg.values"].dtype == np.uint8 and z["seg.values"].shape == (N_TICKS, 8, 12)
        assert z["prob.values"].dtype == np.float16 and z["prob.values"].shape == (N_TICKS, 3, 4, 6)
        assert z["emb.vector"].shape == (N_TICKS, 16)
        assert z["cls.logits"].shape == (N_TICKS, 4)
        assert z["grid.values"].shape == (N_TICKS, 2, 5, 7)
        assert z["stats.values"].dtype.names == ("speed", "turn", "extra")
        assert np.all(np.isnan(z["emb.vector"][5]))

    om = back.meta["outputs"]
    assert om["seg"] == {"kind": "Mask", "camera": "near", "classes": ["floor", "wall", "thing"],
                         "stride": 4, "crop": [0, 32, 0, 48], "dtype": "uint8"}
    assert om["prob"]["crop"] is None and om["prob"]["dtype"] == "float16"
    assert om["emb"] == {"kind": "Embedding", "source": "timm:fake_tiny", "D": 16}
    assert om["cls"]["names"] == ["w", "x", "y", "z"]
    assert om["grid"] == {"kind": "Grid", "cell_m": 0.1, "origin_xy": [0.0, -0.35], "channels": ["occ", "free"]}
    assert om["stats"]["keys"] == ["speed", "turn", "extra"]

    for i in (0, 1, 6):
        a, b = r.outputs["path"][i], back.outputs["path"][i]
        np.testing.assert_array_equal(a.pts, b.pts)
        np.testing.assert_allclose(a.weights, b.weights, rtol=1e-6)   # float64 in, float32 stored
        np.testing.assert_array_equal(a.best, b.best)
        a, b = r.outputs["seg"][i], back.outputs["seg"][i]
        np.testing.assert_array_equal(a.values, b.values)
        assert (b.camera, b.classes, b.stride, b.crop) == ("near", ("floor", "wall", "thing"), 4, (0, 32, 0, 48))
        a, b = r.outputs["prob"][i], back.outputs["prob"][i]
        np.testing.assert_allclose(a.values, b.values.astype(np.float32), atol=1e-3)
        a, b = r.outputs["emb"][i], back.outputs["emb"][i]
        np.testing.assert_array_equal(a.vector, b.vector)
        assert b.source == "timm:fake_tiny"
        a, b = r.outputs["cls"][i], back.outputs["cls"][i]
        np.testing.assert_array_equal(a.logits, b.logits)
        assert b.names == ("w", "x", "y", "z")
        a, b = r.outputs["grid"][i], back.outputs["grid"][i]
        np.testing.assert_array_equal(a.values, b.values)
        assert (b.cell_m, b.origin_xy, b.channels) == (0.1, (0.0, -0.35), ("occ", "free"))
        a, b = r.outputs["stats"][i], back.outputs["stats"][i]
        # the structured array carries the union of keys; absent ones are NaN
        for k, v in a.values.items():
            assert b.values[k] == v
        for k in set(b.values) - set(a.values):
            assert math.isnan(b.values[k])

    # the stacked view of a fresh Run equals what was saved
    arrays, meta = r.stacked("emb")
    np.testing.assert_array_equal(arrays["vector"], back.stacked("emb")[0]["vector"])
    assert meta == om["emb"]


def test_run_json_is_plain_json(session, tmp_path):
    r = run(session, FakeCV(skip_ticks=(0,)), save=False)
    r.save(tmp_path / "j")
    meta = json.loads((tmp_path / "j" / "run.json").read_text())
    assert meta["t_ns"] == [int(t) for t in r.t_ns]
    assert len(meta["timing"]["step_ms"]) == N_TICKS
    assert meta["n_ticks"] == N_TICKS
    assert meta["outputs"]["path"]["kind"] == "Path2D"


def test_save_rejects_mixed_path_grids(session, tmp_path):
    r = run(session, FakeCV(), save=False)
    bad = r.outputs["path"][2]
    r.outputs["path"][2] = Path2D(bad.rel_t[:-1], bad.pts[:, :-1])
    with pytest.raises(ValueError, match="Path2D shape"):
        r.save(tmp_path / "bad")


def test_pack_rejects_class_ids_outside_uint8():
    """Negative ids (the usual -1 ignore index) must not wrap to 255."""
    r = Run([1], {"seg": [Mask("near", np.array([[-1, 0], [1, 2]]))]}, {})
    with pytest.raises(ValueError, match="-1..2"):
        r.stacked("seg")
    r = Run([1], {"seg": [Mask("near", np.array([[0, 256]]))]}, {})
    with pytest.raises(ValueError, match="uint8"):
        r.stacked("seg")
    r = Run([1], {"seg": [Mask("near", np.array([[0, 255]]))]}, {})
    assert r.stacked("seg")[0]["values"].dtype == np.uint8


def test_load_rejects_another_producer(session, tmp_path):
    r = run(session, FakeCV(), save=False)
    d = r.save(tmp_path / "old")
    meta = json.loads((d / "run.json").read_text())
    meta["producer"] = "runner@0"
    (d / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="runner@0"):
        Run.load(d)


def test_run_constructor_checks_lengths():
    with pytest.raises(ValueError, match="2 entries for 3 ticks"):
        Run(np.arange(3), {"path": [None, None]}, {})


def test_paths_needs_a_single_path_output(session):
    r = run(session, FakeMultiModel(), save=False)
    rel_t, pts, _ = r.paths()           # exactly one Path2D output: fine
    assert pts.shape[1:] == (3, 6, 2)
    with pytest.raises(KeyError):
        r.paths("nope")
    r.outputs["path2"] = list(r.outputs["path"])
    with pytest.raises(KeyError, match="exactly one Path2D"):
        r.paths()
