"""FastAPI dashboard API tests on the synthetic session tree.

Reuses the same fixture shape as test_dashboard_core.py (tiny npz/JPEG
session) but exercises the HTTP contract via TestClient — the React app
consumes exactly these routes. Model-validation routes are tested with a
zero-weight ONNX built on the fly when onnxruntime is available; skipped
otherwise so the core API contract stays testable everywhere.
"""

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import dashboard_api  # noqa: E402
from dashboard_core import DatasetDashboard  # noqa: E402

S20 = 1788716672926769002
STAMP = str(S20)
STRIDE = 4
COLOR_W, COLOR_H = 64, 48


def _tiny_grid(v):
    return np.full((60, 60), v, np.uint8)


def _tiny_jpeg(value):
    ok, jpg = cv2.imencode(".jpg", np.full((COLOR_H, COLOR_W, 3), value, np.uint8))
    assert ok
    return jpg.tobytes()


@pytest.fixture
def client(tmp_path):
    """API client against a synthetic 2-tick session (fake ray LUT injected
    so the overlay route works without config/raylut_near.npz)."""
    sess = tmp_path / "navd_session_test"
    for sub in ("labels", "color", "color_far", "depth"):
        (sess / sub).mkdir(parents=True)
    rows = []
    for k, off in enumerate((0, 100_000_000)):
        stamp = S20 + off
        s20 = f"{stamp:020d}"
        np.savez_compressed(
            sess / "labels" / f"{s20}.npz", fused=_tiny_grid(k % 3),
            teacher=_tiny_grid(1), sem_near=_tiny_grid(0),
            sem_far=_tiny_grid(0), floor_near=_tiny_grid(1),
            floor_far=_tiny_grid(1),
            disagree=_tiny_grid(1 if k else 0), unconfirmed=_tiny_grid(0))
        depth = np.full((480, 848), 1500, np.uint16)
        depth[:10, :10] = 0
        np.savez_compressed(sess / "depth" / f"{s20}.npz", near=depth, far=depth)
        (sess / "color" / f"{s20}.jpg").write_bytes(_tiny_jpeg(40 + k))
        (sess / "color_far" / f"{s20}.jpg").write_bytes(_tiny_jpeg(90))
        rows.append({"stamp_ns": stamp,
                     "cmd_vel": {"vx": 0.1 * k, "wz": 0.0},
                     "odom": {"x": 0.0, "y": 0.0, "theta": 0.0},
                     "goal": {"type": "none"}})
    with open(sess / "manifest.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_px = (COLOR_H // STRIDE) * (COLOR_W // STRIDE)
    lc = np.full(n_px, -1, np.int32)
    lc[n_px // 4:n_px // 2] = 0
    lc[n_px // 2:] = 61
    dash = DatasetDashboard(
        tmp_path, load_lut=lambda role: dict(land_cells=lc, n_px=n_px))
    dashboard_api.dash = dash
    dashboard_api._model["sess"] = None
    dashboard_api._validate_jobs.clear()
    return TestClient(dashboard_api.app)


def test_sessions_and_ticks(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    s = r.json()["sessions"]
    assert [x["name"] for x in s] == ["navd_session_test"]
    r = client.get("/api/session/navd_session_test/ticks")
    rows = r.json()
    assert [x["stamp"] for x in rows] == \
        [f"{S20 + o:020d}" for o in (0, 100_000_000)]


def test_tick_payload_roundtrip(client):
    r = client.get(f"/api/tick/navd_session_test/{STAMP}")
    assert r.status_code == 200
    p = r.json()
    assert p["stamp"] == STAMP
    jpg = base64.b64decode(p["color_near"])
    assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) is not None


def test_tick_unknown_session_404(client):
    assert client.get("/api/tick/no_such/123").status_code == 404


def test_overlay_route(client):
    r = client.get(f"/api/overlay/navd_session_test/{STAMP}")
    assert r.status_code == 200, r.json()
    assert r.json()["src"] == "fused"


def test_hand_save_and_clear_route(client):
    hand = np.where(np.mgrid[0:60, 0:60][0] < 30, 1, 0).astype(np.uint8)
    r = client.post(f"/api/tick/navd_session_test/{STAMP}/hand",
                    json={"grid": hand.tolist()})
    assert r.status_code == 200 and r.json()["ok"]
    with np.load(dashboard_api.dash.root / "navd_session_test" /
                 "labels" / f"{int(STAMP):020d}.npz") as z:
        assert "hand" in z.files
    r = client.post(f"/api/tick/navd_session_test/{STAMP}/hand",
                    json={"clear": True})
    assert r.status_code == 200
    with np.load(dashboard_api.dash.root / "navd_session_test" /
                 "labels" / f"{int(STAMP):020d}.npz") as z:
        assert "hand" not in z.files


def test_gridtex_route(client):
    """The projected-image texture: 60x60 PNG covering exactly the cells
    the fake LUT lands in (flat 0 -> r0c0, flat 61 -> r1c1)."""
    r = client.get(f"/api/gridtex/navd_session_test/{STAMP}")
    assert r.status_code == 200, r.json()
    out = r.json()
    assert out["png"] is not None
    tex = cv2.imdecode(
        np.frombuffer(base64.b64decode(out["png"]), np.uint8),
        cv2.IMREAD_COLOR)
    assert tex.shape == (60, 60, 3)
    assert out["coverage"] == pytest.approx(2 / 3600, abs=1e-3)
    # flat 0 -> (row 0, col 0); flat 61 -> (row 1, col 1) — non-black there
    assert tex[0, 0].max() > 0
    assert tex[1, 1].max() > 0
    assert tex[30, 30].max() == 0          # uncovered cells stay black
    # stamp without a color image -> clean error, not 500
    r = client.get("/api/gridtex/navd_session_test/123")
    assert r.status_code in (200, 404)


def test_tick_payload_grids_param(client):
    r = client.get(f"/api/tick/navd_session_test/{STAMP}?grids=false")
    assert r.status_code == 200
    assert r.json()["grids"] == {}
    assert r.json()["color_near"] is not None   # media still present
    r = client.get(f"/api/tick/navd_session_test/{STAMP}")
    assert "fused" in r.json()["grids"]


def test_model_load_missing_file(client):
    r = client.post("/api/model/load", json={"path": "weights/nope.onnx"})
    assert r.status_code == 404


def test_validate_requires_model(client):
    r = client.get(f"/api/validate/navd_session_test/{STAMP}")
    assert r.status_code == 400
    assert "no model loaded" in r.json()["detail"]


def _build_zero_onnx(tmp_path):
    """Export a tiny stand-in ONNX with the navd input/output contract.

    Real NavdUNet needs torch weights; a faithful shape contract is all
    the API test needs (preprocessing + route plumbing), so export the
    actual NavdUNet architecture with random init — inference output is
    garbage but well-formed.
    """
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    ort = pytest.importorskip("onnxruntime")
    import onnx.helper as h

    def tensor(name, dims):
        return h.make_tensor_value_info(name, onnx.TensorProto.FLOAT, dims)

    # identity-of-argmax stand-in: constant logits favoring class 1
    graph = h.make_graph(
        [h.make_node("Identity", ["logits_c"], ["logits"])], "navd_test",
        [tensor("depth_near", [1, 1, 240, 424]),
         tensor("depth_far", [1, 1, 240, 424]),
         tensor("color", [1, 3, 240, 424]),
         tensor("goal", [1, 1, 60, 60])],
        [tensor("logits", [1, 3, 60, 60])],
        initializer=[h.make_tensor(
            "logits_c", onnx.TensorProto.FLOAT, [1, 3, 60, 60],
            np.stack([np.full((60, 60), 0.0), np.full((60, 60), 1.0),
                      np.full((60, 60), 0.0)]).astype(np.float32)
            .reshape(1, 3, 60, 60).ravel().tolist())])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    model.ir_version = 10
    p = tmp_path / "navd_test.onnx"
    onnx.save(model, str(p))
    # sanity: session must run
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    sess.run(None, {"depth_near": np.zeros((1, 1, 240, 424), np.float32),
                    "depth_far": np.zeros((1, 1, 240, 424), np.float32),
                    "color": np.zeros((1, 3, 240, 424), np.float32),
                    "goal": np.zeros((1, 1, 60, 60), np.float32)})
    return p


def test_validate_tick_and_sweep(client, tmp_path):
    """Full validation path with a stand-in ONNX (all-navigable model)."""
    p = _build_zero_onnx(tmp_path)
    r = client.post("/api/model/load", json={"path": str(p)})
    assert r.status_code == 200 and r.json()["ok"]

    r = client.get(f"/api/validate/navd_session_test/{STAMP}")
    assert r.status_code == 200, r.json()
    v = r.json()
    model = np.asarray(v["model"], np.uint8)
    assert model.shape == (60, 60)
    assert model.max() == 1                      # stand-in favors class 1
    assert v["label_key"] == "fused"
    assert v["frac_navigable"] == pytest.approx(1.0)
    # the runtime's self dead-disc carve is mirrored in replay: the
    # stand-in is already all-navigable, so it's a no-op here — but the
    # field must be present and match the rig's min_range disc
    assert v["self_carved_cells"] == pytest.approx(192, abs=40)
    # all-navigable stand-in trips the runtime's plausibility gate — the
    # grid STILL comes back (that is the whole point: see what the model
    # made + why the drive node would hold)
    assert v["gate_rejected"] is True
    assert "drive node would hold" in v["gate_reason"]
    assert np.asarray(v["prob"][1], np.uint8).shape == (60, 60)
    agree = np.asarray(v["agreement"], np.uint8)
    # label tick 0 is all class 0 -> model (class 1) differs everywhere
    assert agree.max() == 2
    assert v["agreement_pct"] == pytest.approx(0.0)

    # sweep over both ticks, then results file round-trips
    r = client.post("/api/validate/navd_session_test/run",
                    json={"path": str(p)})
    assert r.status_code == 200
    import time
    for _ in range(100):
        st = client.get("/api/validate/navd_session_test/status").json()
        if not st.get("running"):
            break
        time.sleep(0.05)
    assert not st.get("running"), st
    assert st.get("error") is None, st
    files = st["results_files"]
    assert files and files[0].startswith("validation_")
    res = client.get(f"/api/validate/navd_session_test/results?file={files[0]}").json()
    assert res["ticks_scored"] == 2
    assert res["ticks_total"] == 2
    assert len(res["per_class_iou"]) == 3
    assert res["failures"] == []
    # both ticks tripped the frac gate (all-navigable stand-in)
    assert len(res["gate_rejected"]) == 2
    assert "hold" in res["gate_rejected"][0]["reason"]
    # bad results filename refused (path-traversal guard)
    assert client.get("/api/validate/navd_session_test/results?file=../x.json"
                      ).status_code == 400
