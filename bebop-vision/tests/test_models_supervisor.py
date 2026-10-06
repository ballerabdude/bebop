"""Tests for the model supervisor registry/API (no docker/GPU needed)."""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bebop_vision import models_supervisor as ms  # noqa: E402


def test_supervisor_api(monkeypatch, tmp_path):
    # Force the built-in registry (ignore any repo config).
    monkeypatch.setattr(ms, "REGISTRY_PATH", tmp_path / "absent.yaml")
    app = ms.build_app(ms.Supervisor())
    client = TestClient(app)

    health = client.get("/healthz").json()
    assert health["ok"] is True
    assert health["budget_gb"] > 0

    backends = client.get("/models").json()["backends"]
    ids = {b["id"] for b in backends}
    assert {"omni-vllm", "vision"} <= ids
    omni = next(b for b in backends if b["id"] == "omni-vllm")
    assert omni["kind"] == "container" and omni["state"] == "stopped"
    vision = next(b for b in backends if b["id"] == "vision")
    assert vision["kind"] == "tracked"

    # A tracked backend "loads" as a no-op.
    r = client.post("/models/vision/load")
    assert r.status_code == 200 and r.json()["state"] == "tracked"

    # Unknown backend -> 400.
    assert client.post("/models/nope/load").status_code == 400


def test_supervisor_sleep_wake(monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "REGISTRY_PATH", tmp_path / "absent.yaml")
    app = ms.build_app(ms.Supervisor())
    client = TestClient(app)

    calls: list[tuple[str, dict]] = []

    class _Resp:
        status_code = 200
        text = "ok"

    def fake_post(url, json=None, timeout=None):  # noqa: A002 - httpx signature
        calls.append((url, json or {}))
        return _Resp()

    monkeypatch.setattr("httpx.post", fake_post)

    r = client.post("/models/omni-vllm/sleep?level=2")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "sleeping"
    assert calls[-1][0].endswith("/v1/omni/sleep") and calls[-1][1]["level"] == 2

    r = client.post("/models/omni-vllm/wake")
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "ready"
    assert calls[-1][0].endswith("/v1/omni/wakeup")

    # A tracked backend (lives in bebop-vision) can't be slept.
    assert client.post("/models/vision/sleep").status_code == 400
