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
