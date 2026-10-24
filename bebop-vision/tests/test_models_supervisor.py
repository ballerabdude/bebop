"""Tests for the model supervisor registry/API (no GPU or Edge-LLM needed)."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bebop_vision import models_supervisor as ms  # noqa: E402

# The registry bebop-models.service loads from the checkout.
SHIPPED_REGISTRY = (Path(ms.__file__).resolve().parent.parent
                    / "config" / "models_supervisor.yaml")


class FakeProc:
    """Stand-in for a `tensorrt-edgellm-serve` child."""

    def __init__(self, *args, **kwargs):
        self.pid = 4242
        self.terminated = False
        self.killed = False
        self.returncode = None

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.killed = True


@pytest.fixture()
def spawned(monkeypatch):
    """Capture spawned processes; a backend becomes healthy once spawned."""
    procs: list[FakeProc] = []
    health = {"up": False}

    def fake_popen(*args, **kwargs):
        proc = FakeProc(*args, **kwargs)
        procs.append(proc)
        health["up"] = True
        return proc

    monkeypatch.setattr(ms.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ms.Backend, "health_ok", lambda self: health["up"])
    return procs


def test_supervisor_api(monkeypatch):
    monkeypatch.setattr(ms, "REGISTRY_PATH", SHIPPED_REGISTRY)
    app = ms.build_app(ms.Supervisor())
    client = TestClient(app)

    health = client.get("/healthz").json()
    assert health["ok"] is True
    assert health["budget_gb"] > 0

    backends = client.get("/models").json()["backends"]
    ids = {b["id"] for b in backends}
    assert {"omni-edgellm", "vision"} <= ids
    omni = next(b for b in backends if b["id"] == "omni-edgellm")
    assert omni["kind"] == "edgellm" and omni["state"] == "stopped"
    assert omni["base_url"] == "http://127.0.0.1:8000"
    vision = next(b for b in backends if b["id"] == "vision")
    assert vision["kind"] == "tracked"

    # A tracked backend "loads" as a no-op.
    r = client.post("/models/vision/load")
    assert r.status_code == 200 and r.json()["state"] == "tracked"

    # Unknown backend -> 400.
    assert client.post("/models/nope/load").status_code == 400


def test_edgellm_argv(monkeypatch):
    monkeypatch.setattr(ms, "resolve_edgellm_bin", lambda: "/opt/fake-serve")
    b = ms.Backend(
        {
            "id": "brain",
            "kind": "edgellm",
            "model": "RadixArk/Qwen3.8-27B-NVFP4",
            "cache_dir": "/var/lib/bebop-voice/edgellm",
            "port": 8012,
            "serve_args": ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"],
        }
    )
    argv = b.argv()
    assert argv[0] == "/opt/fake-serve"
    assert argv[1] == "RadixArk/Qwen3.8-27B-NVFP4"
    assert "--cache-dir" in argv and "/var/lib/bebop-voice/edgellm" in argv
    assert argv[argv.index("--port") + 1] == "8012"
    assert argv[-2:] == ["--tool-call-parser", "qwen3_xml"]


def test_edgellm_load_start_and_unload_stop(spawned, monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "REGISTRY_PATH", tmp_path / "absent.yaml")
    sup = ms.Supervisor()
    backend = ms.Backend(
        {"id": "omni", "kind": "edgellm", "model": "m", "port": 8000}
    )
    sup.backends = {"omni": backend}

    app = ms.build_app(sup)
    client = TestClient(app)

    r = client.post("/models/omni/load")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "ready" and body["pid"] == 4242
    assert len(spawned) == 1

    # Loading an already-ready backend is a no-op (no second spawn).
    client.post("/models/omni/load")
    assert len(spawned) == 1

    r = client.post("/models/omni/unload")
    assert r.status_code == 200 and r.json()["state"] == "stopped"
    assert spawned[0].terminated is True


def test_unsupported_kind_is_an_error():
    b = ms.Backend({"id": "x", "kind": "quantum", "port": 9000})
    b.start()
    assert b.state == "error" and "unsupported kind" in b.detail


def test_sleep_endpoint_is_gone(monkeypatch):
    """The vLLM-Omni sleep/wake API was removed with the container runtime."""
    monkeypatch.setattr(ms, "REGISTRY_PATH", SHIPPED_REGISTRY)
    client = TestClient(ms.build_app(ms.Supervisor()))
    assert client.post("/models/omni-edgellm/sleep").status_code == 404
    assert client.post("/models/omni-edgellm/wake").status_code == 404


def test_load_evicts_when_the_budget_is_tight(spawned, monkeypatch, tmp_path):
    monkeypatch.setattr(ms, "REGISTRY_PATH", tmp_path / "absent.yaml")
    monkeypatch.setattr(ms, "GPU_BUDGET_GB", 30.0)
    monkeypatch.setattr(ms, "_mem", lambda: {"MemAvailable": 100.0})
    sup = ms.Supervisor()
    sup.backends = {
        "first": ms.Backend({"id": "first", "kind": "edgellm", "footprint_gb": 25, "port": 8001}),
        "second": ms.Backend({"id": "second", "kind": "edgellm", "footprint_gb": 25, "port": 8002}),
    }
    sup.load("first")
    assert sup.backends["first"].state == "ready"

    sup.load("second")  # 25 + 25 > 30 -> evict `first`
    assert sup.backends["first"].state == "stopped"
    assert sup.backends["second"].state == "ready"


def test_cors_headers_and_preflight(monkeypatch):
    """The app fetches :9094 cross-origin; GETs carry ACAO and POSTs preflight."""
    monkeypatch.setattr(ms, "REGISTRY_PATH", SHIPPED_REGISTRY)
    client = TestClient(ms.build_app(ms.Supervisor()))
    origin = "http://127.0.0.1:5173"

    r = client.get("/models", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"
    r = client.get("/healthz", headers={"Origin": origin})
    assert r.headers["access-control-allow-origin"] == "*"

    pre = client.options(
        "/models/vision/load",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert pre.status_code == 200
    assert pre.headers["access-control-allow-origin"] == "*"
    assert "POST" in pre.headers["access-control-allow-methods"]

    r = client.post("/models/vision/load", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "*"
