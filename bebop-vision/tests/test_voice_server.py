"""Tests for the voice gateway protocol (stub mode; no model/GPU needed)."""

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bebop_vision import models, voice_server  # noqa: E402


def test_pcm16_to_wav_roundtrip():
    wav = voice_server.pcm16_to_wav(b"\x00\x00\x01\x00\x02\x00", 16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_stub_voice_turn_streams_text_and_audio():
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    with client.websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "utterance_start", "sample_rate": 16000}))
        ws.send_bytes(b"\x00\x00" * 400)
        ws.send_text(json.dumps({"type": "utterance_end"}))

        text = ""
        audio_bytes = 0
        saw_audio_start = False
        saw_done = False
        while True:
            msg = ws.receive()
            if msg.get("type") == "websocket.close":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                audio_bytes += len(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                ev = json.loads(msg["text"])
                if ev["type"] == "text":
                    text += ev["delta"]
                elif ev["type"] == "audio_start":
                    saw_audio_start = True
                    assert ev["sample_rate"] == 24000
                elif ev["type"] == "done":
                    saw_done = True
                    break

    assert saw_done
    assert saw_audio_start
    assert "stub" in text
    assert audio_bytes > 0


def test_stub_voice_empty_utterance_is_done():
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    with client.websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "utterance_start", "sample_rate": 16000}))
        ws.send_text(json.dumps({"type": "utterance_end"}))
        msg = ws.receive()
        assert json.loads(msg["text"])["type"] == "done"


def test_healthz_reports_phase_and_detail():
    status = voice_server.VoiceStatus(
        phase="building",
        detail="building thinker layer 3/48",
        model="qwen3-omni-30b",
        model_downloaded=True,
    )
    app = voice_server.build_app(None, stub=False, status=status)
    client = TestClient(app)
    body = client.get("/healthz").json()
    assert body["phase"] == "building"
    assert body["ok"] is False
    assert "thinker" in body["detail"]
    assert body["model_downloaded"] is True
    assert body["model"] == "qwen3-omni-30b"


def test_healthz_ready_is_ok():
    status = voice_server.VoiceStatus(phase="ready", model="qwen3-omni-30b")
    app = voice_server.build_app(None, stub=False, status=status)
    assert TestClient(app).get("/healthz").json()["ok"] is True


def test_voice_rejects_when_not_ready():
    status = voice_server.VoiceStatus(phase="error", detail="tokenizer conversion failed")
    app = voice_server.build_app(None, stub=False, status=status)
    client = TestClient(app)
    with client.websocket_connect("/voice") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "error"
        assert "not ready" in msg["message"]


def test_friendly_detail_strips_logger_prefix():
    assert (
        voice_server._friendly_detail(
            "16:47:09 INFO builder.qwen3_omni_moe.thinker: building thinker layer 13/48"
        )
        == "building thinker layer 13/48"
    )


def test_progress_detail_tracks_components():
    status = voice_server.VoiceStatus()
    server = voice_server.EdgeLLMServer("m", Path("/tmp/does-not-matter"), status)
    assert (
        server._progress_detail(
            "17:00:00 INFO experimental.builder: Building component talker"
        )
        == "building talker engine"
    )
    assert status.component == "talker"
    # Generic TRT lines must not clobber the meaningful component progress.
    assert (
        server._progress_detail("[TRT] Compiler backend is used during engine build.")
        is None
    )
    assert status.component == "talker"
    assert (
        server._progress_detail("17:01:00 INFO builder: Build completed in 10.0 s (x)")
        == "talker engine built"
    )
    assert status.components_done == ["talker"]
    assert status.component == ""


def test_status_snapshot_has_elapsed_and_heartbeat():
    status = voice_server.VoiceStatus(phase="building")
    status.started_ms = int(time.time() * 1000) - 5000
    status.beat()
    snap = status.snapshot()
    assert snap["phase"] == "building"
    assert snap["elapsed_s"] >= 4
    assert snap["heartbeat_age_s"] is not None and snap["heartbeat_age_s"] < 5


def test_resolve_model_dir_prefers_nvfp4(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "WEIGHTS_DIR", tmp_path)
    spec = models.ModelSpec(
        id="qwen3-omni-30b",
        kind="hf",
        repo="a/b",
        download="snapshot",
        dest="qwen3-omni-30b",
    )
    raw = tmp_path / "qwen3-omni-30b"
    raw.mkdir()
    # No quantization yet -> the raw snapshot.
    assert voice_server.resolve_model_dir(spec) == raw
    assert voice_server.quantized_dir(spec) == tmp_path / "qwen3-omni-30b-nvfp4"

    quant = tmp_path / "qwen3-omni-30b-nvfp4"
    quant.mkdir()
    (quant / "config.json").write_text("{}")
    assert voice_server.resolve_model_dir(spec) == quant


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[tuple[str, object]] = []

    async def send_text(self, s: str) -> None:
        self.sent.append(("text", s))

    async def send_bytes(self, b: bytes) -> None:
        self.sent.append(("bytes", len(b)))


def test_run_turn_history_text_only_by_default():
    import asyncio

    ws = _FakeWS()
    history: list[dict] = []
    cfg = voice_server.VoiceConfig(
        system_prompt="sys", history_turns=2, keep_audio_history=False
    )
    asyncio.run(
        voice_server._run_turn(
            ws, history, b"\x00\x00" * 100, 16000, None, stub=True, config=cfg
        )
    )
    assert any(m["role"] == "assistant" for m in history)
    assert not any(m["role"] == "user" for m in history)
    assert len(history) <= 2


def test_run_turn_keeps_audio_when_enabled():
    import asyncio

    ws = _FakeWS()
    history: list[dict] = []
    cfg = voice_server.VoiceConfig(history_turns=2, keep_audio_history=True)
    asyncio.run(
        voice_server._run_turn(
            ws, history, b"\x00\x00" * 100, 16000, None, stub=True, config=cfg
        )
    )
    roles = [m["role"] for m in history]
    assert "assistant" in roles and "user" in roles
    assert len(history) <= 4


def test_config_post_and_get(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "CONFIG_PATH", tmp_path / "voice-config.json")
    monkeypatch.setattr(
        voice_server, "_VOICE_CACHE", {"ts": 0.0, "voices": ["aiden", "chelsie", "ethan"]}
    )
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    r = client.post(
        "/config",
        json={
            "voice": "ethan",
            "history_turns": 6,
            "keep_audio_history": True,
            "system_prompt": "be terse",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["voice"] == "ethan"
    assert body["history_turns"] == 6
    assert body["keep_audio_history"] is True
    assert body["system_prompt"] == "be terse"
    assert body["voices"] == ["aiden", "chelsie", "ethan"]
    assert client.get("/config").json()["voice"] == "ethan"
    assert client.post("/config", json={"voice": "nope"}).status_code == 400


def test_config_clamps_history_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "CONFIG_PATH", tmp_path / "voice-config.json")
    monkeypatch.setattr(voice_server, "_VOICE_CACHE", {"ts": 0.0, "voices": []})
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    assert client.post("/config", json={"history_turns": 999}).json()["history_turns"] == 20
    assert client.post("/config", json={"history_turns": -5}).json()["history_turns"] == 0


def test_format_snapshot_json():
    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as rt

    snap = rt.Snapshot(mode=1, estop_latched=False)
    snap.vision.running = True
    out = json.loads(voice_server._format_snapshot(snap))
    assert out["mode"] == "IDLE"
    assert out["estop"] is False
    assert out["vision_running"] is True


def test_tool_loop_executes_then_speaks(monkeypatch):
    import asyncio

    ws = _FakeWS()
    history: list[dict] = []
    cfg = voice_server.VoiceConfig(history_turns=2, tools_enabled=True)

    seen_speak: list[dict] = []

    async def fake_stream(server, messages, voice, tools, stub, *, audio=True):
        if audio:
            seen_speak[:] = list(messages)
            yield {"type": "text", "delta": "Battery is at 88 percent."}
        else:
            yield {
                "type": "tool_call",
                "index": 0,
                "id": "c1",
                "name": "get_robot_state",
                "arguments": "{}",
            }

    called: dict[str, str] = {}

    async def fake_execute(name, arguments, server):
        called["name"] = name
        return '{"battery_pct": 88}'

    monkeypatch.setattr(voice_server, "_stream", fake_stream)
    monkeypatch.setattr(voice_server, "execute_tool", fake_execute)

    asyncio.run(
        voice_server._run_turn(
            ws, history, b"\x00\x00" * 100, 16000, None, stub=False, config=cfg
        )
    )
    assert called.get("name") == "get_robot_state"
    assert any(
        m["role"] == "assistant" and "Battery" in str(m.get("content")) for m in history
    )
    assert any('"type": "tool"' in s for _, s in ws.sent)
    # The speaking pass must not replay tool_calls/tool messages (the model
    # re-emits the XML if it sees them); the result is injected as text.
    assert all(m.get("role") != "tool" for m in seen_speak)
    assert "get_robot_state" in seen_speak[0]["content"]


def test_unknown_tool_is_refused():
    import asyncio

    result = asyncio.run(voice_server.execute_tool("rm_rf", "{}", None))
    assert "unknown tool" in result


def test_describe_scene_uses_both_cameras(monkeypatch):
    import asyncio
    import httpx

    fetched: list[str] = []

    async def fake_fetch(stream):
        fetched.append(stream)
        return b"\xff\xd8jpeg-bytes"

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "A chair and a doorway."}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return FakeResponse()

    class FakeServer:
        base_url = "http://127.0.0.1:8000"

    monkeypatch.setattr(voice_server, "_fetch_snapshot", fake_fetch)
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    out = asyncio.run(voice_server.tool_describe_scene(FakeServer()))
    assert fetched == ["color_near", "color_far"]
    parts = captured["json"]["messages"][1]["content"]
    images = [p for p in parts if p.get("type") == "image_url"]
    assert len(images) == 2
    assert "chair" in out


def test_describe_scene_errors_when_both_cameras_fail(monkeypatch):
    import asyncio

    async def fake_fetch(stream):
        return None

    async def fake_enable():
        return False

    monkeypatch.setattr(voice_server, "_fetch_snapshot", fake_fetch)
    monkeypatch.setattr(voice_server, "_enable_vision", fake_enable)

    class FakeServer:
        base_url = "http://127.0.0.1:8000"

    out = asyncio.run(voice_server.tool_describe_scene(FakeServer()))
    assert "cameras are not available" in out


def test_describe_scene_enables_vision_and_retries(monkeypatch):
    import asyncio
    import httpx

    state = {"on": False}

    async def fake_fetch(stream):
        return b"\xff\xd8jpeg" if state["on"] else None

    async def fake_enable():
        state["on"] = True
        return True

    async def no_sleep(_seconds):
        return None

    captured: dict = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "A window and a desk."}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, json):
            captured["json"] = json
            return FakeResponse()

    class FakeServer:
        base_url = "http://127.0.0.1:8000"

    monkeypatch.setattr(voice_server, "_fetch_snapshot", fake_fetch)
    monkeypatch.setattr(voice_server, "_enable_vision", fake_enable)
    monkeypatch.setattr(voice_server.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    out = asyncio.run(voice_server.tool_describe_scene(FakeServer()))
    assert state["on"] is True
    assert "window" in out
    assert len(captured["json"]["messages"][1]["content"]) >= 2


def test_config_tools_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "CONFIG_PATH", tmp_path / "c.json")
    monkeypatch.setattr(voice_server, "_VOICE_CACHE", {"ts": 0.0, "voices": []})
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    assert client.get("/config").json()["tools_enabled"] is True
    assert client.post("/config", json={"tools_enabled": False}).json()["tools_enabled"] is False
