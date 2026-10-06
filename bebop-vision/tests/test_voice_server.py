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


def test_config_supervisor_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "CONFIG_PATH", tmp_path / "voice-config.json")
    monkeypatch.setattr(voice_server, "_VOICE_CACHE", {"ts": 0.0, "voices": []})
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    r = client.post(
        "/config",
        json={
            "omni_supervisor_url": "http://127.0.0.1:9094",
            "omni_backend_id": "omni-vllm",
            "omni_unload_on_stop": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["omni_supervisor_url"] == "http://127.0.0.1:9094"
    assert body["omni_backend_id"] == "omni-vllm"
    assert body["omni_unload_on_stop"] is True
    # Persisted and reloaded.
    reloaded = voice_server.VoiceConfig.load(tmp_path / "voice-config.json")
    assert reloaded.omni_supervisor_url == "http://127.0.0.1:9094"


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


def test_run_turn_injects_live_context(monkeypatch):
    import asyncio

    ws = _FakeWS()
    history: list[dict] = []
    cfg = voice_server.VoiceConfig(history_turns=2, tools_enabled=True)

    async def fake_context(status=None):
        return '{"battery_pct": 88, "mode": "IDLE"}', [
            ("color_near", b"\xff\xd8near"),
            ("color_far", b"\xff\xd8far"),
        ]

    seen: list[dict] = []

    async def fake_stream(server, messages, voice, tools, stub, *, audio=True):
        seen[:] = list(messages)
        yield {"type": "text", "delta": "The person is wearing red."}

    monkeypatch.setattr(voice_server, "live_context", fake_context)
    monkeypatch.setattr(voice_server, "_stream", fake_stream)

    asyncio.run(
        voice_server._run_turn(
            ws, history, b"\x00\x00" * 100, 16000, None, stub=False, config=cfg
        )
    )
    # status + camera instructions go in the system prompt
    assert seen[0]["role"] == "system"
    assert "battery_pct" in seen[0]["content"]
    assert "cameras" in seen[0]["content"]
    # both camera frames + the audio ride the user turn
    user = seen[-1]
    assert user["role"] == "user"
    kinds = [p["type"] for p in user["content"]]
    assert kinds.count("image_url") == 2
    assert "input_audio" in kinds
    assert all(m.get("role") != "tool" for m in seen)
    assert any(m["role"] == "assistant" and "red" in m["content"] for m in history)


def test_resolve_cascade_voice(monkeypatch):
    monkeypatch.setattr(voice_server, "cascade_voices", lambda tts: ["dylan", "serena"])
    assert voice_server.resolve_cascade_voice(None, "dylan") == "dylan"
    assert voice_server.resolve_cascade_voice(None, "ethan") == "dylan"
    assert voice_server.resolve_cascade_voice(None, "") == "dylan"


def test_run_turn_cascade(monkeypatch):
    import asyncio

    ws = _FakeWS()
    history: list[dict] = []
    cfg = voice_server.VoiceConfig(voice="aiden", tools_enabled=False)
    asr = voice_server.StageServer("asr", "m", Path("/tmp"), 1, capability="transcription")
    brain = voice_server.StageServer("brain", "m", Path("/tmp"), 2, capability="chat")
    tts = voice_server.StageServer("tts", "m", Path("/tmp"), 3, capability="speech")
    pipe = voice_server.CascadePipeline(asr, brain, tts, voice_server.VoiceStatus(phase="ready"))

    async def fake_transcribe(stage, wav):
        return "hello robot"

    seen: dict = {}

    async def fake_chat(stage, messages):
        seen["messages"] = messages
        return "Hi there!"

    async def fake_speak(stage, text, voice):
        for chunk in (b"\x00\x01", b"\x02\x03"):
            yield chunk

    monkeypatch.setattr(voice_server, "_cascade_transcribe", fake_transcribe)
    monkeypatch.setattr(voice_server, "_cascade_chat", fake_chat)
    monkeypatch.setattr(voice_server, "_cascade_speak", fake_speak)

    asyncio.run(
        voice_server._run_turn_cascade(
            ws, history, b"\x00\x00" * 100, 16000, pipe, stub=False, config=cfg
        )
    )
    kinds = []
    text_deltas = []
    for kind, payload in ws.sent:
        if kind == "text":
            ev = json.loads(payload)
            kinds.append(ev.get("type"))
            if ev.get("type") == "text":
                text_deltas.append(ev["delta"])
    audio_bytes = sum(n for kind, n in ws.sent if kind == "bytes")
    assert "transcript" in kinds
    assert any("Hi there" in d for d in text_deltas)
    assert audio_bytes == 4
    assert history and history[0]["role"] == "assistant" and history[0]["content"] == "Hi there!"
    # the transcript is what the brain sees as the user turn
    assert seen["messages"][-1]["content"][0]["text"] == "hello robot"


def test_run_turn_skips_context_when_disabled(monkeypatch):
    import asyncio

    called = {"n": 0}

    async def fake_context(status=None):
        called["n"] += 1
        return "", []

    async def fake_stream(server, messages, voice, tools, stub, *, audio=True):
        yield {"type": "text", "delta": "hi"}

    monkeypatch.setattr(voice_server, "live_context", fake_context)
    monkeypatch.setattr(voice_server, "_stream", fake_stream)

    cfg = voice_server.VoiceConfig(tools_enabled=False)
    asyncio.run(
        voice_server._run_turn(
            _FakeWS(), [], b"\x00\x00" * 100, 16000, None, stub=False, config=cfg
        )
    )
    assert called["n"] == 0


def test_wav_bytes_to_pcm16():
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b"\x01\x02\x03\x04")
    assert voice_server.wav_bytes_to_pcm16(buf.getvalue()) == b"\x01\x02\x03\x04"


def test_config_tools_toggle(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_server, "CONFIG_PATH", tmp_path / "c.json")
    monkeypatch.setattr(voice_server, "_VOICE_CACHE", {"ts": 0.0, "voices": []})
    app = voice_server.build_app(None, stub=True)
    client = TestClient(app)
    assert client.get("/config").json()["tools_enabled"] is True
    assert client.post("/config", json={"tools_enabled": False}).json()["tools_enabled"] is False
