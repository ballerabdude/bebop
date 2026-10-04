"""Tests for the voice gateway protocol (stub mode; no model/GPU needed)."""

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bebop_vision import voice_server  # noqa: E402


def test_pcm16_to_wav_roundtrip():
    wav = voice_server.pcm16_to_wav(b"\x00\x00\x01\x00\x02\x00", 16000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"


def test_stub_voice_turn_streams_text_and_audio():
    app = voice_server.build_app(None, stub=True, voice="x")
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
    app = voice_server.build_app(None, stub=True, voice="x")
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
    app = voice_server.build_app(None, stub=False, voice="x", status=status)
    client = TestClient(app)
    body = client.get("/healthz").json()
    assert body["phase"] == "building"
    assert body["ok"] is False
    assert "thinker" in body["detail"]
    assert body["model_downloaded"] is True
    assert body["model"] == "qwen3-omni-30b"


def test_healthz_ready_is_ok():
    status = voice_server.VoiceStatus(phase="ready", model="qwen3-omni-30b")
    app = voice_server.build_app(None, stub=False, voice="x", status=status)
    assert TestClient(app).get("/healthz").json()["ok"] is True


def test_voice_rejects_when_not_ready():
    status = voice_server.VoiceStatus(phase="error", detail="tokenizer conversion failed")
    app = voice_server.build_app(None, stub=False, voice="x", status=status)
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
