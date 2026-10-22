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
            "omni_backend_id": "omni-edgellm",
            "omni_unload_on_stop": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["omni_supervisor_url"] == "http://127.0.0.1:9094"
    assert body["omni_backend_id"] == "omni-edgellm"
    assert body["omni_unload_on_stop"] is True
    # Persisted and reloaded.
    reloaded = voice_server.VoiceConfig.load(tmp_path / "voice-config.json")
    assert reloaded.omni_supervisor_url == "http://127.0.0.1:9094"


def test_config_migrates_dead_omni_backend_id(tmp_path):
    """`omni-vllm` (the removed vLLM container) becomes the live omni-edgellm."""
    p = tmp_path / "voice-config.json"
    p.write_text(json.dumps({"backend": "omni", "omni_backend_id": "omni-vllm"}))
    cfg = voice_server.VoiceConfig.load(p)
    assert cfg.omni_backend_id == "omni-edgellm"
    # persisted, so the fix is durable and the file stops carrying a dead id
    assert json.loads(p.read_text())["omni_backend_id"] == "omni-edgellm"


def test_config_keeps_other_backend_ids(tmp_path):
    p = tmp_path / "voice-config.json"
    p.write_text(json.dumps({"omni_backend_id": "some-other-backend"}))
    assert voice_server.VoiceConfig.load(p).omni_backend_id == "some-other-backend"


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

    async def fake_chat_stream(stage, messages, tools=None):
        seen["messages"] = messages
        for delta in ("Hi ", "there!", " Bye", " now."):
            yield {"type": "text", "delta": delta}

    spoken: list[str] = []

    async def fake_speak(stage, text, voice):
        spoken.append(text)
        for chunk in (b"\x00\x01", b"\x02\x03"):
            yield chunk

    monkeypatch.setattr(voice_server, "_cascade_transcribe", fake_transcribe)
    monkeypatch.setattr(voice_server, "_cascade_chat_stream", fake_chat_stream)
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
    assert "".join(text_deltas) == "Hi there! Bye now."
    # pipelined: one TTS call per finished sentence, not one for the whole reply
    assert spoken == ["Hi there!", "Bye now."]
    assert audio_bytes == 8  # 2 sentences x 2 chunks x 2 bytes
    assert history and history[0]["role"] == "assistant"
    assert history[0]["content"] == "Hi there! Bye now."
    # the transcript is what the brain sees as the user turn
    assert seen["messages"][-1]["content"][0]["text"] == "hello robot"


def test_run_turn_cascade_runs_a_places_tool(monkeypatch):
    """A tool-calling round is executed and its result fed back to the brain."""
    import asyncio

    ws = _FakeWS()
    history: list[dict] = []
    cfg = voice_server.VoiceConfig(voice="aiden", tools_enabled=True)
    asr = voice_server.StageServer("asr", "m", Path("/tmp"), 1, capability="transcription")
    brain = voice_server.StageServer("brain", "m", Path("/tmp"), 2, capability="chat")
    tts = voice_server.StageServer("tts", "m", Path("/tmp"), 3, capability="speech")
    pipe = voice_server.CascadePipeline(asr, brain, tts, voice_server.VoiceStatus(phase="ready"))

    async def fake_transcribe(stage, wav):
        return "where are we"

    async def fake_context(status=None):
        return "", []

    rounds = {"n": 0}
    advertised: list = []

    async def fake_chat_stream(stage, messages, tools=None):
        rounds["n"] += 1
        advertised.append(tools)
        if rounds["n"] == 1:
            yield {"type": "tool_call", "index": 0, "id": "c1",
                   "name": "where_am_i", "arguments": ""}
            return
        # the tool result must have been handed back to the model
        assert any(m.get("role") == "tool" for m in messages)
        yield {"type": "text", "delta": "You are in the office room."}

    async def fake_speak(stage, text, voice):
        yield b"\x00\x01"

    ran: dict = {}

    async def fake_tool(name, arguments):
        ran["name"] = name
        return "Recognised place: office room"

    monkeypatch.setattr(voice_server, "_cascade_transcribe", fake_transcribe)
    monkeypatch.setattr(voice_server, "live_context", fake_context)
    monkeypatch.setattr(voice_server, "_cascade_chat_stream", fake_chat_stream)
    monkeypatch.setattr(voice_server, "_cascade_speak", fake_speak)
    monkeypatch.setattr(voice_server, "_run_tool", fake_tool)

    asyncio.run(
        voice_server._run_turn_cascade(
            ws, history, b"\x00\x00" * 100, 16000, pipe, stub=False, config=cfg
        )
    )
    assert ran["name"] == "where_am_i"
    names = [t["function"]["name"] for t in advertised[0]]  # first pass
    assert "where_am_i" in names and "robot_status" in names
    assert "move_relative" not in names  # motion is its own switch
    assert rounds["n"] == 2  # tool round, then the spoken answer
    events = [json.loads(p) for k, p in ws.sent if k == "text"]
    assert [e["name"] for e in events if e.get("type") == "tool"] == ["where_am_i"]
    said = "".join(e["delta"] for e in events if e.get("type") == "text")
    assert said == "You are in the office room."
    assert history[-1]["content"] == "You are in the office room."


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = {"content-type": "application/json"}

    def json(self):
        return self._payload


class _FakePlaces:
    """Minimal httpx.AsyncClient stand-in that records what the tool sent."""

    def __init__(self, monkeypatch, responses):
        self.calls = []
        self.bodies = []
        self.responses = responses
        monkeypatch.setattr("httpx.AsyncClient", lambda *a, **k: self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def _dispatch(self, method, url):
        self.calls.append((method, url))
        for (m, frag), resp in self.responses.items():
            if m == method and frag in url:
                return resp
        return _Resp(404, {"error": "no such place"})

    async def get(self, url, **kwargs):
        return self._dispatch("GET", url)

    async def post(self, url, json=None, **kwargs):
        self.bodies.append(json)
        return self._dispatch("POST", url)

    async def delete(self, url, **kwargs):
        return self._dispatch("DELETE", url)


def test_places_write_tools_are_advertised():
    names = {t["function"]["name"] for t in voice_server.PLACES_TOOLS}
    assert {"where_am_i", "list_places", "place_facts",
            "remember_place", "add_place_fact",
            "rename_place", "forget_place"} <= names


def test_remember_place_posts_and_reports_capture(monkeypatch):
    import asyncio

    fake = _FakePlaces(monkeypatch, {
        ("POST", "/places"): _Resp(201, {"views": 2, "descriptors": 2}),
    })
    out = asyncio.run(voice_server._run_tool(
        "remember_place", json.dumps({"name": "the study",
                                      "facts": ["has a standing desk"]})))
    assert "the study" in out and "2 appearance descriptor" in out
    assert fake.calls == [("POST", f"{voice_server.PLACES_URL}/places")]
    assert fake.bodies[0] == {"name": "the study",
                              "facts": ["has a standing desk"]}


def test_remember_place_warns_when_no_usable_view(monkeypatch):
    import asyncio

    _FakePlaces(monkeypatch, {
        ("POST", "/places"): _Resp(201, {"views": 0, "descriptors": 0}),
    })
    out = asyncio.run(voice_server._run_tool(
        "remember_place", json.dumps({"name": "nowhere"})))
    assert "could not take a usable view" in out


def test_places_write_requires_a_name(monkeypatch):
    import asyncio

    fake = _FakePlaces(monkeypatch, {})
    for tool in ("remember_place", "forget_place", "add_place_fact"):
        out = asyncio.run(voice_server._run_tool(tool, "{}"))
        assert "required" in out or "No place name" in out or "No name" in out
    assert fake.calls == []  # nothing was sent without a name


class _FakeState:
    def __init__(self):
        self.connected = True
        self.mode = voice_server.pb.MODE_IDLE
        self.estop_latched = False
        self.estop_reason = ""
        self.odom = (0.0, 0.0, 0.0)
        self.wheel_armed = {"left": False, "right": False}


class _FakeRobot:
    """Simulates a chassis that converges ~0.25 m per commanded twist."""

    def __init__(self, monkeypatch, *, estop=False, arm=True):
        self.state = _FakeState()
        self.state.estop_latched = estop
        self._arm = arm
        self.twists = []
        self.armed_calls = []
        self.modes = []
        self.estop_resets = []
        self.stopped = False

    def start(self):
        return self

    def await_connection(self, timeout=5.0):
        return True

    def set_mode(self, mode):
        self.modes.append(mode)
        self.state.mode = mode
        return True

    def arm_wheels(self, enabled):
        self.armed_calls.append(enabled)
        if enabled and self._arm:
            self.state.wheel_armed = {"left": True, "right": True}
        elif not enabled:
            self.state.wheel_armed = {"left": False, "right": False}
        return True

    def send_twist(self, vx, wz):
        self.twists.append((round(vx, 4), round(wz, 4)))
        x, y, th = self.state.odom
        self.state.odom = (x + vx * 0.25, y, th + wz * 0.25)
        return True

    def reset_estop(self):
        self.estop_resets.append(True)
        return True

    def stop(self, timeout=3.0):
        self.stopped = True


def _fake_robot(monkeypatch, **kwargs):
    fake = _FakeRobot(monkeypatch, **kwargs)
    monkeypatch.setattr("bebop_vision.robot.RobotClient",
                        lambda *a, **k: fake)
    return fake


def test_move_relative_refuses_when_estop_latched(monkeypatch):
    import asyncio

    fake = _fake_robot(monkeypatch, estop=True)
    out = asyncio.run(voice_server._move_relative({"forward_m": 1.0}))
    assert "E-STOP" in out
    assert fake.modes == []          # never even changed mode
    assert fake.armed_calls == []    # never armed
    assert fake.estop_resets == []   # an E-STOP is the operator's to clear
    assert fake.stopped is True


def test_move_relative_requires_armed_wheels(monkeypatch):
    import asyncio

    fake = _fake_robot(monkeypatch, arm=False)
    out = asyncio.run(voice_server._move_relative({"forward_m": 1.0}))
    assert "did not arm" in out
    assert fake.modes[0] == voice_server.pb.MODE_DIAL_IN
    assert not [t for t in fake.twists if t != (0.0, 0.0)]  # nothing driven


def test_move_relative_drives_clamped_and_always_stops(monkeypatch):
    import asyncio

    fake = _fake_robot(monkeypatch)
    out = asyncio.run(voice_server._move_relative({"forward_m": 99.0}))
    # clamped to the envelope, and the reply reports what telemetry showed
    assert "m forward" in out, out
    moved_m = float(out.split("Moved ")[1].split(" m")[0])
    assert 1.9 <= moved_m <= voice_server.MOTION_MAX_FORWARD_M, out
    assert fake.modes[0] == voice_server.pb.MODE_DIAL_IN
    assert True in fake.armed_calls and fake.armed_calls[-1] is False  # disarmed
    assert fake.twists[-1] == (0.0, 0.0)                              # stopped
    drove_forward = [t for t in fake.twists if t[0] > 0]
    assert drove_forward and max(t[0] for t in drove_forward) <= \
        voice_server.MOTION_LINEAR_MS


def test_no_model_tool_can_control_the_estop():
    """E-STOP control is operator-only; the model gets no tool for it."""
    every_tool = (voice_server.PLACES_TOOLS
                  + [voice_server.ROBOT_STATUS_TOOL, voice_server.MOVE_TOOL])
    names = [t["function"]["name"] for t in every_tool]
    forbidden = ("estop", "e_stop", "reset", "clear", "arm_wheels", "arm")
    for name in names:
        assert not any(bad in name for bad in forbidden), name
    assert "move_relative" in names      # motion exists...
    assert "robot_status" in names       # ...reading it does


def test_move_relative_no_op(monkeypatch):
    import asyncio

    fake = _fake_robot(monkeypatch)
    out = asyncio.run(voice_server._move_relative({}))
    assert "No movement" in out
    assert fake.twists == []


def test_robot_status_is_routed(monkeypatch):
    import asyncio

    async def fake_state():
        return '{"mode": "IDLE"}'

    monkeypatch.setattr(voice_server, "robot_state_text", fake_state)
    assert asyncio.run(voice_server._run_tool("robot_status", "")) == \
        '{"mode": "IDLE"}'


def test_places_tool_reports_unavailable(monkeypatch):
    import asyncio

    import httpx

    def boom(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "AsyncClient", boom)
    out = asyncio.run(voice_server._run_tool("list_places", ""))
    assert "unavailable" in out


def test_robot_state_text_is_cached(monkeypatch):
    """The firmware round trip costs ~2 s; it must not run once per turn."""
    import asyncio

    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as rt

    calls = {"n": 0}

    class FakeWS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def send(self, data):
            pass

        async def recv(self):
            env = rt.ServerRuntimeMessage(request_id=1)
            env.snapshot.SetInParent()
            return env.SerializeToString()

    def fake_connect(*args, **kwargs):
        calls["n"] += 1
        return FakeWS()

    monkeypatch.setattr("websockets.connect", fake_connect)
    monkeypatch.setattr(voice_server, "_STATE_CACHE", {"ts": 0.0, "text": ""})

    first = asyncio.run(voice_server.robot_state_text())
    second = asyncio.run(voice_server.robot_state_text())
    assert first and first == second
    assert calls["n"] == 1  # the second call is served from the cache


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


# --- navigation tools --------------------------------------------------------


def test_navigation_tools_are_advertised_behind_the_motion_switch():
    names = [t["function"]["name"] for t in voice_server.NAV_TOOLS]
    assert names == ["navigate", "navigation_status", "stop_navigation"]
    for t in voice_server.NAV_TOOLS:
        assert not any(k in t["function"]["name"] for k in ("estop", "reset", "arm"))


def test_navigate_tool_relays_command_and_summary(monkeypatch):
    import asyncio
    calls = []

    async def fake_request(method, path, body=None):
        calls.append((method, path, body))
        if path == "/navigate" and method == "POST":
            return {"ok": True, "status": {"state": "running",
                                           "summary": "Heading for the dresser, about 3.0 metres away."}}
        if path == "/navigate":
            return {"state": "arrived", "summary": "Arrived at dresser."}
        if path == "/navigate/stop":
            return {"ok": True, "status": {"summary": "Stopped on the way to dresser."}}
        return {"error": "nope"}

    monkeypatch.setattr(voice_server, "_nav_request", fake_request)
    out = asyncio.run(voice_server._run_tool("navigate", json.dumps({"command": "go to the dresser"})))
    assert out == "Heading for the dresser, about 3.0 metres away."
    assert calls[0] == ("POST", "/navigate", {"command": "go to the dresser"})
    assert asyncio.run(voice_server._run_tool("navigation_status", "{}")) == "Arrived at dresser."
    assert asyncio.run(voice_server._run_tool("stop_navigation", "{}")) == "Stopped on the way to dresser."
    assert asyncio.run(voice_server._run_tool("navigate", "{}")) == "No destination was given."


def test_navigate_tool_reports_service_errors(monkeypatch):
    import asyncio

    async def fake_request(method, path, body=None):
        return {"error": "no taught route from 'office room' to 'attic'"}

    monkeypatch.setattr(voice_server, "_nav_request", fake_request)
    out = asyncio.run(voice_server._run_tool("navigate", json.dumps({"command": "go to the attic"})))
    assert out.startswith("I cannot do that: no taught route")
