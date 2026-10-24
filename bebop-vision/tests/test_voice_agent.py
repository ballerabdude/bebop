"""Agent turn lifecycle on the voice gateway: text turns, typed events,
cancel, busy rejection, history and the motion tool's cancel/seat rules.

No models, GPU or robot: the cascade stages are monkeypatched fakes.
"""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bebop_vision import voice_server as vs  # noqa: E402


def _pipeline():
    asr = vs.StageServer("asr", "m", Path("/tmp"), 1, capability="transcription")
    brain = vs.StageServer("brain", "m", Path("/tmp"), 2, capability="chat")
    tts = vs.StageServer("tts", "m", Path("/tmp"), 3, capability="speech")
    return vs.CascadePipeline(asr, brain, tts, vs.VoiceStatus(phase="ready"))


def _app(monkeypatch, *, tools=True, motion=False):
    async def no_context(status=None):
        return "", []

    monkeypatch.setattr(vs, "live_context", no_context)
    monkeypatch.setattr(vs, "cascade_voices", lambda tts: ["aiden"])
    cfg = vs.VoiceConfig(voice="aiden", tools_enabled=tools,
                         motion_enabled=motion, history_turns=4)
    status = vs.VoiceStatus(phase="ready")
    app = vs.build_app(None, stub=False, config=cfg, status=status,
                       pipeline=_pipeline(), pipeline_status=status)
    return app, cfg


def _read_turn(ws, *, until="turn_end"):
    """Collect JSON events (and count audio bytes) up to `until`."""
    events, audio = [], 0
    while True:
        msg = ws.receive()
        if msg.get("bytes") is not None:
            audio += len(msg["bytes"])
            continue
        ev = json.loads(msg["text"])
        events.append(ev)
        if ev["type"] == until:
            return events, audio


def _new_events(events):
    legacy = {"text", "tool", "done", "error"}
    return [e for e in events if e["type"] not in legacy]


def test_cascade_is_the_default_backend(tmp_path):
    assert vs.VoiceConfig().backend == "cascade"
    # a pre-v2 file persisted the old default on every save: migrate it
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"backend": "omni", "voice": "aiden"}))
    assert vs.VoiceConfig.load(old).backend == "cascade"
    # a v2 file keeps an explicit omni choice
    cfg = vs.VoiceConfig(backend="omni")
    cfg.save(tmp_path / "new.json")
    assert vs.VoiceConfig.load(tmp_path / "new.json").backend == "omni"


def test_healthz_reports_turns_and_tool_support(monkeypatch):
    app, _ = _app(monkeypatch)
    body = TestClient(app).get("/healthz").json()
    assert body["turn_active"] is False and body["turns_active"] == 0
    assert body["tools_supported"] is True
    assert "motion_enabled" in body and "tools_enabled" in body
    cfg = TestClient(app).get("/config").json()
    assert cfg["tools_supported"] is True and "motion_enabled" in cfg
    # omni (here: the stub, no cascade pipeline) has no tools
    omni = vs.build_app(None, stub=True)
    assert TestClient(omni).get("/healthz").json()["tools_supported"] is False


def test_text_turn_goes_to_the_brain_without_asr_or_tts(monkeypatch):
    app, _ = _app(monkeypatch)
    seen = {}

    async def no_asr(stage, wav):
        raise AssertionError("a typed turn must not run ASR")

    async def brain(stage, messages, tools=None):
        seen["messages"] = messages
        for delta in ("Hello ", "there."):
            yield {"type": "text", "delta": delta}

    async def no_tts(stage, text, voice):
        raise AssertionError("speak=false must not synthesise")
        yield b""  # pragma: no cover

    monkeypatch.setattr(vs, "_cascade_transcribe", no_asr)
    monkeypatch.setattr(vs, "_cascade_chat_stream", brain)
    monkeypatch.setattr(vs, "_cascade_speak", no_tts)

    with TestClient(app).websocket_connect("/voice?session=s1") as ws:
        ws.send_text(json.dumps({"type": "user_text", "text": "hi bebop",
                                 "speak": False, "turn_id": "abc"}))
        events, audio = _read_turn(ws)
    assert audio == 0
    kinds = [e["type"] for e in _new_events(events)]
    assert kinds == ["turn_start", "transcript", "assistant_delta",
                     "assistant_delta", "turn_end"]
    assert all(e["turn_id"] == "abc" for e in _new_events(events))
    assert events[0] == {"type": "turn_start", "turn_id": "abc",
                         "source": "text", "speak": False}
    assert _new_events(events)[1]["text"] == "hi bebop"
    assert events[-1]["status"] == "done"
    # the legacy `done` precedes turn_end; no audio framing at all
    assert [e["type"] for e in events][-2:] == ["done", "turn_end"]
    assert "audio_start" not in kinds
    assert seen["messages"][-1]["content"][0]["text"] == "hi bebop"


def test_text_turn_speaks_when_asked(monkeypatch):
    app, _ = _app(monkeypatch)

    async def brain(stage, messages, tools=None):
        yield {"type": "text", "delta": "One. Two."}

    async def tts(stage, text, voice):
        yield b"\x00\x01"

    monkeypatch.setattr(vs, "_cascade_chat_stream", brain)
    monkeypatch.setattr(vs, "_cascade_speak", tts)
    with TestClient(app).websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "user_text", "text": "count"}))
        events, audio = _read_turn(ws)
    kinds = [e["type"] for e in _new_events(events)]
    assert kinds.index("audio_start") < kinds.index("audio_end") < kinds.index("turn_end")
    assert audio == 4  # two sentences x one 2-byte chunk


def test_tool_events_and_history_keep_the_whole_turn(monkeypatch):
    app, _ = _app(monkeypatch)
    rounds = {"n": 0}

    async def brain(stage, messages, tools=None):
        rounds["n"] += 1
        if rounds["n"] == 1:
            yield {"type": "text", "delta": "Let me check. "}
            yield {"type": "tool_call", "index": 0, "id": "c1",
                   "name": "list_places", "arguments": ""}
            yield {"type": "tool_call", "index": 0, "id": "",
                   "name": "", "arguments": '{"x": 1}'}
            return
        yield {"type": "text", "delta": "Kitchen and office."}

    async def tool(name, arguments):
        return "Remembered places: kitchen, office"

    monkeypatch.setattr(vs, "_cascade_chat_stream", brain)
    monkeypatch.setattr(vs, "_run_tool", tool)
    client = TestClient(app)
    with client.websocket_connect("/voice?session=hist") as ws:
        ws.send_text(json.dumps({"type": "user_text", "text": "what rooms?",
                                 "speak": False}))
        events, _ = _read_turn(ws)
    new = _new_events(events)
    kinds = [e["type"] for e in new]
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("turn_end")
    call = next(e for e in new if e["type"] == "tool_call")
    assert call["name"] == "list_places" and call["call_id"] == "c1"
    assert call["args"] == {"x": 1}
    result = next(e for e in new if e["type"] == "tool_result")
    assert result["ok"] is True and result["call_id"] == "c1"
    assert "kitchen" in result["text"] and isinstance(result["ms"], int)
    # legacy {"type":"tool"} still announced
    assert any(e["type"] == "tool" for e in events)

    history = vs_sessions(app, "hist")
    roles = [m["role"] for m in history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert history[0]["content"] == "what rooms?"
    assert history[1]["tool_calls"][0]["id"] == "c1"
    assert history[2]["tool_call_id"] == "c1"
    assert history[3]["content"] == "Kitchen and office."


def vs_sessions(app, key):
    """The session's history list."""
    return app.state.sessions.get(key)


def _blocking_brain(started, release=None):
    async def brain(stage, messages, tools=None):
        yield {"type": "text", "delta": "Thinking"}
        started.set()
        try:
            await asyncio.sleep(30)  # a long generation; cancel lands here
        finally:
            if release is not None:
                release.set()
        yield {"type": "text", "delta": "never"}
    return brain


def test_cancel_ends_the_turn_and_frees_the_session(monkeypatch):
    app, _ = _app(monkeypatch)
    started, cleaned = threading.Event(), threading.Event()
    monkeypatch.setattr(vs, "_cascade_chat_stream", _blocking_brain(started, cleaned))
    client = TestClient(app)
    with client.websocket_connect("/voice?session=c") as ws:
        ws.send_text(json.dumps({"type": "user_text", "text": "go", "speak": False}))
        events, _ = _read_turn(ws, until="assistant_delta")
        assert started.wait(5)
        busy = client.get("/healthz").json()
        assert busy["turn_active"] is True and busy["turns_active"] == 1

        # the receive loop is still reading: a second turn is rejected...
        ws.send_text(json.dumps({"type": "user_text", "text": "again"}))
        rej, _ = _read_turn(ws, until="turn_rejected")
        assert rej[-1]["reason"] == "busy"
        # no legacy `error` (an old client would show an error mid-turn)
        assert not [e for e in rej if e["type"] == "error"]
        assert rej[-1]["active_turn_id"] == events[0]["turn_id"]

        # ...and cancel lands mid-generation
        ws.send_text(json.dumps({"type": "cancel"}))
        end, _ = _read_turn(ws)
        assert end[-1]["status"] == "cancelled"
        assert cleaned.wait(5)

        # the session is free again; the cut-off reply is in history
        monkeypatch.setattr(vs, "_cascade_chat_stream", _quick_brain("ok"))
        ws.send_text(json.dumps({"type": "user_text", "text": "next", "speak": False}))
        nxt, _ = _read_turn(ws)
        assert nxt[-1]["status"] == "done"
    assert client.get("/healthz").json()["turns_active"] == 0
    # finished turns are dropped, so closed sockets are not pinned
    assert app.state.turns.by_session == {}
    history = vs_sessions(app, "c")
    assert history[1]["content"].endswith("(interrupted)")
    assert history[-1]["content"] == "ok"


def _quick_brain(text):
    async def brain(stage, messages, tools=None):
        yield {"type": "text", "delta": text}
    return brain


def test_disconnect_cancels_the_running_turn(monkeypatch):
    app, _ = _app(monkeypatch)
    started, cleaned = threading.Event(), threading.Event()
    monkeypatch.setattr(vs, "_cascade_chat_stream", _blocking_brain(started, cleaned))
    client = TestClient(app)
    with client.websocket_connect("/voice?session=d") as ws:
        ws.send_text(json.dumps({"type": "user_text", "text": "go", "speak": False}))
        assert started.wait(5)
    assert cleaned.wait(5)
    deadline = time.monotonic() + 5
    while client.get("/healthz").json()["turns_active"] and time.monotonic() < deadline:
        time.sleep(0.05)
    assert client.get("/healthz").json()["turn_active"] is False


def test_voice_turn_still_works_and_reports_its_transcript(monkeypatch):
    app, _ = _app(monkeypatch)

    async def asr(stage, wav):
        return "hello robot"

    async def tts(stage, text, voice):
        yield b"\x00\x01"

    monkeypatch.setattr(vs, "_cascade_transcribe", asr)
    monkeypatch.setattr(vs, "_cascade_chat_stream", _quick_brain("Hi."))
    monkeypatch.setattr(vs, "_cascade_speak", tts)
    with TestClient(app).websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "utterance_start", "sample_rate": 16000}))
        ws.send_bytes(b"\x00\x00" * 200)
        ws.send_text(json.dumps({"type": "utterance_end"}))
        events, audio = _read_turn(ws)
    new = _new_events(events)
    assert new[0]["type"] == "turn_start" and new[0]["source"] == "voice"
    transcript = next(e for e in new if e["type"] == "transcript")
    assert transcript["text"] == "hello robot" and transcript["final"] is True
    assert audio == 2
    assert new[-1] == {"type": "turn_end", "turn_id": new[0]["turn_id"],
                       "status": "done"}


def test_brain_failure_ends_the_turn_as_error(monkeypatch):
    app, _ = _app(monkeypatch)

    async def broken(stage, messages, tools=None):
        raise RuntimeError("brain down")
        yield {}  # pragma: no cover

    monkeypatch.setattr(vs, "_cascade_chat_stream", broken)
    with TestClient(app).websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "user_text", "text": "hi", "speak": False}))
        events, _ = _read_turn(ws)
        assert events[-1]["status"] == "error" and "brain down" in events[-1]["error"]
        assert any(e["type"] == "error" for e in events)  # legacy
        # the socket stays usable
        monkeypatch.setattr(vs, "_cascade_chat_stream", _quick_brain("fine"))
        ws.send_text(json.dumps({"type": "user_text", "text": "hi", "speak": False}))
        again, _ = _read_turn(ws)
        assert again[-1]["status"] == "done"


def test_trim_history_counts_whole_turns():
    h = []
    for i in range(3):
        h += [{"role": "user", "content": f"u{i}"},
              {"role": "assistant", "content": None,
               "tool_calls": [{"id": f"c{i}", "type": "function",
                               "function": {"name": "x", "arguments": ""}}]},
              {"role": "tool", "tool_call_id": f"c{i}", "content": "r"},
              {"role": "assistant", "content": f"a{i}"}]
    vs._trim_history(h, 2)
    assert h[0] == {"role": "user", "content": "u1"} and len(h) == 8
    # omni without audio history keeps assistant-only turns
    a = [{"role": "assistant", "content": str(i)} for i in range(5)]
    vs._trim_history(a, 2)
    assert [m["content"] for m in a] == ["3", "4"]
    vs._trim_history(a, 0)
    assert a == []


def test_close_tool_calls_answers_cut_off_calls():
    msgs = [{"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "a", "type": "function", "function": {}},
                {"id": "b", "type": "function", "function": {}}]},
            {"role": "tool", "tool_call_id": "a", "content": "done"}]
    vs._close_tool_calls(msgs)
    assert [m.get("tool_call_id") for m in msgs if m["role"] == "tool"] == ["a", "b"]


# --- motion: cancel, seat, armed-by-human ---------------------------------


class _State:
    def __init__(self):
        self.connected = True
        self.mode = vs.pb.MODE_IDLE
        self.estop_latched = False
        self.estop_reason = ""
        self.odom = (0.0, 0.0, 0.0)
        self.wheel_armed = {"left": False, "right": False}
        self.has_active_operator = False
        self.you_are_active_operator = False


class _Robot:
    """Chassis that moves 0.05 m per 0.2 m/s twist (a slow, steady drive)."""

    def __init__(self):
        self.state = _State()
        self.twists, self.armed_calls, self.modes = [], [], []
        self.stopped = False

    def start(self):
        return self

    def await_connection(self, timeout=5.0):
        return True

    def set_mode(self, mode):
        self.modes.append(mode)
        self.state.mode = mode

    def arm_wheels(self, enabled):
        self.armed_calls.append(enabled)
        self.state.wheel_armed = {"left": enabled, "right": enabled}

    def send_twist(self, vx, wz):
        self.twists.append((round(vx, 4), round(wz, 4)))
        x, y, th = self.state.odom
        self.state.odom = (x + vx * 0.05, y, th + wz * 0.05)

    def stop(self, timeout=3.0):
        self.stopped = True


@pytest.fixture
def robot(monkeypatch):
    fake = _Robot()
    monkeypatch.setattr("bebop_vision.robot.RobotClient", lambda *a, **k: fake)
    return fake


class _Out:
    def __init__(self):
        self.events = []

    async def send_text(self, s):
        self.events.append(json.loads(s))

    async def send_bytes(self, b):
        pass


def _with_turn(coro_fn):
    """Run `coro_fn(turn)` with a live AgentTurn installed (as a turn task)."""
    out = _Out()
    turn = vs.AgentTurn(turn_id="t1", source="text", out=vs.TurnEmitter(out))

    async def main():
        vs._CURRENT_TURN.set(turn)
        vs._TOOL_STATUS.set({"ok": True, "call_id": "c9"})
        result = await coro_fn(turn)
        return result, vs._TOOL_STATUS.get()["ok"]

    result, ok = asyncio.run(main())
    return result, ok, out.events


def test_move_cancel_flag_stops_within_a_tick_and_disarms(robot):
    async def run(turn):
        async def cancel_soon():
            await asyncio.sleep(0.3)
            turn.cancel.set()
        asyncio.get_running_loop().create_task(cancel_soon())
        t0 = time.monotonic()
        out = await vs._move_relative({"forward_m": 2.0})
        return out, time.monotonic() - t0

    (out, secs), ok, events = _with_turn(run)
    assert "cancelled" in out and ok is False
    assert secs < 0.3 + 0.1 + 0.3  # cancel + one tick + cleanup flush
    assert robot.twists[-1] == (0.0, 0.0)
    assert robot.armed_calls == [True, False]           # armed by voice -> disarmed
    assert robot.modes == [vs.pb.MODE_DIAL_IN, vs.pb.MODE_IDLE]  # mode restored
    assert robot.stopped
    progress = [e for e in events if e["type"] == "motion_progress"]
    assert progress and progress[-1]["call_id"] == "c9"
    assert progress[-1]["total_m"] == 2.0 and 0 < progress[-1]["done_m"] < 2.0


def test_move_task_cancel_still_disarms(robot):
    async def main():
        task = asyncio.ensure_future(vs._move_relative({"forward_m": 2.0}))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert robot.twists[-1] == (0.0, 0.0)
    assert robot.armed_calls[-1] is False and robot.modes[-1] == vs.pb.MODE_IDLE


def test_move_refuses_while_another_operator_drives(robot):
    robot.state.has_active_operator = True
    out, ok, _ = _with_turn(lambda turn: vs._move_relative({"forward_m": 0.5}))
    assert "another operator is driving" in out and ok is False
    assert robot.modes == [] and robot.armed_calls == []
    assert not [t for t in robot.twists if t != (0.0, 0.0)]
    robot.state.active_operator_name = "Pixel 8"
    out, _, _ = _with_turn(lambda turn: vs._move_relative({"turn_deg": 30}))
    assert out.startswith("Not moving: Pixel 8 is driving")


def test_move_leaves_human_armed_wheels_armed(robot):
    robot.state.mode = vs.pb.MODE_DIAL_IN
    robot.state.wheel_armed = {"left": True, "right": True}
    out, ok, _ = _with_turn(lambda turn: vs._move_relative({"forward_m": 0.2}))
    assert out.startswith("Moved") and ok is True
    assert robot.armed_calls == []   # never armed, never disarmed
    assert robot.modes == []         # mode untouched
    assert robot.twists[-1] == (0.0, 0.0)


def test_parse_speak_accepts_bools_and_falsey_strings():
    assert vs._parse_speak(True) is True and vs._parse_speak(False) is False
    assert vs._parse_speak(None) is True
    for v in ("false", "False", "0", "no", "off", 0):
        assert vs._parse_speak(v) is False
    for v in ("true", "1", "yes", 1):
        assert vs._parse_speak(v) is True


def test_move_stops_when_preempted_mid_move(robot):
    # The firmware hands the seat to another client only after voice's input
    # has been idle past operator_grace_ms (it is not a human override while
    # voice streams twists); voice must still stop and leave the new holder's
    # wheels and mode alone.
    async def run(turn):
        async def human_grabs():
            await asyncio.sleep(0.3)
            robot.state.has_active_operator = True
        asyncio.get_running_loop().create_task(human_grabs())
        return await vs._move_relative({"forward_m": 2.0})

    out, ok, _ = _with_turn(run)
    assert "another operator took the controls" in out and ok is False
    assert robot.twists[-1] == (0.0, 0.0)
    assert robot.armed_calls == [True]   # armed by voice, not disarmed under the new holder
    assert robot.modes == [vs.pb.MODE_DIAL_IN]  # mode not restored either
