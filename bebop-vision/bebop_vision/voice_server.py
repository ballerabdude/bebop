"""On-device agent gateway: voice and text turns (TensorRT Edge-LLM).

`bebop-voice.service` runs this as root from `bebop-vision/.venv`. It owns
two things:

1. **The model servers.** The default `cascade` backend supervises three
   `tensorrt-edgellm-serve` stages (ASR, a VLM brain with tools, TTS); the
   `omni` backend (chat only, no tools) serves the catalog's `voice`
   checkpoint (Qwen3-Omni-30B-A3B). The first start builds TensorRT engines
   into `--cache-dir`; that can take many minutes.
2. **The app gateway.** A turn-based WebSocket on :9093 for the app's agent
   console: spoken turns (16 kHz PCM16 in) and typed turns, answered with
   streamed text, tool events and optional 24 kHz PCM16 speech.

The runtime WebSocket on :9090 carries only control (`SetVoiceEnabled`) and
status (`VoiceState`); audio never flows through the firmware. See
`docs/voice.md`.

Wire protocol (:9093, `/voice?session=<id>`): voice and typed turns share
one socket. Each turn runs as its own task, so `cancel` lands mid-turn:

    app -> {"type":"utterance_start","sample_rate":16000,"format":"pcm16"}
    app -> <binary PCM16 frames>
    app -> {"type":"utterance_end"}                 (starts a voice turn)
    app -> {"type":"user_text","text":"...","speak":false}   (a typed turn)
    app -> {"type":"cancel"}  /  {"type":"reset"}
    svc -> turn_start, transcript, assistant_delta, tool_call, tool_result,
           motion_progress, audio_start, <binary PCM16 24 kHz>, audio_end,
           turn_end{status: done|cancelled|error}, turn_rejected
    svc -> done (an empty utterance), error{message} (outside a turn: not
           ready, a bad frame, an empty user_text)

The full contract is in docs/voice.md ("Agent protocol").

`GET /healthz` is the status surface the app polls:

    {"ok": bool, "phase": "starting|building|ready|error|stub",
     "detail": "<last meaningful log line>", "model": "...",
     "backend": "cascade|omni", "tools_supported": bool,
     "turn_active": bool, "turns_active": int, ...}

The gateway serves `/healthz` immediately, even while the model server is
still building or has failed, so the operator sees progress instead of a
silent "running". The `--stub` flag runs a tiny echo/tone generator instead of
the model, so the gateway + app protocol can be exercised without the 60 GB
checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
import contextvars
import io
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import models
from .edge_serve import resolve_edgellm_bin
from .proto.bebop.runtime.v1 import bebop_runtime_pb2 as pb
from .session_notes import TASKS

# --- defaults -------------------------------------------------------------

EDGELLM_HOST = "127.0.0.1"
EDGELLM_PORT = 8000
EDGELLM_CACHE_DIR = Path("/var/lib/bebop-voice/edgellm")

VOICE_PURPOSE = "voice"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9093
DEFAULT_VOICE = "chelsie"
# The model emits 24 kHz; the app sends 16 kHz. Both mono s16le.
INPUT_RATE = 16000
OUTPUT_RATE = 24000
# How many prior turns to keep in the conversation. Audio history is base64
# in the prompt, so keep this small to bound prefill cost.
DEFAULT_HISTORY_TURNS = 4
# How many model-server log lines to retain for `/healthz` and the journal.
LOG_RING = 200
# First build can take a long time (six TensorRT engines); only give up after
# this long without the server answering `/health`.
BUILD_TIMEOUT_S = 5400.0
READY_PHASES = ("ready", "stub")

SYSTEM_PROMPT = (
    "You are Bebop, a friendly robot companion. Reply naturally and briefly, "
    "as if speaking aloud. Keep answers to a sentence or two."
)


def _log(msg: str) -> None:
    print(f"[voice] {msg}", flush=True)


def voice_spec() -> models.ModelSpec | None:
    """The catalog entry that serves the `voice` purpose, if any."""
    try:
        catalog = models.load_catalog()
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator
        _log(f"catalog load failed: {exc}")
        return None
    for spec in catalog.values():
        if spec.purpose == VOICE_PURPOSE:
            return spec
    return None


def read_hf_token() -> str | None:
    """The operator's HF token (root-only), for the Edge-LLM child."""
    return models.load_token()


# Qwen3-Omni is only supported at NVFP4 / INT4-AWQ; the raw FP16 snapshot is
# quantized once on-robot into a sibling directory with this suffix.
QUANT_SUFFIX = "-nvfp4"


def quantized_dir(spec: models.ModelSpec) -> Path:
    base = spec.target_dir
    return base.parent / f"{base.name}{QUANT_SUFFIX}"


def resolve_model_dir(spec: models.ModelSpec) -> Path:
    """Serve the on-robot NVFP4 quantization when present, else the snapshot."""
    quant = quantized_dir(spec)
    if (quant / "config.json").is_file():
        return quant
    return spec.target_dir


def pcm16_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _silence_wav_b64(seconds: float = 0.2, rate: int = INPUT_RATE) -> str:
    """A short silent WAV, base64, for the startup warm-up audio turn."""
    pcm = b"\x00\x00" * int(rate * seconds)
    return base64.b64encode(pcm16_to_wav(pcm, rate)).decode("ascii")


# --- operator settings + session context ---------------------------------


# Persisted operator settings (voice, persona, context). Written by the app
# via `POST /config`; root-owned under the service's state dir.
CONFIG_PATH = Path("/var/lib/bebop-voice/config.json")

# Saved config values that no longer name anything. `omni-vllm` was the
# vLLM-Omni container (removed with the vLLM runtime); the same model is now
# registered by `bebop-models` as `omni-edgellm`. Rewritten on load.
_BACKEND_ID_MIGRATIONS = {"omni-vllm": "omni-edgellm"}
MAX_SESSIONS = 32
# Bumped when a saved field changes meaning (see `VoiceConfig.load`).
# 2: cascade became the default backend.
CONFIG_VERSION = 2


def _clamp_turns(n: Any) -> int:
    try:
        value = int(n)
    except (TypeError, ValueError):
        value = DEFAULT_HISTORY_TURNS
    return max(0, min(value, 20))


@dataclass
class VoiceConfig:
    """Operator-tunable settings, edited from the app's Voice page."""

    voice: str = DEFAULT_VOICE
    system_prompt: str = SYSTEM_PROMPT
    history_turns: int = DEFAULT_HISTORY_TURNS
    keep_audio_history: bool = False
    tools_enabled: bool = True
    # Let the brain move the robot (`move_relative`). Deliberately
    # separate from `tools_enabled`: that switch is the app's "live
    # context" toggle, and arming wheels should not ride on it.
    motion_enabled: bool = False
    # "cascade" (ASR -> brain -> TTS; the only engine with tools) is the
    # default. "omni" (end-to-end Qwen3-Omni) stays selectable as a chat-only
    # engine: it has no tools, so no place memory and no motion.
    backend: str = "cascade"
    # When set, the omni backend uses this already-running OpenAI-compatible
    # server instead of supervising an Edge-LLM child. A debug escape hatch.
    omni_url: str = ""
    # When set, the omni backend is owned by the `bebop-models` supervisor: on
    # start the voice service asks the supervisor to `load` this backend and
    # uses the URL it returns — one control plane owns the GPU budget, eviction
    # and the child process.
    omni_supervisor_url: str = ""
    omni_backend_id: str = "omni-edgellm"
    omni_unload_on_stop: bool = False

    @classmethod
    def load(cls, path: Path | None = None) -> "VoiceConfig":
        path = path or CONFIG_PATH
        try:
            raw = json.loads(path.read_text())
        except Exception:  # noqa: BLE001 - missing/invalid -> defaults
            return cls()
        cfg = cls()
        if isinstance(raw.get("voice"), str) and raw["voice"]:
            cfg.voice = raw["voice"]
        if isinstance(raw.get("system_prompt"), str) and raw["system_prompt"].strip():
            cfg.system_prompt = raw["system_prompt"]
        if "history_turns" in raw:
            cfg.history_turns = _clamp_turns(raw["history_turns"])
        if isinstance(raw.get("keep_audio_history"), bool):
            cfg.keep_audio_history = raw["keep_audio_history"]
        if isinstance(raw.get("tools_enabled"), bool):
            cfg.tools_enabled = raw["tools_enabled"]
        if isinstance(raw.get("motion_enabled"), bool):
            cfg.motion_enabled = raw["motion_enabled"]
        if raw.get("backend") in ("omni", "cascade"):
            cfg.backend = raw["backend"]
        # Files written before CONFIG_VERSION 2 persisted the old default
        # ("omni") on every save, so a saved "omni" there is not a choice.
        # Cascade became the default (omni has no tools and underperformed);
        # migrate once. A v2 file keeps whatever the operator picked.
        version = raw.get("config_version", 1)
        legacy = not isinstance(version, int) or version < CONFIG_VERSION
        if legacy and cfg.backend == "omni":
            _log("config: pre-v2 file; switching the default backend "
                 "omni -> cascade (omni stays selectable)")
            cfg.backend = "cascade"
        if isinstance(raw.get("omni_url"), str):
            cfg.omni_url = raw["omni_url"].strip()
        if isinstance(raw.get("omni_supervisor_url"), str):
            cfg.omni_supervisor_url = raw["omni_supervisor_url"].strip()
        if isinstance(raw.get("omni_backend_id"), str) and raw["omni_backend_id"].strip():
            cfg.omni_backend_id = raw["omni_backend_id"].strip()
        # Forward old backend ids. `omni-vllm` was the vLLM-Omni container,
        # removed with the vLLM runtime (58fd661); robots provisioned before
        # that still carry the dead id, which would misdirect a
        # supervisor-delegated omni start at the backend that actually exists.
        migrated = _BACKEND_ID_MIGRATIONS.get(cfg.omni_backend_id)
        if migrated is not None:
            _log(f"config: migrating omni_backend_id "
                 f"{cfg.omni_backend_id!r} -> {migrated!r}")
            cfg.omni_backend_id = migrated
            try:
                cfg.save(path)
            except OSError as exc:  # noqa: BLE001 - migration is best-effort
                _log(f"config: could not persist migration: {exc}")
        if isinstance(raw.get("omni_unload_on_stop"), bool):
            cfg.omni_unload_on_stop = raw["omni_unload_on_stop"]
        return cfg

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2))
        os.replace(tmp, path)

    def snapshot(self) -> dict[str, Any]:
        return {
            "config_version": CONFIG_VERSION,
            "voice": self.voice,
            "system_prompt": self.system_prompt,
            "history_turns": self.history_turns,
            "keep_audio_history": self.keep_audio_history,
            "tools_enabled": self.tools_enabled,
            "motion_enabled": self.motion_enabled,
            "backend": self.backend,
            "omni_url": self.omni_url,
            "omni_supervisor_url": self.omni_supervisor_url,
            "omni_backend_id": self.omni_backend_id,
            "omni_unload_on_stop": self.omni_unload_on_stop,
        }


class SessionStore:
    """Per-session conversation history, keyed by a client-supplied id.

    History holds only turns *after* the system prompt (which is prepended from
    the live config each request, so persona edits apply immediately). A
    client-supplied `?session=<id>` id keeps context across reconnects; each
    anonymous connection gets a fresh bucket. LRU-capped.
    """

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._lock = threading.Lock()
        self._max = max_sessions
        self._map: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()

    def get(self, key: str) -> list[dict[str, Any]]:
        with self._lock:
            if key in self._map:
                self._map.move_to_end(key)
                return self._map[key]
            history: list[dict[str, Any]] = []
            self._map[key] = history
            while len(self._map) > self._max:
                self._map.popitem(last=False)
            return history


# --- agent turns -------------------------------------------------------------
#
# A turn (one voice utterance or one typed message, and everything the brain
# does about it) runs as its own asyncio.Task, so the WebSocket keeps reading
# while it runs: `cancel` arrives mid-turn, and a disconnect cancels it. See
# docs/voice.md "Agent protocol" for the event contract.

# The turn the current task is serving; tools (motion progress, cancel
# checks) read it without threading it through `_run_tool`'s signature.
_CURRENT_TURN: contextvars.ContextVar["AgentTurn | None"] = contextvars.ContextVar(
    "bebop_voice_turn", default=None)
# Per tool call: {"ok": bool, "call_id": str}. A tool marks a refusal with
# `_tool_failed()` so the app can render it as one without parsing text.
_TOOL_STATUS: contextvars.ContextVar["dict[str, Any] | None"] = contextvars.ContextVar(
    "bebop_voice_tool_status", default=None)


def _tool_failed() -> None:
    status = _TOOL_STATUS.get()
    if status is not None:
        status["ok"] = False


class TurnEmitter:
    """Serialised, disconnect-tolerant sender for one WebSocket.

    The turn task and the receive loop both send; a lock keeps frames whole.
    After the peer goes away every send is a silent no-op, so a turn that is
    winding down (stopping the wheels) never dies on a dead socket.
    """

    def __init__(self, ws: Any) -> None:
        self.ws = ws
        self.closed = False
        self._lock = asyncio.Lock()

    async def json(self, obj: dict[str, Any]) -> None:
        await self._send(text=json.dumps(obj))

    async def audio(self, data: bytes) -> None:
        await self._send(data=data)

    async def _send(self, *, text: str | None = None, data: bytes | None = None) -> None:
        if self.closed or self.ws is None:
            return
        async with self._lock:
            try:
                if text is not None:
                    await self.ws.send_text(text)
                else:
                    await self.ws.send_bytes(data)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - peer gone; stop sending
                self.closed = True


_TURN_SEQ = {"n": 0}


def _new_turn_id() -> str:
    _TURN_SEQ["n"] += 1
    return f"t{int(time.time() * 1000):x}-{_TURN_SEQ['n']}"


@dataclass(eq=False)  # identity semantics: turns live in sets
class AgentTurn:
    """One turn: its id, where it came from, and how to talk to the client."""

    turn_id: str
    source: str  # "voice" | "text"
    out: TurnEmitter
    speak: bool = True
    session_key: str = ""
    owner: Any = None  # the WebSocket that started it (disconnect cancels)
    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    task: "asyncio.Task[None] | None" = None
    audio_open: bool = False

    async def emit(self, kind: str, **fields: Any) -> None:
        await self.out.json({"type": kind, "turn_id": self.turn_id, **fields})

    async def delta(self, text: str) -> None:
        await self.emit("assistant_delta", text=text)

    async def audio_start(self) -> None:
        if self.speak and not self.audio_open:
            self.audio_open = True
            await self.emit("audio_start", sample_rate=OUTPUT_RATE, format="pcm16")

    async def audio_end(self) -> None:
        if self.audio_open:
            self.audio_open = False
            await self.emit("audio_end")

    async def audio(self, data: bytes) -> None:
        if self.speak:
            await self.out.audio(data)

    def request_cancel(self) -> None:
        self.cancel.set()
        if self.task is not None and not self.task.done():
            self.task.cancel()

    @property
    def running(self) -> bool:
        return self.task is not None and not self.task.done()


def _parse_speak(value: Any) -> bool:
    """`speak` flag: a JSON bool; strings "false"/"0"/"no"/"off" read as false."""
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in ("false", "0", "no", "off", "")


class TurnRegistry:
    """Server-wide view of running turns (`/healthz` `turn_active`).

    At most one turn per session runs at a time. A new turn while one is
    running is rejected (`turn_rejected`); the client sends `cancel` first.
    A cancelled turn that is still winding down (stopping the wheels) does not
    block the next one: the new turn waits for it before starting.
    """

    def __init__(self) -> None:
        self.by_session: dict[str, AgentTurn] = {}
        self.running: set[AgentTurn] = set()

    @property
    def count(self) -> int:
        return len(self.running)

    def busy(self, session_key: str) -> AgentTurn | None:
        turn = self.by_session.get(session_key)
        if turn is not None and turn.running and not turn.cancel.is_set():
            return turn
        return None



async def run_agent_turn(
    turn: AgentTurn,
    registry: TurnRegistry | None,
    body: Any,
    *,
    previous: AgentTurn | None = None,
) -> None:
    """Run one turn's body with the lifecycle events around it.

    Emits `turn_start`, then whatever the body emits, then
    `turn_end{status}`. Cancellation (the `cancel` message or a disconnect)
    ends the turn as `cancelled`; the body's own `finally` blocks (stop +
    disarm, history commit) run first.
    """
    if registry is not None:
        registry.running.add(turn)
    token = _CURRENT_TURN.set(turn)
    status, error = "done", None
    try:
        if previous is not None and previous.task is not None:
            # A cancelled predecessor may still be stopping the wheels.
            await asyncio.wait({previous.task}, timeout=10.0)
        await turn.emit("turn_start", source=turn.source, speak=turn.speak)
        await body()
    except asyncio.CancelledError:
        status = "cancelled"
    except Exception as exc:  # noqa: BLE001 - reported to the client
        status, error = "error", str(exc) or type(exc).__name__
        _log(f"turn {turn.turn_id} failed: {error}")
    finally:
        _CURRENT_TURN.reset(token)
        if registry is not None:
            registry.running.discard(turn)
    if turn.cancel.is_set() and status == "done":
        status = "cancelled"
    try:
        await turn.audio_end()
        end: dict[str, Any] = {"status": status}
        if error:
            end["error"] = error
        await turn.emit("turn_end", **end)
    finally:
        # Drop the finished turn so the registry does not pin closed sockets
        # (every anonymous connection has its own session key).
        if registry is not None and registry.by_session.get(turn.session_key) is turn:
            registry.by_session.pop(turn.session_key, None)


def _turn_starts(history: list[dict[str, Any]]) -> list[int]:
    """Indices where a turn begins in a flat message history.

    A turn starts at a user message; histories that keep no user messages
    (omni without audio history) start a turn at each assistant reply that
    does not follow a tool result or a tool-calling assistant message.
    """
    starts: list[int] = []
    for i, msg in enumerate(history):
        role = msg.get("role")
        if role == "user":
            starts.append(i)
        elif role == "assistant":
            prev = history[i - 1] if i else None
            if prev is None or (prev.get("role") == "assistant"
                                and not prev.get("tool_calls")):
                starts.append(i)
    return starts


def _trim_history(history: list[dict[str, Any]], turns: int) -> None:
    """Keep the last `turns` whole turns (user, tool calls/results, reply)."""
    if turns <= 0:
        history.clear()
        return
    starts = _turn_starts(history)
    if len(starts) > turns:
        del history[:starts[-turns]]


def _close_tool_calls(msgs: list[dict[str, Any]]) -> None:
    """Give every advertised tool call a result (a cancel can cut one off).

    Chat templates reject an assistant `tool_calls` message whose calls have
    no matching `tool` messages, so a cut-off call gets a placeholder.
    """
    answered = {m.get("tool_call_id") for m in msgs if m.get("role") == "tool"}
    for msg in list(msgs):
        for call in msg.get("tool_calls") or []:
            if call["id"] not in answered:
                msgs.append({"role": "tool", "tool_call_id": call["id"],
                             "content": "Cancelled by the operator before it ran."})
                answered.add(call["id"])


def _tool_args(arguments: str) -> dict[str, Any]:
    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except ValueError:
        return {}
    return args if isinstance(args, dict) else {}


# Cache of the model's speaker ids (GET /v1/voices), so `/config` can validate
# a requested voice without a round-trip every time.
_VOICE_CACHE: dict[str, Any] = {"ts": 0.0, "voices": []}


def known_voices(server: "EdgeLLMServer | None") -> list[str]:
    if server is None:
        return list(_VOICE_CACHE["voices"])
    now = time.time()
    if _VOICE_CACHE["voices"] and now - _VOICE_CACHE["ts"] < 60:
        return list(_VOICE_CACHE["voices"])
    import httpx

    try:
        r = httpx.get(f"{server.base_url}/v1/voices", timeout=5.0)
        r.raise_for_status()
        voices = [
            v["id"]
            for v in r.json().get("data", [])
            if isinstance(v, dict) and v.get("id")
        ]
    except Exception:  # noqa: BLE001
        voices = []
    if voices:
        _VOICE_CACHE["voices"] = voices
        _VOICE_CACHE["ts"] = now
    return list(_VOICE_CACHE["voices"])


# --- live context (cameras + robot status) --------------------------------
#
# With the operator's "live context" switch on (`tools_enabled`), every turn
# carries the robot's current near/far camera frames in the user message and
# a one-line robot status in the system prompt. Attaching them directly
# needs no tool round-trip: omni (chat only) gets them in its single model
# call, and the cascade brain gets them next to its tools.

ROBOT_WS_URL = "ws://127.0.0.1:9090/ws"
VISION_SNAPSHOT_BASE = "http://127.0.0.1:9092/snapshot?stream="
# The rig has two cameras (near + far); both are attached to every turn.
VISION_STREAMS = ("color_near", "color_far")

_MODE_NAMES = {0: "UNSPECIFIED", 1: "IDLE", 2: "DIAL_IN"}


def _format_snapshot(snapshot: Any) -> str:
    """Compact JSON view of a firmware Snapshot for the model, split out for
    tests."""
    armed = [w.name for w in snapshot.wheels if w.armed]
    state: dict[str, Any] = {
        "mode": _MODE_NAMES.get(snapshot.mode, str(snapshot.mode)),
        "estop": bool(snapshot.estop_latched),
        "wheels_armed": armed,
        "drive": bool(snapshot.drive.present),
    }
    if snapshot.estop_reason:
        state["estop_reason"] = snapshot.estop_reason
    if snapshot.drive.present:
        state["odom"] = {
            "x_m": round(snapshot.drive.odom_x, 2),
            "y_m": round(snapshot.drive.odom_y, 2),
            "heading_rad": round(snapshot.drive.odom_theta, 2),
        }
    if snapshot.power.present:
        state["battery_pct"] = round(snapshot.power.state_of_charge_pct, 1)
        state["battery_v"] = round(snapshot.power.battery_voltage_v, 2)
    state["vision_running"] = bool(snapshot.vision.running)
    return json.dumps(state)


# `robot_state_text()` opens a fresh WebSocket to the firmware, which measured a
# flat ~2 s (the snapshot reply is tied to the firmware's telemetry tick). That
# one call dominated a voice turn — more than ASR, vision prefill and the first
# sentence combined — so it is cached and kept warm by `_refresh_state_loop`.
_STATE_TTL_S = 15.0
_STATE_REFRESH_S = 5.0
_STATE_CACHE: dict[str, Any] = {"ts": 0.0, "text": ""}


async def robot_state_text(*, refresh: bool = False) -> str:
    """One-line robot status for the system prompt; '' when unavailable.

    Served from a short-lived cache; `refresh=True` forces a firmware round
    trip (used by the background refresher).
    """
    import websockets
    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as rt

    if not refresh and _STATE_CACHE["text"] and (
        time.monotonic() - _STATE_CACHE["ts"] < _STATE_TTL_S
    ):
        return _STATE_CACHE["text"]
    try:
        async with websockets.connect(
            ROBOT_WS_URL, open_timeout=5, close_timeout=2, max_size=None
        ) as ws:
            msg = rt.ClientRuntimeMessage(request_id=1)
            msg.get_snapshot.SetInParent()
            await ws.send(msg.SerializeToString())
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                data = await asyncio.wait_for(
                    ws.recv(), timeout=max(0.2, deadline - time.monotonic())
                )
                env = rt.ServerRuntimeMessage()
                env.ParseFromString(data)
                if env.request_id == 1 and env.WhichOneof("payload") == "snapshot":
                    text = _format_snapshot(env.snapshot)
                    _STATE_CACHE["text"] = text
                    _STATE_CACHE["ts"] = time.monotonic()
                    return text
    except Exception as exc:  # noqa: BLE001 - status is best-effort
        _log(f"robot state unavailable: {exc}")
    # A stale status line beats none.
    return str(_STATE_CACHE["text"])


async def _refresh_state_loop() -> None:
    """Keep `robot_state_text()` warm so a turn never pays the ~2 s fetch."""
    while True:
        try:
            await robot_state_text(refresh=True)
        except Exception as exc:  # noqa: BLE001 - never let the loop die
            _log(f"state refresh failed: {exc}")
        await asyncio.sleep(_STATE_REFRESH_S)


async def _fetch_snapshot(stream: str, attempts: int = 2) -> bytes | None:
    import httpx

    last: Exception | None = None
    for _ in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{VISION_SNAPSHOT_BASE}{stream}")
                r.raise_for_status()
                return r.content
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            await asyncio.sleep(0.3)
    _log(f"snapshot {stream} unavailable: {last}")
    return None


async def _enable_vision() -> bool:
    """Ask the firmware to start the vision service (mirrors the app's toggle).

    Best-effort; the resulting state is reflected in telemetry, so the app's
    vision card follows. Returns True if the request was sent.
    """
    import websockets
    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as rt

    try:
        async with websockets.connect(
            ROBOT_WS_URL, open_timeout=5, close_timeout=2, max_size=None
        ) as ws:
            msg = rt.ClientRuntimeMessage(request_id=1)
            msg.set_vision_enabled.enabled = True
            await ws.send(msg.SerializeToString())
            await asyncio.sleep(0.3)
        _log("cameras down; asked firmware to enable vision")
        return True
    except Exception as exc:  # noqa: BLE001
        _log(f"could not enable vision: {exc}")
        return False


async def _snapshot_views() -> list[tuple[str, bytes]]:
    # Sequential: the snapshot endpoint is only lightly loaded, and serializing
    # avoids piling concurrent grabs onto the vision process.
    views: list[tuple[str, bytes]] = []
    for stream in VISION_STREAMS:
        jpeg = await _fetch_snapshot(stream)
        if jpeg is not None:
            views.append((stream, jpeg))
    return views


async def live_context() -> tuple[str, list[tuple[str, bytes]]]:
    """Gather the per-turn context: robot status + current camera frames.

    If the cameras are down, ask the firmware to start vision and wait a
    bounded time for the first frames. Returns `("", [])` style best-effort
    values rather than raising.
    """
    state = await robot_state_text()
    views = await _snapshot_views()
    if not views and await _enable_vision():
        for _ in range(10):
            await asyncio.sleep(1.5)
            views = await _snapshot_views()
            if views:
                break
    return state, views


# --- cascade (multi-hop: ASR -> VLM brain -> TTS) -------------------------
#
# Instead of one end-to-end omni model, the cascade uses the best model for
# each stage, each as its own Edge-LLM OpenAI server:
#   ASR   (Qwen3-ASR)              audio -> transcript      /v1/audio/transcriptions
#   Brain (Qwen3.8-27B NVFP4 VLM)  text+image(+tools)->text /v1/chat/completions
#   TTS   (Qwen3-TTS)              text -> speech           /v1/audio/speech
#
# A bonus: the brain is a text/VLM model, so native tool calling works (the
# omni server rejects tools combined with audio output).

CASCADE_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
CASCADE_BRAIN_MODEL = "RadixArk/Qwen3.8-27B-NVFP4"
# Multi-token prediction for the brain. `mtp` uses the checkpoint's own MTP
# layers, so no draft model is needed (per the server CLI: "MTP is the only
# method that legitimately has none"). The mode is baked into the engine
# bundle profile name (the `<spec_type>` slot), so enabling it triggers ONE
# rebuild; the previous bundle is kept, so removing the flag reverts instantly.
# `num_speculative_tokens` is the number of tokens drafted per step.
CASCADE_BRAIN_SPECULATIVE = '{"method": "mtp", "num_speculative_tokens": 3}'
CASCADE_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CASCADE_ASR_PORT = 8011
CASCADE_BRAIN_PORT = 8012
CASCADE_TTS_PORT = 8013


_TTS_VOICE_CACHE: dict[str, Any] = {"ts": 0.0, "voices": []}


class StageServer:
    """Supervises one stage's `tensorrt-edgellm-serve` process."""

    def __init__(
        self,
        name: str,
        model: str,
        cache_dir: Path,
        port: int,
        *,
        capability: str,
        extra_args: list[str] | None = None,
        patched_launcher: bool = False,
    ) -> None:
        self.name = name
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.port = port
        self.capability = capability
        self.extra_args = extra_args or []
        self.patched_launcher = patched_launcher
        self.proc: subprocess.Popen[str] | None = None
        self.ready = False
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://{EDGELLM_HOST}:{self.port}"

    def health(self) -> dict[str, Any] | None:
        import httpx

        try:
            r = httpx.get(f"{self.base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001 - not up yet
            pass
        return None

    def is_ready(self) -> bool:
        h = self.health()
        if not h:
            return False
        caps = h.get("capabilities", {})
        return bool(caps.get(self.capability))

    def _cmd(self) -> list[str]:
        args = [
            self.model,
            "--cache-dir",
            str(self.cache_dir),
            "--port",
            str(self.port),
            "--host",
            EDGELLM_HOST,
            *self.extra_args,
        ]
        if self.patched_launcher:
            return [sys.executable, "-m", "bebop_vision.edge_serve", *args]
        return [resolve_edgellm_bin(), *args]

    def start_and_wait(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.is_ready():
            self.ready = True
            _log(f"{self.name}: already up on {self.base_url}")
            return
        cmd = self._cmd()
        _log(f"{self.name}: starting ({self.model}) on port {self.port}")
        env = os.environ.copy()
        token = read_hf_token()
        if token:
            env["HF_TOKEN"] = token
        # `-m bebop_vision.edge_serve` needs the package root importable.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        with self._lock:
            self.proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            proc = self.proc
        atexit.register(self.stop)
        threading.Thread(
            target=self._read_output, name=f"{self.name}-logs", daemon=True
        ).start()
        deadline = time.monotonic() + BUILD_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.is_ready():
                self.ready = True
                _log(f"{self.name}: ready on {self.base_url}")
                return
            if proc.poll() is not None:
                _log(f"{self.name}: exited rc={proc.returncode}")
                return
            time.sleep(2.0)
        _log(f"{self.name}: timed out waiting for {self.capability}")

    def _read_output(self) -> None:
        proc = self.proc
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            line = line.rstrip()
            if not line or "httpx:" in line:
                continue
            _log(f"[{self.name}] {line}")

    def stop(self) -> None:
        with self._lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        _log(f"{self.name}: stopping")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


class CascadePipeline:
    """Owns the stage servers and reports an aggregate status."""

    def __init__(
        self,
        asr: StageServer,
        brain: StageServer,
        tts: StageServer,
        status: "VoiceStatus",
    ) -> None:
        self.asr = asr
        self.brain = brain
        self.tts = tts
        self.status = status

    def start_background(self) -> None:
        threading.Thread(
            target=self._run, name="cascade-supervisor", daemon=True
        ).start()

    def _run(self) -> None:
        self.status.set(phase="starting", detail="starting cascade")
        # Sequential: one engine build at a time keeps GPU memory bounded.
        for stage in (self.asr, self.tts, self.brain):
            self.status.set(detail=f"starting {stage.name} ({stage.model})")
            stage.start_and_wait()
            if not stage.ready:
                self.status.set(
                    phase="error", detail=f"{stage.name} failed to start"
                )
                return
        self.status.set(phase="ready", detail="cascade ready")
        self.status.beat()

    @property
    def ready(self) -> bool:
        return self.asr.ready and self.tts.ready and self.brain.ready

    def stop(self) -> None:
        for stage in (self.asr, self.brain, self.tts):
            stage.stop()


async def _cascade_transcribe(stage: StageServer, wav_bytes: bytes) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(
            f"{stage.base_url}/v1/audio/transcriptions",
            files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
            data={"response_format": "json"},
        )
        r.raise_for_status()
        return str(r.json().get("text", "")).strip()


async def _cascade_chat_stream(
    stage: StageServer,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the brain's stream as events.

    Events: `{"type":"text","delta"}` and `{"type":"tool_call","index","id",
    "name","arguments"}` (the server parses the model's XML via `qwen3_xml`).

    The cascade used to wait for the whole reply before synthesising, so the
    first audio lagged "user stopped talking" by the entire generation time
    (~5.8 s measured). Streaming lets TTS start on the first sentence.
    """
    import httpx

    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", f"{stage.base_url}/v1/chat/completions", json=payload
        ) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield {"type": "text", "delta": text}
                for tc in delta.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    yield {
                        "type": "tool_call",
                        "index": int(tc.get("index", 0) or 0),
                        "id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "arguments": fn.get("arguments") or "",
                    }


def _sentence_chunks(buf: str) -> tuple[list[str], str]:
    """Split finished sentences off a streaming buffer.

    Returns ``(sentences, remainder)``. A sentence ends at `.`/`!`/`?`
    followed by whitespace, so the remainder always holds an unfinished
    sentence (or the tail after the last terminator).
    """
    ready: list[str] = []
    start = 0
    for match in re.finditer(r"(?<=[.!?])\s+", buf):
        ready.append(buf[start:match.end()].strip())
        start = match.end()
    return [s for s in ready if s], buf[start:]


# --- place-memory tools -----------------------------------------------------
# The places service lives inside `bebop-vision` (:9095) — a different process —
# so the brain reaches it over the same HTTP API the app uses. Reads answer
# questions about memory; writes (remember/fact/rename/forget) act only on an
# explicit request, and reuse that API's guards (name uniqueness, per-place
# growth caps).
PLACES_URL = "http://127.0.0.1:9095"
# bebop-vision video server: also hosts the drive-note routes that write
# into the active MCAP segment (POST /intent, POST /event).
VISION_URL = "http://127.0.0.1:9092"

PLACES_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "where_am_i",
            "description": (
                "Recognise which remembered room the robot is in right now, by "
                "appearance. Use when asked where you are, or whether you "
                "recognise the current room."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_places",
            "description": "List the rooms/places the robot has been taught.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_facts",
            "description": "What the robot knows about one remembered place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Place name, e.g. 'office room'.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_place",
            "description": (
                "Create a new place from what the robot can see right now. Use "
                "when the user asks to remember/save the room they are in or "
                "the current view, e.g. 'remember this as the study'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name to save it under.",
                    },
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional facts to record with it.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_place_fact",
            "description": (
                "Record something about a place the robot already knows, e.g. "
                "'the office has a whiteboard'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Place name."},
                    "text": {"type": "string", "description": "The fact."},
                },
                "required": ["name", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rename_place",
            "description": "Give a remembered place a different name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Current name."},
                    "new_name": {"type": "string", "description": "New name."},
                },
                "required": ["name", "new_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_place",
            "description": (
                "Delete a remembered place and its keyframes. Only when the "
                "user asks to forget/remove it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Place name."}
                },
                "required": ["name"],
            },
        },
    },
]


def _place_arg(args: dict[str, Any]) -> str:
    return str(args.get("name") or "").strip()


async def _drive_note(name: str, args: dict[str, Any]) -> str:
    """note_drive_intent / mark_drive_event -> the recorder's note routes."""
    import httpx

    if name == "note_drive_intent":
        path = "/intent"
        given = {key: str(args.get(arg) or "").strip()
                 for key, arg in (("text", "text"), ("from", "from_place"),
                                  ("to", "to_place"), ("task", "task"),
                                  ("target", "target"))}
        mode = str(args.get("mode") or "").strip().lower()
        if mode == "continue":
            # A continue edits the open run: send only what the model gave,
            # so unspoken fields (the places, say) keep their values.
            payload = {k: v for k, v in given.items() if v}
            payload["mode"] = "continue"
        elif not any(given.values()):
            # Nothing at all: an explicit clear, so the spoken reply and the
            # recorded /intent agree.
            payload = {"clear": True}
        else:
            # A new run: text and places always go (as before), task, target
            # and mode only when given, so an old-style call posts exactly
            # what it always did.
            payload = {"text": given["text"], "from": given["from"],
                       "to": given["to"]}
            for key in ("task", "target"):
                if given[key]:
                    payload[key] = given[key]
            if mode:
                payload["mode"] = mode
    else:
        path = "/event"
        payload = {"kind": str(args.get("kind") or "note"),
                   "note": str(args.get("note") or "")}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{VISION_URL}{path}", json=payload)
    except Exception as exc:  # noqa: BLE001 - report, never break the turn
        return f"The recorder is unavailable ({exc})."
    if r.status_code == 409:
        if path == "/intent":
            return "No run is open to continue; say it again as a new run."
        return "Not recording right now, so nothing was marked."
    if r.status_code != 200:
        detail = ""
        if r.status_code == 400:
            try:
                detail = f": {r.json().get('error')}"
            except Exception:  # noqa: BLE001 - the status alone is enough
                detail = ""
        return f"The recorder refused the note (HTTP {r.status_code}){detail}."
    if path == "/intent":
        if payload.get("clear"):
            return "Cleared the drive intent."
        # Describe the intent the recorder now holds (after a continue that
        # is the merged run), falling back to what was sent.
        try:
            shown = r.json().get("intent")
        except Exception:  # noqa: BLE001 - the payload is enough
            shown = None
        if not isinstance(shown, dict):
            shown = payload
        route = (f" ({shown.get('from', '')} -> {shown.get('to', '')})"
                 if shown.get("from") or shown.get("to") else "")
        task = ""
        if shown.get("task"):
            task = f" [{shown['task']}"
            task += f": {shown['target']}]" if shown.get("target") else "]"
        return f"Noted: this drive is {shown.get('text', '')!r}{route}{task}."
    note = f": {payload['note']}" if payload["note"] else ""
    return f"Marked {payload['kind']}{note}."


async def _run_tool(name: str, arguments: str) -> str:
    """Run one voice tool; return a short result for the model.

    Reads answer questions about memory; writes act on an explicit request
    ("remember this as the study"). The write path is the same one the app
    uses, so it inherits its guards (name uniqueness, per-place growth caps)
    rather than inventing new ones.
    """
    import httpx

    try:
        args = json.loads(arguments) if arguments.strip() else {}
    except ValueError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    # Robot tools do not go through the places HTTP API.
    if name == "robot_status":
        return await _robot_status_text()
    if name == "move_relative":
        return await _move_relative(args)
    if name in ("note_drive_intent", "mark_drive_event"):
        return await _drive_note(name, args)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if name == "remember_place":
                label = _place_arg(args)
                if not label:
                    return "No name was given for the new place."
                facts = args.get("facts") or []
                if isinstance(facts, str):
                    facts = [facts]
                r = await client.post(f"{PLACES_URL}/places",
                                      json={"name": label,
                                            "facts": [str(f) for f in facts]})
                if r.status_code not in (200, 201):
                    body = r.json() if r.headers.get("content-type", "").startswith(
                        "application/json") else {}
                    return (f"Could not save {label!r}: "
                            f"{body.get('error') or r.status_code}")
                saved = r.json()
                views = saved.get("views", 0)
                desc = saved.get("descriptors", 0)
                note = ("" if desc else
                        " (I could not take a usable view just now, so I "
                        "cannot recognise it later without another look.)")
                return (f"Saved {label!r} from {views} camera view(s), "
                        f"{desc} appearance descriptor(s){note}.")
            if name == "add_place_fact":
                label, text = _place_arg(args), str(args.get("text") or "").strip()
                if not label or not text:
                    return "Both a place name and a fact are required."
                r = await client.post(f"{PLACES_URL}/places/{label}/facts",
                                      json={"text": text})
                if r.status_code not in (200, 201):
                    return f"No place called {label!r} is remembered."
                return f"Noted about {label!r}: {text}"
            if name == "rename_place":
                label = _place_arg(args)
                new_name = str(args.get("new_name") or "").strip()
                if not label or not new_name:
                    return "Both the current and the new name are required."
                r = await client.post(f"{PLACES_URL}/places/{label}/rename",
                                      json={"name": new_name})
                if r.status_code != 200:
                    return f"Could not rename {label!r} (no such place, or {new_name!r} is taken)."
                return f"Renamed {label!r} to {new_name!r}."
            if name == "forget_place":
                label = _place_arg(args)
                if not label:
                    return "No place name was given."
                r = await client.delete(f"{PLACES_URL}/places/{label}")
                if r.status_code != 200:
                    return f"No place called {label!r} is remembered."
                return f"Forgot {label!r}."
            if name == "where_am_i":
                d = (await client.get(f"{PLACES_URL}/whereami")).json()
                if not d.get("matched"):
                    return "The current room is not recognised (no taught place matches)."
                place = d["place"]
                facts = "; ".join(f["text"] for f in place.get("facts", []))
                return (f"Recognised place: {place['name']} "
                        f"(appearance match {d.get('score')}). "
                        f"Facts: {facts or 'none recorded'}.")
            if name == "list_places":
                d = (await client.get(f"{PLACES_URL}/places")).json()
                names = [p["name"] for p in d.get("places", [])]
                return "Remembered places: " + (", ".join(names) or "none")
            if name == "place_facts":
                target = str(args.get("name") or "").strip()
                if not target:
                    return "No place name was given."
                r = await client.get(f"{PLACES_URL}/places/{target}")
                if r.status_code != 200:
                    return f"No place called {target!r} is remembered."
                place = r.json()["place"]
                facts = "; ".join(f["text"] for f in place.get("facts", []))
                return f"{place['name']} — {facts or 'no facts recorded'}."
    except Exception as exc:  # noqa: BLE001 - report, never break the turn
        _tool_failed()
        return f"The places service is unavailable ({exc})."
    _tool_failed()
    return f"Unknown tool {name!r}."


# --- motion -----------------------------------------------------------------
# Moving the robot is the one capability here that cannot be undone, so it is
# deliberately narrow:
#   * its own config switch (`motion_enabled`), NOT the casual "live context"
#     toggle — nobody should arm a robot by flipping a context setting;
#   * hard clamps on what a single call may ask for;
#   * arming is explicit and temporary: mode -> DIAL_IN (the firmware's
#     `Supervisor::arm_wheel` refuses any other mode), enable all wheels, move,
#     then stop + disarm + restore the mode. Every exit path stops the wheels.
#   * the firmware is the backstop, not this code: its watchdog zeroes a stale
#     twist (so a crashed tool coasts to a stop, it does not run away), it
#     slews every wheel command and clamps it to the wheel's `vel_max`, and at
#     most one WS client may hold the drive assignment.
#   * NO E-STOP control, deliberately. There is no tool to latch, clear or
#     reset one: an E-STOP means a person must go and look at the robot, so the
#     brain may only *report* it and refuse to move (`move_relative` bails
#     before arming; the mode ends at IDLE and the wheels stay disarmed).
#     `RobotClient` has no E-STOP reset on purpose. Do not add one here — a
#     misheard "clear the e-stop" that actually clears one defeats the whole
#     mechanism.
# It is odom-relative only. There is no map: "move to the door" needs metric
# localisation this robot does not have (see bebop_vision/places/__init__.py).
MOTION_MAX_FORWARD_M = 2.0
MOTION_MAX_TURN_DEG = 180.0
MOTION_LINEAR_MS = 0.2          # gentle; 1 m takes ~5 s
MOTION_ANGULAR_RAD_S = 0.6
MOTION_TOL_M = 0.03
MOTION_TOL_RAD = 0.06           # ~3.5 deg
MOTION_ARM_WAIT_S = 4.0

# Drive notes: the operator says what a drive is for ("I'm driving from the
# living room to the kitchen") or marks a moment ("mark that sock"); the
# recorder writes them into the drive MCAP (/intent, /event) so the data
# carries its own labels.  Read-write but harmless: they never move anything.
DRIVE_NOTE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "note_drive_intent",
            "description": (
                "Record what the current drive is for, so the recording "
                "carries it: e.g. 'from the living room to the kitchen'. Use "
                "when the operator says where they are driving to or what "
                "this run is about. A call with no text, places, task or "
                "target clears the intent. Each call starts a new run unless "
                "mode is 'continue' (the operator is correcting the current "
                "run; give only the fields that change)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string",
                             "description": "the mission in the operator's words"},
                    "from_place": {"type": "string"},
                    "to_place": {"type": "string"},
                    "task": {"type": "string", "enum": list(TASKS),
                             "description": "optional: the kind of run"},
                    "target": {"type": "string",
                               "description": (
                                   "optional, for the task: a place name "
                                   "(go_to_place), left|right|ahead "
                                   "(go_through_door), the step text "
                                   "(follow_step), the object (look_for), "
                                   "keep|left|right|stop|turn-around "
                                   "(maneuver)")},
                    "mode": {"type": "string",
                             "enum": ["new", "repeat", "continue"],
                             "description": "optional, default new"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_drive_event",
            "description": (
                "Drop a marker into the current recording. kind: 'obstacle' "
                "(a planted doll, sock, cable...), 'recovery' (starting "
                "off-lane on purpose), 'exclude' (that stretch was a "
                "mistake), or 'note'. Use when the operator says to mark or "
                "note something during a drive. Markers only land while a "
                "drive is being recorded (wheels armed); mark 'exclude' "
                "before disarming."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["obstacle", "recovery", "exclude", "note"]},
                    "note": {"type": "string"},
                },
                "required": ["kind"],
            },
        },
    },
]

ROBOT_STATUS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "robot_status",
        "description": (
            "The robot's own state: drive mode, whether the E-STOP is latched, "
            "which wheels are armed, battery voltage, and its odometry pose. "
            "Use before or after moving, or when asked how the robot is doing. "
            "A latched E-STOP can only be cleared by a person at the robot — "
            "you cannot reset it."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

MOVE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "move_relative",
        "description": (
            "Drive the robot relative to where it is now: forward/back by a "
            "distance, and/or turn on the spot. There is no map, so it cannot "
            "accept a destination like 'the door' or a room name — use small "
            "bounded moves plus what you can see. Only when the user asks the "
            "robot to move. It refuses if the E-STOP is latched or the wheels "
            "will not arm; both need a person to sort out."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "forward_m": {
                    "type": "number",
                    "description": (f"Metres forward (negative = back); "
                                    f"magnitude ≤ {MOTION_MAX_FORWARD_M}."),
                },
                "turn_deg": {
                    "type": "number",
                    "description": (f"Degrees to turn on the spot, + = left; "
                                    f"magnitude ≤ {MOTION_MAX_TURN_DEG}."),
                },
            },
        },
    },
}


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


async def _robot_status_text() -> str:
    """Read-only robot state (mode, estop, armed wheels, odom, battery)."""
    text = await robot_state_text()
    return text or "Robot status is unavailable."


def _seat_taken(st: Any) -> bool:
    """Another firmware client holds the drive seat (DriveState arbitration)."""
    return bool(getattr(st, "has_active_operator", False)) and not bool(
        getattr(st, "you_are_active_operator", False))


async def _move_relative(args: dict[str, Any]) -> str:
    """Drive a bounded odom-relative move: forward/back then turn, or a turn.

    Refuses while a person holds the drive seat. Arms the chassis for the
    duration only when it was not already armed, and on every exit path
    stops; it then undoes only what it changed itself (disarm if it armed,
    restore the mode if it switched it) — wheels a person armed stay armed.
    """
    forward_m = float(args.get("forward_m") or 0.0)
    turn_deg = float(args.get("turn_deg") or 0.0)
    if abs(forward_m) < 0.01 and abs(turn_deg) < 0.5:
        _tool_failed()
        return "No movement was requested."
    forward_m = max(-MOTION_MAX_FORWARD_M, min(MOTION_MAX_FORWARD_M, forward_m))
    turn_deg = max(-MOTION_MAX_TURN_DEG, min(MOTION_MAX_TURN_DEG, turn_deg))

    from .robot import RobotClient

    # Announced as "voice" (ClientHello), so the app shows who is driving,
    # and as autonomous: any person's drive input (a stop included) takes the
    # seat from this connection at once — see _run_move for how voice yields.
    robot = RobotClient(url=ROBOT_WS_URL, hello_name="voice",
                        autonomous=True).start()
    before_mode = None
    changed_mode = False
    armed_by_us = False
    try:
        # Blocking wait: keep it off the event loop so `cancel` still lands.
        if not await asyncio.to_thread(robot.await_connection, 5.0):
            _tool_failed()
            return "I cannot reach the robot's runtime, so I did not move."
        st = robot.state
        before_mode = st.mode
        if st.estop_latched:
            _tool_failed()
            return (f"Not moving: the E-STOP is latched "
                    f"({st.estop_reason or 'no reason given'}).")
        if _seat_taken(st):
            _tool_failed()
            who = getattr(st, "active_operator_name", "") or "another operator"
            return (f"Not moving: {who} is driving the robot right now. They "
                    "have to let go of the controls first.")

        already_armed = bool(st.wheel_armed) and all(st.wheel_armed.values())
        if already_armed:
            # A person armed the wheels: drive as-is and leave them armed.
            if st.mode != pb.MODE_DIAL_IN:
                _tool_failed()
                return ("Not moving: the wheels are armed in "
                        f"{_MODE_NAMES.get(st.mode, st.mode)} mode, not dial-in.")
        else:
            # Arming requires DIAL_IN, the bring-up / teleop mode with the
            # slew limit and watchdog enforced.
            if before_mode != pb.MODE_DIAL_IN:
                robot.set_mode(pb.MODE_DIAL_IN)
                changed_mode = True
            deadline = time.monotonic() + MOTION_ARM_WAIT_S
            while time.monotonic() < deadline and st.mode != pb.MODE_DIAL_IN:
                await asyncio.sleep(0.05)
            if st.mode != pb.MODE_DIAL_IN:
                _tool_failed()
                return "Could not switch to dial-in mode, so I did not move."

            robot.arm_wheels(True)
            armed_by_us = True  # from here on, the cleanup disarms
            deadline = time.monotonic() + MOTION_ARM_WAIT_S
            armed = False
            while time.monotonic() < deadline:
                if st.wheel_armed and all(st.wheel_armed.values()):
                    armed = True
                    break
                if st.estop_latched:
                    _tool_failed()
                    return "Not moving: the E-STOP latched while arming."
                await asyncio.sleep(0.05)
            if not armed:
                _tool_failed()
                return "The wheels did not arm, so I did not move."

        dist, turned, secs, stopped = await _run_move(robot, forward_m, turn_deg)
        took_over = stopped is not None and stopped.startswith(_TOOK_OVER_PREFIX)
        # Report honestly from telemetry, not from what was asked for.
        direction = "forward" if forward_m > 0 else "back"
        if abs(forward_m) >= 0.01:
            body = f"{dist:.2f} m {direction}"
        else:
            body = f"{abs(math.degrees(turned)):.0f} deg {'left' if turn_deg > 0 else 'right'}"
        where = (f"now at x={st.odom[0]:.2f} m, y={st.odom[1]:.2f} m, "
                 f"heading {math.degrees(st.odom[2]):.0f} deg")
        if took_over:
            # A person took the controls mid-move: report it plainly; the
            # turn goes on (or ends) with this as the tool result.
            _tool_failed()
            return f"{stopped} after {body} ({where}). I let them drive."
        if stopped is not None:
            _tool_failed()
            return f"I stopped early ({stopped}) after {body} ({where})."
        return f"Moved {body} in {secs:.1f}s ({where})."
    finally:
        # Synchronous submits, in order, on the client's FIFO outbox: stop,
        # then undo only what this call changed.
        try:
            robot.send_twist(0.0, 0.0)
            # If another client now holds the drive seat it owns the armed
            # state: do not pull the wheels or the mode out from under it.
            if not _seat_taken(robot.state):
                if armed_by_us:
                    robot.arm_wheels(False)
                if changed_mode and before_mode is not None:
                    robot.set_mode(before_mode)
            else:
                _log("move_relative: another operator holds the seat; "
                     "leaving wheels/mode as they are")
            await _flush_window(0.2)
        finally:
            robot.stop()


async def _flush_window(seconds: float) -> None:
    """Give the robot client's sender time to flush the cleanup commands.

    Runs in `finally` blocks of a turn that may be cancelled a second time
    (cancel, then a disconnect); a cancelled sleep must not skip the flush
    or `robot.stop()` would drop the disarm, so it falls back to blocking.
    """
    t0 = time.monotonic()
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        time.sleep(max(0.0, seconds - (time.monotonic() - t0)))
        raise


MOTION_PROGRESS_S = 0.2  # motion_progress event period
_TOOK_OVER_PREFIX = "stopped: "  # `stop_reason` when someone took the seat


def _took_over_reason(st: Any) -> str:
    who = getattr(st, "active_operator_name", "") or "another operator"
    return f"{_TOOK_OVER_PREFIX}{who} took over"


async def _run_move(
    robot: Any, forward_m: float, turn_deg: float
) -> tuple[float, float, float, str | None]:
    """Closed loop on wheel odometry.

    Returns (distance_m, turned_rad, seconds, stopped) where `stopped` is None
    when the move completed, else why it stopped early: "E-STOP latched",
    "link to the runtime lost", "stopped: <name> took over" or "cancelled by
    the operator". The turn's cancel flag is checked every control tick
    (50 ms); task cancellation also lands at each sleep. Either way the
    `finally` zeroes the twist.

    Preemption: voice's connection is autonomous (`ClientHello.autonomous`),
    so the firmware gives the seat to a person the moment their app sends any
    drive command, a stop included, and applies the person's twist. Telemetry
    then shows `has_active_operator && !you_are_active_operator`; the next
    tick sees it and returns without another non-zero twist — voice never
    fights for the seat (the firmware would reject it anyway).
    """
    st = robot.state
    t0 = time.monotonic()
    start = tuple(st.odom)
    turn = _CURRENT_TURN.get()
    tool = _TOOL_STATUS.get() or {}
    total_m, total_deg = abs(forward_m), abs(turn_deg)
    last_progress = [0.0]

    async def progress(force: bool = False) -> None:
        if turn is None:
            return
        now = time.monotonic()
        if not force and now - last_progress[0] < MOTION_PROGRESS_S:
            return
        last_progress[0] = now
        x, y, th = st.odom
        await turn.emit(
            "motion_progress", call_id=tool.get("call_id", ""),
            done_m=round(math.hypot(x - start[0], y - start[1]), 3),
            total_m=total_m,
            done_deg=round(abs(math.degrees(_wrap_pi(th - start[2]))), 1),
            total_deg=total_deg,
        )

    def stop_reason() -> str | None:
        if turn is not None and turn.cancel.is_set():
            return "cancelled by the operator"
        if st.estop_latched:
            return "E-STOP latched"
        if not st.connected:
            return "link to the runtime lost"
        if _seat_taken(st):
            return _took_over_reason(st)
        return None

    async def settle(axis: str, target: float) -> str | None:
        """Drive one axis to `target` from where this phase starts."""
        ref_x, ref_y, ref_th = st.odom
        budget = (abs(target) / max(MOTION_LINEAR_MS, 1e-6)) + 8.0
        end = time.monotonic() + budget
        while time.monotonic() < end:
            reason = stop_reason()
            if reason is not None:
                return reason
            x, y, th = st.odom
            if axis == "turn":
                err = _wrap_pi(ref_th + target - th)
                if abs(err) <= MOTION_TOL_RAD:
                    return None
                robot.send_twist(0.0, max(-MOTION_ANGULAR_RAD_S,
                                          min(MOTION_ANGULAR_RAD_S, 1.5 * err)))
            else:
                # Distance along the heading this phase set out on.
                travelled = (x - ref_x) * math.cos(ref_th) + \
                            (y - ref_y) * math.sin(ref_th)
                err = target - travelled
                if abs(err) <= MOTION_TOL_M:
                    return None
                # A small heading hold keeps it straight.
                hold = _wrap_pi(ref_th - th)
                robot.send_twist(
                    max(-MOTION_LINEAR_MS, min(MOTION_LINEAR_MS, 1.2 * err)),
                    max(-0.4, min(0.4, 1.5 * hold)),
                )
            await progress()
            await asyncio.sleep(0.05)
        return None  # out of time: report what actually happened

    stopped: str | None = None
    try:
        if abs(turn_deg) >= 0.5:
            stopped = await settle("turn", math.radians(turn_deg))
        if stopped is None and abs(forward_m) >= 0.01:
            stopped = await settle("forward", forward_m)
    finally:
        robot.send_twist(0.0, 0.0)
    await progress(force=True)
    x, y, th = st.odom
    dist = math.hypot(x - start[0], y - start[1])
    return dist, _wrap_pi(th - start[2]), time.monotonic() - t0, stopped


async def _cascade_speak(
    stage: StageServer, text: str, voice: str
) -> AsyncIterator[bytes]:
    import httpx

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{stage.base_url}/v1/audio/speech",
            json={"input": text, "voice": voice, "response_format": "pcm"},
        ) as r:
            if r.status_code != 200:
                body = (await r.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"TTS HTTP {r.status_code}: {body[:200]}")
            async for chunk in r.aiter_bytes():
                if chunk:
                    yield chunk


def cascade_voices(tts: StageServer | None) -> list[str]:
    """Available TTS speaker ids (cached)."""
    if tts is None:
        return list(_TTS_VOICE_CACHE["voices"])
    import httpx

    now = time.time()
    if _TTS_VOICE_CACHE["voices"] and now - _TTS_VOICE_CACHE["ts"] < 120:
        return list(_TTS_VOICE_CACHE["voices"])
    try:
        r = httpx.get(f"{tts.base_url}/v1/voices", timeout=5.0)
        r.raise_for_status()
        ids = [v["id"] for v in r.json().get("data", []) if v.get("id")]
        if ids:
            _TTS_VOICE_CACHE["voices"] = ids
            _TTS_VOICE_CACHE["ts"] = now
    except Exception:  # noqa: BLE001 - fall back to any cached/default
        pass
    return list(_TTS_VOICE_CACHE["voices"])


def resolve_cascade_voice(tts: StageServer | None, requested: str) -> str:
    voices = cascade_voices(tts)
    if requested and (not voices or requested in voices):
        return requested
    if "aiden" in voices:
        return "aiden"
    return voices[0] if voices else "aiden"


async def _run_turn_cascade(
    turn: AgentTurn,
    history: list[dict[str, Any]],
    pcm: bytes,
    sample_rate: int,
    pipeline: CascadePipeline | None,
    *,
    config: "VoiceConfig",
    text: str | None = None,
) -> None:
    """One cascade turn: (ASR) -> brain with tools -> (TTS), streamed.

    `text` set means a typed turn: no ASR, the text goes straight to the
    brain. `turn.speak=False` skips TTS (no audio frames at all). Lifecycle
    events (`turn_start`/`turn_end`) belong to `run_agent_turn`; this emits
    the turn's content. History gets the user's words, every tool call and
    result, and the reply, committed even when the turn is cancelled.
    """
    if pipeline is None:
        raise RuntimeError("cascade backend is not running")

    # 1) ASR — audio to transcript (skipped for a typed turn).
    if text is None:
        wav_bytes = pcm16_to_wav(pcm, sample_rate)
        transcript = await _cascade_transcribe(pipeline.asr, wav_bytes)
    else:
        transcript = text
    await turn.emit("transcript", text=transcript, final=True, source=turn.source)
    user_text = transcript or "(no speech detected)"

    # 2) Brain — text + current camera views -> reply text.
    state = ""
    views: list[tuple[str, bytes]] = []
    if config.tools_enabled:
        state, views = await live_context()
    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for stream, jpeg in views:
        label = "near" if "near" in stream else "far"
        b64 = base64.b64encode(jpeg).decode("ascii")
        content.append({"type": "text", "text": f"Camera ({label}):"})
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    system = config.system_prompt
    if state:
        system += f"\n\nCurrent robot status: {state}."
    if views:
        system += (
            "\n\nYou can see through two cameras (near and far); the user's "
            "message includes current views. Use them to answer questions about "
            "what is around you."
        )
    if text is not None and not turn.speak:
        system += "\n\nThe user typed this message and reads your reply as text."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": content},
    ]
    # 3) TTS — stream the reply as it is generated, synthesising one finished
    # sentence at a time, so the first audio does not wait for the whole reply.
    # Measured on the Thor: first audio 5.8-6.7 s waiting for the full text vs
    # ~2 s pipelined. TTS runs ~2x faster than playback, so it stays ahead.
    voice = resolve_cascade_voice(pipeline.tts, config.voice) if turn.speak else ""
    await turn.audio_start()

    async def say(sentence: str) -> None:
        if turn.speak and sentence:
            async for chunk in _cascade_speak(pipeline.tts, sentence, voice):
                await turn.audio(chunk)

    # Read-only place-memory tools ride the same switch as live context, so the
    # brain can answer "where are we?" / "what do you know about the office?".
    tools: list[dict[str, Any]] = []
    if config.tools_enabled:
        tools += PLACES_TOOLS + [ROBOT_STATUS_TOOL] + DRIVE_NOTE_TOOLS
    if config.motion_enabled:
        tools += [MOVE_TOOL]
    # What this turn adds to the history (images are never kept).
    turn_msgs: list[dict[str, Any]] = [{"role": "user", "content": user_text}]
    pending = ""
    round_text = ""
    finished = False
    convo = list(messages)
    try:
        for rnd in range(3):  # at most: tool call(s), then the spoken answer
            calls: dict[int, dict[str, str]] = {}
            round_text = ""
            async for ev in _cascade_chat_stream(pipeline.brain, convo, tools or None):
                if ev["type"] == "tool_call":
                    slot = calls.setdefault(
                        ev["index"], {"id": "", "name": "", "arguments": ""}
                    )
                    if ev["id"]:
                        slot["id"] = ev["id"]
                    if ev["name"]:
                        slot["name"] = ev["name"]
                    slot["arguments"] += ev["arguments"]
                    continue
                delta = ev["delta"]
                round_text += delta
                pending += delta
                await turn.delta(delta)
                ready, pending = _sentence_chunks(pending)
                for sentence in ready:
                    await say(sentence)
            if not calls:
                break
            # Anything said before the tool call ("let me check…") is its own
            # utterance; flush it so it does not merge with the final answer.
            await say(pending.strip())
            pending = ""
            ordered = [calls[i] for i in sorted(calls)]
            for i, c in enumerate(ordered):
                c["id"] = c["id"] or f"{turn.turn_id}-{rnd}-{i}"
            call_msg = {
                "role": "assistant",
                "content": round_text or None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in ordered
                ],
            }
            convo.append(call_msg)
            turn_msgs.append(call_msg)
            round_text = ""
            for c in ordered:
                result_msg = await _execute_tool_call(turn, c)
                convo.append(result_msg)
                turn_msgs.append(result_msg)
        await say(pending.strip())
        pending = ""
        await turn.audio_end()
        finished = True
    finally:
        # Commit whatever happened, even on cancel: the model should know it
        # already moved, or was cut off, next turn.
        _close_tool_calls(turn_msgs)
        reply = round_text.strip()
        if reply or not finished:
            if not finished:
                reply = (reply + " (interrupted)").strip()
            turn_msgs.append({"role": "assistant", "content": reply})
        history.extend(turn_msgs)
        _trim_history(history, config.history_turns)


async def _execute_tool_call(turn: AgentTurn, call: dict[str, str]) -> dict[str, Any]:
    """Run one tool call with `tool_call`/`tool_result` events around it."""
    args = _tool_args(call["arguments"])
    await turn.emit("tool_call", call_id=call["id"], name=call["name"], args=args)
    status: dict[str, Any] = {"ok": True, "call_id": call["id"]}
    token = _TOOL_STATUS.set(status)
    t0 = time.monotonic()
    try:
        result = await _run_tool(call["name"], call["arguments"])
    except asyncio.CancelledError:
        await turn.emit("tool_result", call_id=call["id"], ok=False,
                        text="cancelled", ms=int((time.monotonic() - t0) * 1000))
        raise
    except Exception as exc:  # noqa: BLE001 - the model hears the failure
        status["ok"] = False
        result = f"The {call['name']} tool failed ({exc})."
    finally:
        _TOOL_STATUS.reset(token)
    await turn.emit("tool_result", call_id=call["id"], ok=bool(status["ok"]),
                    text=result, ms=int((time.monotonic() - t0) * 1000))
    return {"role": "tool", "tool_call_id": call["id"], "content": result}


# --- status ---------------------------------------------------------------


@dataclass
class VoiceStatus:
    """Operator-facing state, polled by the app via `GET /healthz`."""

    phase: str = "idle"  # idle|starting|building|ready|error|stub
    detail: str = ""
    model: str = ""
    log_tail: list[str] = field(default_factory=list)
    # Liveness / progress. `heartbeat_ms` is refreshed by the supervisor loop
    # every couple of seconds even when the build logs go quiet, so the app can
    # tell "compiling silently" from "hung". `started_ms` anchors the elapsed
    # timer; `component`/`components_done` track the six-engine build.
    started_ms: int = 0
    heartbeat_ms: int = 0
    component: str = ""
    components_done: list[str] = field(default_factory=list)

    def set(self, phase: str | None = None, detail: str | None = None) -> None:
        if phase is not None:
            self.phase = phase
        if detail is not None:
            self.detail = detail

    def beat(self) -> None:
        self.heartbeat_ms = int(time.time() * 1000)

    def snapshot(self) -> dict[str, Any]:
        now = int(time.time() * 1000)
        return {
            "ok": self.phase in READY_PHASES,
            "phase": self.phase,
            "detail": self.detail,
            "model": self.model,
            "heartbeat_ms": self.heartbeat_ms,
            "heartbeat_age_s": round((now - self.heartbeat_ms) / 1000, 1)
            if self.heartbeat_ms
            else None,
            "elapsed_s": round((now - self.started_ms) / 1000, 1)
            if self.started_ms
            else 0,
            "component": self.component,
            "components_done": self.components_done,
            "log_tail": self.log_tail[-20:],
        }


# --- model server supervision --------------------------------------------


class EdgeLLMServer:
    """Supervises the `tensorrt-edgellm-serve` child process.

    The child is started on a background thread so the gateway can serve
    `/healthz` (and report build progress) immediately. Its stdout/stderr are
    captured line-by-line: echoed to the journal and folded into
    `VoiceStatus.detail` so the app can show what's happening.
    """

    def __init__(
        self,
        model: str,
        cache_dir: Path,
        status: VoiceStatus,
        port: int = EDGELLM_PORT,
        extra_args: list[str] | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.status = status
        self.port = port
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._component: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{EDGELLM_HOST}:{self.port}"

    def health_ok(self) -> bool:
        import httpx

        try:
            r = httpx.get(f"{self.base_url}/health", timeout=1.5)
            return r.status_code == 200
        except Exception:  # noqa: BLE001 - any failure means "not up"
            return False

    def start_background(self) -> None:
        """Kick off start + monitor. Returns immediately."""
        threading.Thread(
            target=self._run, name="edgellm-supervisor", daemon=True
        ).start()

    def _run(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.status.started_ms = int(time.time() * 1000)
        self.status.beat()
        if self.health_ok():
            _log(f"Edge-LLM server already up on {self.base_url}")
            self._warmup_then_ready()
            return
        cmd = [
            resolve_edgellm_bin(),
            self.model,
            "--cache-dir",
            str(self.cache_dir),
            "--port",
            str(self.port),
            # Tool-call parsing (the model's XML calls -> OpenAI
            # `tool_calls`). Omni turns are chat-only and send no tools, so
            # these stay inert.
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            "qwen3_xml",
            *self.extra_args,
        ]
        _log("starting: " + " ".join(cmd))
        self.status.set(phase="starting", detail="launching TensorRT Edge-LLM server")
        env = os.environ.copy()
        token = read_hf_token()
        if token:
            env["HF_TOKEN"] = token
        with self._lock:
            self.proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            proc = self.proc
        atexit.register(self.stop)
        threading.Thread(
            target=self._read_output, name="edgellm-logs", daemon=True
        ).start()

        deadline = time.monotonic() + BUILD_TIMEOUT_S
        while time.monotonic() < deadline:
            self.status.beat()
            if proc.poll() is not None:
                self.status.set(
                    phase="error",
                    detail=self._error_detail(proc.returncode),
                )
                _log(f"Edge-LLM server exited rc={proc.returncode}")
                return
            if self.health_ok():
                _log("Edge-LLM server ready")
                self._warmup_then_ready()
                return
            time.sleep(2.0)
        self.status.set(
            phase="error",
            detail="timed out waiting for the Edge-LLM server to become healthy",
        )

    def _warmup_then_ready(self) -> None:
        """Prime the runtime, then mark ready.

        The first Omni generation after engine load is garbled (observed on
        Thor: repeated characters + runaway Talker audio); every turn after is
        clean. A throwaway text turn plus a short silent audio turn settles it,
        so the operator's first utterance is good.
        """
        self.status.set(phase="starting", detail="warming up the model")
        self._warmup()
        self.status.set(phase="ready", detail="")
        _log("Edge-LLM warm-up complete; ready")

    def _warmup(self) -> None:
        import httpx

        url = f"{self.base_url}/v1/chat/completions"
        try:
            httpx.post(
                url,
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4,
                    "temperature": 0.0,
                },
                timeout=180.0,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"warm-up (text) failed: {exc}")
        try:
            httpx.post(
                url,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": _silence_wav_b64(),
                                        "format": "wav",
                                    },
                                }
                            ],
                        }
                    ],
                    "modalities": ["text", "audio"],
                    "audio": {"voice": DEFAULT_VOICE, "format": "pcm16"},
                    "max_tokens": 4,
                    "temperature": 0.0,
                },
                timeout=300.0,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"warm-up (audio) failed: {exc}")

    def _read_output(self) -> None:
        with self._lock:
            proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            line = raw.rstrip()
            if not line:
                continue
            self.status.log_tail.append(line)
            del self.status.log_tail[:-LOG_RING]
            _log(line)
            # First output means the builder has actually begun.
            phase = "building" if self.status.phase in ("starting", "building") else None
            self.status.set(phase=phase, detail=self._progress_detail(line))

    def _progress_detail(self, line: str) -> str | None:
        """Derive a persistent progress line from a build log line.

        The TensorRT compile phase is log-silent for minutes, so once a
        component starts we keep showing it rather than letting stray `[TRT]`
        warnings replace the meaningful status. Returns None to leave the
        current detail untouched.
        """
        if "Building component " in line:
            name = line.split("Building component ", 1)[1].strip()
            self._component = name
            self.status.component = name
            return f"building {name} engine"
        if "Build completed in" in line and self._component:
            done = self._component
            self.status.components_done.append(done)
            self._component = None
            self.status.component = ""
            return f"{done} engine built"
        if self._component:
            return None
        return _friendly_detail(line)

    def _error_detail(self, rc: int | None) -> str:
        # Surface the most useful recent line: a traceback's final exception,
        # else the last non-empty log line.
        for line in reversed(self.status.log_tail):
            stripped = line.strip()
            if stripped.startswith(("RuntimeError", "ValueError", "ImportError",
                                    "ModuleNotFoundError", "FileNotFoundError",
                                    "OSError", "CUDA", "[TRT] Error", "Error")):
                return stripped
        if self.status.log_tail:
            return self.status.log_tail[-1]
        return f"Edge-LLM server exited (rc={rc})"

    def stop(self) -> None:
        with self._lock:
            proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        _log("stopping Edge-LLM server")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


class ExternalServer:
    """A backend that is already running, at an OpenAI-compatible base URL.

    Unlike `EdgeLLMServer`, it does not supervise a child process; it just
    points the gateway at an existing server (a hand-run Edge-LLM child, or one
    on the workstation). An escape hatch for debugging — the normal path is
    `edgellm` (voice) or `SupervisedServer` (the `bebop-models` supervisor).
    """

    def __init__(self, base_url: str, status: "VoiceStatus") -> None:
        self.base_url = base_url.rstrip("/")
        self.status = status

    def health_ok(self) -> bool:
        import httpx

        try:
            return httpx.get(f"{self.base_url}/v1/models", timeout=3.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def start_background(self) -> None:
        def _run() -> None:
            self.status.set(phase="starting", detail=f"connecting to {self.base_url}")
            for _ in range(300):
                if self.health_ok():
                    self.status.set(phase="ready", detail="external server ready")
                    self.status.beat()
                    return
                time.sleep(2.0)
            self.status.set(
                phase="error", detail=f"external server not reachable at {self.base_url}"
            )

        threading.Thread(target=_run, name="external-server", daemon=True).start()

    def stop(self) -> None:
        return


class SupervisedServer:
    """Omni backend whose lifecycle is owned by the `bebop-models` supervisor.

    On start it asks the supervisor to `load` a backend and uses that backend's
    `base_url`; on stop it optionally `unload`s it. The supervisor handles the
    GPU budget and eviction, so the voice service does not manage the child
    process itself.
    """

    def __init__(
        self,
        supervisor_url: str,
        backend_id: str,
        status: "VoiceStatus",
        *,
        unload_on_stop: bool = False,
    ) -> None:
        self.supervisor_url = supervisor_url.rstrip("/")
        self.backend_id = backend_id
        self.status = status
        self.unload_on_stop = unload_on_stop
        self._base_url = ""

    @property
    def base_url(self) -> str:
        return self._base_url or self.supervisor_url

    def health_ok(self) -> bool:
        import httpx

        try:
            return httpx.get(
                f"{self.base_url}/v1/models", timeout=3.0
            ).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def start_background(self) -> None:
        threading.Thread(
            target=self._run, name="supervisor-client", daemon=True
        ).start()

    def _run(self) -> None:
        import httpx

        atexit.register(self.stop)
        self.status.set(
            phase="starting", detail=f"supervisor: loading {self.backend_id}"
        )
        try:
            # `load` blocks until the backend is ready (cold start can be long).
            r = httpx.post(
                f"{self.supervisor_url}/models/{self.backend_id}/load", timeout=None
            )
            data = r.json()
            if isinstance(data, dict) and data.get("base_url"):
                self._base_url = str(data["base_url"])
        except Exception as exc:  # noqa: BLE001
            self.status.set(
                phase="error", detail=f"supervisor load failed: {exc}"
            )
            return
        if self.health_ok():
            self.status.set(phase="ready", detail="supervisor backend ready")
            self.status.beat()
        else:
            self.status.set(phase="error", detail="backend not reachable after load")

    def stop(self) -> None:
        if not self.unload_on_stop:
            return
        import httpx

        try:
            httpx.post(
                f"{self.supervisor_url}/models/{self.backend_id}/unload", timeout=30.0
            )
        except Exception:  # noqa: BLE001
            pass


def _friendly_detail(line: str) -> str:
    """Trim a raw builder log line for display in the app."""
    text = line.strip()
    # Drop the leading `HH:MM:SS LEVEL logger.name:` prefix when present.
    parts = text.split(":", 3)
    if len(parts) >= 3 and parts[0][:2].isdigit():
        text = parts[-1].strip()
    if len(text) > 160:
        text = text[:157] + "..."
    return text or line.strip()


# --- generation -----------------------------------------------------------


async def stream_completion(
    server: EdgeLLMServer,
    messages: list[dict[str, Any]],
    voice: str,
    *,
    audio: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Yield events from the Edge-LLM chat stream (omni, chat only).

    Events: `{"type":"text","delta"}` · `{"type":"audio","data"}`. Parses the
    OpenAI SSE schema; audio rides in `delta.audio.data` (base64 PCM16).
    `audio=False` requests a text-only reply (a typed turn with speak=false).
    """
    import httpx

    payload: dict[str, Any] = {
        "messages": messages,
        "stream": True,
        "max_tokens": 512,
    }
    if audio:
        payload["modalities"] = ["text", "audio"]
        payload["audio"] = {"voice": voice, "format": "pcm16"}
    else:
        payload["modalities"] = ["text"]
    url = f"{server.base_url}/v1/chat/completions"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"Edge-LLM HTTP {resp.status_code}: {body[:300]}")
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or choice.get("message") or {}
                    text = delta.get("content")
                    if text:
                        yield {"type": "text", "delta": text}
                    audio = delta.get("audio")
                    if isinstance(audio, dict) and audio.get("data"):
                        try:
                            yield {
                                "type": "audio",
                                "data": base64.b64decode(audio["data"]),
                            }
                        except Exception:  # noqa: BLE001 - skip a bad chunk
                            pass


async def stream_stub(
    messages: list[dict[str, Any]], voice: str
) -> AsyncIterator[dict[str, Any]]:
    """Offline stand-in for the model: a short text reply + a 440 Hz tone."""
    import math
    import struct

    del messages, voice
    reply = "Voice gateway stub: audio transport is working."
    for word in reply.split():
        yield {"type": "text", "delta": word + " "}
        await asyncio.sleep(0.05)
    frames = 24_000 // 2  # 0.5 s
    for i in range(0, frames, 480):
        samples = [
            int(6000 * math.sin(2 * math.pi * 440 * (i + j) / OUTPUT_RATE))
            for j in range(480)
        ]
        yield {"type": "audio", "data": struct.pack(f"<{len(samples)}h", *samples)}
        await asyncio.sleep(0.01)


def _stream(
    server: EdgeLLMServer | None,
    messages: list[dict[str, Any]],
    voice: str,
    stub: bool,
    *,
    audio: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Select the stub or the real model stream (monkeypatchable in tests)."""
    if stub or server is None:
        return stream_stub(messages, voice)
    return stream_completion(server, messages, voice, audio=audio)


# --- gateway app ----------------------------------------------------------


def build_app(
    server: EdgeLLMServer | None,
    *,
    stub: bool,
    config: VoiceConfig | None = None,
    sessions: SessionStore | None = None,
    status: VoiceStatus | None = None,
    pipeline: CascadePipeline | None = None,
) -> Any:
    """Construct the FastAPI app.

    FastAPI resolves the endpoint's annotations with `typing.get_type_hints`,
    and this module uses `from __future__ import annotations` (string
    annotations). `WebSocket` therefore must be a *module* global, not a local
    import, or FastAPI fails to recognize the websocket parameter and the
    route rejects every upgrade with 403.
    """
    if status is None:
        spec = voice_spec()
        status = VoiceStatus(
            phase="stub" if stub else "idle",
            model=spec.id if spec else "",
        )
    if config is None:
        config = VoiceConfig(voice="stub" if stub else DEFAULT_VOICE)
    if sessions is None:
        sessions = SessionStore()
    app = FastAPI(title="bebop-voice", docs_url=None, redoc_url=None)
    # The app polls `GET /healthz` from a different origin (tauri://… or a dev
    # http://localhost); WebSockets aren't CORS-gated but the fetch is.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _start_state_refresher() -> None:
        # Keep the per-turn robot status warm: the firmware round trip costs
        # ~2 s, which otherwise lands in front of every turn's first audio.
        if not stub:
            asyncio.create_task(_refresh_state_loop())

    @app.on_event("shutdown")
    async def _teardown_backend() -> None:
        # uvicorn runs lifespan shutdown on SIGTERM (systemctl stop / the app's
        # Voice toggle), which is the one place that reliably fires — the code
        # after uvicorn.run() and atexit did not. Tear down whatever this
        # service owns so stopping Voice frees the GPU: kill the Edge-LLM child
        # / cascade stages (or ask the supervisor to unload).
        if server is not None:
            await asyncio.to_thread(server.stop)
        if pipeline is not None:
            await asyncio.to_thread(pipeline.stop)

    registry = TurnRegistry()
    app.state.turns = registry
    app.state.sessions = sessions

    def use_cascade() -> bool:
        """Turns go to the cascade (the engine with tools)."""
        return (config.backend == "cascade" and pipeline is not None) or (
            server is None and pipeline is not None
        )

    def voices_list() -> list[str]:
        if config.backend == "cascade" and pipeline is not None:
            return cascade_voices(pipeline.tts)
        if server is not None:
            return known_voices(server)
        if pipeline is not None:
            return cascade_voices(pipeline.tts)
        return known_voices(None)

    def config_view() -> dict[str, Any]:
        snap = config.snapshot()
        snap["voices"] = voices_list()
        # Both are selectable; the service loads only the chosen one.
        snap["backends"] = ["omni", "cascade"]
        snap["tools_supported"] = use_cascade()
        return snap

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        snap = status.snapshot()
        snap["backend"] = config.backend
        # Only the cascade brain has tools (place memory, robot status,
        # motion); omni is chat-only.
        snap["tools_supported"] = use_cascade()
        snap["tools_enabled"] = config.tools_enabled
        snap["motion_enabled"] = config.motion_enabled
        # In-flight turns.
        snap["turns_active"] = registry.count
        snap["turn_active"] = registry.count > 0
        return JSONResponse(snap)

    @app.get("/config")
    async def get_config() -> JSONResponse:
        return JSONResponse(config_view())

    @app.post("/config")
    async def set_config(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 - treat a bad body as empty
            body = {}
        if not isinstance(body, dict):
            body = {}
        if isinstance(body.get("voice"), str):
            voice = body["voice"]
            available = voices_list()
            if available and voice not in available:
                return JSONResponse(
                    {"error": f"unknown voice {voice!r} (have {available})"},
                    status_code=400,
                )
            config.voice = voice
        if isinstance(body.get("system_prompt"), str) and body["system_prompt"].strip():
            config.system_prompt = body["system_prompt"].strip()
        if "history_turns" in body:
            config.history_turns = _clamp_turns(body["history_turns"])
        if "keep_audio_history" in body:
            config.keep_audio_history = bool(body["keep_audio_history"])
        if "tools_enabled" in body:
            config.tools_enabled = bool(body["tools_enabled"])
        if isinstance(body.get("motion_enabled"), bool):
            config.motion_enabled = body["motion_enabled"]
        if body.get("backend") in ("omni", "cascade"):
            config.backend = body["backend"]
        if isinstance(body.get("omni_url"), str):
            config.omni_url = body["omni_url"].strip()
        if isinstance(body.get("omni_supervisor_url"), str):
            config.omni_supervisor_url = body["omni_supervisor_url"].strip()
        if isinstance(body.get("omni_backend_id"), str) and body["omni_backend_id"].strip():
            config.omni_backend_id = body["omni_backend_id"].strip()
        if isinstance(body.get("omni_unload_on_stop"), bool):
            config.omni_unload_on_stop = body["omni_unload_on_stop"]
        try:
            config.save()
        except OSError as exc:
            return JSONResponse({"error": f"save failed: {exc}"}, status_code=500)
        return JSONResponse(config_view())

    @app.websocket("/voice")
    async def voice_ws(ws: WebSocket) -> None:
        await ws.accept()
        if status.phase not in READY_PHASES:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"voice service not ready ({status.phase}): {status.detail}",
                    }
                )
            )
            await ws.close()
            return
        params = ws.query_params
        # `?session=<id>` keeps context across reconnects; anonymous
        # connections get a fresh bucket each time.
        session_key = params.get("session") or f"anon-{id(ws)}"
        history = sessions.get(session_key)
        out = TurnEmitter(ws)
        # The utterance being recorded: None, or its control frame + chunks.
        utterance: dict[str, Any] | None = None

        async def start_turn(ctl: dict[str, Any], *, source: str,
                             text: str | None, pcm: bytes,
                             sample_rate: int) -> None:
            busy = registry.busy(session_key)
            if busy is not None:
                # No `error` alongside: the running turn is unaffected and
                # still ends with its own `turn_end`.
                await out.json({"type": "turn_rejected", "reason": "busy",
                                "active_turn_id": busy.turn_id, "source": source})
                return
            client_id = ctl.get("turn_id")
            turn = AgentTurn(
                turn_id=(client_id if isinstance(client_id, str) and 0 < len(client_id) <= 64
                         else _new_turn_id()),
                source=source,
                out=out,
                speak=_parse_speak(ctl.get("speak", True)),
                session_key=session_key,
                owner=ws,
            )
            previous = registry.by_session.get(session_key)
            registry.by_session[session_key] = turn
            if use_cascade():
                async def body() -> None:
                    await _run_turn_cascade(
                        turn, history, pcm, sample_rate, pipeline,
                        config=config, text=text)
            elif server is not None or stub:
                async def body() -> None:
                    await _run_turn(
                        turn, history, pcm, sample_rate, server, stub=stub,
                        config=config, text=text)
            else:
                async def body() -> None:
                    raise RuntimeError("no voice backend available")
            turn.task = asyncio.create_task(
                run_agent_turn(turn, registry, body,
                               previous=previous if previous is not None
                               and previous.running else None),
                name=f"voice-turn-{turn.turn_id}",
            )

        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    if utterance is not None:
                        utterance["chunks"].append(msg["bytes"])
                    continue
                if msg.get("text") is None:
                    continue
                try:
                    ctl = json.loads(msg["text"])
                except ValueError:
                    await out.json({"type": "error", "message": "bad JSON frame"})
                    continue
                if not isinstance(ctl, dict):
                    continue
                kind = ctl.get("type")
                if kind == "cancel":
                    turn = registry.by_session.get(session_key)
                    if turn is not None and turn.running:
                        turn.request_cancel()
                elif kind == "reset":
                    turn = registry.by_session.get(session_key)
                    if turn is not None and turn.running:
                        # A turn commits its history when it ends; let it
                        # end first so the reset sticks.
                        turn.request_cancel()
                        await asyncio.wait({turn.task}, timeout=10.0)
                    history.clear()
                    await out.json({"type": "reset"})
                elif kind == "utterance_start":
                    utterance = {"ctl": ctl, "chunks": []}
                elif kind == "utterance_end":
                    if utterance is None:
                        continue
                    pcm = b"".join(utterance["chunks"])
                    start_ctl = utterance["ctl"]
                    utterance = None
                    if not pcm:
                        await out.json({"type": "done"})
                        continue
                    await start_turn(
                        start_ctl, source="voice", text=None, pcm=pcm,
                        sample_rate=int(start_ctl.get("sample_rate", INPUT_RATE)))
                elif kind == "user_text":
                    text = str(ctl.get("text") or "").strip()
                    if not text:
                        await out.json({"type": "error", "message": "empty user_text"})
                        continue
                    await start_turn(ctl, source="text", text=text, pcm=b"",
                                     sample_rate=INPUT_RATE)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - report and close cleanly
            await out.json({"type": "error", "message": str(exc)})
        finally:
            out.closed = True
            # A disconnect cancels the turns this socket started (the motion
            # tool's own `finally` still stops and disarms).
            for turn in list(registry.running):
                if turn.owner is ws:
                    turn.request_cancel()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    return app


async def _run_turn(
    turn: AgentTurn,
    history: list[dict[str, Any]],
    pcm: bytes,
    sample_rate: int,
    server: EdgeLLMServer | None,
    *,
    stub: bool,
    config: VoiceConfig,
    text: str | None = None,
) -> None:
    """One omni turn: a single model call, chat only (no tools). `text` set
    means a typed turn; `turn.speak=False` asks for a text-only reply."""
    voice = config.voice

    # Live context: current camera frames + robot status, attached to the same
    # request as the user's audio so every turn is a single model call.
    state = ""
    views: list[tuple[str, bytes]] = []
    if config.tools_enabled and not stub:
        state, views = await live_context()

    if text is not None:
        await turn.emit("transcript", text=text, final=True, source=turn.source)
    wav_b64 = ("" if text is not None else
               base64.b64encode(pcm16_to_wav(pcm, sample_rate)).decode("ascii"))
    content: list[dict[str, Any]] = []
    if views:
        content.append({"type": "text", "text": "Your current camera views:"})
        for stream, jpeg in views:
            label = "near" if "near" in stream else "far"
            b64 = base64.b64encode(jpeg).decode("ascii")
            content.append({"type": "text", "text": f"Camera ({label}):"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )
    if text is not None:
        content.append({"type": "text", "text": text})
    else:
        content.append(
            {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}
        )
    user_msg = {"role": "user", "content": content}

    # The system prompt comes from the live config (persona edits apply on the
    # next turn); `history` holds only the turns after it.
    system = config.system_prompt
    if state:
        system += f"\n\nCurrent robot status: {state}."
    if views:
        system += (
            "\n\nYou can see through two cameras (near and far); the user's "
            "message includes your current views. Use them to answer questions "
            "about what is around you."
        )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *history,
        user_msg,
    ]

    await turn.audio_start()
    final_text = ""
    finished = False
    try:
        async for ev in _stream(server, messages, voice, stub, audio=turn.speak):
            if ev["type"] == "text":
                final_text += ev["delta"]
                await turn.delta(ev["delta"])
            elif ev["type"] == "audio":
                await turn.audio(ev["data"])
        await turn.audio_end()
        finished = True
    finally:
        # Bound the rolling history. The user's words (typed turns) and the
        # assistant's text are always kept; the user's (large) audio only when
        # the operator opts in, so the model can re-hear prior turns at the
        # cost of re-prefilling them. Camera frames are never kept.
        if text is not None:
            history.append({"role": "user", "content": text})
        elif config.keep_audio_history:
            history.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": wav_b64, "format": "wav"},
                        }
                    ],
                }
            )
        reply = final_text.strip()
        if not finished:
            reply = (reply + " (interrupted)").strip()
        if reply:
            history.append({"role": "assistant", "content": reply})
        _trim_history(history, config.history_turns)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS)
    parser.add_argument(
        "--keep-audio-history",
        action="store_true",
        help="keep the user's audio in the conversation context (more faithful, "
        "but re-prefills it every turn)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="disable live context (camera views + robot status in each turn)",
    )
    parser.add_argument("--cache-dir", type=Path, default=EDGELLM_CACHE_DIR)
    parser.add_argument(
        "--stub",
        action="store_true",
        help="run the offline echo/tone generator instead of the model",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print resolved config + readiness and exit",
    )
    parser.add_argument(
        "--edgellm-arg",
        action="append",
        default=[],
        help="extra argument passed through to tensorrt-edgellm-serve",
    )
    parser.add_argument(
        "--backend",
        choices=["omni", "cascade"],
        default="",
        help="voice backend: 'cascade' (ASR -> VLM brain -> TTS, with tools; "
        "the default) or 'omni' (end-to-end Qwen3-Omni, chat only, no tools). "
        "Overrides the saved config.",
    )
    parser.add_argument(
        "--omni-url",
        default="",
        help="use an already-running OpenAI-compatible omni server instead of "
        "supervising an Edge-LLM child (debug escape hatch)",
    )
    parser.add_argument(
        "--omni-supervisor-url",
        default="",
        help="URL of the bebop-models supervisor (e.g. http://127.0.0.1:9094); "
        "its lifecycle owns the omni backend",
    )
    parser.add_argument(
        "--omni-backend-id",
        default="",
        help="backend id to load from the supervisor (default omni-edgellm)",
    )
    args = parser.parse_args(argv)

    _saved = VoiceConfig.load() if CONFIG_PATH.exists() else None
    if _saved is not None:
        config = _saved
    else:
        config = VoiceConfig(
            voice=args.voice,
            system_prompt=args.system_prompt,
            history_turns=_clamp_turns(args.history_turns),
            keep_audio_history=args.keep_audio_history,
            tools_enabled=not args.no_tools,
        )
    if args.backend:
        config.backend = args.backend
    if args.omni_url:
        config.omni_url = args.omni_url
    if args.omni_supervisor_url:
        config.omni_supervisor_url = args.omni_supervisor_url
    if args.omni_backend_id:
        config.omni_backend_id = args.omni_backend_id
    sessions = SessionStore()

    spec = voice_spec()
    model_dir = resolve_model_dir(spec) if spec is not None else None
    quantized = bool(spec is not None and model_dir != spec.target_dir)
    if spec is None:
        downloaded = False
    elif quantized:
        downloaded = (model_dir / "config.json").is_file()
    else:
        downloaded = (spec.target_dir / ".bebop-complete").is_file()
    precision = "nvfp4" if quantized else "fp16"

    if args.check:
        print(
            json.dumps(
                {
                    "model": spec.id if spec else "",
                    "model_dir": str(model_dir) if model_dir else "",
                    "precision": precision,
                    "downloaded": downloaded,
                    "stub": args.stub,
                    **config.snapshot(),
                },
                indent=2,
            )
        )
        return 0

    # Start ONLY the selected backend — the other model is not loaded, so no
    # GPU memory or compute is spent on it. The app's Engine selector saves
    # `config.backend`; restart the voice service (Voice toggle) to apply.
    backend = config.backend
    server: EdgeLLMServer | None = None
    pipeline: CascadePipeline | None = None

    if args.stub:
        status = VoiceStatus(model="stub")
        status.set(phase="stub", detail="stub mode (no models)")
    elif backend == "cascade":
        status = VoiceStatus(model="cascade")
        asr = StageServer(
            "asr", CASCADE_ASR_MODEL, args.cache_dir / "asr", CASCADE_ASR_PORT,
            capability="transcription",
        )
        tts = StageServer(
            "tts", CASCADE_TTS_MODEL, args.cache_dir / "tts", CASCADE_TTS_PORT,
            capability="speech", patched_launcher=True,
        )
        brain = StageServer(
            "brain", CASCADE_BRAIN_MODEL, args.cache_dir / "brain", CASCADE_BRAIN_PORT,
            capability="chat",
            extra_args=[
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
                "--max-input-len", "8192", "--max-kv-cache-capacity", "16384",
                "--speculative-config", CASCADE_BRAIN_SPECULATIVE,
            ],
        )
        pipeline = CascadePipeline(asr, brain, tts, status)
        pipeline.start_background()
    elif config.omni_supervisor_url:
        # Preferred long-term mode: the bebop-models supervisor owns the backend
        # lifecycle + GPU budget. We ask it to load the backend and use its URL.
        status = VoiceStatus(
            model=f"qwen3-omni (supervised:{config.omni_backend_id})")
        server = SupervisedServer(
            config.omni_supervisor_url,
            config.omni_backend_id,
            status,
            unload_on_stop=config.omni_unload_on_stop,
        )
        server.start_background()
    elif config.omni_url:
        # External omni backend — no supervision (debug escape hatch).
        status = VoiceStatus(model="qwen3-omni (external)")
        server = ExternalServer(config.omni_url, status)
        server.start_background()
    else:
        status = VoiceStatus(model=spec.id if spec else "qwen3-omni-30b")
        if spec is not None and downloaded:
            server = EdgeLLMServer(
                str(model_dir), args.cache_dir, status, extra_args=args.edgellm_arg
            )
            server.start_background()
        else:
            status.set(
                phase="error",
                detail="omni checkpoint not provisioned (see docs/voice.md)",
            )

    _log(
        f"voice backend: {backend} voice={config.voice} turns={config.history_turns}"
    )

    app = build_app(
        server,
        stub=args.stub,
        config=config,
        sessions=sessions,
        status=status,
        pipeline=pipeline,
    )
    import uvicorn

    _log(f"listening on ws://{args.host}:{args.port}/voice")
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        # systemd stops this unit with SIGTERM. uvicorn traps SIGTERM and shuts
        # down gracefully, but may exit via SystemExit, so tear our children
        # down here rather than relying on the lines after run() (and on
        # atexit, which was not firing). This is what stops the Edge-LLM
        # child (or the cascade stages) when Voice is toggled off.
        if server is not None:
            server.stop()
        if pipeline is not None:
            pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
