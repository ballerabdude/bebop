"""On-device speech-to-speech gateway (Qwen3-Omni via TensorRT Edge-LLM).

`bebop-voice.service` runs this as root from `bebop-vision/.venv-voice`. It
owns two things:

1. **The model server.** It supervises `tensorrt-edgellm-serve`, which serves
   the catalog's `voice` checkpoint (Qwen3-Omni-30B-A3B) as an
   OpenAI-compatible HTTP API on 127.0.0.1:8000. The first start builds the
   checkpoint's TensorRT engines into `--cache-dir`; that can take many
   minutes.
2. **The app gateway.** It exposes a turn-based WebSocket on :9093 that the
   Tauri app's Voice page talks to. The app streams 16 kHz mono PCM16 for one
   utterance, then receives streamed text and 24 kHz mono PCM16 speech back.

The runtime WebSocket on :9090 carries only control (`SetVoiceEnabled`) and
status (`VoiceState`); audio never flows through the firmware. See
`docs/voice.md`.

Wire protocol (:9093), one utterance per connection turn:

    app -> {"type":"utterance_start","sample_rate":16000,"format":"pcm16"}
    app -> <binary PCM16 frames>
    app -> {"type":"utterance_end"}
    svc -> {"type":"text","delta":"..."}          (zero or more)
    svc -> {"type":"audio_start","sample_rate":24000,"format":"pcm16"}
    svc -> <binary PCM16 frames>
    svc -> {"type":"audio_end"}
    svc -> {"type":"done"}
    svc -> {"type":"error","message":"..."}       (on failure)

`GET /healthz` is the status surface the app polls:

    {"ok": bool, "phase": "starting|building|ready|error|stub",
     "detail": "<last meaningful log line>", "model": "qwen3-omni-30b",
     "model_downloaded": bool, "updated_ms": 123, "log_tail": [...]}

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
import io
import json
import os
import shutil
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

# --- defaults -------------------------------------------------------------

EDGELLM_BIN = "tensorrt-edgellm-serve"
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


def resolve_edgellm_bin() -> str:
    """Locate the `tensorrt-edgellm-serve` console script.

    Under systemd the venv's `bin/` is not on `PATH`, so prefer the sibling of
    the running interpreter (the venv that owns this process), then fall back
    to `PATH`.
    """
    sibling = Path(sys.executable).parent / EDGELLM_BIN
    if sibling.exists():
        return str(sibling)
    return shutil.which(EDGELLM_BIN) or EDGELLM_BIN


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
MAX_SESSIONS = 32


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
    backend: str = "omni"  # "omni" (end-to-end) | "cascade" (ASR->brain->TTS)
    # When set, the omni backend uses this already-running OpenAI-compatible
    # server (e.g. a vLLM-Omni container) instead of supervising Edge-LLM.
    omni_url: str = ""
    # When true, the omni backend *manages* a Docker container (start/stop with
    # the voice service) instead of pointing at an external URL. Used for the
    # vLLM-Omni image; also the pattern for future VLA containers.
    omni_container: bool = False
    omni_image: str = "bebop-vllm-omni-thor"
    omni_model_dir: str = "/home/bebop/qwen3-omni-talker-safe"
    omni_deploy_config: str = "/home/bebop/qwen3_omni_1gpu.yaml"
    omni_port: int = 8101

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
        if raw.get("backend") in ("omni", "cascade"):
            cfg.backend = raw["backend"]
        if isinstance(raw.get("omni_url"), str):
            cfg.omni_url = raw["omni_url"].strip()
        if isinstance(raw.get("omni_container"), bool):
            cfg.omni_container = raw["omni_container"]
        for key in ("omni_image", "omni_model_dir", "omni_deploy_config"):
            if isinstance(raw.get(key), str) and raw[key].strip():
                setattr(cfg, key, raw[key].strip())
        if isinstance(raw.get("omni_port"), int):
            cfg.omni_port = raw["omni_port"]
        return cfg

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2))
        os.replace(tmp, path)

    def snapshot(self) -> dict[str, Any]:
        return {
            "voice": self.voice,
            "system_prompt": self.system_prompt,
            "history_turns": self.history_turns,
            "keep_audio_history": self.keep_audio_history,
            "tools_enabled": self.tools_enabled,
            "backend": self.backend,
            "omni_url": self.omni_url,
            "omni_container": self.omni_container,
            "omni_image": self.omni_image,
            "omni_model_dir": self.omni_model_dir,
            "omni_deploy_config": self.omni_deploy_config,
            "omni_port": self.omni_port,
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

    def clear(self, key: str) -> None:
        with self._lock:
            self._map.pop(key, None)


# Cache of the model's speaker ids (GET /v1/voices), so `/config` and the WS
# can validate a requested voice without a round-trip every time.
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
# Every voice turn is a single model call: the user's audio plus the robot's
# current near/far camera frames, with a one-line robot status in the system
# prompt. No tool calls — the model server rejects tools combined with audio
# output, and attaching the frames directly is simpler and avoids the
# tool-call round-trip (and its bugs) entirely.

ROBOT_WS_URL = "ws://127.0.0.1:9090/ws"
VISION_SNAPSHOT_BASE = "http://127.0.0.1:9092/snapshot?stream="
# The rig has two cameras (near + far); both are attached to every turn.
VISION_STREAMS = ("color_near", "color_far")

_MODE_NAMES = {0: "UNSPECIFIED", 1: "IDLE", 2: "DIAL_IN", 3: "RUN_POLICY"}


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


async def robot_state_text() -> str:
    """One-line robot status for the system prompt; '' when unavailable."""
    import websockets
    from bebop_vision.proto.bebop.runtime.v1 import bebop_runtime_pb2 as rt

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
                    return _format_snapshot(env.snapshot)
    except Exception as exc:  # noqa: BLE001 - status is best-effort
        _log(f"robot state unavailable: {exc}")
    return ""


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


async def live_context(
    status: Any | None = None,
) -> tuple[str, list[tuple[str, bytes]]]:
    """Gather the per-turn context: robot status + current camera frames.

    If the cameras are down, ask the firmware to start vision and wait a
    bounded time for the first frames. Returns `("", [])` style best-effort
    values rather than raising.
    """
    state = await robot_state_text()
    views = await _snapshot_views()
    if not views and await _enable_vision():
        if status is not None:
            status.set(detail="waiting for cameras to start")
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
        status: "VoiceStatus | None" = None,
    ) -> None:
        self.name = name
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.port = port
        self.capability = capability
        self.extra_args = extra_args or []
        self.patched_launcher = patched_launcher
        self.status = status
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
    """Owns the stage servers and reports an aggregate status.

    `brain` is optional: when a cloud brain (OpenRouter) is selected we skip
    the local VLM entirely, so no GPU memory is spent on it.
    """

    def __init__(
        self,
        asr: StageServer,
        brain: StageServer | None,
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
        stages = [s for s in (self.asr, self.tts, self.brain) if s is not None]
        for stage in stages:
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
        return (
            self.asr.ready
            and self.tts.ready
            and (self.brain is None or self.brain.ready)
        )

    def stop(self) -> None:
        for stage in (self.asr, self.brain, self.tts):
            if stage is not None:
                stage.stop()

    def model_ids(self) -> list[str]:
        return [s.model for s in (self.asr, self.brain, self.tts) if s is not None]


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


async def _cascade_chat(stage: StageServer, messages: list[dict[str, Any]]) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(
            f"{stage.base_url}/v1/chat/completions",
            json={"messages": messages, "max_tokens": 512, "temperature": 0.7},
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        return str(msg.get("content") or "").strip()


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
    ws: Any,
    history: list[dict[str, Any]],
    pcm: bytes,
    sample_rate: int,
    pipeline: CascadePipeline,
    *,
    stub: bool,
    config: "VoiceConfig",
    voice_override: str | None = None,
) -> None:
    if stub:
        await ws.send_text(
            json.dumps({"type": "audio_start", "sample_rate": OUTPUT_RATE, "format": "pcm16"})
        )
        async for ev in stream_stub([], config.voice):
            if ev["type"] == "text":
                await ws.send_text(json.dumps({"type": "text", "delta": ev["delta"]}))
            elif ev["type"] == "audio":
                await ws.send_bytes(ev["data"])
        await ws.send_text(json.dumps({"type": "audio_end"}))
        await ws.send_text(json.dumps({"type": "done"}))
        return

    wav_bytes = pcm16_to_wav(pcm, sample_rate)

    # 1) ASR — audio to transcript.
    transcript = await _cascade_transcribe(pipeline.asr, wav_bytes)
    await ws.send_text(
        json.dumps({"type": "transcript", "text": transcript, "final": True})
    )

    # 2) Brain — text + current camera views -> reply text.
    state = ""
    views: list[tuple[str, bytes]] = []
    if config.tools_enabled:
        state, views = await live_context()
    content: list[dict[str, Any]] = [
        {"type": "text", "text": transcript or "(no speech detected)"}
    ]
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
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": content},
    ]
    reply = await _cascade_chat(pipeline.brain, messages)

    # 3) TTS — reply text to speech.
    voice = resolve_cascade_voice(pipeline.tts, voice_override or config.voice)
    await ws.send_text(
        json.dumps({"type": "text", "delta": reply})
    )
    await ws.send_text(
        json.dumps({"type": "audio_start", "sample_rate": OUTPUT_RATE, "format": "pcm16"})
    )
    async for chunk in _cascade_speak(pipeline.tts, reply, voice):
        await ws.send_bytes(chunk)
    await ws.send_text(json.dumps({"type": "audio_end"}))
    await ws.send_text(json.dumps({"type": "done"}))

    if reply:
        history.append({"role": "assistant", "content": reply})
    if config.keep_audio_history:
        history.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                            "format": "wav",
                        },
                    }
                ],
            }
        )
    per_turn = 2 if config.keep_audio_history else 1
    keep = max(0, config.history_turns) * per_turn
    if keep == 0:
        history.clear()
    elif len(history) > keep:
        del history[:-keep]


# --- status ---------------------------------------------------------------


@dataclass
class VoiceStatus:
    """Operator-facing state, polled by the app via `GET /healthz`."""

    phase: str = "idle"  # idle|starting|building|ready|error|stub
    detail: str = ""
    model: str = ""
    precision: str = ""
    model_downloaded: bool = False
    updated_ms: int = 0
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
        self.updated_ms = int(time.time() * 1000)

    def beat(self) -> None:
        self.heartbeat_ms = int(time.time() * 1000)

    def snapshot(self) -> dict[str, Any]:
        now = int(time.time() * 1000)
        return {
            "ok": self.phase in READY_PHASES,
            "phase": self.phase,
            "detail": self.detail,
            "model": self.model,
            "precision": self.precision,
            "model_downloaded": self.model_downloaded,
            "updated_ms": self.updated_ms,
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
            # Agentic tool calling: the server parses the model's XML calls
            # into OpenAI `tool_calls` (used by the read-only tools).
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
    """A backend that is already running (e.g. a vLLM-Omni container).

    Unlike `EdgeLLMServer`, it does not supervise a child process; it just
    points the gateway at an OpenAI-compatible base URL.
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


class ContainerServer:
    """Supervises a model server running in a Docker container.

    Unlike `EdgeLLMServer` (a child process), the server runs detached via
    `docker run -d`; `stop()` removes it. Used for the vLLM-Omni image and the
    pattern for future VLA containers. Requires the gateway to run as root (it
    does, under `bebop-voice.service`).
    """

    NAME = "bebop-model-serve"
    CACHE_DIR = "/home/bebop/.cache"

    def __init__(
        self,
        image: str,
        port: int,
        status: "VoiceStatus",
        *,
        model_dir: str,
        deploy_config: str = "",
        extra_args: list[str] | None = None,
    ) -> None:
        self.image = image
        self.port = port
        self.status = status
        self.model_dir = model_dir
        self.deploy_config = deploy_config
        self.extra_args = extra_args or []
        self.base_url = f"http://127.0.0.1:{port}"

    def _docker(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=False
        )

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
            target=self._run, name="container-supervisor", daemon=True
        ).start()

    def _run(self) -> None:
        atexit.register(self.stop)
        self.status.set(phase="starting", detail=f"starting {self.image}")
        self._docker("rm", "-f", self.NAME)  # replace a stale container
        args = [
            "run", "-d", "--name", self.NAME,
            "--runtime", "nvidia", "--network", "host",
            "-v", f"{self.model_dir}:/models/model:ro",
            "-v", f"{self.CACHE_DIR}/vllm:/root/.cache/vllm",
            "-v", f"{self.CACHE_DIR}/huggingface:/root/.cache/huggingface",
        ]
        if self.deploy_config:
            args += ["-v", f"{self.deploy_config}:/cfg/deploy.yaml:ro"]
        args += [self.image, "/models/model", "--omni"]
        if self.deploy_config:
            args += ["--deploy-config", "/cfg/deploy.yaml"]
        args += [
            "--host", "0.0.0.0", "--port", str(self.port),
            "--init-timeout", "3600", "--stage-init-timeout", "3600",
            *self.extra_args,
        ]
        res = self._docker(*args)
        if res.returncode != 0:
            self.status.set(
                phase="error", detail=f"docker run failed: {res.stderr.strip()[:180]}"
            )
            return
        # vLLM-Omni cold start is long (model load + compile); poll generously.
        for _ in range(720):
            if self.health_ok():
                self.status.set(phase="ready", detail="container ready")
                self.status.beat()
                return
            running = self._docker("ps", "-q", "-f", f"name={self.NAME}").stdout.strip()
            if not running:
                self.status.set(phase="error", detail="container exited during startup")
                return
            time.sleep(5.0)
        self.status.set(phase="error", detail="container did not become ready in time")

    def stop(self) -> None:
        self._docker("rm", "-f", self.NAME)


def _friendly_detail(line: str) -> str:
    """Trim a raw builder log line for display in the app."""
    text = line.strip()
    # Drop the leading `HH:MM:SS LEVEL logger.name:` prefix when present.
    parts = text.split(":", 3)
    if len(parts) == 4 and parts[0].count(":") == 2:
        text = parts[3].strip()
    elif len(parts) >= 3 and parts[0][:2].isdigit():
        text = parts[-1].strip()
    if len(text) > 160:
        text = text[:157] + "..."
    return text or line.strip()


# --- generation -----------------------------------------------------------


def wav_bytes_to_pcm16(wav_bytes: bytes) -> bytes:
    """Extract raw PCM16 frames from a WAV container.

    vLLM-Omni streams audio as a base64 WAV inside `delta.content` (tagged
    `modality: "audio"`), unlike Edge-LLM which sends raw PCM in
    `delta.audio.data`. We normalize both to raw PCM16 for the app.
    """
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes())


async def stream_completion(
    server: EdgeLLMServer,
    messages: list[dict[str, Any]],
    voice: str,
    tools: list[dict[str, Any]] | None = None,
    *,
    audio: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Yield events from the Edge-LLM chat stream.

    Events: `{"type":"text","delta"}` · `{"type":"audio","data"}` ·
    `{"type":"tool_call","index","id","name","arguments"}`. Parses the OpenAI
    SSE schema; audio rides in `delta.audio.data` (base64 PCM16), tool calls in
    `delta.tool_calls` (the server parses the model's XML via qwen3_xml).

    `audio=False` requests a text-only turn — required for tool calls, because
    the server rejects `tools` combined with audio output
    (`unsupported_feature`). `_run_turn` therefore plans/executes tools
    text-only, then does one audio pass (no tools) to speak the answer.
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
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
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
                modality = chunk.get("modality")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or choice.get("message") or {}
                    text = delta.get("content")
                    if text:
                        if modality == "audio":
                            # vLLM-Omni: base64 WAV in content.
                            try:
                                yield {
                                    "type": "audio",
                                    "data": wav_bytes_to_pcm16(base64.b64decode(text)),
                                }
                            except Exception:  # noqa: BLE001 - skip a bad chunk
                                pass
                        else:
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
                    for tc in delta.get("tool_calls") or []:
                        fn = tc.get("function") or {}
                        yield {
                            "type": "tool_call",
                            "index": int(tc.get("index", 0) or 0),
                            "id": tc.get("id") or "",
                            "name": fn.get("name") or "",
                            "arguments": fn.get("arguments") or "",
                        }


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
    tools: list[dict[str, Any]] | None,
    stub: bool,
    *,
    audio: bool = True,
) -> AsyncIterator[dict[str, Any]]:
    """Select the stub or the real model stream (monkeypatchable in tests)."""
    if stub or server is None:
        return stream_stub(messages, voice)
    return stream_completion(server, messages, voice, tools, audio=audio)


# --- gateway app ----------------------------------------------------------


def build_app(
    server: EdgeLLMServer | None,
    *,
    stub: bool,
    config: VoiceConfig | None = None,
    sessions: SessionStore | None = None,
    status: VoiceStatus | None = None,
    pipeline: CascadePipeline | None = None,
    pipeline_status: VoiceStatus | None = None,
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

    def current_status() -> VoiceStatus:
        """Status of the backend selected right now."""
        active_cascade = (config.backend == "cascade" and pipeline is not None) or (
            server is None and pipeline is not None
        )
        if active_cascade:
            return pipeline_status or status
        return status

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
        return snap

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        snap = current_status().snapshot()
        snap["backend"] = config.backend
        snap["available"] = {"omni": server is not None, "cascade": pipeline is not None}
        return JSONResponse(snap)

    @app.get("/voices")
    async def voices() -> JSONResponse:
        return JSONResponse({"voices": voices_list(), "default": config.voice})

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
        if body.get("backend") in ("omni", "cascade"):
            config.backend = body["backend"]
        if isinstance(body.get("omni_url"), str):
            config.omni_url = body["omni_url"].strip()
        if isinstance(body.get("omni_container"), bool):
            config.omni_container = body["omni_container"]
        for key in ("omni_image", "omni_model_dir", "omni_deploy_config"):
            if isinstance(body.get(key), str) and body[key].strip():
                setattr(config, key, body[key].strip())
        if isinstance(body.get("omni_port"), int):
            config.omni_port = body["omni_port"]
        try:
            config.save()
        except OSError as exc:
            return JSONResponse({"error": f"save failed: {exc}"}, status_code=500)
        return JSONResponse(config_view())

    @app.websocket("/voice")
    async def voice_ws(ws: WebSocket) -> None:
        await ws.accept()
        if current_status().phase not in READY_PHASES:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"voice service not ready ({current_status().phase}): {current_status().detail}",
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
        requested_voice = params.get("voice")
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "text" in msg and msg["text"] is not None:
                    ctl = json.loads(msg["text"])
                    kind = ctl.get("type")
                    if kind == "reset":
                        history.clear()
                        await ws.send_text(json.dumps({"type": "reset"}))
                        continue
                    if kind != "utterance_start":
                        continue
                    sample_rate = int(ctl.get("sample_rate", INPUT_RATE))
                    pcm = await _collect_pcm(ws)
                    if not pcm:
                        await ws.send_text(json.dumps({"type": "done"}))
                        continue
                    use_cascade = (
                        config.backend == "cascade" and pipeline is not None
                    ) or (server is None and pipeline is not None)
                    if use_cascade:
                        await _run_turn_cascade(
                            ws,
                            history,
                            pcm,
                            sample_rate,
                            pipeline,
                            stub=stub,
                            config=config,
                            voice_override=requested_voice,
                        )
                    elif server is not None or stub:
                        await _run_turn(
                            ws,
                            history,
                            pcm,
                            sample_rate,
                            server,
                            stub=stub,
                            config=config,
                            voice_override=requested_voice,
                        )
                    else:
                        raise RuntimeError("no voice backend available")
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001 - report and close cleanly
            try:
                await ws.send_text(
                    json.dumps({"type": "error", "message": str(exc)})
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    return app


async def _collect_pcm(ws: Any) -> bytes:
    """Accumulate binary PCM frames until `utterance_end`."""
    chunks: list[bytes] = []
    while True:
        msg = await ws.receive()
        if msg.get("type") == "websocket.disconnect":
            return b"".join(chunks)
        if "bytes" in msg and msg["bytes"] is not None:
            chunks.append(msg["bytes"])
        elif "text" in msg and msg["text"] is not None:
            ctl = json.loads(msg["text"])
            if ctl.get("type") == "utterance_end":
                return b"".join(chunks)


async def _run_turn(
    ws: Any,
    history: list[dict[str, Any]],
    pcm: bytes,
    sample_rate: int,
    server: EdgeLLMServer | None,
    *,
    stub: bool,
    config: VoiceConfig,
    voice_override: str | None = None,
) -> None:
    voice = config.voice
    if voice_override:
        available = known_voices(server)
        if not available or voice_override in available:
            voice = voice_override

    # Live context: current camera frames + robot status, attached to the same
    # request as the user's audio so every turn is a single model call.
    state = ""
    views: list[tuple[str, bytes]] = []
    if config.tools_enabled and not stub:
        state, views = await live_context()

    wav_b64 = base64.b64encode(pcm16_to_wav(pcm, sample_rate)).decode("ascii")
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

    await ws.send_text(
        json.dumps({"type": "audio_start", "sample_rate": OUTPUT_RATE, "format": "pcm16"})
    )
    final_text = ""
    async for ev in _stream(server, messages, voice, None, stub, audio=True):
        if ev["type"] == "text":
            final_text += ev["delta"]
            await ws.send_text(json.dumps({"type": "text", "delta": ev["delta"]}))
        elif ev["type"] == "audio":
            await ws.send_bytes(ev["data"])

    await ws.send_text(json.dumps({"type": "audio_end"}))
    await ws.send_text(json.dumps({"type": "done"}))

    # Bound the rolling history. The assistant's text is always kept; the
    # user's (large) audio is kept only when the operator opts in, so the model
    # can re-hear prior turns at the cost of re-prefilling them. Camera frames
    # are never kept — only the current turn's views are attached.
    if final_text.strip():
        history.append({"role": "assistant", "content": final_text.strip()})
    if config.keep_audio_history:
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
    per_turn = 2 if config.keep_audio_history else 1
    keep = max(0, config.history_turns) * per_turn
    if keep == 0:
        history.clear()
    elif len(history) > keep:
        del history[:-keep]


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
        help="voice backend: 'omni' (end-to-end Qwen3-Omni) or 'cascade' "
        "(ASR -> VLM brain -> TTS). Defaults to the saved config.",
    )
    parser.add_argument(
        "--omni-url",
        default="",
        help="use an already-running OpenAI-compatible omni server (e.g. a "
        "vLLM-Omni container) instead of supervising Edge-LLM",
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
        status = VoiceStatus(model="stub", model_downloaded=False)
        status.set(phase="stub", detail="stub mode (no models)")
    elif backend == "cascade":
        status = VoiceStatus(model="cascade", precision="mixed", model_downloaded=True)
        asr = StageServer(
            "asr", CASCADE_ASR_MODEL, args.cache_dir / "asr", CASCADE_ASR_PORT,
            capability="transcription", status=status,
        )
        tts = StageServer(
            "tts", CASCADE_TTS_MODEL, args.cache_dir / "tts", CASCADE_TTS_PORT,
            capability="speech", patched_launcher=True, status=status,
        )
        brain = StageServer(
            "brain", CASCADE_BRAIN_MODEL, args.cache_dir / "brain", CASCADE_BRAIN_PORT,
            capability="chat",
            extra_args=[
                "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
                "--max-input-len", "8192", "--max-kv-cache-capacity", "16384",
            ],
            status=status,
        )
        pipeline = CascadePipeline(asr, brain, tts, status)
        pipeline.start_background()
    elif config.omni_container:
        # bebop-voice owns the container: start on service start, remove on stop.
        status = VoiceStatus(
            model="qwen3-omni (container)", precision=precision, model_downloaded=True
        )
        server = ContainerServer(
            config.omni_image,
            config.omni_port,
            status,
            model_dir=config.omni_model_dir,
            deploy_config=config.omni_deploy_config,
        )
        server.start_background()
    elif config.omni_url:
        # External omni backend (e.g. a vLLM-Omni container) — no supervision.
        status = VoiceStatus(
            model="qwen3-omni (external)", precision=precision, model_downloaded=True
        )
        server = ExternalServer(config.omni_url, status)
        server.start_background()
    else:
        status = VoiceStatus(
            model=spec.id if spec else "qwen3-omni-30b",
            precision=precision,
            model_downloaded=downloaded,
        )
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
        pipeline_status=status,
    )
    import uvicorn

    _log(f"listening on ws://{args.host}:{args.port}/voice")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    if server is not None:
        server.stop()
    if pipeline is not None:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
