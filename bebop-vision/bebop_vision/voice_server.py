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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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


async def stream_completion(
    server: EdgeLLMServer,
    messages: list[dict[str, Any]],
    voice: str,
) -> AsyncIterator[tuple[str, bytes | None]]:
    """Yield `(text_delta, audio_bytes)` from the Edge-LLM chat stream.

    Parses the OpenAI SSE schema; audio rides in `delta.audio.data` (base64
    PCM16). The parser is defensive about where the audio object appears
    because the experimental server's exact chunk shape is still settling.
    """
    import httpx

    payload = {
        "messages": messages,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "pcm16"},
        "stream": True,
        "max_tokens": 512,
    }
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
                    audio_b64 = None
                    audio = delta.get("audio")
                    if isinstance(audio, dict):
                        audio_b64 = audio.get("data")
                    if text:
                        yield text, None
                    if audio_b64:
                        try:
                            yield "", base64.b64decode(audio_b64)
                        except Exception:  # noqa: BLE001 - skip a bad chunk
                            pass


async def stream_stub(
    messages: list[dict[str, Any]], voice: str
) -> AsyncIterator[tuple[str, bytes | None]]:
    """Offline stand-in for the model: a short text reply + a 440 Hz tone."""
    import math
    import struct

    del messages, voice
    reply = "Voice gateway stub: audio transport is working."
    for word in reply.split():
        yield word + " ", None
        await asyncio.sleep(0.05)
    frames = 24_000 // 2  # 0.5 s
    for i in range(0, frames, 480):
        samples = [
            int(6000 * math.sin(2 * math.pi * 440 * (i + j) / OUTPUT_RATE))
            for j in range(480)
        ]
        yield "", struct.pack(f"<{len(samples)}h", *samples)
        await asyncio.sleep(0.01)


# --- gateway app ----------------------------------------------------------


def build_app(
    server: EdgeLLMServer | None,
    *,
    stub: bool,
    voice: str,
    status: VoiceStatus | None = None,
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
    app = FastAPI(title="bebop-voice", docs_url=None, redoc_url=None)
    # The app polls `GET /healthz` from a different origin (tauri://… or a dev
    # http://localhost); WebSockets aren't CORS-gated but the fetch is.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(status.snapshot())

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
        history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if "text" in msg and msg["text"] is not None:
                    ctl = json.loads(msg["text"])
                    if ctl.get("type") != "utterance_start":
                        continue
                    sample_rate = int(ctl.get("sample_rate", INPUT_RATE))
                    pcm = await _collect_pcm(ws)
                    if not pcm:
                        await ws.send_text(json.dumps({"type": "done"}))
                        continue
                    await _run_turn(
                        ws, history, pcm, sample_rate, server, stub=stub, voice=voice
                    )
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
    voice: str,
) -> None:
    wav_b64 = base64.b64encode(pcm16_to_wav(pcm, sample_rate)).decode("ascii")
    user_msg = {
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": wav_b64, "format": "wav"}}
        ],
    }
    messages = [*history, user_msg]

    await ws.send_text(
        json.dumps({"type": "audio_start", "sample_rate": OUTPUT_RATE, "format": "pcm16"})
    )
    assistant_text = ""
    stream = stream_stub(messages, voice) if stub else stream_completion(server, messages, voice)  # type: ignore[arg-type]
    async for text, audio in stream:
        if text:
            assistant_text += text
            await ws.send_text(json.dumps({"type": "text", "delta": text}))
        if audio:
            await ws.send_bytes(audio)
    await ws.send_text(json.dumps({"type": "audio_end"}))
    await ws.send_text(json.dumps({"type": "done"}))

    # Bounded rolling history. Keep the user *audio* out of history (it is
    # large); retain the assistant text so the model has conversational
    # context without re-prefilling minutes of audio.
    if assistant_text.strip():
        history.append({"role": "assistant", "content": assistant_text.strip()})
    keep = DEFAULT_HISTORY_TURNS * 2
    if len(history) > keep + 1:
        history[:] = [history[0], *history[-(keep):]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
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
    args = parser.parse_args(argv)

    spec = voice_spec()
    if spec is None:
        _log("no catalog entry serves purpose 'voice' — see config/models.yaml")
        return 1
    if not spec.is_snapshot:
        _log(f"voice model {spec.id} must use download: snapshot")
        return 1
    model_dir = resolve_model_dir(spec)
    quantized = model_dir != spec.target_dir
    if quantized:
        downloaded = (model_dir / "config.json").is_file()
    else:
        downloaded = (spec.target_dir / ".bebop-complete").is_file()
    precision = "nvfp4" if quantized else "fp16"
    _log(f"voice model {spec.id} -> {model_dir} ({precision})")

    status = VoiceStatus(
        model=spec.id, precision=precision, model_downloaded=downloaded
    )

    if args.check:
        print(
            json.dumps(
                {
                    "model": spec.id,
                    "model_dir": str(model_dir),
                    "precision": precision,
                    "downloaded": downloaded,
                    "stub": args.stub,
                },
                indent=2,
            )
        )
        return 0

    server: EdgeLLMServer | None = None
    if args.stub:
        status.set(phase="stub", detail="stub mode (no model)")
    elif not downloaded:
        status.set(
            phase="error",
            detail=(
                f"checkpoint not ready at {model_dir}; download it from the "
                "Models page and run the NVFP4 quantization step (docs/voice.md)"
            ),
        )
    else:
        server = EdgeLLMServer(
            str(model_dir), args.cache_dir, status, extra_args=args.edgellm_arg
        )
        # Non-blocking: the gateway serves /healthz immediately and reports
        # build progress while the engines compile.
        server.start_background()

    app = build_app(server, stub=args.stub, voice=args.voice, status=status)
    import uvicorn

    _log(f"listening on ws://{args.host}:{args.port}/voice")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    if server is not None:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
