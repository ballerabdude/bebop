"""On-device speech-to-speech gateway (Qwen3-Omni via TensorRT Edge-LLM).

`bebop-voice.service` runs this as root from `bebop-vision/.venv-voice`. It
owns two things:

1. **The model server.** It supervises `tensorrt-edgellm-serve`, which serves
   the catalog's `voice` checkpoint (Qwen3-Omni-30B-A3B) as an
   OpenAI-compatible HTTP API on 127.0.0.1:8000. The first start builds the
   checkpoint's TensorRT engines into `--cache-dir`; that can take minutes.
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

`GET /healthz` reports readiness (model server up, checkpoint present).

The `--stub` flag runs a tiny local echo/tone generator instead of the model,
so the gateway + app protocol can be exercised without the 60 GB checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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


def pcm16_to_wav(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


class EdgeLLMServer:
    """Supervises the `tensorrt-edgellm-serve` child process."""

    def __init__(
        self,
        model: str,
        cache_dir: Path,
        port: int = EDGELLM_PORT,
        extra_args: list[str] | None = None,
    ) -> None:
        self.model = model
        self.cache_dir = cache_dir
        self.port = port
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{EDGELLM_HOST}:{self.port}"

    def _already_running(self) -> bool:
        import httpx

        try:
            r = httpx.get(f"{self.base_url}/health", timeout=1.0)
            return r.status_code == 200
        except Exception:  # noqa: BLE001 - any failure means "not up"
            return False

    def start(self) -> None:
        if self._already_running():
            _log(f"Edge-LLM server already up on {self.base_url}")
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            EDGELLM_BIN,
            self.model,
            "--cache-dir",
            str(self.cache_dir),
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        _log("starting: " + " ".join(cmd))
        env = os.environ.copy()
        token = read_hf_token()
        if token:
            env["HF_TOKEN"] = token
        self.proc = subprocess.Popen(cmd, env=env)
        atexit.register(self.stop)

    def wait_ready(self, timeout_s: float = 1800.0) -> bool:
        """Block until /health is up. Generous: first run builds engines."""
        import httpx

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                _log(f"Edge-LLM server exited rc={self.proc.returncode}")
                return False
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    _log("Edge-LLM server ready")
                    return True
            except Exception:  # noqa: BLE001 - keep polling
                pass
            time.sleep(2.0)
        _log("timed out waiting for the Edge-LLM server")
        return False

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        _log("stopping Edge-LLM server")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


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


def build_app(server: EdgeLLMServer | None, *, stub: bool, voice: str) -> Any:
    """Construct the FastAPI app.

    FastAPI resolves the endpoint's annotations with `typing.get_type_hints`,
    and this module uses `from __future__ import annotations` (string
    annotations). `WebSocket` therefore must be a *module* global, not a local
    import, or FastAPI fails to recognize the websocket parameter and the
    route rejects every upgrade with 403.
    """
    app = FastAPI(title="bebop-voice", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        spec = voice_spec()
        model_present = spec is not None and (
            spec.target_dir / ".bebop-complete"
        ).is_file()
        model_ready = stub or (server is not None and server._already_running())
        return JSONResponse(
            {
                "ok": model_ready,
                "stub": stub,
                "model": spec.id if spec else None,
                "model_downloaded": model_present,
                "model_ready": model_ready,
            }
        )

    @app.websocket("/voice")
    async def voice_ws(ws: WebSocket) -> None:
        await ws.accept()
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
    model_dir = spec.target_dir
    _log(f"voice model {spec.id} -> {model_dir}")

    if args.check:
        present = (model_dir / ".bebop-complete").is_file()
        print(
            json.dumps(
                {
                    "model": spec.id,
                    "model_dir": str(model_dir),
                    "downloaded": present,
                    "stub": args.stub,
                },
                indent=2,
            )
        )
        return 0

    server: EdgeLLMServer | None = None
    if not args.stub:
        if not (model_dir / ".bebop-complete").is_file():
            _log(
                f"checkpoint not downloaded at {model_dir}; "
                "download it from the app's Models page first"
            )
            return 1
        server = EdgeLLMServer(
            str(model_dir), args.cache_dir, extra_args=args.edgellm_arg
        )
        server.start()
        # Don't block the listener on engine build: serve /healthz immediately
        # and let the first turn wait. But do a bounded wait so a fast start is
        # ready before the operator connects.
        server.wait_ready(timeout_s=1800.0)

    app = build_app(server, stub=args.stub, voice=args.voice)
    import uvicorn

    _log(f"listening on ws://{args.host}:{args.port}/voice")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    if server is not None:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
