"""bebop-models: the on-robot model supervisor.

The robot has one GPU (Thor, ~122 GB unified) and several model consumers:
voice (omni / cascade) and a future VLA. This service owns the **GPU budget and
model lifecycle** so consumers do not each invent it.

Every generative model is a **TensorRT Edge-LLM** process (see
`bebop_vision/edge_serve.py`): a checkpoint plus an engine cache plus a port.
That is the only runtime kind this service starts.

Backend kinds:

    edgellm   supervise a `tensorrt-edgellm-serve` child process
    external  an already-running OpenAI-compatible server (debug escape hatch)
    tracked   runs in another process (bebop-vision); counted, not managed

HTTP API (default :9094):

    GET  /healthz            supervisor + GPU status
    GET  /models             registry + live state
    POST /models/{id}/load   start a backend (evicting others to fit budget)
    POST /models/{id}/unload stop a backend

Registry entries come from `config/models_supervisor.yaml` (falls back to a
built-in default).
"""

from __future__ import annotations

import asyncio
import atexit
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import models
from .edge_serve import resolve_edgellm_bin

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9094
# Leave headroom for the OS, the camera pipeline, and CUDA context overhead.
GPU_BUDGET_GB = float(os.environ.get("BEBOP_GPU_BUDGET_GB", "108"))
# An Edge-LLM backend's *first* start builds its TensorRT engines (tens of
# minutes on the Thor); later starts are deserialization (~40 s).
BUILD_TIMEOUT_S = 5400.0
REGISTRY_PATH = Path(
    os.environ.get(
        "BEBOP_MODELS_CONFIG",
        str(Path(__file__).resolve().parent.parent / "config" / "models_supervisor.yaml"),
    )
)

# Built-in registry used when no config file is present.
DEFAULT_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "omni-edgellm",
        "purpose": "voice",
        "kind": "edgellm",
        "footprint_gb": 20,
        "model": "/home/bebop/bebop/bebop-vision/weights/qwen3-omni-30b-nvfp4",
        "cache_dir": "/var/lib/bebop-voice/edgellm",
        "port": 8000,
        "serve_args": ["--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml"],
    },
    {
        "id": "vision",
        "purpose": "vision",
        "kind": "tracked",  # lives inside bebop-vision; counted, not managed here
        "footprint_gb": 6,
    },
]


def _log(msg: str) -> None:
    print(f"[models] {msg}", flush=True)


def _mem() -> dict[str, float]:
    """Unified-memory figures in GB (Thor has no separate VRAM)."""
    info: dict[str, float] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            val = rest.strip().split()[0]
            if key in ("MemTotal", "MemAvailable"):
                info[key] = int(val) / 1024 / 1024
    except Exception:  # noqa: BLE001
        pass
    return info


class Backend:
    """A model backend the supervisor can start/stop or just track."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.id = spec["id"]
        self.kind = spec.get("kind", "external")
        self.footprint_gb = float(spec.get("footprint_gb", 0))
        self.port = int(spec.get("port", 0))
        self.base_url = spec.get("base_url", "")
        if not self.base_url and self.port:
            self.base_url = f"http://127.0.0.1:{self.port}"
        self.state = "stopped" if self.kind != "tracked" else "tracked"
        self.detail = ""
        self.last_used = time.time()
        self.proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    # -- tensorrt edge-llm child ------------------------------------------

    def argv(self) -> list[str]:
        """The `tensorrt-edgellm-serve` command line for this backend."""
        spec = self.spec
        model = str(spec.get("model") or spec.get("model_dir") or "")
        argv = [
            resolve_edgellm_bin(),
            model,
            "--cache-dir", str(spec.get("cache_dir", "")),
            "--host", str(spec.get("host", "127.0.0.1")),
            "--port", str(self.port),
        ]
        argv += [str(a) for a in spec.get("serve_args", [])]
        return argv

    def health_ok(self) -> bool:
        if self.kind == "tracked":
            return True
        import httpx

        try:
            return httpx.get(f"{self.base_url}/v1/models", timeout=3.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def start(self) -> None:
        if self.kind == "tracked":
            self.state = "tracked"
            return
        if self.kind == "external":
            self.state = "ready" if self.health_ok() else "error"
            self.detail = "external" if self.state == "ready" else "unreachable"
            return
        if self.kind != "edgellm":
            self.state = "error"
            self.detail = f"unsupported kind {self.kind}"
            return
        self._start_edgellm()

    def _start_edgellm(self) -> None:
        if self.health_ok():
            with self._lock:
                self.state = "ready"
                self.detail = "already running"
            return
        cache = str(self.spec.get("cache_dir") or "")
        if cache:
            Path(cache).mkdir(parents=True, exist_ok=True)
        argv = self.argv()
        env = os.environ.copy()
        token = models.load_token()
        if token:
            env["HF_TOKEN"] = token
        _log("starting: " + " ".join(argv))
        with self._lock:
            self.state = "starting"
            self.detail = "launching TensorRT Edge-LLM server"
        try:
            # The child inherits our stdout/stderr, so its log lands in the
            # bebop-models journal alongside this service's own lines.
            self.proc = subprocess.Popen(argv, env=env, text=True)
        except OSError as exc:
            with self._lock:
                self.state = "error"
                self.detail = f"spawn failed: {exc}"
            return
        atexit.register(self.stop)
        proc = self.proc
        deadline = time.monotonic() + BUILD_TIMEOUT_S
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                with self._lock:
                    self.state = "error"
                    self.detail = f"server exited rc={proc.returncode}"
                return
            if self.health_ok():
                with self._lock:
                    self.state = "ready"
                    self.detail = "ready"
                _log(f"{self.id} ready on {self.base_url}")
                return
            time.sleep(5.0)
        with self._lock:
            self.state = "error"
            self.detail = "did not become ready in time"

    def stop(self) -> None:
        if self.kind == "tracked":
            return
        proc = self.proc
        if proc is not None and proc.poll() is None:
            _log(f"stopping {self.id} (pid {proc.pid})")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        self.proc = None
        with self._lock:
            self.state = "stopped"
            self.detail = ""

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "purpose": self.spec.get("purpose", ""),
            "kind": self.kind,
            "state": self.state,
            "detail": self.detail,
            "footprint_gb": self.footprint_gb,
            "base_url": self.base_url or None,
            "pid": self.proc.pid if self.proc is not None else None,
            "idle_s": round(max(0.0, time.time() - self.last_used), 1),
        }


def _load_registry() -> list[dict[str, Any]]:
    if REGISTRY_PATH.is_file():
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(REGISTRY_PATH.read_text()) or {}
            entries = data.get("backends") or []
            if entries:
                return entries
        except Exception as exc:  # noqa: BLE001
            _log(f"registry parse failed ({exc}); using defaults")
    return DEFAULT_REGISTRY


class Supervisor:
    def __init__(self) -> None:
        self.backends = {spec["id"]: Backend(spec) for spec in _load_registry()}
        self._lock = threading.Lock()

    def resident_gb(self) -> float:
        return sum(b.footprint_gb for b in self.backends.values() if b.state in ("ready", "tracked"))

    def free_gb(self) -> float:
        return _mem().get("MemAvailable", 0.0)

    def load(self, backend_id: str) -> dict[str, Any]:
        b = self.backends.get(backend_id)
        if b is None:
            return {"error": f"unknown backend {backend_id!r}"}
        if b.state in ("ready", "starting", "tracked"):
            return b.snapshot()
        # Evict resident backends until the new one fits the budget. With
        # Edge-LLM footprints this rarely fires; it is the seam for a future
        # large VLA that genuinely cannot co-reside.
        needed = b.footprint_gb
        with self._lock:
            for other in self.backends.values():
                if other.id == b.id or other.kind == "tracked":
                    continue
                if other.state in ("ready", "starting"):
                    if self.resident_gb() + needed <= GPU_BUDGET_GB and self.free_gb() >= needed:
                        break
                    _log(f"evicting {other.id} to fit {b.id}")
                    other.stop()
        b.start()
        return b.snapshot()

    def unload(self, backend_id: str) -> dict[str, Any]:
        b = self.backends.get(backend_id)
        if b is None:
            return {"error": f"unknown backend {backend_id!r}"}
        b.stop()
        return b.snapshot()


def build_app(supervisor: Supervisor | None = None) -> Any:
    sup = supervisor or Supervisor()
    app = FastAPI(title="bebop-models", docs_url=None, redoc_url=None)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        mem = _mem()
        return JSONResponse(
            {
                "ok": True,
                "budget_gb": GPU_BUDGET_GB,
                "resident_gb": round(sup.resident_gb(), 1),
                "mem": {k: round(v, 1) for k, v in mem.items()},
            }
        )

    @app.get("/models")
    async def models_list() -> JSONResponse:
        return JSONResponse(
            {"backends": [b.snapshot() for b in sup.backends.values()]}
        )

    @app.post("/models/{backend_id}/load")
    async def load(backend_id: str) -> JSONResponse:
        res = await asyncio.to_thread(sup.load, backend_id)
        return JSONResponse(res, status_code=400 if "error" in res else 200)

    @app.post("/models/{backend_id}/unload")
    async def unload(backend_id: str) -> JSONResponse:
        res = await asyncio.to_thread(sup.unload, backend_id)
        return JSONResponse(res, status_code=400 if "error" in res else 200)

    return app


def main() -> int:
    import uvicorn

    host = os.environ.get("BEBOP_MODELS_HOST", DEFAULT_HOST)
    port = int(os.environ.get("BEBOP_MODELS_PORT", str(DEFAULT_PORT)))
    _log(f"listening on http://{host}:{port} (budget {GPU_BUDGET_GB} GB)")
    uvicorn.run(build_app(), host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
