"""bebop-models: the on-robot model supervisor.

The robot has one GPU (Thor, ~122 GB unified) and several would-be consumers:
voice (Qwen3-Omni, the cascade), vision (navd / SAM / DINOv3), navigation, and
a future VLA model. The big models can't co-reside, so this service owns the
**GPU budget and model lifecycle** and lets those consumers time-share it.

It is a *control plane*, not one inference process: backends stay as they are
(Edge-LLM C++ children, Docker containers, external servers) and this service
starts/stops them and reports status. Camera-coupled real-time vision inference
stays inside `bebop-vision` (it is registered here as `tracked` for budget
accounting only).

HTTP API (default :9094):

    GET  /healthz            supervisor + GPU status
    GET  /models             registry + live state
    POST /models/{id}/load   start a backend (evicting others to fit budget)
    POST /models/{id}/unload stop a backend

Registry entries come from `config/models_supervisor.yaml` (falls back to a
built-in default: the vLLM-Omni container plus a tracked `vision` entry).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9094
# Leave headroom for the OS, the camera pipeline, and CUDA context overhead.
GPU_BUDGET_GB = float(os.environ.get("BEBOP_GPU_BUDGET_GB", "108"))
REGISTRY_PATH = Path(
    os.environ.get(
        "BEBOP_MODELS_CONFIG",
        str(Path(__file__).resolve().parent.parent / "config" / "models_supervisor.yaml"),
    )
)

# Built-in registry used when no config file is present.
DEFAULT_REGISTRY: list[dict[str, Any]] = [
    {
        "id": "omni-vllm",
        "purpose": "voice",
        "kind": "container",
        "footprint_gb": 50,
        "image": "ghcr.io/ballerabdude/bebop-vllm-omni-thor:latest",
        "model_dir": "/home/bebop/qwen3-omni-talker-safe",
        "deploy_config": "/home/bebop/qwen3_omni_1gpu.yaml",
        "port": 8101,
        "serve_args": [
            "--omni",
            "--init-timeout", "3600",
            "--stage-init-timeout", "3600",
            # Parallel safetensors load (vLLM defaults to one reader).
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 14}',
            # Disable the flaky sm_110 MoE autotuner (it can fatally crash).
            "--kernel-config",
            '{"enable_flashinfer_autotune": false}',
            # Sleep/wake: free the GPU when idle (level 2: ~84 GB in 1 s).
            "--enable-sleep-mode",
        ],
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


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=False)


class Backend:
    """A model backend the supervisor can start/stop or just track."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.id = spec["id"]
        self.kind = spec.get("kind", "external")
        self.footprint_gb = float(spec.get("footprint_gb", 0))
        self.container = spec.get("container_name", f"bebop-model-{self.id}")
        self.port = int(spec.get("port", 0))
        self.base_url = spec.get("base_url", "")
        if not self.base_url and self.port:
            self.base_url = f"http://127.0.0.1:{self.port}"
        self.state = "stopped" if self.kind != "tracked" else "tracked"
        self.detail = ""
        self.last_used = time.time()
        self._lock = threading.Lock()

    @property
    def _stage_ids(self) -> list[int]:
        ids = self.spec.get("stage_ids")
        return [int(i) for i in ids] if ids else [0, 1, 2]

    def sleep(self, level: int = 2) -> dict[str, Any]:
        """Ask a running OpenAI-compatible omni server to release the GPU.

        vLLM-Omni exposes `/v1/omni/sleep`; level 2 frees the weights (~84 GB on
        Thor in ~1 s) and wake reloads them (~4 min). Requires the server to
        have been started with `--enable-sleep-mode`.
        """
        if self.kind not in ("container", "external"):
            return {"error": f"{self.id} does not support sleep"}
        import httpx

        try:
            r = httpx.post(
                f"{self.base_url}/v1/omni/sleep",
                json={"stage_ids": self._stage_ids, "level": level},
                timeout=180.0,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:120]}"}
        with self._lock:
            self.state = "sleeping"
            self.detail = f"asleep (level {level})"
            self.last_used = time.time()
        return self.snapshot()

    def wake(self) -> dict[str, Any]:
        if self.kind not in ("container", "external"):
            return {"error": f"{self.id} does not support wake"}
        import httpx

        try:
            r = httpx.post(
                f"{self.base_url}/v1/omni/wakeup",
                json={"stage_ids": self._stage_ids},
                timeout=900.0,
            )
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:120]}"}
        with self._lock:
            self.state = "ready"
            self.detail = "awake"
            self.last_used = time.time()
        return self.snapshot()

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
        if self.kind != "container":
            self.state = "error"
            self.detail = f"unsupported kind {self.kind}"
            return
        # container
        cache = Path("/home/bebop/.cache")
        for sub in (
            "vllm", "huggingface", "flashinfer",
            "torch_extensions", "torch", "triton",
        ):
            (cache / sub).mkdir(parents=True, exist_ok=True)
        _docker("rm", "-f", self.container)
        img = self.spec["image"]
        if "/" in img:
            _docker("pull", img)
        args = [
            "run", "-d", "--name", self.container,
            "--runtime", "nvidia", "--network", "host",
            "-v", f"{self.spec['model_dir']}:/models/model:ro",
            "-v", f"{cache}/vllm:/root/.cache/vllm",
            "-v", f"{cache}/huggingface:/root/.cache/huggingface",
            # Persist kernel/JIT caches: otherwise a fresh container re-JITs
            # flashinfer/Triton (minutes) on every start. See docs/vllm-omni.md.
            "-v", f"{cache}/flashinfer:/root/.cache/flashinfer",
            "-v", f"{cache}/torch_extensions:/root/.cache/torch_extensions",
            "-v", f"{cache}/torch:/root/.cache/torch",
            "-v", f"{cache}/triton:/root/.triton",
        ]
        if self.spec.get("deploy_config"):
            args += ["-v", f"{self.spec['deploy_config']}:/cfg/deploy.yaml:ro"]
        args += [img, "/models/model", *self.spec.get("serve_args", ["--omni"])]
        if self.spec.get("deploy_config"):
            args += ["--deploy-config", "/cfg/deploy.yaml"]
        args += ["--host", "0.0.0.0", "--port", str(self.port)]
        with self._lock:
            self.state = "starting"
            self.detail = f"starting {img}"
        res = _docker(*args)
        if res.returncode != 0:
            with self._lock:
                self.state = "error"
                self.detail = f"docker run failed: {res.stderr.strip()[:160]}"
            return
        # vLLM-Omni cold start is long.
        for _ in range(720):
            if self.health_ok():
                with self._lock:
                    self.state = "ready"
                    self.detail = "ready"
                return
            if not _docker("ps", "-q", "-f", f"name={self.container}").stdout.strip():
                with self._lock:
                    self.state = "error"
                    self.detail = "container exited during startup"
                return
            time.sleep(5.0)
        with self._lock:
            self.state = "error"
            self.detail = "did not become ready in time"

    def stop(self) -> None:
        if self.kind == "tracked":
            return
        if self.kind == "container":
            _docker("rm", "-f", self.container)
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
        # Evict resident backends until the new one fits the budget.
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

    def sleep(self, backend_id: str, level: int = 2) -> dict[str, Any]:
        b = self.backends.get(backend_id)
        if b is None:
            return {"error": f"unknown backend {backend_id!r}"}
        return b.sleep(level=level)

    def wake(self, backend_id: str) -> dict[str, Any]:
        b = self.backends.get(backend_id)
        if b is None:
            return {"error": f"unknown backend {backend_id!r}"}
        return b.wake()


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
    async def models() -> JSONResponse:
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

    @app.post("/models/{backend_id}/sleep")
    async def sleep(backend_id: str, level: int = 2) -> JSONResponse:
        res = await asyncio.to_thread(sup.sleep, backend_id, level)
        return JSONResponse(res, status_code=400 if "error" in res else 200)

    @app.post("/models/{backend_id}/wake")
    async def wake(backend_id: str) -> JSONResponse:
        res = await asyncio.to_thread(sup.wake, backend_id)
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
