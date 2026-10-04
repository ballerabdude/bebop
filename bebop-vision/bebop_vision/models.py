"""Shared model catalog + provisioning helpers.

The catalog (`bebop-vision/config/models.yaml`) is the single source of truth
for downloadable/preprovisioned models, shared with the Rust firmware
(`bebop-linux/src/model.rs`). This module parses it and owns the on-robot
paths (weights dir, HF token, per-model download status) plus token loading.

Nothing here imports torch/HF; it stays import-light so the firmware-side
tooling and tests can use it cheaply.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = REPO_ROOT / "config" / "models.yaml"
WEIGHTS_DIR = REPO_ROOT / "weights"

# Where the Rust firmware stores the operator-supplied HF token (root-only),
# and where the downloader writes per-model progress for the firmware to read.
HF_TOKEN_PATH = Path("/etc/bebop/hf_token")
STATUS_DIR = Path("/run/bebop/models")
LEGACY_ENV_PATH = REPO_ROOT / ".env"


@dataclass(frozen=True)
class ModelSpec:
    """One catalog entry."""

    id: str
    name: str = ""
    description: str = ""
    kind: str = "hf"
    purpose: str = ""
    repo: str = ""
    files: tuple[str, ...] = field(default_factory=tuple)
    revision: str = "main"
    gated: bool = False
    bytes: int = 0
    path: str = ""
    # kind="hf" fetch strategy: "files" (named `files`) or "snapshot" (whole
    # repo). Snapshot entries are for checkpoints a runtime builds from.
    download: str = "files"
    # Snapshot destination subdir under WEIGHTS_DIR (defaults to `id`).
    dest: str = ""

    @property
    def downloadable(self) -> bool:
        return self.kind == "hf"

    @property
    def is_snapshot(self) -> bool:
        return self.kind == "hf" and self.download == "snapshot"

    @property
    def target_dir(self) -> Path:
        """Where a snapshot download lands (``weights/<dest or id>/``)."""
        return WEIGHTS_DIR / (self.dest or self.id)


def _as_tuple(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def spec_from_mapping(raw: dict) -> ModelSpec:
    return ModelSpec(
        id=str(raw["id"]).strip(),
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        kind=str(raw.get("kind", "hf")),
        purpose=str(raw.get("purpose", "")),
        repo=str(raw.get("repo", "")),
        files=_as_tuple(raw.get("files")),
        revision=str(raw.get("revision", "main")),
        gated=bool(raw.get("gated", False)),
        bytes=int(raw.get("bytes", 0) or 0),
        path=str(raw.get("path", "")),
        download=str(raw.get("download", "files")),
        dest=str(raw.get("dest", "")),
    )


def load_catalog(path: Path | str = CATALOG_PATH) -> dict[str, ModelSpec]:
    """Load the catalog as an ordered ``{id: ModelSpec}`` mapping."""
    import yaml

    text = Path(path).read_text()
    raw = yaml.safe_load(text) or {}
    models = raw.get("models") or []
    out: dict[str, ModelSpec] = {}
    for entry in models:
        spec = spec_from_mapping(entry)
        if not spec.id:
            raise ValueError(f"catalog entry with empty id in {path}")
        if spec.kind == "hf":
            if not spec.repo:
                raise ValueError(f"catalog entry {spec.id!r}: kind=hf requires a repo")
            if spec.download not in ("files", "snapshot"):
                raise ValueError(
                    f"catalog entry {spec.id!r}: unsupported download "
                    f"{spec.download!r} (expected 'files' or 'snapshot')"
                )
            if spec.download == "files" and not spec.files:
                raise ValueError(
                    f"catalog entry {spec.id!r}: kind=hf download=files requires files"
                )
        if spec.id in out:
            raise ValueError(f"duplicate catalog id {spec.id!r}")
        out[spec.id] = spec
    return out


def load_token() -> str | None:
    """HF token: ``HF_TOKEN`` env, then ``/etc/bebop/hf_token``, then ``.env``."""
    token = os.environ.get("HF_TOKEN")
    if token and token.strip():
        return token.strip()
    try:
        value = HF_TOKEN_PATH.read_text().strip()
        if value:
            return value
    except OSError:
        pass
    if LEGACY_ENV_PATH.exists():
        for line in LEGACY_ENV_PATH.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return value
    return None


def status_path(model_id: str) -> Path:
    return STATUS_DIR / f"{model_id}.json"


def write_status(
    model_id: str,
    *,
    state: str,
    detail: str = "",
    bytes_downloaded: int = 0,
    bytes_total: int = 0,
) -> None:
    """Atomically publish progress for the firmware to read."""
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "detail": detail,
        "bytes_downloaded": int(bytes_downloaded),
        "bytes_total": int(bytes_total),
    }
    path = status_path(model_id)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload))
    os.replace(tmp, path)


def read_status(model_id: str) -> dict | None:
    try:
        return json.loads(status_path(model_id).read_text())
    except (OSError, ValueError):
        return None
