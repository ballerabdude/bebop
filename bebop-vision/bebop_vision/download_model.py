"""Download catalog model weights from Hugging Face.

The robot does not ship weights. This is the worker the firmware starts as
``bebop-model-download@<id>.service`` (root) with the operator-supplied token
stored at ``/etc/bebop/hf_token``. It resolves the repo/files from the shared
catalog (``bebop_vision.models``) and publishes progress to
``/run/bebop/models/<id>.json`` for the firmware to relay as `ModelState`.

Usage:
    python -m bebop_vision.download_model sam3.1
    python -m bebop_vision.download_model --all
"""

from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path

from . import models
from .models import ModelSpec


def _final_bytes(spec: ModelSpec, dest: Path) -> int:
    total = 0
    for name in spec.files:
        p = dest / name
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _downloaded_bytes(spec: ModelSpec, dest: Path) -> int:
    """Best-effort progress: completed files + in-flight ``.incomplete`` blobs."""
    total = _final_bytes(spec, dest)
    cache = dest / ".cache"
    if cache.is_dir():
        wanted = {Path(name).name for name in spec.files}
        try:
            incomplete = list(cache.rglob("*.incomplete"))
        except OSError:
            incomplete = []
        for p in incomplete:
            if any(w in p.name for w in wanted):
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
    return total


def _classify(exc: BaseException) -> str:
    name = type(exc).__name__
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if "Gated" in name or code in (401, 403):
        return "unauthorized"
    return "failed"


def _detail(spec: ModelSpec, exc: BaseException) -> str:
    name = type(exc).__name__
    if "Gated" in name:
        return (
            f"Access to {spec.repo} is gated: accept the license at "
            f"https://huggingface.co/{spec.repo} for the token's account, then retry."
        )
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if code in (401, 403):
        return f"Hugging Face rejected the token (HTTP {code}) for {spec.repo}."
    if "NotFound" in name:
        return f"{spec.repo} was not found, or this token cannot access it."
    return f"{name}: {exc}"


def _ensure_space(spec: ModelSpec, dest: Path, total: int) -> None:
    """Refuse a download that can't fit, rather than filling the disk."""
    if total <= 0:
        return
    try:
        free = shutil.disk_usage(dest).free
    except OSError:
        return
    headroom = 256 * 1024 * 1024  # metadata + the HF cache copy
    if free < total + headroom:
        raise RuntimeError(
            f"insufficient disk space for {spec.id}: {free} bytes free, need ~{total + headroom}"
        )


def _preflight_size(spec: ModelSpec, token: str | None) -> int:
    """Expected total size from HF metadata, falling back to the catalog hint."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(spec.repo, files_metadata=True, token=token)
    sizes = {s.rfilename: getattr(s, "size", None) for s in (info.siblings or [])}
    total = 0
    found = False
    for filename in spec.files:
        size = sizes.get(filename)
        if size is None:
            base = Path(filename).name
            for name, value in sizes.items():
                if Path(name).name == base:
                    size = value
                    break
        if size:
            total += int(size)
            found = True
    return total if found else spec.bytes


def _download_file(spec: ModelSpec, filename: str, token: str | None, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=spec.repo,
        filename=filename,
        revision=spec.revision,
        local_dir=str(dest),
        token=token,
    )


def _download_with_progress(spec: ModelSpec, token: str | None, dest: Path, total: int) -> None:
    last = 0
    for filename in spec.files:
        done = threading.Event()
        errors: list[BaseException] = []

        def worker(name: str = filename) -> None:
            try:
                _download_file(spec, name, token, dest)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while not done.wait(0.5):
            last = _downloaded_bytes(spec, dest)
            models.write_status(
                spec.id,
                state="downloading",
                detail=f"{spec.repo}/{filename}",
                bytes_downloaded=last,
                bytes_total=total,
            )
        thread.join()
        if errors:
            raise errors[0]
    models.write_status(
        spec.id,
        state="downloading",
        detail=f"{spec.repo} (finalizing)",
        bytes_downloaded=_downloaded_bytes(spec, dest),
        bytes_total=total,
    )


def download_one(spec: ModelSpec, token: str | None) -> None:
    dest = models.WEIGHTS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    models.write_status(spec.id, state="downloading", detail=f"contacting {spec.repo}", bytes_total=spec.bytes)

    try:
        total = _preflight_size(spec, token) or spec.bytes
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator/status file
        models.write_status(
            spec.id,
            state=_classify(exc),
            detail=_detail(spec, exc),
            bytes_total=spec.bytes,
        )
        raise

    _ensure_space(spec, dest, total)

    models.write_status(spec.id, state="downloading", detail=spec.repo, bytes_total=total)
    try:
        _download_with_progress(spec, token, dest, total)
    except Exception as exc:  # noqa: BLE001
        models.write_status(
            spec.id,
            state=_classify(exc),
            detail=_detail(spec, exc),
            bytes_downloaded=_downloaded_bytes(spec, dest),
            bytes_total=total,
        )
        raise

    downloaded = _final_bytes(spec, dest)
    models.write_status(
        spec.id,
        state="ready",
        detail="",
        bytes_downloaded=downloaded,
        bytes_total=downloaded or total,
    )
    print(f"[model] {spec.id} ready -> {dest}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0 if ("--help" in argv or "-h" in argv) else 2

    catalog = models.load_catalog()
    if "--all" in argv:
        ids = [spec.id for spec in catalog.values() if spec.downloadable]
    else:
        ids = [a for a in argv if not a.startswith("-")]

    token = models.load_token()
    rc = 0
    for model_id in ids:
        spec = catalog.get(model_id)
        if spec is None:
            print(f"[model] unknown id {model_id!r} (see config/models.yaml)")
            rc = 1
            continue
        if not spec.downloadable:
            print(f"[model] {spec.id}: kind={spec.kind}; nothing to download")
            continue
        if spec.gated and not token:
            models.write_status(spec.id, state="failed", detail="no Hugging Face token set")
            print(f"[model] {spec.id}: no Hugging Face token — set one from the app first")
            rc = 1
            continue
        try:
            download_one(spec, token)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next model
            print(f"[model] {spec.id}: FAILED: {exc}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
