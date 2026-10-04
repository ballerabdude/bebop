"""Tests for the shared model catalog + downloader helpers (no network/HF)."""

import json

import pytest

from bebop_vision import download_model, models


def test_real_catalog_loads_and_has_expected_entries():
    catalog = models.load_catalog()
    assert "sam3.1" in catalog
    sam = catalog["sam3.1"]
    assert sam.kind == "hf"
    assert sam.repo == "facebook/sam3.1"
    assert sam.gated is True
    assert sam.purpose == "segmentation"
    assert sam.downloadable

    # Local entries exist and are not downloadable.
    local = [s for s in catalog.values() if s.kind == "local"]
    assert local, "expected at least one local catalog entry"
    assert all(not s.downloadable for s in local)
    # Two segmentation models -> the operator can pick between them.
    assert sum(1 for s in catalog.values() if s.purpose == "segmentation") >= 2


def test_spec_from_mapping_defaults():
    spec = models.spec_from_mapping({"id": "x", "repo": "a/b", "files": "f.bin"})
    assert spec.kind == "hf"
    assert spec.revision == "main"
    assert spec.files == ("f.bin",)
    assert spec.gated is False


def test_load_catalog_rejects_hf_without_repo(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - id: bad\n    kind: hf\n    files: [a]\n")
    with pytest.raises(ValueError):
        models.load_catalog(path)


def test_load_catalog_rejects_duplicates(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - id: dup\n    kind: local\n  - id: dup\n    kind: local\n")
    with pytest.raises(ValueError):
        models.load_catalog(path)


def test_token_precedence_env_then_file_then_envfile(monkeypatch, tmp_path):
    token_file = tmp_path / "hf_token"
    legacy = tmp_path / ".env"

    monkeypatch.setattr(models, "HF_TOKEN_PATH", token_file)
    monkeypatch.setattr(models, "LEGACY_ENV_PATH", legacy)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # File wins when env is unset.
    token_file.write_text("hf_from_file\n")
    assert models.load_token() == "hf_from_file"

    # Env wins over the file.
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")
    assert models.load_token() == "hf_from_env"

    # Falling back to .env when neither env nor file is present.
    monkeypatch.delenv("HF_TOKEN")
    token_file.unlink()
    legacy.write_text("HF_TOKEN=hf_from_dotenv\nOTHER=1\n")
    assert models.load_token() == "hf_from_dotenv"


def test_status_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "STATUS_DIR", tmp_path)
    models.write_status("sam3.1", state="downloading", detail="repo", bytes_downloaded=10, bytes_total=100)
    path = tmp_path / "sam3.1.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["state"] == "downloading"
    assert payload["bytes_downloaded"] == 10
    assert models.read_status("sam3.1") == payload
    assert models.read_status("missing") is None


def test_classify_and_detail_map_auth_errors():
    class GatedRepoError(Exception):
        pass

    exc = GatedRepoError("gated")
    assert download_model._classify(exc) == "unauthorized"
    detail = download_model._detail(
        models.ModelSpec(id="sam3.1", repo="facebook/sam3.1"), exc
    )
    assert "accept the license" in detail

    class NotFound(Exception):
        pass

    assert download_model._classify(NotFound("x")) == "failed"


def test_download_progress_counts_cache_growth(monkeypatch, tmp_path):
    """The partial blob is etag-named; progress must come from cache growth."""
    spec = models.ModelSpec(id="m", kind="hf", repo="a/b", files=("f.bin",), bytes=100)
    writes = []
    monkeypatch.setattr(
        download_model.models, "write_status", lambda _id, **kw: writes.append(kw)
    )

    def fake_download(_spec, filename, _token, dest):
        cache = dest / ".cache" / "huggingface" / "download"
        cache.mkdir(parents=True, exist_ok=True)
        # etag-named partial, NOT the model filename
        (cache / "apa6_blob.incomplete").write_bytes(b"x" * 40)
        import time as _time

        _time.sleep(0.7)  # let the polling loop observe the partial
        (dest / filename).write_bytes(b"y" * 100)

    monkeypatch.setattr(download_model, "_download_file", fake_download)
    download_model._download_with_progress(spec, None, tmp_path, 100)

    partial = [w for w in writes if 0 < w.get("bytes_downloaded", 0) < 100]
    assert partial, f"expected an in-flight progress sample, got {writes}"
    assert writes[-1]["bytes_downloaded"] >= 100


def test_real_catalog_has_voice_snapshot():
    catalog = models.load_catalog()
    omni = catalog["qwen3-omni-30b"]
    assert omni.is_snapshot
    assert omni.downloadable
    assert omni.purpose == "voice"
    assert omni.target_dir.name == "qwen3-omni-30b"


def test_spec_snapshot_defaults_dest_to_id():
    spec = models.spec_from_mapping({"id": "x", "repo": "a/b", "download": "snapshot"})
    assert spec.is_snapshot
    assert spec.files == ()
    assert spec.target_dir.name == "x"


def test_load_catalog_rejects_bad_download_mode(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - id: bad\n    repo: a/b\n    download: torrent\n")
    with pytest.raises(ValueError):
        models.load_catalog(path)


def test_load_catalog_allows_snapshot_without_files(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - id: ok\n    repo: a/b\n    download: snapshot\n")
    assert models.load_catalog(path)["ok"].is_snapshot


def test_tree_bytes_excludes_hf_cache(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 300)
    cache = tmp_path / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "blob.incomplete").write_bytes(b"y" * 999)
    assert download_model._tree_bytes(tmp_path) == 300


def test_snapshot_progress_reports_total(monkeypatch, tmp_path):
    spec = models.ModelSpec(
        id="omni", kind="hf", repo="a/b", download="snapshot", bytes=1000
    )
    writes = []
    monkeypatch.setattr(
        download_model.models, "write_status", lambda _id, **kw: writes.append(kw)
    )
    dest = tmp_path / "omni"

    def fake_snapshot(_spec, _token, _dest):
        _dest.mkdir(parents=True, exist_ok=True)
        (_dest / "config.json").write_text("{}")
        import time as _time

        _time.sleep(0.7)
        (_dest / "model.safetensors").write_bytes(b"x" * 500)

    monkeypatch.setattr(download_model, "_download_snapshot", fake_snapshot)
    download_model._download_snapshot_with_progress(spec, None, dest, 1000)
    assert writes[-1]["bytes_total"] == 1000
    assert writes[-1]["bytes_downloaded"] >= 500


def test_snapshot_download_one_writes_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "WEIGHTS_DIR", tmp_path)
    spec = models.ModelSpec(
        id="omni", kind="hf", repo="a/b", download="snapshot", dest="omni", bytes=10
    )
    monkeypatch.setattr(download_model, "_preflight_size", lambda _s, _t: 10)
    monkeypatch.setattr(download_model.models, "write_status", lambda _id, **kw: None)

    def fake_download(_spec, _token, _dest, _total):
        (_dest / "config.json").write_text("{}")

    monkeypatch.setattr(download_model, "_download_with_progress", fake_download)
    download_model.download_one(spec, None)
    assert (tmp_path / "omni" / download_model.SNAPSHOT_MARKER).exists()
