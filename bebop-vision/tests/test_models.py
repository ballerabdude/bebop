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
