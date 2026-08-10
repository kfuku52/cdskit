"""
Tests for pretrained localize model alias resolution.
"""

import hashlib
from pathlib import Path

import pytest

from cdskit.localize_models import (
    PRETRAINED_LOCALIZE_MODELS,
    localize_model_cache_dir,
    resolve_localize_model_path,
)


class FakeResponse:
    def __init__(self, chunks, content_length=None, error=None):
        self._chunks = iter(chunks)
        self._error = error
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, _size):
        if self._error is not None:
            raise self._error
        return next(self._chunks, b"")


def register_downloadable_model(
    monkeypatch, content, *, published=True, url="https://example.test/tiny.pt"
):
    spec = {
        "name": "downloadable",
        "version": "v1",
        "filename": "tiny.pt",
        "aliases": ("downloadable",),
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "published": published,
    }
    monkeypatch.setitem(PRETRAINED_LOCALIZE_MODELS, "downloadable-v1", spec)
    return spec


def test_resolve_localize_model_path_prefers_existing_path(temp_dir):
    model_path = temp_dir / "model.pt"
    model_path.write_bytes(b"not a real model")

    assert resolve_localize_model_path(str(model_path)) == str(model_path)


def test_resolve_localize_model_path_rejects_empty_value():
    with pytest.raises(ValueError, match="--model is required"):
        resolve_localize_model_path("")


def test_resolve_localize_model_path_reports_unknown_path(temp_dir):
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_localize_model_path(str(temp_dir / "missing_model.pt"))

    assert "Localize model path not found" in str(exc_info.value)
    assert "targeting5" in str(exc_info.value)


def test_targeting5_alias_respects_disabled_download(temp_dir, monkeypatch):
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_localize_model_path("targeting5", allow_download=False)

    assert "model download is disabled" in str(exc_info.value)


def test_published_perox_alias_respects_disabled_download(temp_dir, monkeypatch):
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_localize_model_path(
            "targeting5-perox-deeploc21-et-v1", allow_download=False
        )

    spec = PRETRAINED_LOCALIZE_MODELS["targeting5-perox-deeploc21-et-v1"]
    assert spec["published"] is True
    assert (
        spec["sha256"]
        == "d0998df8819d975b4392342ab78dccc0dd95cf301e4d2df8f38c73d0b5aab445"
    )
    assert "model download is disabled" in str(exc_info.value)


def test_cached_alias_returns_verified_cache_path(temp_dir, monkeypatch):
    content = b"cached model bytes"
    sha256 = hashlib.sha256(content).hexdigest()
    spec = {
        "name": "tiny",
        "version": "v1",
        "filename": "tiny.pt",
        "aliases": ("tiny",),
        "url": "",
        "sha256": sha256,
        "published": False,
    }
    monkeypatch.setitem(PRETRAINED_LOCALIZE_MODELS, "tiny-v1", spec)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))
    cache_path = localize_model_cache_dir() / "localize" / "tiny" / "v1" / "tiny.pt"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(content)

    assert resolve_localize_model_path("tiny") == str(cache_path)


def test_registered_alias_wins_over_same_named_working_directory_file(
    temp_dir,
    monkeypatch,
):
    content = b"cached model bytes"
    spec = {
        "name": "tiny",
        "version": "v1",
        "filename": "tiny.pt",
        "aliases": ("tiny",),
        "url": "",
        "sha256": hashlib.sha256(content).hexdigest(),
        "published": False,
    }
    monkeypatch.setitem(PRETRAINED_LOCALIZE_MODELS, "tiny-v1", spec)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir / "cache"))
    cache_path = localize_model_cache_dir() / "localize" / "tiny" / "v1" / "tiny.pt"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(content)
    (temp_dir / "tiny").write_bytes(b"untrusted local shadow")
    monkeypatch.chdir(temp_dir)

    assert resolve_localize_model_path("tiny") == str(cache_path)


def test_cached_alias_checksum_mismatch_is_rejected(temp_dir, monkeypatch):
    spec = {
        "name": "bad",
        "version": "v1",
        "filename": "bad.pt",
        "aliases": ("bad",),
        "url": "",
        "sha256": hashlib.sha256(b"expected").hexdigest(),
        "published": False,
    }
    monkeypatch.setitem(PRETRAINED_LOCALIZE_MODELS, "bad-v1", spec)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))
    cache_path = localize_model_cache_dir() / "localize" / "bad" / "v1" / "bad.pt"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"observed")

    with pytest.raises(ValueError) as exc_info:
        resolve_localize_model_path("bad")

    assert "checksum mismatch" in str(exc_info.value)


def test_cache_dir_honors_xdg_cache_home(temp_dir, monkeypatch):
    monkeypatch.delenv("CDSKIT_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(temp_dir))

    assert localize_model_cache_dir() == temp_dir / "cdskit" / "models"


def test_published_alias_downloads_atomically(temp_dir, monkeypatch):
    content = b"downloaded model"
    register_downloadable_model(monkeypatch, content)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))
    monkeypatch.setattr(
        "cdskit.localize_models.urllib.request.urlopen",
        lambda request, timeout: FakeResponse([content], len(content)),
    )

    resolved = Path(resolve_localize_model_path("downloadable"))

    assert resolved.read_bytes() == content
    assert list(resolved.parent.glob("*.download.*.tmp")) == []


@pytest.mark.parametrize("stream_limit", [False, True])
def test_oversized_download_is_rejected_and_cleaned_up(
    temp_dir, monkeypatch, stream_limit
):
    content = b"four"
    register_downloadable_model(monkeypatch, content)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))
    monkeypatch.setattr("cdskit.localize_models.MAX_MODEL_DOWNLOAD_BYTES", 3)
    response = (
        FakeResponse([content], content_length=None)
        if stream_limit
        else FakeResponse([], content_length=len(content))
    )
    monkeypatch.setattr(
        "cdskit.localize_models.urllib.request.urlopen",
        lambda request, timeout: response,
    )

    with pytest.raises(ValueError, match="safety limit"):
        resolve_localize_model_path("downloadable")

    assert list(temp_dir.rglob("*.tmp")) == []
    assert list(temp_dir.rglob("tiny.pt")) == []


def test_download_error_removes_partial_file(temp_dir, monkeypatch):
    content = b"expected"
    register_downloadable_model(monkeypatch, content)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))
    monkeypatch.setattr(
        "cdskit.localize_models.urllib.request.urlopen",
        lambda request, timeout: FakeResponse([], error=OSError("network failed")),
    )

    with pytest.raises(OSError, match="network failed"):
        resolve_localize_model_path("downloadable")

    assert list(temp_dir.rglob("*.tmp")) == []


def test_unpublished_alias_is_not_downloaded(temp_dir, monkeypatch):
    register_downloadable_model(monkeypatch, b"content", published=False)
    monkeypatch.setenv("CDSKIT_MODEL_DIR", str(temp_dir))

    with pytest.raises(FileNotFoundError, match="not published yet"):
        resolve_localize_model_path("downloadable")
