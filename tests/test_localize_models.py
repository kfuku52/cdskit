"""
Tests for pretrained localize model alias resolution.
"""

import hashlib

import pytest

from cdskit.localize_models import (
    PRETRAINED_LOCALIZE_MODELS,
    localize_model_cache_dir,
    resolve_localize_model_path,
)


def test_resolve_localize_model_path_prefers_existing_path(temp_dir):
    model_path = temp_dir / 'model.pt'
    model_path.write_bytes(b'not a real model')

    assert resolve_localize_model_path(str(model_path)) == str(model_path)


def test_resolve_localize_model_path_reports_unknown_path(temp_dir):
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_localize_model_path(str(temp_dir / 'missing_model.pt'))

    assert 'Localize model path not found' in str(exc_info.value)
    assert 'targeting5' in str(exc_info.value)


def test_targeting5_alias_respects_disabled_download(temp_dir, monkeypatch):
    monkeypatch.setenv('CDSKIT_MODEL_DIR', str(temp_dir))

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_localize_model_path('targeting5', allow_download=False)

    assert 'model download is disabled' in str(exc_info.value)


def test_cached_alias_returns_verified_cache_path(temp_dir, monkeypatch):
    content = b'cached model bytes'
    sha256 = hashlib.sha256(content).hexdigest()
    spec = {
        'name': 'tiny',
        'version': 'v1',
        'filename': 'tiny.pt',
        'aliases': ('tiny',),
        'url': '',
        'sha256': sha256,
        'published': False,
    }
    monkeypatch.setitem(PRETRAINED_LOCALIZE_MODELS, 'tiny-v1', spec)
    monkeypatch.setenv('CDSKIT_MODEL_DIR', str(temp_dir))
    cache_path = localize_model_cache_dir() / 'localize' / 'tiny' / 'v1' / 'tiny.pt'
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(content)

    assert resolve_localize_model_path('tiny') == str(cache_path)


def test_cached_alias_checksum_mismatch_is_rejected(temp_dir, monkeypatch):
    spec = {
        'name': 'bad',
        'version': 'v1',
        'filename': 'bad.pt',
        'aliases': ('bad',),
        'url': '',
        'sha256': hashlib.sha256(b'expected').hexdigest(),
        'published': False,
    }
    monkeypatch.setitem(PRETRAINED_LOCALIZE_MODELS, 'bad-v1', spec)
    monkeypatch.setenv('CDSKIT_MODEL_DIR', str(temp_dir))
    cache_path = localize_model_cache_dir() / 'localize' / 'bad' / 'v1' / 'bad.pt'
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b'observed')

    with pytest.raises(ValueError) as exc_info:
        resolve_localize_model_path('bad')

    assert 'checksum mismatch' in str(exc_info.value)
