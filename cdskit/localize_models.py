"""
Helpers for resolving cdskit localize pretrained model aliases.
"""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path


TARGETING5_V1_FILENAME = 'cdskit-localize-targeting5-v1.pt'
TARGETING5_PEROX_DEEPLOC21_HGB_V1_FILENAME = (
    'cdskit-localize-targeting5-perox-deeploc21-hgb-v1.pt'
)

PRETRAINED_LOCALIZE_MODELS = {
    'targeting5-v1': {
        'name': 'targeting5-v1',
        'version': 'v1',
        'filename': TARGETING5_V1_FILENAME,
        'aliases': ('targeting5', 'targeting5-v1', 'targeting5:v1'),
        'description': 'CPU-inference TargetP-compatible noTP/SP/mTP/cTP/lTP model.',
        'url': (
            'https://github.com/kfuku52/cdskit/releases/download/'
            'localize-targeting5-v1/{}'.format(TARGETING5_V1_FILENAME)
        ),
        'sha256': 'ddaeab7093533a213ee58117b70ad0f45b0c126cf82c77df32e369eaff2beeb2',
        'published': True,
    },
    'targeting5-perox-deeploc21-hgb-v1': {
        'name': 'targeting5-perox-deeploc21-hgb-v1',
        'version': 'v1',
        'filename': TARGETING5_PEROX_DEEPLOC21_HGB_V1_FILENAME,
        'aliases': (
            'targeting5-perox-deeploc21-hgb-v1',
            'targeting5-perox-deeploc21-hgb',
            'targeting5-perox-deeploc21',
        ),
        'description': (
            'Unpublished CPU-inference targeting5 model with an experimental '
            'DeepLoc21-trained peroxisome sequence-label head.'
        ),
        'url': (
            'https://github.com/kfuku52/cdskit/releases/download/'
            'localize-targeting5-perox-deeploc21-hgb-v1/{}'.format(
                TARGETING5_PEROX_DEEPLOC21_HGB_V1_FILENAME
            )
        ),
        'sha256': '1e081e26623e73317a1f04bf130ddb9fc717563b98dd8cafbf77383882af375f',
        'published': False,
    },
}


def _truthy(value):
    return str(value or '').strip().lower() in {'1', 'true', 't', 'yes', 'y', 'on'}


def _alias_map():
    out = {}
    for key, spec in PRETRAINED_LOCALIZE_MODELS.items():
        aliases = list(spec.get('aliases', ()))
        aliases.append(key)
        for alias in aliases:
            out[str(alias).strip().lower()] = spec
    return out


def known_pretrained_localize_models():
    """Return known pretrained localize aliases."""

    return sorted(_alias_map().keys())


def localize_model_cache_dir():
    """Return the root cache directory for downloaded cdskit models."""

    override = os.environ.get('CDSKIT_MODEL_DIR', '').strip()
    if override:
        return Path(override).expanduser()

    xdg_cache = os.environ.get('XDG_CACHE_HOME', '').strip()
    if xdg_cache:
        return Path(xdg_cache).expanduser() / 'cdskit' / 'models'

    return Path.home() / '.cache' / 'cdskit' / 'models'


def _cache_path_for_spec(spec):
    return (
        localize_model_cache_dir()
        / 'localize'
        / str(spec.get('name', '')).strip()
        / str(spec.get('version', '')).strip()
        / str(spec.get('filename', '')).strip()
    )


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path, spec):
    expected = str(spec.get('sha256', '') or '').strip().lower()
    if not expected:
        return

    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(
            'Cached pretrained localize model checksum mismatch at {}. '
            'Expected {}, observed {}. Remove the cached file to download it again.'.format(
                path,
                expected,
                observed,
            )
        )


def _download_to_cache(spec, cache_path):
    url = str(spec.get('url', '') or '').strip()
    expected_sha256 = str(spec.get('sha256', '') or '').strip()
    if url == '' or expected_sha256 == '':
        raise FileNotFoundError(
            'Pretrained localize model "{}" is registered but its release URL or '
            'sha256 checksum is not ready yet. Use an explicit --model PATH for now.'.format(
                spec.get('name', 'unknown')
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix='{}.download.'.format(cache_path.name),
        suffix='.tmp',
        dir=str(cache_path.parent),
    )
    try:
        with os.fdopen(fd, 'wb') as out:
            request = urllib.request.Request(
                url,
                headers={'User-Agent': 'cdskit-localize-model-downloader'},
            )
            with urllib.request.urlopen(request, timeout=60) as inp:
                while True:
                    chunk = inp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
        _verify_checksum(tmp_path, spec)
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def resolve_localize_model_path(model, allow_download=True):
    """
    Resolve a localize model argument to a local file path.

    Existing paths are returned unchanged. Known aliases are resolved from the
    cdskit model cache and downloaded only after the registry entry is marked
    as published and has a checksum.
    """

    model_text = str(model or '').strip()
    if model_text == '':
        raise ValueError('--model is required.')

    path = Path(model_text).expanduser()
    if path.exists():
        return str(path)

    spec = _alias_map().get(model_text.lower())
    if spec is None:
        known = ', '.join(known_pretrained_localize_models())
        raise FileNotFoundError(
            'Localize model path not found: {}. Known pretrained model aliases: {}'.format(
                model_text,
                known,
            )
        )

    cache_path = _cache_path_for_spec(spec)
    if cache_path.exists():
        _verify_checksum(cache_path, spec)
        return str(cache_path)

    if not allow_download or _truthy(os.environ.get('CDSKIT_OFFLINE', '')):
        raise FileNotFoundError(
            'Pretrained localize model "{}" is not present in the cache at {} '
            'and model download is disabled.'.format(
                spec.get('name', model_text),
                cache_path,
            )
        )

    if not bool(spec.get('published', False)):
        raise FileNotFoundError(
            'Pretrained localize model "{}" is registered but not published yet. '
            'Use an explicit --model PATH until the GitHub Release asset and '
            'sha256 checksum are finalized.'.format(spec.get('name', model_text))
        )

    _download_to_cache(spec, cache_path)
    _verify_checksum(cache_path, spec)
    return str(cache_path)
