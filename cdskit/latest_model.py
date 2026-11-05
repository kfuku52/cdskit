#!/usr/bin/env python3
"""Select the newest published CDSKIT localization checkpoint once per cache."""

import argparse
from filelock import FileLock
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from pathlib import Path

RELEASES_URL = "https://api.github.com/repos/kfuku52/cdskit/releases"
DOWNLOAD_PREFIX = "https://github.com/kfuku52/cdskit/releases/download/"


def open_url(url):
    return urllib.request.urlopen(
        urllib.request.Request(
            url,
            headers={"User-Agent": "cdskit", "Accept": "application/vnd.github+json"},
        ),
        timeout=120,
    )


def _asset_metadata(asset):
    """Optional explicit selector evidence stored in the GitHub asset label."""
    try:
        value = json.loads(asset.get("label") or "{}")
    except (ValueError, TypeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key]
        for key in ("model_family", "feature_schema")
        if isinstance(value.get(key), str) and value[key]
    }


def _matches_selection(spec, model_family, feature_schema):
    return all(
        expected is None or spec.get(key) == expected
        for key, expected in (
            ("model_family", model_family),
            ("feature_schema", feature_schema),
        )
    )


def latest_model(
    *,
    family="localize",
    schema=None,
    model_family=None,
    feature_schema=None,
    opener=None,
):
    if family != "localize" or schema not in {None, "checkpoint"}:
        raise ValueError("Unsupported latest model family/schema")
    releases = []
    page = 1
    while True:
        with (opener or open_url)(
            f"{RELEASES_URL}?per_page=100&page={page}"
        ) as response:
            batch = json.load(response)
        releases.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    candidates = []
    for release in releases:
        if (
            release["draft"]
            or release["prerelease"]
            or not release["tag_name"].startswith("localize-")
        ):
            continue
        assets = [
            a
            for a in release["assets"]
            if re.fullmatch(r"cdskit-localize-[\w.-]+\.pt", a["name"])
            and _matches_selection(_asset_metadata(a), model_family, feature_schema)
        ]
        if len(assets) != 1:
            continue
        candidates.append((release["published_at"], release, assets[0]))
    if not candidates:
        raise ValueError("No published CDSKIT localization checkpoint found")
    _, release, asset = max(candidates, key=lambda item: item[0])
    digest = asset.get("digest") or ""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Latest CDSKIT model has no SHA-256 release digest")
    if not asset["browser_download_url"].startswith(DOWNLOAD_PREFIX):
        raise ValueError("Unexpected CDSKIT model download URL")
    return {
        "release": release["tag_name"],
        "published_at": release["published_at"],
        "filename": asset["name"],
        "url": asset["browser_download_url"],
        "sha256": digest[7:],
        **_asset_metadata(asset),
    }


def verify(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"CDSKIT model checksum mismatch: {path}")


def resolve(
    cache_dir,
    allow_download=True,
    *,
    namespace="latest",
    selector=None,
    opener=None,
    model_family=None,
    feature_schema=None,
):
    if (
        not namespace
        or namespace in {".", ".."}
        or "/" in namespace
        or "\\" in namespace
    ):
        raise ValueError("Model selection namespace must be one directory name")
    cache = Path(cache_dir).resolve() / namespace
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "selection.json"
    with FileLock(str(cache / "selection.lock")):
        if manifest.exists():
            spec = json.loads(manifest.read_text())
            if Path(spec["filename"]).name != spec["filename"]:
                raise ValueError("Invalid cached CDSKIT model filename")
            if not _matches_selection(spec, model_family, feature_schema):
                raise ValueError(
                    "Frozen model selection does not match the requested family/schema; use a separate namespace"
                )
            path = cache / spec["filename"]
            verify(path, spec["sha256"])
            return path
        if not allow_download or os.environ.get("CDSKIT_OFFLINE", "").lower() in {
            "1",
            "true",
            "yes",
            "t",
            "y",
            "on",
        }:
            raise FileNotFoundError(
                "No selected CDSKIT model in cache and model download is disabled"
            )
        spec = (
            selector()
            if selector is not None
            else latest_model(
                model_family=model_family, feature_schema=feature_schema, opener=opener
            )
        )
        if not _matches_selection(spec, model_family, feature_schema):
            raise ValueError(
                "Selected checkpoint lacks the requested family/schema evidence"
            )
        path = cache / spec["filename"]
        # Publish the selection only after a complete, verified download.
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with (opener or open_url)(spec["url"]) as response:
                    total = 0
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        total += len(chunk)
                        if total > 2 * 1024**3:
                            raise ValueError("CDSKIT checkpoint exceeds 2 GiB")
                        stream.write(chunk)
                stream.close()
                verify(temporary, spec["sha256"])
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=cache, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(spec, stream, indent=2)
                stream.write("\n")
                stream.close()
                temporary.replace(manifest)
            finally:
                temporary.unlink(missing_ok=True)
        return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-download", choices=("yes", "no"), default="yes")
    args = parser.parse_args()
    print(resolve(args.cache_dir, args.model_download == "yes"))


if __name__ == "__main__":
    main()
