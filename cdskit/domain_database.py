"""Resolve and atomically cache NCBI RPS-BLAST profile databases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse
from urllib.request import urlopen

from filelock import SoftFileLock

PFAM_URL = "https://ftp.ncbi.nlm.nih.gov/pub/mmdb/cdd/little_endian/Pfam_LE.tar.gz"
PFAM_FILE = re.compile(r"Pfam(?:\.\d+)?\.[A-Za-z0-9]+")
VOLUME_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class ProfileDatabase:
    prefix: Path
    identity: dict


def database_files(prefix: Path) -> list[Path]:
    """Check every volume behind a split or single-volume RPS database."""
    parent = prefix.parent
    alias = parent / (prefix.name + ".pal")
    names = [prefix.name]
    files = []
    if alias.is_file():
        files.append(alias)
        names = [
            name
            for line in alias.read_text(encoding="utf-8").splitlines()
            if (fields := shlex.split(line, comments=True)) and fields[0] == "DBLIST"
            for name in fields[1:]
        ]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Invalid database volume list: {alias}")
    for name in names:
        if not VOLUME_NAME.fullmatch(name):
            raise ValueError(f"Database volume must be in {parent}: {name}")
        for extension in (".loo", ".rps"):
            file = parent / (name + extension)
            if not file.is_file() or not file.stat().st_size:
                raise FileNotFoundError(f"Missing RPS-BLAST database volume: {file}")
        files.extend(f for f in parent.glob(name + ".*") if f.is_file() and f != alias)
    return sorted(set(files))


def _describe_database(prefix: Path) -> dict:
    return {
        f.name: {"bytes": (info := f.stat()).st_size, "mtime_ns": info.st_mtime_ns}
        for f in database_files(prefix)
    }


def _extract_pfam(archive_path: Path, staging: Path) -> None:
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not PFAM_FILE.fullmatch(path.name):
                continue
            if not member.isfile():
                raise ValueError(f"Expected a regular DB member: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unreadable DB member: {member.name}")
            with source, (staging / path.name).open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _probe_database(prefix: Path) -> None:
    subprocess.run(
        ["rpsblast", "-db", str(prefix), "-outfmt", "6"],
        input=b">probe\nMKALAEKLGIGALMNVQGATITQYQQTLN\n",
        stdout=subprocess.DEVNULL,
        check=True,
    )


def _download(url: str, staging: Path) -> dict:
    archive = staging / "source.tar.gz"
    digest = hashlib.sha256()
    scheme = urlparse(url).scheme
    with urlopen(url, timeout=60) as response, archive.open("wb") as handle:
        if urlparse(response.geturl()).scheme not in (scheme, "https"):
            raise ValueError(
                "Pfam database archive redirected to an unexpected scheme."
            )
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
    _extract_pfam(archive, staging)
    prefix = staging / "Pfam"
    files = _describe_database(prefix)
    _probe_database(prefix)
    archive.unlink()
    return dict(
        source_url=url,
        archive_sha256=digest.hexdigest(),
        files=files,
    )


def _cached_database(root: Path, url: str) -> ProfileDatabase:
    receipt = json.loads((root / "database.json").read_text(encoding="utf-8"))
    prefix = root / "Pfam"
    if receipt["source_url"] != url or receipt["files"] != _describe_database(prefix):
        raise ValueError(f"Cached RPS-BLAST database has changed: {root}")
    return ProfileDatabase(prefix, receipt)


def resolve_database(args) -> ProfileDatabase:
    if args.db_prefix:
        prefix = Path(args.db_prefix).resolve()
        return ProfileDatabase(prefix, {"files": _describe_database(prefix)})
    for candidate in ["/usr/local/db/Pfam_LE/Pfam", *(args.db_preferred or [])]:
        prefix = Path(candidate).resolve()
        if not any(
            (prefix.parent / (prefix.name + ext)).exists() for ext in (".pal", ".loo")
        ):
            continue
        return ProfileDatabase(prefix, {"files": _describe_database(prefix)})
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = Path(args.db_cache or cache_home / "cdskit" / "rpsblast").resolve()
    url = args.db_url or PFAM_URL
    if urlparse(url).scheme not in ("https", "http", "file"):
        raise ValueError(
            "RPS-BLAST database archive URL must use HTTPS, HTTP, or file."
        )
    root = cache / ("pfam-le-" + hashlib.sha256(url.encode()).hexdigest()[:16])
    if root.exists():
        return _cached_database(root, url)
    if args.no_download:
        raise FileNotFoundError(
            f"RPS-BLAST database is unavailable in {cache}; supply --db_prefix."
        )
    cache.mkdir(parents=True, exist_ok=True)
    # Use exclusive creation on the cache filesystem. Kernel flock can admit
    # multiple writers on shared mounts even when all callers receive success.
    with SoftFileLock(str(root) + ".lock", timeout=3600, poll_interval=0.25):
        if not root.exists():
            # A killed downloader can leave a staging directory. Only remove
            # directories covered by this database's lock.
            staging_prefix = f".{root.name}-download-"
            for abandoned in cache.glob(staging_prefix + "*"):
                if abandoned.is_dir() and not abandoned.is_symlink():
                    shutil.rmtree(abandoned)
            with tempfile.TemporaryDirectory(prefix=staging_prefix, dir=cache) as work:
                staging = Path(work) / "database"
                staging.mkdir()
                receipt = _download(url, staging)
                (staging / "database.json").write_text(
                    json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
                )
                staging.rename(root)
    return _cached_database(root, url)
