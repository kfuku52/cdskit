"""Read compressed annotation text independently of pipeline settings."""

import bz2
import gzip
from pathlib import Path


def open_text(path, mode="rt", errors="strict"):
    path = Path(path)
    with path.open("rb") as probe:
        magic = probe.read(3)
    opener = (
        gzip.open
        if magic[:2] == b"\x1f\x8b" or path.suffix == ".gz"
        else bz2.open
        if magic == b"BZh" or path.suffix == ".bz2"
        else open
    )
    return opener(path, mode, encoding="utf-8", errors=errors)
