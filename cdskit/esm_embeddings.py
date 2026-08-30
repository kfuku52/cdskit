"""Bounded, label-independent cache for frozen ESM sequence embeddings."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def encoder_cache_key(
    source: str, revision: str | None, max_len: int, pooling: str
) -> str:
    local_files = []
    if revision is None:
        root = Path(source).expanduser().resolve()
        source = str(root)
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    info = path.stat()
                    local_files.append(
                        (str(path.relative_to(root)), info.st_size, info.st_mtime_ns)
                    )
    payload = [source, revision, int(max_len), pooling, local_files]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ESMEmbeddingCache:
    """An in-memory LRU shared across epochs, folds and classifier stages.

    Only frozen, unlabelled features are cached; no fitted head or training labels
    cross folds. The byte budget includes a conservative per-entry overhead.
    """

    def __init__(self, max_bytes: int = 256 * 1024 * 1024) -> None:
        if max_bytes < 0:
            raise ValueError("ESM embedding cache budget must be nonnegative.")
        self.max_bytes = int(max_bytes)
        self.size_bytes = 0
        self._entries: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()

    @staticmethod
    def _key(encoder: str, sequence: str) -> tuple[str, str]:
        return encoder, hashlib.sha256(sequence.encode()).hexdigest()

    def get(self, encoder: str, sequence: str) -> np.ndarray | None:
        key = self._key(encoder, sequence)
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(self, encoder: str, sequence: str, value: np.ndarray) -> None:
        key = self._key(encoder, sequence)
        size = value.nbytes + 512
        if size > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.size_bytes -= previous.nbytes + 512
        while self._entries and self.size_bytes + size > self.max_bytes:
            _, removed = self._entries.popitem(last=False)
            self.size_bytes -= removed.nbytes + 512
        copied = value.copy()
        copied.flags.writeable = False
        self._entries[key] = copied
        self.size_bytes += size
