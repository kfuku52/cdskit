"""Incremental hashes of the established JSON row-list representation."""

import hashlib
import json


def json_rows_digest(rows, *, sort_keys=False):
    digest = hashlib.sha256()
    digest.update(b"[")
    separator = b""
    for row in rows:
        digest.update(separator)
        digest.update(json.dumps(row, ensure_ascii=True, sort_keys=sort_keys).encode())
        separator = b", "
    digest.update(b"]")
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
