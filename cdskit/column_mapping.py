"""Explicit alignment-column maps and complete subsequence correspondence.

Coordinates are zero-based aligned amino-acid columns, including gap columns.
"""

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
import re
import sys

_COLUMN_BLOCK_BYTES = 1024 * 1024
_COLD_TRANSPOSE_MIN_CELLS = 8_000_000


@dataclass(frozen=True)
class MappingAnalysis:
    status: str
    leftmost: list[int]
    rightmost: list[int]


def alignment_columns(sequences: list[str]) -> list[str]:
    if len({len(sequence) for sequence in sequences}) > 1:
        raise ValueError("Sequences must be aligned before column mapping.")
    normalized = [sequence.upper().replace(".", "-") for sequence in sequences]
    cells = len(normalized) * len(normalized[0]) if normalized else 0
    if (
        type(sequences) in (list, tuple)
        and len(normalized) >= 16
        and cells >= 65_536
        and ("numpy" in sys.modules or cells >= _COLD_TRANSPOSE_MIN_CELLS)
        # ASCII normalization keeps the original widths. Unicode uppercasing
        # may expand a residue, so preserve the strict zip path in that case.
        and all(type(row) is str and row.isascii() for row in sequences)
    ):
        return _ascii_columns(normalized)
    return ["".join(column) for column in zip(*normalized, strict=True)]


def _ascii_columns(rows: list[str]) -> list[str]:
    import numpy as np

    width = len(rows[0])
    height = len(rows)
    block_width = max(1, _COLUMN_BLOCK_BYTES // height)
    columns: list[str] = []
    for start in range(0, width, block_width):
        stop = min(start + block_width, width)
        block = "".join(row[start:stop] for row in rows).encode("ascii")
        transposed = np.frombuffer(block, dtype=np.uint8).reshape(height, -1).T
        text = transposed.tobytes().decode("ascii")
        columns.extend(text[i : i + height] for i in range(0, len(text), height))
    return columns


def _earliest(source: list[str], target: list[str]) -> list[int]:
    sites = []
    cursor = 0
    source_length = len(source)
    for column in target:
        while cursor < source_length and source[cursor] != column:
            cursor += 1
        if cursor == source_length:
            return []
        sites.append(cursor)
        cursor += 1
    return sites


def analyze_mapping(source: list[str], target: list[str]) -> MappingAnalysis:
    """Classify complete order-preserving maps; local duplicates are insufficient.

    Every complete map is componentwise bounded by the earliest and latest maps.
    Thus their equality proves uniqueness without enumerating all embeddings.
    """
    left = _earliest(source, target)
    if len(left) != len(target):
        return MappingAnalysis("unmatched", [], [])
    # A complete forward map guarantees that the backward scan finds a match
    # for each column. Keep the original column indexes throughout the scan.
    right = []
    cursor = len(source) - 1
    for column in reversed(target):
        while source[cursor] != column:
            cursor -= 1
        right.append(cursor)
        cursor -= 1
    right.reverse()
    return MappingAnalysis("unique" if left == right else "ambiguous", left, right)


def validate_sites(sites: list[int], source_length: int) -> None:
    if any(
        type(site) is not int or site < 0 or site >= source_length for site in sites
    ):
        raise ValueError("Kept sites must be integers within the source AA alignment.")
    if any(first >= second for first, second in pairwise(sites)):
        raise ValueError("Kept sites must be strictly increasing without duplicates.")


def _integer(value: str) -> int:
    if re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"Invalid column index: {value!r}.")
    return int(value)


def _clipkit_sites(lines: list[str], source_length: int) -> list[int]:
    sites = []
    positions = []
    for line in lines:
        fields = line.split()
        if len(fields) != 4 or fields[1] not in {"keep", "trim"}:
            raise ValueError(
                "Expected a four-column ClipKIT log with keep/trim actions."
            )
        position = _integer(fields[0]) - 1
        positions.append(position)
        if fields[1] == "keep":
            sites.append(position)
    if positions != list(range(source_length)):
        raise ValueError(
            "ClipKIT log must cover every source AA column once, in order."
        )
    return sites


def read_kept_sites(path: str, format_name: str, source_length: int) -> list[int]:
    """Parse explicitly selected formats; never guess coordinate origin.

    trimAl input is the standalone #ColumnsMap record, not mixed stdout.
    """
    lines = [
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if format_name in {"indices0", "indices1"}:
        origin = int(format_name[-1])
        sites = [_integer(line) - origin for line in lines]
    elif format_name == "clipkit-log":
        sites = _clipkit_sites(lines, source_length)
    elif format_name == "trimal-colnumbering":
        if len(lines) != 1 or not re.match(r"^#ColumnsMap(?:\s|$)", lines[0]):
            raise ValueError("Expected exactly one trimAl #ColumnsMap record.")
        payload = lines[0][len("#ColumnsMap") :].strip()
        sites = (
            [_integer(item.strip()) for item in payload.split(",")] if payload else []
        )
    else:
        raise ValueError("Specify --kept_sites_format for --kept_sites.")
    validate_sites(sites, source_length)
    return sites


def validate_mapping(source: list[str], target: list[str], sites: list[int]) -> None:
    validate_sites(sites, len(source))
    if len(sites) != len(target):
        raise ValueError("Kept site count must equal the trimmed AA alignment length.")
    if [source[site] for site in sites] != target:
        raise ValueError("Selected columns do not match the trimmed AA alignment.")
