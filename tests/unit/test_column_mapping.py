"""Independent enumeration oracle for column identifiability."""

from itertools import combinations, product

import pytest

from cdskit import column_mapping
from cdskit.column_mapping import (
    alignment_columns,
    analyze_mapping,
    read_kept_sites,
    validate_mapping,
)


def test_complete_mapping_matches_exhaustive_oracle():
    words = [
        list(chars) for length in range(5) for chars in product("AB", repeat=length)
    ]
    for source in words:
        for target in words:
            maps = [
                list(sites)
                for sites in combinations(range(len(source)), len(target))
                if [source[site] for site in sites] == target
            ]
            result = analyze_mapping(source, target)
            expected = (
                "unmatched" if not maps else "unique" if len(maps) == 1 else "ambiguous"
            )
            assert result.status == expected, (source, target, maps)
            if maps:
                assert result.leftmost == maps[0]
                assert result.rightmost == maps[-1]


def test_normalization_and_alignment_validation():
    assert alignment_columns(["a.", "A-"]) == ["AA", "--"]
    with pytest.raises(ValueError, match="aligned"):
        alignment_columns(["A", "AA"])


def test_ascii_column_transpose_preserves_normalization_across_blocks(monkeypatch):
    monkeypatch.setattr(column_mapping, "_COLUMN_BLOCK_BYTES", 512)
    monkeypatch.setattr(column_mapping, "_COLD_TRANSPOSE_MIN_CELLS", 0)
    rows = ["aM.x-?" * 120] * 64 + ["Tq.Y*n" * 120] * 32
    expected = ["".join(column) for column in zip(*rows, strict=True)]
    assert alignment_columns(rows) == [
        column.upper().replace(".", "-") for column in expected
    ]
    rows[-1] = rows[-1][:-1] + "ß"
    with pytest.raises(ValueError, match="zip"):
        alignment_columns(rows)


@pytest.mark.parametrize(
    "format_name,content",
    [
        ("indices0", "1\n3\n"),
        ("indices1", "2\n4\n"),
        ("clipkit-log", "1 trim nPI 0\n2 keep PI 0\n3 trim nPI 1\n4 keep PI 0\n"),
        ("trimal-colnumbering", "#ColumnsMap\t1, 3\n"),
    ],
)
def test_formats(tmp_path, format_name, content):
    path = tmp_path / "sites"
    path.write_text(content)
    assert read_kept_sites(str(path), format_name, 4) == [1, 3]


@pytest.mark.parametrize(
    "format_name,content",
    [
        ("indices0", "-1"),
        ("indices0", "4"),
        ("indices0", "1\n1"),
        ("indices0", "1.0"),
        ("clipkit-log", "1 keep PI 0\n"),
        ("trimal-colnumbering", "#ColumnsMap\t0, 1,\n"),
        ("unknown", "0"),
    ],
)
def test_invalid_maps_rejected(tmp_path, format_name, content):
    path = tmp_path / "sites"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_kept_sites(str(path), format_name, 4)


@pytest.mark.parametrize("sites", [[True], [0, 0], [-1], [2]])
def test_mapping_validation(sites):
    with pytest.raises(ValueError):
        validate_mapping(["A", "B"], ["A"], sites)


@pytest.mark.parametrize(
    "format_name,content",
    [
        ("indices0", ""),
        ("indices1", ""),
        ("clipkit-log", ""),
        ("trimal-colnumbering", "#ColumnsMap\n"),
    ],
)
def test_empty_maps(tmp_path, format_name, content):
    path = tmp_path / "sites"
    path.write_text(content)
    assert read_kept_sites(str(path), format_name, 0) == []


def test_clipkit_log_keeps_order_and_nonblank_line_numbers(tmp_path):
    path = tmp_path / "clipkit.log"
    path.write_text("1 trim PI 0\r\n\r\n2 keep PI 0\u20283 keep PI 0\v4 trim PI 0")
    assert read_kept_sites(str(path), "clipkit-log", 4) == [1, 2]
    path.write_text("1 trim PI 0\n3 keep PI 0\n2 keep PI 0\n4 trim PI 0")
    with pytest.raises(ValueError, match="every source AA column once, in order"):
        read_kept_sites(str(path), "clipkit-log", 4)


def test_map_read_error_takes_precedence_over_early_format_error(tmp_path):
    path = tmp_path / "clipkit.log"
    path.write_bytes(b"bad line\n" + b"\n" * 70_000 + b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_kept_sites(str(path), "clipkit-log", 4)
