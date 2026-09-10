"""Independent enumeration oracle for column identifiability."""

from itertools import combinations, product

import pytest

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
        ("indices1", "0"),
        ("indices0", "4"),
        ("indices0", "1\n1"),
        ("indices0", "2\n1"),
        ("indices0", "1.0"),
        ("indices0", "1 2"),
        ("indices0", "True"),
        ("clipkit-log", "1 keep PI 0\n"),
        ("clipkit-log", "1 keep PI 0\n1 keep PI 0\n"),
        ("clipkit-log", "1 kept PI 0\n"),
        ("trimal-colnumbering", "#ColumnsMap\t0, 1,\n"),
        ("trimal-colnumbering", "#ColumnsMap\t0\n#ColumnsMap\t1\n"),
        ("trimal-colnumbering", ">seq\nAA\n#ColumnsMap\t0"),
        ("trimal-colnumbering", "#ColumnsMapping\t0"),
        ("unknown", "0"),
    ],
)
def test_invalid_maps_rejected(tmp_path, format_name, content):
    path = tmp_path / "sites"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_kept_sites(str(path), format_name, 4)


@pytest.mark.parametrize("sites", [[True], [0, 0], [-1], [2], [0, 1], [1]])
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
