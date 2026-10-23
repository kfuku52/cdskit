"""Clipped duplicate IDs share suffixes without changing their assignment order."""

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pytest

from cdskit.label import uniquify_label_ids


def _records(names):
    return [
        SeqRecord(
            Seq("ATG"),
            id=name,
            name=f"record_{index}",
            description=f"description {index}",
        )
        for index, name in enumerate(names)
    ]


def test_distinct_originals_share_clipped_suffixes_and_reserved_gaps():
    names = ["AAA_1", "AAA_3", "AAA_7", "AA_10", "AA_12"] + [
        name for name in ("AAA00", "AAA01", "AAA02", "AAA03") for _ in range(2)
    ]
    records = _records(names)

    result = uniquify_label_ids(records, clip_len=5)

    assert result == (8, ["AAA00", "AAA01", "AAA02", "AAA03"])
    assert [record.id for record in records] == [
        *names[:5],
        "AAA_2",
        "AAA_4",
        "AAA_5",
        "AAA_6",
        "AAA_8",
        "AAA_9",
        "AA_11",
        "AA_13",
    ]
    assert [record.description for record in records] == [
        f"description {index}" for index in range(5)
    ] + [""] * 8
    assert [record.name for record in records] == [
        f"record_{index}" for index in range(len(names))
    ]
    assert all(str(record.seq) == "ATG" for record in records)


def test_shared_suffixes_preserve_prefix_changes_across_three_digit_widths():
    records = _records(
        [name for index in range(55) for name in (f"AAA{index:02}",) * 2]
    )

    result = uniquify_label_ids(records, clip_len=5)

    assert result == (110, [f"AAA{index:02}" for index in range(55)])
    assert [record.id for record in records] == [
        ("AAA" if index < 10 else "AA" if index < 100 else "A") + f"_{index}"
        for index in range(1, 111)
    ]


def test_shared_clipped_suffix_namespaces_keep_unrelated_prefixes_independent():
    records = _records(["AAA00", "BBB00", "AAA01", "AAA00", "BBB00", "AAA01"])

    result = uniquify_label_ids(records, clip_len=5)

    assert result == (6, ["AAA00", "BBB00", "AAA01"])
    assert [record.id for record in records] == [
        "AAA_1",
        "BBB_1",
        "AAA_2",
        "AAA_3",
        "BBB_2",
        "AAA_4",
    ]


@pytest.mark.parametrize("clip_len", [0, 60])
def test_suffixes_with_no_effective_clipping_preserve_original_names(clip_len):
    records = _records(["AAA00_1", "AAA00", "AAA01", "AAA00", "AAA01"])

    result = uniquify_label_ids(records, clip_len=clip_len)

    assert result == (4, ["AAA00", "AAA01"])
    assert [record.id for record in records] == [
        "AAA00_1",
        "AAA00_2",
        "AAA01_1",
        "AAA00_3",
        "AAA01_2",
    ]
    assert records[0].description == "description 0"
    assert all(record.description == "" for record in records[1:])


def test_exhausted_clipped_suffixes_preserve_mutations_before_error():
    names = ["AA", "AA", "AB", "AB", "BA", "BA", "BB", "BB", "AC", "AC"]
    records = _records(names)

    with pytest.raises(
        ValueError,
        match=r"^--clip_len=2 is too small to create unique labels\.$",
    ):
        uniquify_label_ids(records, clip_len=2)

    assert [record.id for record in records] == [
        f"_{index}" for index in range(1, 10)
    ] + ["AC"]
    assert [record.description for record in records] == [""] * 9 + ["description 9"]


class _UnhashableSlice(str):
    def __eq__(self, other):
        return str.__eq__(self, other)


class _UnhashableSliceId(str):
    def __getitem__(self, key):
        return _UnhashableSlice(str.__getitem__(self, key))


class _EqualSlice(str):
    def __eq__(self, other):
        return isinstance(other, str)

    def __hash__(self):
        return 42


class _EqualSliceId(str):
    def __getitem__(self, key):
        return _EqualSlice(str.__getitem__(self, key))


class _StatefulSliceId(str):
    def __new__(cls, value):
        result = super().__new__(cls, value)
        result.slices = 0
        return result

    def __getitem__(self, key):
        self.slices += 1
        return "AAA" if self.slices == 1 else "BBB"


class _StatefulClip(int):
    def __new__(cls):
        result = super().__new__(cls, 5)
        result.calls = 0
        return result

    def __sub__(self, other):
        self.calls += 1
        return 3 if self.calls <= 8 else 2


def test_unhashable_string_subclass_slices_keep_literal_suffix_search():
    records = _records(
        ["AAA_1", _UnhashableSliceId("AAA00"), _UnhashableSliceId("AAA00")]
    )

    assert uniquify_label_ids(records, clip_len=5) == (2, ["AAA00"])
    assert [record.id for record in records] == ["AAA_1", "AAA_2", "AAA_3"]


def test_equal_string_subclass_slices_keep_each_original_suffix_search():
    reserved = [*[f"AAA_{index}" for index in range(1, 6)], "BBB_1"]
    records = _records(
        [
            *reserved,
            *[_EqualSliceId(name) for name in ["AAA00", "AAA00", "BBB00", "BBB00"]],
        ]
    )

    assert uniquify_label_ids(records, clip_len=5) == (4, ["AAA00", "BBB00"])
    assert [record.id for record in records] == [
        *reserved,
        "AAA_6",
        "AAA_7",
        "BBB_2",
        "BBB_3",
    ]


def test_stateful_string_subclass_slices_preserve_each_slice_call():
    reserved = [f"AAA_{index}" for index in range(1, 6)]
    records = _records(
        [
            *reserved,
            "AAA00",
            "AAA00",
            _StatefulSliceId("BBB00"),
            _StatefulSliceId("BBB00"),
        ]
    )

    assert uniquify_label_ids(records, clip_len=5) == (4, ["AAA00", "BBB00"])
    assert [record.id for record in records] == [
        *reserved,
        "AAA_6",
        "AAA_7",
        "BBB_2",
        "BBB_4",
    ]


def test_integer_subclass_clipping_preserves_each_subtraction_call():
    reserved = [f"AAA_{index}" for index in range(1, 6)]
    records = _records([*reserved, "AAA00", "AAA00", "AAA01", "AAA01"])

    assert uniquify_label_ids(records, clip_len=_StatefulClip()) == (
        4,
        ["AAA00", "AAA01"],
    )
    assert [record.id for record in records] == [
        *reserved,
        "AAA_6",
        "AAA_7",
        "AA_2",
        "AA_3",
    ]
