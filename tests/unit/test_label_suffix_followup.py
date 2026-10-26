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
