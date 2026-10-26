"""Keep sequence statistics equivalent across input sizes and representations."""

import random
from types import SimpleNamespace

from Bio.Seq import Seq
import pytest

from cdskit.stats import num_masked_bp, record_stats


def reference_counts(sequence):
    sequence = str(sequence)
    upper = sequence.upper()
    return {
        "bp_masked": sum(base in "abcdefghijklmnopqrstuvwxyz" for base in sequence),
        "bp_all": len(sequence),
        "bp_G": upper.count("G"),
        "bp_C": upper.count("C"),
        "bp_N": upper.count("N"),
        "bp_gap": sequence.count("-"),
    }


@pytest.mark.parametrize("length", [0, 1, 90, 8191, 8192, 8193, 65535, 65536, 65537])
@pytest.mark.parametrize("pattern", ["ACGT", "ACGTacgtNNry-?.", "n---rn", "gggg"])
def test_record_stats_preserves_counts_across_sizes(length, pattern):
    sequence = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
    actual = record_stats(SimpleNamespace(seq=Seq(sequence)))
    expected = reference_counts(sequence)
    assert actual == expected
    assert list(actual) == list(expected)
    assert all(type(value) is int for value in actual.values())


@pytest.mark.parametrize("length", [90, 9000, 70000])
def test_record_stats_accepts_all_ascii_characters(length):
    generator = random.Random(8472)
    sequence = "".join(chr(generator.randrange(128)) for _ in range(length))
    assert record_stats(SimpleNamespace(seq=sequence)) == reference_counts(sequence)


@pytest.mark.parametrize(
    "sequence",
    [
        "aéßŉng-C𐀀",
        "aéßŉng-C𐀀" * 10000,
        "g\ud800n-C",
        b"GCN-nAcg",
        Seq("GCN-nAcg"),
    ],
    ids=["unicode_short", "unicode_long", "surrogate", "bytes", "bio_seq"],
)
def test_record_stats_preserves_non_ascii_and_string_conversion(sequence):
    assert record_stats(SimpleNamespace(seq=sequence)) == reference_counts(sequence)
    assert num_masked_bp(sequence) == reference_counts(sequence)["bp_masked"]
