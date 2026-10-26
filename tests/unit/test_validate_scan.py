from itertools import product
import random

from Bio.Seq import Seq
import pytest

from cdskit import validate
from cdskit.codonutil import summarize_codons


def reference_counts(sequence):
    ambiguous = 0
    evaluable = 0
    for start in range(0, len(sequence) - 2, 3):
        codon = sequence[start : start + 3]
        if any(char in "-?." for char in codon):
            continue
        evaluable += 1
        ambiguous += any(char not in "ACGTacgt" for char in codon)
    return ambiguous, evaluable


@pytest.mark.parametrize(
    "sequence",
    [
        "",
        "A",
        "NN",
        "ACGacg---..?NNNXxXTAR!aC",
        "-?.NXnx" * 1000,
        "ACGTacgt" * 1000,
        "ATGGCNTAR---...?NNNacg" * 1000 + "N?",
        "ACGé中N---" * 1000 + "?",
        "A\ud800N" * 1000,
        "".join(map("".join, product("ACGTacgtNXry-?.!", repeat=3))),
    ],
)
def test_ambiguity_scan_matches_independent_reference(sequence):
    result = validate.sequence_ambiguous_codon_counts(sequence)
    assert result == reference_counts(sequence)
    assert type(result) is tuple
    assert all(type(value) is int for value in result)


@pytest.mark.parametrize("length", [383, 384, 385, 786431, 786432, 786433, 786437])
def test_ambiguity_scan_preserves_frame_at_fast_path_and_chunk_boundaries(length):
    pattern = "ACG-NNXx?.taa!"
    sequence = (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]
    assert validate.sequence_ambiguous_codon_counts(sequence) == reference_counts(
        sequence
    )


def test_ambiguity_scan_handles_every_ascii_character():
    rng = random.Random(20261023)
    sequence = "".join(chr(rng.randrange(128)) for _ in range(30000))
    assert validate.sequence_ambiguous_codon_counts(sequence) == reference_counts(
        sequence
    )


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ("", False),
        ("-?.NXnx", True),
        ("-?.NXnx" * 1000, True),
        ("-?.NXnx" * 1000 + "A", False),
        ("A" + "-?.NXnx" * 1000, False),
        ("-" + "ACG" * 10000, False),
        ("é" + "-?.NXnx" * 1000, False),
        ("-?.NXnx" * 1000 + "中", False),
        (Seq("nN-x?"), True),
        (["-", "N", "?"], True),
        (b"---", False),
    ],
)
def test_gap_scan_preserves_unicode_and_sequence_types(sequence, expected):
    result = validate.is_gap_only_sequence(sequence)
    assert result is expected


@pytest.mark.parametrize(
    "sequence",
    [Seq("ACGNnN---" * 100), tuple("ACGNnN---" * 100)],
    ids=["BioSeq", "tuple"],
)
def test_ambiguity_scan_preserves_non_string_sequences(sequence):
    assert validate.sequence_ambiguous_codon_counts(sequence) == reference_counts(
        sequence
    )


@pytest.mark.parametrize("sequence", [None, 1, ["A", "C", "G"]])
def test_ambiguity_scan_preserves_invalid_input_type_errors(sequence):
    with pytest.raises(TypeError):
        validate.sequence_ambiguous_codon_counts(sequence)


@pytest.mark.parametrize("code", [1, 27, 28, 31])
@pytest.mark.parametrize("tail", ["", "A", "N?"])
def test_scan_counts_preserve_legacy_and_code_aware_summary_contracts(code, tail):
    sequence = "atgTARTGANNn---..?TAAtga" * 40 + tail
    counts = reference_counts(sequence)
    legacy = validate.summarize_single_sequence(
        "example", sequence, stop_codons=validate.get_stop_codons(code)
    )
    modern = validate.summarize_single_sequence("example", sequence, codontable=code)
    semantics = summarize_codons(sequence, code)
    assert type(legacy) is tuple and len(legacy) == 6
    assert type(modern) is tuple and len(modern) == 8
    assert legacy[4:6] == modern[4:6] == counts
    assert modern[3] == semantics["internal_stop"]
    assert modern[6:] == (semantics["possible_stop"], semantics["context_dependent"])
