"""Independent NCBI/IUPAC expansion reference for histogram aggregation."""

from collections import Counter
from itertools import product
from types import SimpleNamespace

from Bio.Data import CodonTable
from Bio.Data.IUPACData import ambiguous_dna_values
import numpy as np
import pytest

from cdskit import codonarray
from cdskit.codonstats import summarize_record


def _reference(sequence, code):
    table = CodonTable.unambiguous_dna_by_id[code]
    usage = Counter()
    complete = missing = ambiguous = stop = possible = dual = 0
    gc = [0, 0, 0]
    denominators = [0, 0, 0]
    for start in range(0, len(sequence), 3):
        codon = sequence[start : start + 3].upper()
        for pos, base in enumerate(codon):
            if base in "ACGT":
                denominators[pos] += 1
                gc[pos] += base in "GC"
        if any(base in "-.?" for base in codon):
            missing += 1
            continue
        concrete = all(base in "ACGT" for base in codon)
        if len(codon) < 3:
            if concrete:
                usage[codon] += 1
            continue
        complete += 1
        if concrete:
            usage[codon] += 1
        else:
            ambiguous += 1
        if all(base in ambiguous_dna_values for base in codon):
            expansions = [
                "".join(letters)
                for letters in product(*(ambiguous_dna_values[base] for base in codon))
            ]
            definite = all(
                triplet in table.stop_codons and triplet not in table.forward_table
                for triplet in expansions
            )
            stop += definite
            possible += (
                any(triplet in table.stop_codons for triplet in expansions)
                and not definite
            )
            dual += any(
                triplet in table.stop_codons and triplet in table.forward_table
                for triplet in expansions
            )

    def percent(numerator, denominator):
        return 100 * (numerator / denominator) if denominator else 0.0

    return {
        "seq_id": "record",
        "nt_length": len(sequence),
        "codons_total": len(sequence) // 3,
        "codons_complete": complete,
        "codons_missing": missing,
        "codons_ambiguous": ambiguous,
        "codons_stop": stop,
        "codons_possible_stop": possible,
        "codons_context_dependent": dual,
        "codon_semantics_version": "2",
        "gc_all": percent(sum(gc), sum(denominators)),
        "gc1": percent(gc[0], denominators[0]),
        "gc2": percent(gc[1], denominators[1]),
        "gc3": percent(gc[2], denominators[2]),
        "usage": usage,
    }


def _assert_summary(actual, expected):
    assert actual == expected
    assert list(actual) == list(expected)
    assert [type(value) for value in actual.values()] == [
        type(value) for value in expected.values()
    ]
    assert list(actual["usage"].items()) == list(expected["usage"].items())
    assert all(type(value) is int for value in actual["usage"].values())


@pytest.mark.parametrize("code", sorted(CodonTable.unambiguous_dna_by_id))
def test_all_triplets_and_table_key_forms_match_independent_reference(code):
    sequence = "".join(
        "".join(letters) for letters in product(".X?N-VHDBMKWSYRGTCA", repeat=3)
    )
    record = SimpleNamespace(seq=sequence.lower(), id="record")
    expected = _reference(sequence, code)
    name = CodonTable.unambiguous_dna_by_id[code].names[0]
    for key in (code, str(code), name):
        _assert_summary(summarize_record(record, key), expected)


@pytest.mark.parametrize(
    "suffix",
    ["", "A", "AT", "N", "?", "ZUA", "A@G", "A\N{GREEK CAPITAL LETTER OMEGA}G"],
)
def test_partial_invalid_and_unicode_direct_inputs_keep_scalar_contract(suffix):
    sequence = "TARtgaGCN---ATG" * 60 + suffix
    _assert_summary(
        summarize_record(SimpleNamespace(seq=sequence, id="record"), 28),
        _reference(sequence, 28),
    )


@pytest.mark.parametrize("length", [255, 256, 257])
def test_scalar_histogram_boundary_matches_reference(length):
    sequence = ("AGCNNNTAR---" * ((length + 3) // 4))[: length * 3]
    _assert_summary(
        summarize_record(SimpleNamespace(seq=sequence, id="record"), 1),
        _reference(sequence, 1),
    )


@pytest.mark.parametrize("sequence", ["", "NNN" * 300, "---" * 300, "???...NNN" * 100])
def test_zero_gc_denominators_and_empty_usage(sequence):
    actual = summarize_record(SimpleNamespace(seq=sequence, id="record"), 1)
    _assert_summary(actual, _reference(sequence, 1))
    assert actual["usage"] == Counter()
    assert [actual[key] for key in ["gc_all", "gc1", "gc2", "gc3"]] == [0.0] * 4


def test_blocks_bound_encoding_and_preserve_late_usage_order(monkeypatch):
    sequence = "TGA" * 65536 + "ATGCCC" + "TGA" * 65536 + "aaaTAA"
    calls = []
    encode = codonarray._triplet_indices

    def record_block(block):
        calls.append(len(block))
        assert len(block) <= 65536 * 3
        return encode(block)

    monkeypatch.setattr(codonarray, "_triplet_indices", record_block)
    actual = summarize_record(SimpleNamespace(seq=sequence, id="record"), 27)
    _assert_summary(actual, _reference(sequence, 27))
    assert len(calls) == 3
    assert list(actual["usage"]) == ["TGA", "ATG", "CCC", "AAA", "TAA"]
    features, states = codonarray.codon_features("TARtgaGCN---", 27)
    np.testing.assert_array_equal(
        features,
        [
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
    )
    np.testing.assert_array_equal(states, [2, 0, 2, 1])
