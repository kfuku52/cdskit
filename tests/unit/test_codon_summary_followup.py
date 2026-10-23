"""Independent row aggregation for bounded shared codon summaries."""

from itertools import product
import random
from types import SimpleNamespace

from Bio.Data.CodonTable import unambiguous_dna_by_id
import pytest

from cdskit import codonarray, codonutil


def reference_summary(sequence, code, policy):
    meanings = [
        codonutil.analyze_codon(sequence[start : start + 3], code)
        for start in range(0, len(sequence) - 2, 3)
    ]
    terminal = None
    if meanings and len(sequence) % 3 == 0:
        terminal = len(meanings) - 1
        if policy == "last_evaluable":
            terminal = next(
                (i for i in reversed(range(len(meanings))) if not meanings[i].missing),
                None,
            )
    internal = sum(m.definite_stop for i, m in enumerate(meanings) if i != terminal)
    return {
        "total": len(meanings),
        "clean": sum(m.clean for m in meanings),
        "missing": sum(m.missing for m in meanings),
        "ambiguous": sum(m.ambiguous or m.invalid for m in meanings),
        "stop": sum(m.definite_stop for m in meanings),
        "evaluable": sum(not m.missing for m in meanings),
        "internal_stop": internal > 0,
        "internal_stop_count": internal,
        "possible_stop": sum(m.possible_stop for m in meanings),
        "context_dependent": sum(m.context_dependent for m in meanings),
        "internal_possible_stop_count": sum(
            m.possible_stop for i, m in enumerate(meanings) if i != terminal
        ),
        "internal_context_dependent_count": sum(
            m.context_dependent for i, m in enumerate(meanings) if i != terminal
        ),
    }


@pytest.mark.parametrize("code", sorted(unambiguous_dna_by_id))
@pytest.mark.parametrize("policy", ["physical", "last_evaluable"])
def test_long_all_triplets_keep_independent_attributes_and_python_types(code, policy):
    sequence = "".join(map("".join, product("ACGTRYSWKMBDHVNX-?.", repeat=3)))
    for value in (sequence, sequence.lower()):
        actual = codonutil.summarize_codons(value, code, policy)
        expected = reference_summary(value, code, policy)
        assert actual == expected
        assert list(actual) == list(expected)
        assert all(type(actual[key]) is type(expected[key]) for key in expected)


@pytest.mark.parametrize("code", [1, 2, 27, 28, 31])
@pytest.mark.parametrize("policy", ["physical", "last_evaluable"])
def test_last_evaluable_and_physical_boundaries_across_blocks(code, policy):
    block_codons = 65536
    cases = [
        "---" * (block_codons + 1),
        "TAR" + "---" * (block_codons + 1),
        "TGA" + "..." * (block_codons + 1),
        "ATG" * (block_codons - 1) + "TAR???---",
        "---" * block_codons + "TAN",
    ]
    for sequence in cases:
        assert codonutil.summarize_codons(sequence, code, policy) == reference_summary(
            sequence, code, policy
        )


@pytest.mark.parametrize("length", [6141, 6144, 6147, 196605, 196608, 196611])
def test_threshold_and_blocks_keep_frame_and_ignore_partial_tail(length):
    rng = random.Random(length)
    sequence = "".join(rng.choices("ACGTN-?.", k=length))
    for tail in ("", "A", "TA"):
        value = sequence + tail
        for policy in ("physical", "last_evaluable"):
            assert codonutil.summarize_codons(value, 1, policy) == reference_summary(
                value, 1, policy
            )


@pytest.mark.parametrize("bad", ["!", "U", "é", "\ud800"])
def test_invalid_and_unicode_keep_scalar_semantics(bad):
    for sequence in (bad * 3 + "ATG" * 3000, "ATG" * 3000 + bad * 3):
        assert codonutil.summarize_codons(sequence, 1) == reference_summary(
            sequence, 1, "last_evaluable"
        )


def test_custom_string_slicing_keeps_scalar_path(monkeypatch):
    class CustomString(str):
        def __getitem__(self, item):
            value = super().__getitem__(item)
            return value.replace("A", "N")

    def forbidden(*args):
        raise AssertionError("custom strings must retain scalar processing")

    monkeypatch.setattr(codonarray, "codon_histogram", forbidden)
    sequence = CustomString("ATG" * 3000)
    assert codonutil.summarize_codons(sequence, 1) == reference_summary(
        sequence, 1, "last_evaluable"
    )


def test_last_evaluable_scan_bounds_temporary_rows(monkeypatch):
    observed = []
    original = codonarray.codon_features

    def inspect(sequence, code):
        observed.append(len(sequence))
        return original(sequence, code)

    monkeypatch.setattr(codonarray, "codon_features", inspect)
    sequence = "ATGTAA" + "---" * 140000
    result = codonutil.summarize_codons(sequence, 1)
    assert not result["internal_stop"]
    assert len(observed) == 3
    assert max(observed) <= 196608


def test_small_empty_and_invalid_policy_contracts():
    assert codonutil.summarize_codons("", "not-a-code") == reference_summary(
        "", "not-a-code", "last_evaluable"
    )
    with pytest.raises(ValueError, match="Unknown terminal policy: invalid"):
        codonutil.summarize_codons("ATG" * 3000, 1, "invalid")


@pytest.mark.parametrize("numpy_loaded", [False, True])
def test_cold_import_gate_uses_measured_sizes(monkeypatch, numpy_loaded):
    calls = []
    original = codonarray.codon_histogram

    def inspect(sequence, code):
        calls.append(len(sequence))
        return original(sequence, code)

    monkeypatch.setattr(codonarray, "codon_histogram", inspect)
    monkeypatch.setattr(
        codonutil,
        "sys",
        SimpleNamespace(modules={"numpy": object()} if numpy_loaded else {}),
    )
    for length in (6141, 6144, 300000, 2999997, 3000000):
        result = codonutil.summarize_codons("ATG" * (length // 3), 1)
        assert result["total"] == result["clean"] == length // 3
    assert calls == ([6144, 300000, 2999997, 3000000] if numpy_loaded else [3000000])
