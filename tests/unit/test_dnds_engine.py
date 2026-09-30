import itertools
import random

import numpy as np
import pytest

from cdskit.codonutil import analyze_codon, get_forward_table, has_internal_stop
from cdskit.dnds import (
    BASES,
    _batch_size,
    _codon_lookup,
    _encode_batch,
    _encode_pair,
    _geometry,
    _finite_column,
    estimate_pairs,
)


def synthetic_pair(code=1, seed=10, size=300, change=0.15):
    rng = random.Random(seed)
    sense = sorted(get_forward_table(code))
    first = [rng.choice(sense) for _ in range(size)]
    second = [rng.choice(sense) if rng.random() < change else codon for codon in first]
    return "".join(first), "".join(second)


@pytest.mark.parametrize("code", [1, 2, 6, 27, 28, 31])
def test_precomputed_equal_sense_paths_match_independent_enumeration(code):
    forward = get_forward_table(code)
    geometry = _geometry(code)
    for first, second in itertools.product(forward, repeat=2):
        changed = [i for i in range(3) if first[i] != second[i]]
        counts = []
        for order in itertools.permutations(changed):
            path = [first]
            for position in order:
                prior = path[-1]
                path.append(prior[:position] + second[position] + prior[position + 1 :])
            if any(codon not in forward for codon in path):
                continue
            result = [0, 0, 0, 0]
            for a, b in itertools.pairwise(path):
                from_base, to_base = next(
                    (x, y) for x, y in zip(a, b, strict=True) if x != y
                )
                transition = (from_base in "TC" and to_base in "TC") or (
                    from_base in "AG" and to_base in "AG"
                )
                index = (0 if forward[a] == forward[b] else 2) + (
                    0 if transition else 1
                )
                result[index] += 1
            counts.append(result)
        expected = (
            np.mean(counts, axis=0) if counts else [0, 0, 0.5, len(changed) - 0.5]
        )

        def encode(codon):
            return sum(
                BASES.index(base) * weight
                for base, weight in zip(codon, (16, 4, 1), strict=True)
            )

        assert geometry["differences"][
            encode(first) * 64 + encode(second)
        ] == pytest.approx(expected)


def test_identical_diverse_cds_has_zero_distances_not_missing():
    first, _ = synthetic_pair()
    result = estimate_pairs([(first, first)])[0]
    assert result["dS"] == result["dN"] == 0
    assert result["status"] == "ok"
    assert result["kappa_defaulted"] is True
    assert result["S"] + result["N"] == pytest.approx(len(first))


@pytest.mark.parametrize("code", [1, 2, 6, 27, 28, 31])
def test_batched_encoding_matches_individual_pair_semantics(code):
    pairs = [synthetic_pair(code, seed=i, size=i + 1) for i in range(32)]
    pairs += [
        ("", ""),
        ("NNN---", "NNNATG"),
        ("ATGTAANNN", "ATGAAANNN"),
        ("ATGTAA---", "ATGAAA---"),
        ("ATGTAAATG", "ATGAAAATG"),
        ("ATGTARGCT", "ATGAAAGCT"),
        ("ATGTRA---", "ATGAAA---"),
        ("ATGAGRNNN", "ATGAAANNN"),
        ("augGCN?.-gct", "ATGGCT---gcc"),
    ]
    geometry = _geometry(code)
    for batch, individual in zip(
        _encode_batch(pairs, geometry),
        [_encode_pair(a, b, geometry) for a, b in pairs],
        strict=True,
    ):
        assert batch[1] == individual[1]
        if individual[0] is None:
            assert batch[0] is None
        else:
            np.testing.assert_array_equal(batch[0], individual[0])
    assert _encode_batch([], geometry) == []


@pytest.mark.parametrize("invalid", ["ATG!AA", "ATGéAA", "ATG1AA", "ATG\u017fAA"])
def test_batch_invalid_alphabet_never_becomes_missing(invalid):
    with pytest.raises(ValueError, match="Invalid DNA alphabet"):
        estimate_pairs([("ATGATG", "ATGATG"), (invalid, "ATGAAA")])


def test_packed_codon_lookup_exhaustively_preserves_ascii_classification():
    # All 128**3 triplets, including controls and invalid punctuation. Lookup
    # validation must reject invalid+ambiguous codons, not hide them as missing.
    keys = np.arange(1 << 21, dtype=np.int32)
    alphabet = _geometry(1)["alphabet"]
    bases = np.stack([alphabet[(keys >> shift) & 127] for shift in (14, 7, 0)], axis=1)
    values = (bases * (16, 4, 1)).sum(axis=1)
    values[np.any(bases < 0, axis=1)] = -1
    values[np.any(bases < -1, axis=1)] = -2
    np.testing.assert_array_equal(_codon_lookup(), values)
    assert not _codon_lookup().flags.writeable
    assert _geometry(1)["codon_lookup"] is _geometry(2)["codon_lookup"]


@pytest.mark.parametrize("code", [1, 2, 6, 27, 28, 31])
def test_symmetric_class_projection_preserves_all_matrix_entries(code):
    geometry = _geometry(code)
    counts = np.random.default_rng(code).integers(0, 1000, size=(7, 4096))
    expected = (counts @ geometry["classes"]).reshape(-1, 2, 4, 4)
    projected = (counts @ geometry["class_projection"]).reshape(-1, 2, 10)
    actual = np.empty_like(expected)
    a, b = geometry["class_triangle"]
    actual[:, :, a, b] = projected
    actual[:, :, b, a] = projected
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "pairs,workers,size",
    [
        (55006, 1, 256),
        (55006, 2, 512),
        (55006, 4, 1024),
        (1000, 4, 256),
        (520, 2, 256),
        (55006, 64, 256),
    ],
)
def test_parallel_batch_size_is_bounded_and_keeps_small_workloads_parallel(
    pairs, workers, size
):
    assert _batch_size(pairs, workers) == size


def test_result_columns_keep_none_separate_from_zero_and_native_python_types():
    values = np.array([0.0, 1.0, np.nan, np.inf, -np.inf])
    assert _finite_column(values) == [0.0, 1.0, None, None, None]
    assert _finite_column(values, np.array([True, False, True, True, True])) == [
        0.0,
        None,
        None,
        None,
        None,
    ]
    assert type(_finite_column(values)[0]) is float


@pytest.mark.parametrize(
    "first,second,status",
    [
        ("NNN---", "NNNATG", "no_aligned_sense_codons"),
        ("ATGTAAATG", "ATGAAAATG", "internal_stop"),
        ("ATGTAANNN", "ATGAAANNN", "internal_stop"),
        ("ATG", "ATG", "no_synonymous_sites"),
    ],
)
def test_missing_and_invalid_estimates_have_explicit_status(first, second, status):
    result = estimate_pairs([(first, second)])[0]
    assert result["status"] == status
    assert result["dS"] is None


@pytest.mark.parametrize("code,stop", [(1, "TAR"), (1, "TRA"), (2, "AGR")])
@pytest.mark.parametrize("tail", ["GCT", "NNN"])
def test_ambiguous_definite_internal_stops_cannot_be_deleted_as_uncertainty(
    code, stop, tail
):
    sequence, _ = synthetic_pair(code=code, size=60)
    first, second = sequence + stop + tail, sequence + "GCT" + tail
    assert has_internal_stop(first, code)
    result = estimate_pairs([(first, second)], code)[0]
    assert result["status"] == "internal_stop"
    assert result["retained_codons"] == 0
    assert result["dS"] is result["dN"] is None


@pytest.mark.parametrize("code,stop", [(1, "TAR"), (1, "TRA"), (2, "AGR")])
def test_ambiguous_definite_terminal_stop_preserves_last_evaluable_policy(code, stop):
    sequence, _ = synthetic_pair(code=code, size=60)
    first, second = sequence + stop + "---", sequence + "GCT---"
    assert not has_internal_stop(first, code)
    result = estimate_pairs([(first, second)], code)[0]
    reference = estimate_pairs([(sequence, sequence)], code)[0]
    assert result["status"] == "ok"
    assert result["retained_codons"] == 60
    assert result["excluded_codons"] == 2
    assert result["dS"] == reference["dS"]
    assert result["dN"] == reference["dN"]


@pytest.mark.parametrize(
    "code,codon", [(1, "TAN"), (1, "AGR"), (27, "TAR"), (28, "TRA"), (31, "TAR")]
)
def test_possible_or_context_dependent_stops_remain_uncertain_not_internal(code, codon):
    sequence, _ = synthetic_pair(code=code, size=60)
    first, second = sequence + codon + "GCT", sequence + "GCTGCT"
    assert not has_internal_stop(first, code)
    result = estimate_pairs([(first, second)], code)[0]
    assert result["status"] == "ok"
    assert result["retained_codons"] == 61


@pytest.mark.parametrize("code", [1, 2, 6, 27, 28, 31])
def test_all_iupac_triplets_share_definite_stop_semantics(code):
    # Independent shared-codon analysis prevents the bulk encoder and its
    # reference path from agreeing on the same incomplete stop classification.
    codons = [
        "".join(parts) for parts in itertools.product("ACGTRYSWKMBDHVN", repeat=3)
    ]
    encoded = _encode_batch(
        [(codon + "GCT", "GCTGCT") for codon in codons], _geometry(code)
    )
    for codon, (_, metadata) in zip(codons, encoded, strict=True):
        assert (metadata["status"] == "internal_stop") == analyze_codon(
            codon, code
        ).definite_stop


def test_pairwise_gap_ambiguity_removal_and_terminal_stop_are_audited():
    first, second = synthetic_pair(size=100)
    clean = estimate_pairs([(first, second)])[0]
    result = estimate_pairs([(first + "---NNNTAA---", second + "AAAGCTTAA---")])[0]
    assert result["aligned_codons"] == 104
    assert result["retained_codons"] == 100
    assert result["excluded_codons"] == 4
    assert result["dS"] == clean["dS"]
    assert result["dN"] == clean["dN"]


@pytest.mark.parametrize("code", [27, 28, 31])
def test_dual_coding_codons_are_sense_in_an_ordinary_alignment(code):
    sequence = "".join(sorted(get_forward_table(code))) * 4
    result = estimate_pairs([(sequence, sequence)], code)[0]
    assert result["retained_codons"] == len(sequence) // 3
    assert result["status"] == "ok"


@pytest.mark.parametrize(
    "first,second", [("AT", "AT"), ("ATG", "ATGATG"), ("AZG", "ATG")]
)
def test_malformed_inputs_fail_instead_of_becoming_zero(first, second):
    with pytest.raises(ValueError):
        estimate_pairs([(first, second)])


def test_saturation_has_na_distance_and_separate_diagnostic():
    forward = get_forward_table(1)
    codons = sorted(forward)
    second = [
        next(
            (
                other
                for other in codons
                if other != codon and forward[other] == forward[codon]
            ),
            codon,
        )
        for codon in codons
    ]
    result = estimate_pairs([("".join(codons) * 20, "".join(second) * 20)])[0]
    assert result["status"] == "saturated"
    assert result["dS"] is None
    assert result["dS_diagnostic"] is not None


def test_batch_parallelism_keeps_results_and_pair_order_exactly():
    pairs = [synthetic_pair(seed=i, size=40) for i in range(520)]
    assert estimate_pairs(pairs, threads=2) == estimate_pairs(pairs, threads=1)


def test_empty_input_and_unknown_code():
    assert estimate_pairs([]) == []
    with pytest.raises((ValueError, KeyError)):
        estimate_pairs([], codon_table=999)


@pytest.mark.parametrize(
    "code,seed,change,dn,ds,kappa",
    [
        (1, 10, 0.15, 0.1190, 0.1142, 0.9802),
        (1, 99, 0.05, 0.0629, 0.0573, 1.0943),
        (1, 3, 0.5, 0.5373, 0.3772, 1.1117),
        (2, 10, 0.15, 0.1275, 0.1208, 0.8828),
        (3, 10, 0.15, 0.1052, 0.0848, 1.4528),
        (4, 10, 0.15, 0.1075, 0.0809, 1.8131),
        (5, 10, 0.15, 0.1088, 0.0777, 1.5598),
        (6, 10, 0.15, 0.1077, 0.0916, 0.9398),
        (9, 10, 0.15, 0.1066, 0.0831, 1.5620),
        (10, 10, 0.15, 0.1066, 0.0833, 1.8143),
        (12, 10, 0.15, 0.1154, 0.1248, 0.9577),
        (13, 10, 0.15, 0.1107, 0.0734, 1.7469),
        (15, 10, 0.15, 0.1039, 0.0902, 1.7990),
    ],
)
def test_synthetic_values_match_independent_paml_yn00_reference(
    code, seed, change, dn, ds, kappa
):
    # Reference: PAML YN00, weighting=0/commonkappa=0/commonf3x4=0;
    # its report rounds to four decimal places. No PAML dependency is needed.
    result = estimate_pairs([synthetic_pair(code, seed, 300, change)], code)[0]
    assert result["status"] == "ok"
    for name, value in (("dN", dn), ("dS", ds), ("kappa", kappa)):
        assert result[name] == pytest.approx(value, abs=0.000051)
