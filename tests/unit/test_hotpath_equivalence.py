"""Independent oracles for optimized alignment calculations."""

from itertools import combinations, product
import random

from Bio.Data.CodonTable import unambiguous_dna_by_id
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pytest

from cdskit.degeneracy import classify_alignment_columns
from cdskit.maxalign import solve_exact


def reference_degeneracy(sequences, code):
    forward = unambiguous_dna_by_id[code].forward_table
    assignments = []
    conflict = unassigned = 0
    for start in range(0, len(sequences[0]), 3):
        codons = [sequence[start : start + 3].upper() for sequence in sequences]
        codons = [codon for codon in codons if codon in forward]
        for position in range(3):
            values = set()
            for codon in codons:
                synonymous = sum(
                    forward.get(codon[:position] + base + codon[position + 1 :])
                    == forward[codon]
                    for base in "ACGT"
                )
                values.add(0 if synonymous == 1 else synonymous)
            assignments.append(next(iter(values)) if len(values) == 1 else None)
            conflict += len(values) > 1
            unassigned += not values
    return {
        "assignments": assignments,
        "counts_by_fold": {fold: assignments.count(fold) for fold in (0, 2, 3, 4)},
        "num_conflict_sites": conflict,
        "num_unassigned_sites": unassigned,
    }


@pytest.mark.parametrize("code", sorted(unambiguous_dna_by_id))
def test_degeneracy_matches_independent_genetic_code_enumeration(code):
    rng = random.Random(19)
    codons = ["".join(parts) for parts in product("ACGT", repeat=3)]
    # Exercise every sense/stop codon individually, then mixed columns with
    # conflicts, lowercase, ambiguity and entire missing columns.
    for sequences in (
        ["".join(codons)],
        [
            "".join(rng.choices([*codons, "---", "TAR", "GCN", "atg"], k=200))
            + "???...NNN"
            for _ in range(8)
        ],
        ["", ""],
    ):
        records = [SeqRecord(Seq(sequence)) for sequence in sequences]
        assert classify_alignment_columns(records, code) == reference_degeneracy(
            sequences, code
        )


@pytest.mark.parametrize("seed", range(12))
def test_maxalign_matches_exhaustive_column_oracle_under_constraints(seed):
    rng = random.Random(seed)
    num_sequences = 8
    matrix = [[rng.random() < 0.6 for _ in range(24)] for _ in range(num_sequences)]
    candidates = sorted(rng.sample(range(num_sequences), 6))
    required = candidates[1:3] if seed % 2 else []
    max_removed = seed % 5 if seed % 3 else None
    feasible = []
    for count in range(1, len(candidates) + 1):
        for kept in combinations(candidates, count):
            if not set(required).issubset(kept):
                continue
            if max_removed is not None and num_sequences - count > max_removed:
                continue
            complete = sum(all(matrix[i][j] for i in kept) for j in range(24))
            feasible.append((count * complete, count, list(kept), complete))
    expected = (
        min(feasible, key=lambda item: (-item[0], -item[1], item[2]))
        if feasible
        else None
    )
    result = solve_exact(
        matrix,
        candidate_indices=candidates,
        required_indices=required,
        max_removed=max_removed,
        threads=4,
    )
    if expected is None:
        assert result is None
    else:
        assert result == dict(
            area=expected[0],
            num_kept=expected[1],
            kept_indices=expected[2],
            complete_codon_columns=expected[3],
            steps=[],
        )


@pytest.mark.parametrize(
    "count", [0, 7, 2**70, np.uint64(2**53 + 1), np.int64(2**53 + 1)]
)
def test_maxalign_preserves_large_integer_counts_and_high_sequence_bits(count):
    result = solve_exact(
        None,
        candidate_indices=[0, 4, 70],
        required_indices=[4, 70],
        support_counts={(1 << 4) | (1 << 70): count, (1 << 71) - 1: 1},
        num_sequences=71,
        total_sequences=3,
    )
    assert result["area"] == max(3, 2 * (int(count) + 1))
    assert type(result["area"]) is int
    assert result["kept_indices"] == ([0, 4, 70] if count == 0 else [4, 70])


def test_maxalign_all_required_and_empty_support():
    assert solve_exact(
        None,
        candidate_indices=[2, 4],
        required_indices=[2, 4],
        support_counts={},
        num_sequences=5,
    ) == dict(
        area=0, num_kept=2, kept_indices=[2, 4], complete_codon_columns=0, steps=[]
    )
