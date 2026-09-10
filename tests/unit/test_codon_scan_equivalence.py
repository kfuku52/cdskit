"""Exhaustive genetic-code checks and positional reference comparisons."""

from itertools import product
import random

from Bio.Data.CodonTable import unambiguous_dna_by_id
from Bio.Data.IUPACData import ambiguous_dna_values
import pytest

from cdskit.codonutil import (
    analyze_codon,
    definite_stop_patterns,
    summarize_codons,
)
from cdskit.longestcds import collect_start_stop_positions_by_frame, get_scan_codons
from cdskit.pad import count_internal_stop_codons


@pytest.mark.parametrize("code", sorted(unambiguous_dna_by_id))
def test_stop_patterns_cover_exactly_all_definite_iupac_stops(code):
    table = unambiguous_dna_by_id[code]
    stops = frozenset(table.stop_codons) - table.forward_table.keys()
    patterns = set(definite_stop_patterns(frozenset(stops)))
    for parts in product(ambiguous_dna_values, repeat=3):
        codon = "".join(parts)
        assert (codon in patterns) == analyze_codon(codon, code).definite_stop


@pytest.mark.parametrize("code", [1, 2, 27, 28, 31])
def test_scans_and_counts_match_positional_reference(code):
    rng = random.Random(7)
    for _ in range(50):
        sequence = "".join(rng.choices("ACGTRYMKWSBDHVNX-.?", k=rng.randrange(1, 300)))
        meanings = [
            analyze_codon(sequence[i : i + 3], code)
            for i in range(0, len(sequence) - 2, 3)
        ]
        for policy in ("physical", "last_evaluable"):
            terminal = None
            if meanings and len(sequence) % 3 == 0:
                if policy == "physical":
                    terminal = len(meanings) - 1
                else:
                    terminal = next(
                        (
                            i
                            for i in reversed(range(len(meanings)))
                            if not meanings[i].missing
                        ),
                        None,
                    )
            summary = summarize_codons(sequence.lower(), code, policy)
            expected = {
                "total": len(meanings),
                "clean": sum(m.clean for m in meanings),
                "missing": sum(m.missing for m in meanings),
                "ambiguous": sum(m.ambiguous or m.invalid for m in meanings),
                "stop": sum(m.definite_stop for m in meanings),
                "evaluable": sum(not m.missing for m in meanings),
                "internal_stop_count": sum(
                    m.definite_stop for i, m in enumerate(meanings) if i != terminal
                ),
                "possible_stop": sum(m.possible_stop for m in meanings),
                "context_dependent": sum(m.context_dependent for m in meanings),
                "internal_possible_stop_count": sum(
                    m.possible_stop for i, m in enumerate(meanings) if i != terminal
                ),
                "internal_context_dependent_count": sum(
                    m.context_dependent for i, m in enumerate(meanings) if i != terminal
                ),
            }
            expected["internal_stop"] = expected["internal_stop_count"] > 0
            assert summary == expected
            if policy == "physical":
                assert (
                    count_internal_stop_codons(sequence, code)
                    == summary["internal_stop_count"]
                )
        starts, stops = get_scan_codons(code)
        found_start, found_stop = collect_start_stop_positions_by_frame(
            sequence, starts, stops
        )
        for frame in range(3):
            positions = list(range(frame, len(sequence) - 2, 3))
            assert found_start[frame] == [
                i for i in positions if sequence[i : i + 3] in starts
            ]
            assert found_stop[frame] == [
                i
                for i in positions
                if analyze_codon(sequence[i : i + 3], code).definite_stop
            ]
