"""Compatibility and block boundaries for the ORF position scanner."""

from dataclasses import asdict
from itertools import product
import random

from Bio.Data.CodonTable import unambiguous_dna_by_id
from Bio.Data.IUPACData import ambiguous_dna_values
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pytest

from cdskit import longestcds
from cdskit.codonutil import definite_stop_patterns


CODES = sorted(unambiguous_dna_by_id)


def positional_reference(sequence, starts, stops):
    """Enumerate each literal occurrence, including empty/non-triplet starts."""
    positions = [[], [], []], [[], [], []]
    for patterns, destination in (
        (starts, positions[0]),
        (definite_stop_patterns(frozenset(stops)), positions[1]),
    ):
        for pattern in patterns:
            for index in range(len(sequence) + 1):
                if sequence.startswith(pattern, index):
                    destination[index % 3].append(index)
        for frame in destination:
            frame.sort()
    return positions


@pytest.mark.parametrize("code", CODES)
def test_block_scan_preserves_all_iupac_patterns_and_offsets(monkeypatch, code):
    monkeypatch.setattr(longestcds, "_POSITION_SCAN_MIN_LENGTH", 0)
    monkeypatch.setattr(longestcds, "_POSITION_SCAN_MIN_PATTERNS", 0)
    # Not a multiple of three: local frame offsets rotate at every block.
    monkeypatch.setattr(longestcds, "_POSITION_SCAN_BLOCK_SIZE", 17)
    triplets = "?".join(
        "".join(parts) for parts in product(ambiguous_dna_values, repeat=3)
    )
    sequence = "ATGATGATGTAR?ATGTAN?éATG\u2028TAR?\ud800ATG?atg" + triplets
    starts, stops = longestcds.get_scan_codons(code)
    assert longestcds.collect_start_stop_positions_by_frame(
        sequence, starts, stops
    ) == positional_reference(sequence, starts, stops)


@pytest.mark.parametrize("boundary_shift", [-2, -1, 0, 1, 2])
def test_scan_includes_every_triplet_across_real_block_boundaries(boundary_shift):
    block_size = longestcds._POSITION_SCAN_BLOCK_SIZE
    bases = list("C" * (block_size * 2 + 11))
    for left, codon in (
        (block_size + boundary_shift, "ATG"),
        (block_size * 2 + boundary_shift, "TAR"),
    ):
        bases[left : left + 3] = codon
    sequence = "".join(bases)
    starts, stops = longestcds.get_scan_codons(1)
    assert longestcds.collect_start_stop_positions_by_frame(
        sequence, starts, stops
    ) == positional_reference(sequence, starts, stops)


def test_scan_bounds_temporary_buffers_per_block(monkeypatch):
    block_size = longestcds._POSITION_SCAN_BLOCK_SIZE
    sequence = "ATGTAR" * (block_size // 2)
    sizes = []
    original = longestcds.np.frombuffer

    def record_buffer(buffer, *args, **kwargs):
        sizes.append(len(buffer))
        return original(buffer, *args, **kwargs)

    monkeypatch.setattr(longestcds.np, "frombuffer", record_buffer)
    starts, stops = longestcds.get_scan_codons(1)
    found = longestcds.collect_start_stop_positions_by_frame(sequence, starts, stops)
    assert found == positional_reference(sequence, starts, stops)
    assert sizes
    assert max(sizes) <= block_size + 2


@pytest.mark.parametrize("code", [1, 11])
def test_short_scans_match_literal_positions(code):
    starts, stops = longestcds.get_scan_codons(code)
    for length in (0, 1, 2, 150, longestcds._POSITION_SCAN_MIN_LENGTH - 1):
        sequence = ("ATGTAR" * (length // 6 + 1))[:length]
        assert longestcds.collect_start_stop_positions_by_frame(
            sequence, starts, stops
        ) == positional_reference(sequence, starts, stops)


@pytest.mark.parametrize("code", [3, 6, 9, 10, 12, 27, 28, 31])
def test_few_pattern_scans_match_literal_positions(code):
    starts, stops = longestcds.get_scan_codons(code)
    sequence = "ATGTAATAGTGA" * 2000
    assert longestcds.collect_start_stop_positions_by_frame(
        sequence, starts, stops
    ) == positional_reference(sequence, starts, stops)


@pytest.mark.parametrize(
    "starts,stops,container",
    [
        (("ATG", "ATG", "CTG", "TTG"), ("TAA", "TAG", "TGA"), tuple),
        (("TTG", "CTG", "ATG"), ("TAA", "TAG", "TGA"), tuple),
        (("ATG", "CTG", "TTG"), ("TAA", "TAG", "TGA"), list),
        (("ATG", "CTG", "TTG"), ("TAA", "TAG", "TGA"), iter),
        (("é", "ATGATG", "A", ""), (), tuple),
    ],
)
def test_external_patterns_keep_literal_semantics(starts, stops, container):
    sequence = "GCC" * 5500 + "éATGATGTAR"
    expected = positional_reference(sequence, starts, stops)
    assert (
        longestcds.collect_start_stop_positions_by_frame(
            sequence, container(starts), container(stops)
        )
        == expected
    )


@pytest.mark.parametrize("selection", ["complete-first", "longest"])
@pytest.mark.parametrize("code", CODES)
def test_block_scan_preserves_candidates_selection_ties_and_report(
    monkeypatch, code, selection
):
    rng = random.Random(2026102208)
    sequence = "ATGATGTARTANATGTAAATGTAA" + "".join(rng.choices("ACGTRYX-.?", k=233))
    record = SeqRecord(Seq(sequence), id="duplicate")

    def snapshot():
        chosen = longestcds.choose_best_candidate(sequence, code, selection)
        return (
            None if chosen is None else asdict(chosen),
            [asdict(c) for c in longestcds.iter_candidates(sequence, code)],
            longestcds.candidate_report(record, code, selection, 2),
        )

    monkeypatch.setattr(longestcds, "_POSITION_SCAN_MIN_LENGTH", len(sequence) + 1)
    expected = snapshot()
    monkeypatch.setattr(longestcds, "_POSITION_SCAN_MIN_LENGTH", 0)
    monkeypatch.setattr(longestcds, "_POSITION_SCAN_MIN_PATTERNS", 0)
    monkeypatch.setattr(longestcds, "_POSITION_SCAN_BLOCK_SIZE", 17)
    assert snapshot() == expected
