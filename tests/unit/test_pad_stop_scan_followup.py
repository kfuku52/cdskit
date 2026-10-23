"""Independent frame-zero references for the blocked padding stop counter."""

from itertools import product

from Bio.Data import CodonTable
import pytest

from cdskit import pad


_ALPHABET = "ACGTRYSWKMBDHVNX"


def _reference_count(sequence, codon_table):
    """Literal triplet membership preserves the original physical-end rule."""
    sequence = str(sequence).upper()
    patterns = set(pad.get_stop_codon_scan_list(codon_table))
    return sum(
        sequence[start : start + 3] in patterns
        for start in range(0, len(sequence) - 3, 3)
    )


@pytest.mark.parametrize("code", sorted(CodonTable.unambiguous_dna_by_id))
def test_long_stop_scan_matches_all_iupac_triplets_in_every_code(code):
    sequence = "".join("".join(codon) for codon in product(_ALPHABET, repeat=3))
    sequence += "ATG"

    expected = _reference_count(sequence, code)

    assert pad.count_internal_stop_codons(sequence, code) == expected
    assert pad.count_internal_stop_codons(sequence.lower(), code) == expected


@pytest.mark.parametrize(
    "length", [4095, 4096, 4097, 65534, 65535, 65536, 65537, 131070]
)
@pytest.mark.parametrize("code", [1, 2, 27])
def test_stop_scan_threshold_and_block_edges_preserve_offsets(length, code):
    pattern = "ATAATAGTARTGA-?.!あß"
    for phase in ("", "A", "AA"):
        sequence = (phase + pattern * ((length + len(pattern)) // len(pattern)))[
            :length
        ]
        assert pad.count_internal_stop_codons(sequence, code) == _reference_count(
            sequence, code
        )


@pytest.mark.parametrize(
    "suffix", ["TAA", "TAAA", "TAATT", "TAA---", "TAR", "TAN", "TGA"]
)
def test_long_stop_scan_preserves_physical_terminal_and_partial_tail_rules(suffix):
    sequence = "ATG" * 2000 + suffix

    assert pad.count_internal_stop_codons(sequence, 1) == _reference_count(sequence, 1)


def test_stop_scan_rechecks_mutated_explicit_unconditional_stop_sets(monkeypatch):
    stops = {"TAA"}
    monkeypatch.setattr(pad, "get_stop_codons", lambda _: stops)
    sequence = "TAATAAAGA" * 1000 + "ATG"

    assert pad.count_internal_stop_codons(sequence, 1) == 2000
    stops.clear()
    stops.add("AGA")
    assert pad.count_internal_stop_codons(sequence, 1) == 1000


@pytest.mark.parametrize("code", [1, 2, 27])
@pytest.mark.parametrize("mode", ["min-stop", "preserve-frame"])
@pytest.mark.parametrize("padchar", ["N", "-"])
@pytest.mark.parametrize("include_report", [False, True])
def test_long_stop_scan_preserves_padding_candidates_and_ties(
    monkeypatch, code, mode, padchar, include_report
):
    sequence = "ATGTAAGGGTAR-?." * 400 + "ATGTA"
    arguments = dict(
        record_name="long record",
        record_seq=sequence,
        codon_table=code,
        padchar=padchar,
        mode=mode,
        include_report=include_report,
    )
    actual = pad.process_record_padding(**arguments)
    monkeypatch.setattr(pad, "count_internal_stop_codons", _reference_count)

    assert actual == pad.process_record_padding(**arguments)


class _CustomFindSequence(str):
    def __str__(self):
        return self

    def upper(self):
        return self

    def find(self, *args):
        return -1


def test_string_subclass_stop_scan_keeps_its_custom_find_behavior():
    sequence = _CustomFindSequence("TAATAA" * 1000)

    assert pad.count_internal_stop_codons(sequence, 1) == 0


def test_unicode_invalid_bases_cannot_create_false_stop_patterns():
    sequence = "TAあTA\ud800TßARtåaT\U0001f600A\uff34AA" * 1000

    assert pad.count_internal_stop_codons(sequence, 1) == _reference_count(sequence, 1)
