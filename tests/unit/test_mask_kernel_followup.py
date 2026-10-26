"""Regressions for the byte-oriented partial-gap masking kernel."""

from itertools import product

import Bio.Data.CodonTable
import pytest

from cdskit.mask import (
    mask_partial_gap_codons,
    mask_sequence_string,
    mask_translated_codons,
)
from cdskit.translate import translate_sequence_scalar


@pytest.mark.parametrize(
    "codontable", sorted(Bio.Data.CodonTable.unambiguous_dna_by_id)
)
@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize(
    "mask_ambiguous,mask_stop", list(product([False, True], repeat=2))
)
def test_mask_kernel_matches_scalar_helpers(
    codontable, lowercase, mask_ambiguous, mask_stop
):
    codons = ["".join(chars) for chars in product("ACGT", repeat=3)]
    codons.extend("".join(chars) for chars in product("A-.", repeat=3))
    codons.extend(["TGY", "TAR", "TRA", "TAN", "NNN", "???", "A?G", "XGC", "AUG"])
    sequence = "".join(codons)
    if lowercase:
        sequence = sequence.lower()
    expected = [sequence[start : start + 3] for start in range(0, len(sequence), 3)]
    mask_partial_gap_codons(expected, "n?n")
    if mask_ambiguous or mask_stop:
        amino_acids = translate_sequence_scalar(sequence, codontable, False)
        mask_translated_codons(expected, amino_acids, "n?n", mask_ambiguous, mask_stop)
    assert mask_sequence_string(
        sequence, codontable, "n?n", mask_ambiguous, mask_stop
    ) == "".join(expected)


@pytest.mark.parametrize("mask_triplet", ["N", "n?n", "\x00\x00\x00"])
def test_mask_replacement_preserves_byte_values_and_single_base_broadcast(mask_triplet):
    replacement = mask_triplet * 3 if len(mask_triplet) == 1 else mask_triplet
    assert mask_sequence_string("atgA-G---", 1, mask_triplet, False, False) == (
        "atg" + replacement + "---"
    )


def test_mask_no_change_preserves_sequence():
    sequence = "atg---...-.-???"
    assert mask_sequence_string(sequence, 1, "NNN", False, False) == sequence
    assert mask_sequence_string("", 1, "NNN", True, True) == ""


def test_mask_direct_api_named_code_and_dual_coding():
    assert (
        mask_sequence_string("TGATAGTAA", "Standard", "NNN", True, True) == "NNNNNNNNN"
    )
    assert mask_sequence_string("TGATAGTAA", 27, "NNN", True, True) == "TGATAGTAA"
    assert mask_sequence_string("TGATAGTAA", 28, "NNN", True, True) == "TGATAGTAA"
    assert mask_sequence_string("TGATAGTAA", 31, "NNN", True, True) == "TGATAGTAA"
