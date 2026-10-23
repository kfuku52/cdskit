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


@pytest.mark.parametrize("mask_triplet", ["", "NN", "NNNN"])
def test_mask_replacement_invalid_width_is_checked_only_when_selected(mask_triplet):
    sequence = "ATG---"
    assert mask_sequence_string(sequence, 1, mask_triplet, False, False) is sequence
    with pytest.raises(ValueError, match="shape mismatch"):
        mask_sequence_string("A-G", 1, mask_triplet, False, False)


def test_mask_no_change_preserves_input_identity_and_ignores_unused_arguments():
    sequence = "atg---...-.-???"
    assert mask_sequence_string(sequence, "invalid", None, False, False) is sequence
    assert mask_sequence_string("", "invalid", None, True, True) == ""


def test_mask_direct_api_preserves_validation_order():
    with pytest.raises(ValueError, match="reshape"):
        mask_sequence_string("AT", "invalid", None, True, True)
    with pytest.raises(UnicodeEncodeError):
        mask_sequence_string("AéG", "invalid", None, True, True)
    with pytest.raises(Bio.Data.CodonTable.TranslationError, match="Invalid codon"):
        mask_sequence_string("ATZ", 1, None, True, True)
    assert mask_sequence_string("ATZ", 1, None, False, False) == "ATZ"


def test_mask_direct_api_named_code_and_dual_coding():
    assert (
        mask_sequence_string("TGATAGTAA", "Standard", "NNN", True, True) == "NNNNNNNNN"
    )
    assert mask_sequence_string("TGATAGTAA", 27, "NNN", True, True) == "TGATAGTAA"
    assert mask_sequence_string("TGATAGTAA", 28, "NNN", True, True) == "TGATAGTAA"
    assert mask_sequence_string("TGATAGTAA", 31, "NNN", True, True) == "TGATAGTAA"


@pytest.mark.parametrize("fail_second_encode", [False, True])
def test_mask_direct_string_subclass_preserves_second_encode(fail_second_encode):
    class ReencodingSequence(str):
        def __new__(cls):
            instance = super().__new__(cls, "ATGA-G")
            instance.encode_calls = 0
            return instance

        def encode(self, encoding="utf-8", errors="strict"):
            assert (encoding, errors) == ("ascii", "strict")
            self.encode_calls += 1
            if self.encode_calls == 1:
                return b"ATGA-G"
            if fail_second_encode:
                raise RuntimeError("second encode failed")
            return b"CCCA-G"

    sequence = ReencodingSequence()
    if fail_second_encode:
        # The second input encode still precedes encoding a bad replacement.
        with pytest.raises(RuntimeError, match="second encode failed"):
            mask_sequence_string(sequence, 1, None, False, False)
    else:
        assert mask_sequence_string(sequence, 1, "NNN", False, False) == "CCCNNN"
    assert sequence.encode_calls == 2
