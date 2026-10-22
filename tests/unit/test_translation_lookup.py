"""Compatibility of compact/lazy translation and the public eager lookup."""

from itertools import product

from Bio.Data.CodonTable import TranslationError
import numpy as np
import pytest

from cdskit import translate


@pytest.fixture(autouse=True)
def fresh_translation_caches(monkeypatch):
    monkeypatch.setattr(translate, "_CODON_TRANSLATOR_CACHE", {})
    monkeypatch.setattr(translate, "_TRANSLATION_KERNEL_CACHE", {})


def test_concrete_translation_does_not_initialize_ambiguity_entries():
    assert translate.translate_sequence_string("atggcttaa", 1, False) == "MA*"
    kernel = translate._TRANSLATION_KERNEL_CACHE[1]
    assert np.count_nonzero(kernel["aa_lut"]) == 64
    assert translate.translate_sequence_string("GCNTAR---TAN", 1, False) == "A*-X"
    assert np.count_nonzero(kernel["aa_lut"]) == 68
    assert translate.translate_sequence_string("GCNTAR---TAN", 1, True) == "A"


def test_public_lookup_stays_complete_and_mutable_after_lazy_translation():
    translate.translate_sequence_string("GCN", 1, False)
    lookup = translate.get_codon_translator(1)
    assert list(lookup) == [
        "alphabet_size",
        "encode_table",
        "aa_lut",
        "error_lut",
        "errors_by_index",
    ]
    assert len(lookup["aa_lut"]) == 19**3
    assert np.all(lookup["aa_lut"] != 0)
    assert lookup is translate.get_codon_translator(1)
    # Direct callers historically receive the live table, rather than a copy.
    alphabet = translate._TRANSLATION_ALPHABET
    first, second, third = map(alphabet.index, "ATG")
    index = first * 19**2 + second * 19 + third
    lookup["aa_lut"][index] = ord("Q")
    assert translate.translate_sequence_string("ATGGCN", 1, False) == "QA"
    lookup["error_lut"][index] = True
    lookup["errors_by_index"][index] = "custom translation error"
    with pytest.raises(TranslationError, match="custom translation error"):
        translate.translate_sequence_string("ATG", 1, False)
    assert translate.translate_sequence_string("TAAATG", 1, True) == ""


def test_public_lookup_custom_metadata_keeps_existing_index_arithmetic():
    lookup = translate.get_codon_translator(1)
    lookup["alphabet_size"] = 64
    lookup["aa_lut"] = np.full(64**3, ord("X"), dtype=np.uint8)
    lookup["error_lut"] = np.zeros(64**3, dtype=bool)
    # Three encoded dots have a custom index greater than uint16's maximum.
    index = 18 * 64**2 + 18 * 64 + 18
    lookup["aa_lut"][index] = ord("W")
    assert translate.translate_sequence_string("...", 1, False) == "W"


@pytest.mark.parametrize("code", [1, 2, 27, 28, 31])
def test_mixed_triplets_match_scalar_and_public_lookup(code):
    codons = ["".join(parts) for parts in product("ACGTRYSWKMBDHVNX-?.", repeat=3)]
    sequence = "".join(codons)
    for to_stop in (False, True):
        expected = translate.translate_sequence_scalar(sequence, code, to_stop)
        assert translate.translate_sequence_string(sequence, code, to_stop) == expected
        translate.get_codon_translator(code)
        assert translate.translate_sequence_string(sequence, code, to_stop) == expected


@pytest.mark.parametrize(
    "sequence,to_stop,expected,error",
    [
        ("TAA!!!", True, "", None),
        ("TAA!!!", False, None, "Invalid codon: '!!!'"),
        ("!!!TAA", True, None, "Invalid codon: '!!!'"),
        ("ATGéAA", False, None, "Invalid codon: 'ÉAA'"),
        ("AUGTAA", False, "M*", None),
        ("ATG--", False, "M-", None),
        ("", False, "", None),
    ],
)
def test_invalid_unicode_rna_stop_and_partial_tail_behavior(
    sequence, to_stop, expected, error
):
    if error is not None:
        with pytest.raises(TranslationError) as raised:
            translate.translate_sequence_string(sequence, 1, to_stop)
        assert str(raised.value) == error
    else:
        assert translate.translate_sequence_string(sequence, 1, to_stop) == expected


def test_direct_code_array_length_error_keeps_exception_and_dtype():
    assert translate.translate_sequence_codes("", 1).dtype == np.uint8
    with pytest.raises(TranslationError) as raised:
        translate.translate_sequence_codes("AT", 1)
    assert str(raised.value) == "Sequence length should be a multiple of three."
