from itertools import product

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import pytest

from cdskit import hammer
from cdskit.translate import translate_sequence_codes


def test_gap_matrix_matches_character_membership():
    codons = ["".join(codon) for codon in product("ACGTRYSWKMBDHVNX-?.", repeat=3)]
    sequence = "".join(codons)
    records = [SeqRecord(Seq(sequence)), SeqRecord(Seq(sequence.lower()))]
    expected = np.array(
        [
            [all(ch in hammer.GAP_ONLY_CHARS for ch in c) for c in row]
            for row in (codons, [c.lower() for c in codons])
        ]
    )
    actual = hammer.build_codon_gap_like_matrix(records)
    assert actual.dtype == np.dtype(bool)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("code", [1, 2, 27, 28, 31])
def test_occupancy_retains_translation_based_missing_rules(code):
    sequences = ["ATGGCNTARTANTGATGGTAA---NNN", "atggcntartantgatggtaa---nnn"]
    records = [SeqRecord(Seq(sequence)) for sequence in sequences]
    translated = translate_sequence_codes("".join(sequences), code).reshape(2, -1)
    expected = np.array(
        [sum(chr(int(aa)) not in "-?X*" for aa in column) for column in translated.T]
    )
    actual = hammer.build_non_missing_site(records, code, threads=1)
    assert actual.dtype == np.dtype(int)
    np.testing.assert_array_equal(actual, expected)


def test_occupancy_lookup_handles_every_byte(monkeypatch):
    codes = np.arange(256, dtype=np.uint8)
    monkeypatch.setattr(hammer, "translate_sequence_codes", lambda **kwargs: codes)
    records = [SeqRecord(Seq("AAA" * len(codes)))]
    actual = hammer.build_non_missing_site(records, 1, threads=1)
    expected = [int(chr(int(aa)) not in "-?X*") for aa in codes]
    np.testing.assert_array_equal(actual, expected)
