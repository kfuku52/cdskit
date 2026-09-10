"""Independent expansion oracle and source-backed genetic-code regressions."""

from itertools import product

import pytest
from Bio.Data import CodonTable

from cdskit.codonutil import analyze_codon, summarize_codons

# IUPAC definitions and NCBI table 27/28/31 examples, checked 2026-09-10:
# https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=cgencodes
IUPAC = dict(
    zip(
        "ACGTRYSWKMBDHVNX",
        [
            "A",
            "C",
            "G",
            "T",
            "AG",
            "CT",
            "CG",
            "AT",
            "GT",
            "AC",
            "CGT",
            "AGT",
            "ACT",
            "ACG",
            "ACGT",
            "ACGT",
        ],
        strict=True,
    )
)


@pytest.mark.parametrize("code", sorted(CodonTable.unambiguous_dna_by_id))
def test_every_iupac_triplet_against_independent_expansion(code):
    table = CodonTable.unambiguous_dna_by_id[code]
    for letters in product(IUPAC, repeat=3):
        codon = "".join(letters)
        expanded = ["".join(x) for x in product(*(IUPAC[x] for x in letters))]
        unconditional = [
            x in table.stop_codons and x not in table.forward_table for x in expanded
        ]
        meaning = analyze_codon(codon, code)
        assert meaning.definite_stop == all(unconditional), (code, codon)
        assert meaning.possible_stop == (
            any(x in table.stop_codons for x in expanded) and not all(unconditional)
        ), (code, codon)
        assert meaning.context_dependent == any(
            x in table.stop_codons and x in table.forward_table for x in expanded
        ), (code, codon)
        assert meaning.terminal_stop_compatible == all(
            x in table.stop_codons for x in expanded
        ), (code, codon)
        assert meaning.amino_acids == tuple(
            sorted(
                {table.forward_table[x] for x in expanded if x in table.forward_table}
            )
        ), (code, codon)


@pytest.mark.parametrize(
    "code,codon,aa,dual",
    [
        (27, "TGA", "W", True),
        (27, "TAA", "Q", False),
        (28, "TAA", "Q", True),
        (28, "TAG", "Q", True),
        (28, "TGA", "W", True),
        (31, "TAA", "E", True),
        (31, "TAG", "E", True),
        (31, "TGA", "W", False),
    ],
)
def test_ncbi_dual_coding_examples(code, codon, aa, dual):
    meaning = analyze_codon(codon, code)
    assert meaning.amino_acids == (aa,)
    assert meaning.context_dependent is dual
    assert meaning.definite_stop is False
    assert meaning.clean is True
    assert analyze_codon(codon, code, "complete_terminal").definite_stop is dual


def test_ambiguity_and_stop_are_independent():
    tar = analyze_codon("tar", 1)
    assert tar.ambiguous and tar.definite_stop and not tar.clean
    tan = analyze_codon("TAN", 1)
    assert tan.ambiguous and tan.possible_stop and not tan.definite_stop
    assert analyze_codon("TAR", 28).context_dependent
    assert not analyze_codon("TAR", 28).definite_stop
    summary = summarize_codons("ATGTARAAA", 1)
    assert summary["ambiguous"] == summary["stop"] == 1
    assert summary["internal_stop"] and summary["clean"] == 2


@pytest.mark.parametrize(
    "codon,attribute",
    [("TA?", "missing"), ("TA!", "invalid"), ("AT", "partial"), ("", "partial")],
)
def test_invalid_missing_and_partial_are_not_clean_or_definite(codon, attribute):
    meaning = analyze_codon(codon, 1)
    assert getattr(meaning, attribute)
    assert not meaning.clean and not meaning.definite_stop


def test_terminal_exception_is_a_position_rule_not_completeness():
    assert not summarize_codons("ATGTAR---", 1)["internal_stop"]
    assert summarize_codons("ATGTAR---", 1, "physical")["internal_stop"]
    assert summarize_codons("ATGTARN", 1)["internal_stop"]
    assert summarize_codons("ATGTARNNN", 1)["internal_stop"]
    assert not summarize_codons("ATGTGA", 27)["internal_stop"]
    assert summarize_codons("ATGTGA", 27)["context_dependent"] == 1
    with pytest.raises(ValueError):
        analyze_codon("ATG", 1, "invented")
    with pytest.raises(ValueError):
        summarize_codons("ATG", 1, "invented")


@pytest.mark.parametrize("code", sorted(CodonTable.unambiguous_dna_by_id))
def test_all_translations_match_biopython_with_x_as_n(code):
    import warnings
    from Bio.Seq import Seq
    from cdskit.translate import translate_sequence_scalar, translate_sequence_string

    # X is an explicitly supported alias for any DNA base in CDSKIT. Some
    # Biopython Seq paths reject X, so compare against its canonical N spelling.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for letters in product(IUPAC, repeat=3):
            codon = "".join(letters)
            expected = str(Seq(codon.replace("X", "N")).translate(table=code))
            assert translate_sequence_string(codon, code, False) == expected
            assert translate_sequence_scalar(codon, code, False) == expected
