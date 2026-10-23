"""Backalignment case preservation and ordered failure diagnostics."""

import pytest

from cdskit.backalign import backalign_sequence_strings


@pytest.mark.parametrize(
    "sequence,protein,expected",
    [
        ("ATGGCNTAA", "m-.?", "ATG------GCN"),
        ("AtGgcNtaA", "m-.?", "AtG------gcN"),
        ("a-.tGg-cN.TaA", "m-.?", "atG------gcN"),
        ("ATGGCNTAA", "m-.?*", "ATG------GCNTAA"),
        ("AtGgcNtaA", "m-.?*", "AtG------gcNtaA"),
    ],
)
def test_backalign_preserves_input_case_and_gap_placement(
    capsys, sequence, protein, expected
):
    result = backalign_sequence_strings(sequence, protein, 1, "sample", True)
    assert result == expected
    warning = capsys.readouterr().err
    assert warning == (
        "Ignored terminal stop codon not present in amino acid alignment: sample\n"
        if not protein.endswith("*")
        else ""
    )


@pytest.mark.parametrize(
    "sequence,protein,message",
    [
        (
            "AtG@@@",
            ".aXK",
            "Amino acid mismatch for sample at aligned position 2: "
            "aa_aln=a, translated=M, codon=AtG",
        ),
        (
            "ATG@@@",
            "m-.XK",
            "Invalid codon for sample at aligned position 4: @@@",
        ),
        (
            "AtG",
            "m-.XK",
            "Protein alignment had too many non-gap sites for sample "
            "at amino acid position 4.",
        ),
        (
            "ATG@@@",
            "m-.",
            "Invalid terminal codon for sample: @@@",
        ),
        (
            "ATG@@@AAA",
            "m-.",
            "2 codons remained unmatched for sample. "
            "The amino acid alignment may be truncated.",
        ),
        (
            "ATGAA",
            "m-.X",
            "Protein alignment had too many non-gap sites for sample "
            "at amino acid position 4.",
        ),
    ],
)
def test_backalign_reports_first_failure_in_alignment_order(
    capsys, sequence, protein, message
):
    with pytest.raises(ValueError) as error:
        backalign_sequence_strings(sequence, protein, 1, "sample", True)
    assert str(error.value) == message
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "code,terminal,sense", [(27, "TGA", "W"), (28, "TAA", "Q"), (31, "TAG", "E")]
)
def test_dual_coding_internal_and_terminal_matching(code, terminal, sense, capsys):
    sequence = "aTg" + terminal.lower() + "AtG" + terminal
    protein = "m." + sense.lower() + "-.m-*"
    assert backalign_sequence_strings(sequence, protein, code, "sample", True) == (
        "aTg---" + terminal.lower() + "------AtG---" + terminal
    )
    assert capsys.readouterr().err == ""
