"""
Tests for cdskit codonstats command.
"""

import pytest
import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.codonstats import codonstats_main, summarize_record


@pytest.mark.parametrize(
    "sequence,expected_complete,expected_missing,expected_ambiguous",
    [
        ("?", 0, 1, 0),
        ("N", 0, 0, 0),
        ("ATG?", 1, 1, 0),
        ("ATGN", 1, 0, 0),
    ],
)
def test_summarize_record_ignores_partial_codon_for_complete_counts(
    sequence,
    expected_complete,
    expected_missing,
    expected_ambiguous,
):
    summary = summarize_record(SeqRecord(Seq(sequence), id="seq1"), codontable=1)

    assert summary["codons_complete"] == expected_complete
    assert summary["codons_missing"] == expected_missing
    assert summary["codons_ambiguous"] == expected_ambiguous


class TestCodonstatsMain:
    """Tests for codonstats_main function."""

    def test_codonstats_summary_mode(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("GCTGCTGCT"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNTAA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            mode="summary",
        )

        codonstats_main(args)
        captured = capsys.readouterr()

        assert captured.out.startswith("seq_id\tnt_length\tcodons_total")
        assert (
            "seq1\t9\t3\t3\t0\t0\t0\t66.666667\t100.000000\t100.000000\t0.000000"
            in captured.out
        )
        assert "seq2\t9\t3\t3\t0\t1\t1" in captured.out

    def test_codonstats_usage_mode(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGATGTTT"), id="seq1", description=""),
            SeqRecord(Seq("ATGTTTTTT"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            mode="usage",
        )

        codonstats_main(args)
        captured = capsys.readouterr()

        assert captured.out.startswith("codon\taa\tcount\tfraction")
        assert "ATG\tM\t3\t0.500000" in captured.out
        assert "TTT\tF\t3\t0.500000" in captured.out

    def test_codonstats_rejects_non_triplet_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAA"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            mode="summary",
        )

        with pytest.raises(ValueError) as exc_info:
            codonstats_main(args)
        assert "multiple of three" in str(exc_info.value)
