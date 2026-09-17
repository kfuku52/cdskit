"""
Tests for cdskit printseq command.
Based on wiki example: https://github.com/kfuku52/cdskit/wiki/cdskit-printseq
"""

import pytest

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.printseq import printseq_main


class TestPrintseqMain:
    """Tests for printseq_main function."""

    def test_printseq_regex_match(self, temp_dir, mock_args, capsys):
        """Test printseq with regex matching - wiki example."""
        input_path = temp_dir / "input.fasta"

        # Wiki example: seq_A, seq_T, seq_G, seq_C
        records = [
            SeqRecord(Seq("AAAAAAAAAAAA"), id="seq_A", name="seq_A", description=""),
            SeqRecord(Seq("TTTTTTTTTTTT"), id="seq_T", name="seq_T", description=""),
            SeqRecord(Seq("GGGGGGGGGGGG"), id="seq_G", name="seq_G", description=""),
            SeqRecord(Seq("CCCCCCCCCCCC"), id="seq_C", name="seq_C", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="seq_[AG]",  # Match seq_A and seq_G
            show_seqname=True,
        )

        printseq_main(args)

        captured = capsys.readouterr()
        # Should print seq_A and seq_G with headers
        assert ">seq_A" in captured.out
        assert "AAAAAAAAAAAA" in captured.out
        assert ">seq_G" in captured.out
        assert "GGGGGGGGGGGG" in captured.out
        # Should NOT print seq_T or seq_C
        assert ">seq_T" not in captured.out
        assert ">seq_C" not in captured.out

    def test_printseq_single_match(self, temp_dir, mock_args, capsys):
        """Test printseq with exact name match."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="target", name="target", description=""),
            SeqRecord(Seq("ATGCCC"), id="other", name="other", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="target",
            show_seqname=True,
        )

        printseq_main(args)

        captured = capsys.readouterr()
        assert ">target" in captured.out
        assert "ATGAAA" in captured.out
        assert "other" not in captured.out

    def test_printseq_no_header(self, temp_dir, mock_args, capsys):
        """Test printseq without showing sequence name."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", name="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="seq1",
            show_seqname=False,
        )

        printseq_main(args)

        captured = capsys.readouterr()
        # Should print sequence without header
        assert "ATGAAACCC" in captured.out
        assert ">seq1" not in captured.out

    def test_printseq_no_match(self, temp_dir, mock_args, capsys):
        """Test printseq when no sequences match."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", name="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="nonexistent",
            show_seqname=True,
        )

        printseq_main(args)

        captured = capsys.readouterr()
        # Should not print anything (except stderr from read_seqs)
        assert "ATGAAA" not in captured.out
        assert "ATGCCC" not in captured.out

    def test_printseq_complex_regex(self, temp_dir, mock_args, capsys):
        """Test printseq with complex regex pattern."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("AAAA"), id="gene_001", name="gene_001", description=""),
            SeqRecord(Seq("TTTT"), id="gene_002", name="gene_002", description=""),
            SeqRecord(
                Seq("GGGG"), id="transcript_001", name="transcript_001", description=""
            ),
            SeqRecord(Seq("CCCC"), id="gene_100", name="gene_100", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="gene_00[12]",  # Match gene_001 and gene_002
            show_seqname=True,
        )

        printseq_main(args)

        captured = capsys.readouterr()
        assert ">gene_001" in captured.out
        assert ">gene_002" in captured.out
        assert "transcript_001" not in captured.out
        assert "gene_100" not in captured.out

    def test_printseq_match_all(self, temp_dir, mock_args, capsys):
        """Test printseq matching all sequences with wildcard."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", name="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="seq.*",  # Match all seq*
            show_seqname=True,
        )

        printseq_main(args)

        captured = capsys.readouterr()
        assert ">seq1" in captured.out
        assert ">seq2" in captured.out

    def test_printseq_rejects_invalid_regex(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", name="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="[",
            show_seqname=True,
        )
        with pytest.raises(ValueError) as exc_info:
            printseq_main(args)
        assert "Invalid regex in --seq_name_regex" in str(exc_info.value)

    def test_printseq_accepts_protein_input_when_seqtype_protein(
        self, temp_dir, mock_args, capsys
    ):
        input_path = temp_dir / "input.fasta"
        records = [
            SeqRecord(Seq("MKT"), id="prot1", name="prot1", description=""),
            SeqRecord(Seq("QQQ"), id="prot2", name="prot2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            seqname="prot1",
            show_seqname=True,
            seqtype="protein",
        )
        printseq_main(args)

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]
        assert lines == [">prot1", "MKT"]
