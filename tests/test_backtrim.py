"""
Tests for cdskit backtrim command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.backtrim import backtrim_main, check_same_seq_num


class TestCheckSameSeqNum:
    """Tests for check_same_seq_num function."""

    def test_same_number(self):
        """Test with same number of sequences."""
        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1")]
        pep_records = [SeqRecord(Seq("MK"), id="seq1")]
        # Should not raise
        check_same_seq_num(cdn_records, pep_records)

    def test_different_number(self):
        """Test with different number of sequences."""
        cdn_records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),
            SeqRecord(Seq("ATGCCC"), id="seq2"),
        ]
        pep_records = [SeqRecord(Seq("MK"), id="seq1")]
        with pytest.raises(AssertionError):
            check_same_seq_num(cdn_records, pep_records)


class TestBacktrimMain:
    """Tests for backtrim_main function."""

    def test_backtrim_basic(self, temp_dir, mock_args):
        """Test basic backtrim functionality."""
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        # Codon sequences: MK (ATG AAA) + P (CCC)
        # Trimmed protein: MK (only first 2 amino acids)
        cdn_records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAAGGG"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        # Trimmed protein alignment
        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("MK"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        backtrim_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have kept only first 2 codons = 6 nucleotides
        assert len(result[0].seq) == 6
        assert str(result[0].seq) == "ATGAAA"

    def test_backtrim_with_gaps(self, temp_dir, mock_args):
        """Test backtrim with gapped alignment."""
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        # Aligned codon sequences with gaps
        cdn_records = [
            SeqRecord(Seq("ATGAAA---CCC"), id="seq1", description=""),  # M K - P
            SeqRecord(Seq("ATGAAAGGGCCC"), id="seq2", description=""),  # M K G P
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        # Trimmed protein - keep only M and K columns
        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("MK"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        backtrim_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have kept only M and K codons
        assert len(result[0].seq) == 6

    def test_backtrim_rejects_non_multiple_of_three(self, temp_dir, mock_args):
        """Test backtrim rejects codon sequences not multiple of 3."""
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [SeqRecord(Seq("ATGAA"), id="seq1", description="")]  # 5 nt
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        pep_records = [SeqRecord(Seq("M"), id="seq1", description="")]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backtrim_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_backtrim_rejects_unaligned_codons(self, temp_dir, mock_args):
        """Test backtrim rejects unaligned codon sequences."""
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),  # 6 nt
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),  # 9 nt - different length
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("MKP"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backtrim_main(args)
        assert "not identical" in str(exc_info.value)

    def test_backtrim_with_test_data_01(self, data_dir, temp_dir, mock_args):
        """Test backtrim with backtrim_01 test data."""
        cdn_path = data_dir / "backtrim_01" / "LUC.cds.aln.fasta"
        pep_path = data_dir / "backtrim_01" / "pep.clipkit.fasta"
        expected_path = data_dir / "backtrim_01" / "out.fasta"
        output_path = temp_dir / "output.fasta"

        if not cdn_path.exists():
            pytest.skip("backtrim_01 test data not found")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        backtrim_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        if expected_path.exists():
            expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))
            assert len(result) == len(expected)
            for r, e in zip(result, expected):
                assert str(r.seq) == str(e.seq), f"Mismatch for {r.id}"

    def test_backtrim_with_test_data_02(self, data_dir, temp_dir, mock_args):
        """Test backtrim with backtrim_02 test data."""
        cdn_path = data_dir / "backtrim_02" / "untrimmed_codon.fasta"
        pep_path = data_dir / "backtrim_02" / "trimmed_aa.fasta"
        expected_path = data_dir / "backtrim_02" / "trimmed_codon.fasta"
        output_path = temp_dir / "output.fasta"

        if not cdn_path.exists():
            pytest.skip("backtrim_02 test data not found")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        backtrim_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        if expected_path.exists():
            expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))
            assert len(result) == len(expected)
