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

from cdskit.backtrim import backtrim_main, build_column_index, check_same_seq_num


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
        with pytest.raises(Exception):
            check_same_seq_num(cdn_records, pep_records)


class TestBuildColumnIndex:
    """Tests for build_column_index helper."""

    def test_empty_input(self):
        """Empty input should produce an empty index."""
        col_index = build_column_index([])
        assert len(col_index) == 0

    def test_duplicate_column_patterns_preserve_order(self):
        """Duplicate column keys should retain all matching positions."""
        col_index = build_column_index(["AA", "AA"])
        assert list(col_index["AA"]) == [0, 1]


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

    def test_backtrim_handles_question_codon_as_missing_not_error(self, temp_dir, mock_args):
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [
            SeqRecord(Seq("ATG???CCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        pep_records = [
            SeqRecord(Seq("MXP"), id="seq1", description=""),
            SeqRecord(Seq("MKP"), id="seq2", description=""),
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
        assert [r.id for r in result] == ["seq1", "seq2"]
        assert [str(r.seq) for r in result] == ["ATG???CCC", "ATGAAACCC"]

    def test_backtrim_reorders_trimmed_alignment_by_id(self, temp_dir, mock_args):
        """Trimmed amino-acid records in different order should still map by ID."""
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),  # MK
            SeqRecord(Seq("ATGCCC"), id="seq2", description=""),  # MP
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        # Same IDs, different order.
        pep_records = [
            SeqRecord(Seq("MP"), id="seq2", description=""),
            SeqRecord(Seq("MK"), id="seq1", description=""),
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
        assert [r.id for r in result] == ["seq1", "seq2"]
        assert [str(r.seq) for r in result] == ["ATGAAA", "ATGCCC"]

    def test_backtrim_rejects_mismatched_ids(self, temp_dir, mock_args):
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", description=""),
        ]
        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("MP"), id="seqX", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        with pytest.raises(Exception) as exc_info:
            backtrim_main(args)
        assert "Sequence IDs did not match between CDS" in str(exc_info.value)

    def test_backtrim_multiple_matches_uses_first_site(self, temp_dir, mock_args, capsys):
        """When multiple codon sites match one protein column, first site is used."""
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGATG"), id="seq1", description=""),  # M M
            SeqRecord(Seq("ATGATG"), id="seq2", description=""),  # M M
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        pep_records = [
            SeqRecord(Seq("M"), id="seq1", description=""),
            SeqRecord(Seq("M"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )

        backtrim_main(args)

        captured = capsys.readouterr()
        assert "multiple matches" in captured.err
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(r.seq) for r in result] == ["ATG", "ATG"]

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

    def test_backtrim_rejects_invalid_codontable(self, temp_dir, mock_args):
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        pep_records = [SeqRecord(Seq("MK"), id="seq1", description="")]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=999,
        )

        with pytest.raises(Exception) as exc_info:
            backtrim_main(args)
        assert "Invalid --codontable" in str(exc_info.value)

    def test_backtrim_empty_inputs_produce_empty_output(self, temp_dir, mock_args):
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        output_path = temp_dir / "output.fasta"

        cdn_path.write_text("")
        pep_path.write_text("")

        args = mock_args(
            seqfile=str(cdn_path),
            outfile=str(output_path),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
        )
        backtrim_main(args)
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 0

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

    def test_backtrim_threads_matches_single_thread(self, temp_dir, mock_args):
        cdn_path = temp_dir / "codon.fasta"
        pep_path = temp_dir / "protein.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        cdn_records = [
            SeqRecord(Seq("ATGAAA---CCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAAGGGCCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGTTT---CCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(cdn_records, str(cdn_path), "fasta")

        pep_records = [
            SeqRecord(Seq("MK"), id="seq1", description=""),
            SeqRecord(Seq("MK"), id="seq2", description=""),
            SeqRecord(Seq("MF"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(pep_records, str(pep_path), "fasta")

        args_single = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_single),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(cdn_path),
            outfile=str(out_threaded),
            trimmed_aa_aln=str(pep_path),
            codontable=1,
            threads=4,
        )

        backtrim_main(args_single)
        backtrim_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]
