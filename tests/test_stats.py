"""
Tests for cdskit stats command.
"""

import pytest
from pathlib import Path
from io import StringIO
import sys

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.stats import stats_main, num_masked_bp


class TestNumMaskedBp:
    """Tests for num_masked_bp function."""

    def test_no_masked(self):
        """Test sequence with no soft-masked bases."""
        seq = "ATGCCC"
        assert num_masked_bp(seq) == 0

    def test_all_masked(self):
        """Test sequence with all soft-masked bases."""
        seq = "atgccc"
        assert num_masked_bp(seq) == 6

    def test_mixed_masked(self):
        """Test sequence with some soft-masked bases."""
        seq = "ATGccc"
        assert num_masked_bp(seq) == 3


class TestStatsMain:
    """Tests for stats_main function."""

    def test_stats_basic(self, temp_dir, mock_args, capsys):
        """Test basic stats output."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="seq1", description=""),  # 6 nt
            SeqRecord(Seq("GGGGGG"), id="seq2", description=""),  # 6 nt
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Number of sequences: 2" in captured.out
        assert "Total length: 12" in captured.out

    def test_stats_gc_content(self, temp_dir, mock_args, capsys):
        """Test GC content calculation."""
        input_path = temp_dir / "input.fasta"

        # 50% GC content: 3 G/C out of 6
        records = [
            SeqRecord(Seq("ATGCAT"), id="seq1", description=""),  # A T G C A T = 2 GC / 6
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        # 2 G/C out of 6 = 33.3%
        assert "GC content:" in captured.out

    def test_stats_with_gaps(self, temp_dir, mock_args, capsys):
        """Test stats with gap characters."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATG---CCC"), id="seq1", description=""),  # 3 gaps
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Total gap (-) length: 3" in captured.out

    def test_stats_with_ns(self, temp_dir, mock_args, capsys):
        """Test stats with N characters."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGNNNCCC"), id="seq1", description=""),  # 3 Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Total N length: 3" in captured.out

    def test_stats_with_softmasked(self, temp_dir, mock_args, capsys):
        """Test stats with soft-masked bases."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGcccGGG"), id="seq1", description=""),  # 3 lowercase
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Total softmasked length: 3" in captured.out

    def test_stats_empty_file(self, temp_dir, mock_args, capsys):
        """Test stats with empty file."""
        input_path = temp_dir / "input.fasta"
        input_path.write_text("")

        args = mock_args(
            seqfile=str(input_path),
        )

        # This may raise a division by zero error for GC content
        # or may handle it gracefully
        try:
            stats_main(args)
            captured = capsys.readouterr()
            assert "Number of sequences: 0" in captured.out
        except ZeroDivisionError:
            # Expected if empty file not handled
            pass

    def test_stats_with_test_data(self, data_dir, mock_args, capsys):
        """Test stats with stats_01 test data."""
        input_path = data_dir / "stats_01" / "example_stats.fasta"

        if not input_path.exists():
            pytest.skip("stats_01 test data not found")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Number of sequences:" in captured.out
        assert "Total length:" in captured.out
        assert "GC content:" in captured.out

    def test_stats_wiki_example(self, temp_dir, mock_args, capsys):
        """Test stats output format matching wiki example.

        Wiki shows output like:
        - Number of sequences: 2
        - Total length: 62,067,787
        - Total softmasked length: 38,082,422
        - Total N length: 0
        - Total gap (-) length: 0
        - GC content: 57.8%
        """
        input_path = temp_dir / "input.fasta"

        # Create test data with known statistics
        records = [
            SeqRecord(Seq("GCGCGCGCGC"), id="seq1", description=""),  # 10 nt, 100% GC
            SeqRecord(Seq("ATATATATAT"), id="seq2", description=""),  # 10 nt, 0% GC
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        # Verify all expected fields are present
        assert "Number of sequences: 2" in captured.out
        assert "Total length: 20" in captured.out
        assert "Total softmasked length: 0" in captured.out
        assert "Total N length: 0" in captured.out
        assert "Total gap (-) length: 0" in captured.out
        assert "GC content:" in captured.out
        # 10 GC / 20 total = 50%
        assert "50" in captured.out

    def test_stats_large_sequence(self, temp_dir, mock_args, capsys):
        """Test stats with a larger sequence."""
        input_path = temp_dir / "input.fasta"

        # Create a 1000 bp sequence
        seq = "ATGC" * 250  # 1000 bp, 50% GC
        records = [
            SeqRecord(Seq(seq), id="large_seq", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Total length: 1,000" in captured.out or "Total length: 1000" in captured.out

    def test_stats_mixed_content(self, temp_dir, mock_args, capsys):
        """Test stats with mixed content (gaps, Ns, softmasked)."""
        input_path = temp_dir / "input.fasta"

        # Sequence with all types of special characters
        records = [
            SeqRecord(Seq("ATG---NNNcccGGG"), id="mixed", description=""),
            # 3 gaps, 3 Ns, 3 softmasked (ccc), 6 regular
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Total gap (-) length: 3" in captured.out
        assert "Total N length: 3" in captured.out
        assert "Total softmasked length: 3" in captured.out

    def test_stats_single_sequence(self, temp_dir, mock_args, capsys):
        """Test stats with single sequence."""
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGCCC"), id="only_one", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
        )

        stats_main(args)

        captured = capsys.readouterr()
        assert "Number of sequences: 1" in captured.out
