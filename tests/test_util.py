"""
Tests for cdskit/util.py utility functions.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from io import StringIO

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit import util


class TestReadSeqs:
    """Tests for read_seqs function."""

    def test_read_fasta_file(self, temp_dir):
        """Test reading FASTA file from path."""
        fasta_path = temp_dir / "test.fasta"
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(fasta_path), "fasta")

        result = util.read_seqs(str(fasta_path), "fasta")
        assert len(result) == 2
        assert str(result[0].seq) == "ATGAAATGA"
        assert result[0].id == "seq1"

    def test_read_empty_file(self, temp_dir):
        """Test reading empty FASTA file."""
        fasta_path = temp_dir / "empty.fasta"
        fasta_path.write_text("")

        result = util.read_seqs(str(fasta_path), "fasta")
        assert len(result) == 0


class TestWriteSeqs:
    """Tests for write_seqs function."""

    def test_write_fasta_file(self, temp_dir):
        """Test writing FASTA to file."""
        fasta_path = temp_dir / "output.fasta"
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
        ]

        util.write_seqs(records, str(fasta_path), "fasta")

        # Read back and verify
        result = list(Bio.SeqIO.parse(str(fasta_path), "fasta"))
        assert len(result) == 1
        assert str(result[0].seq) == "ATGAAATGA"


class TestStopIfNotMultipleOfThree:
    """Tests for stop_if_not_multiple_of_three function."""

    def test_valid_sequences(self):
        """Test with sequences that are multiples of 3."""
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),  # 6 nt
            SeqRecord(Seq("ATGAAATGA"), id="seq2"),  # 9 nt
        ]
        # Should not raise
        util.stop_if_not_multiple_of_three(records)

    def test_invalid_sequence_length(self):
        """Test with sequence not multiple of 3."""
        records = [
            SeqRecord(Seq("ATGAA"), id="seq1"),  # 5 nt
        ]
        with pytest.raises(Exception) as exc_info:
            util.stop_if_not_multiple_of_three(records)
        assert "multiple of three" in str(exc_info.value)

    def test_mixed_sequences(self):
        """Test with mix of valid and invalid sequences."""
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),  # 6 nt - valid
            SeqRecord(Seq("ATGAA"), id="seq2"),  # 5 nt - invalid
        ]
        with pytest.raises(Exception):
            util.stop_if_not_multiple_of_three(records)


class TestStopIfNotAligned:
    """Tests for stop_if_not_aligned function."""

    def test_aligned_sequences(self):
        """Test with aligned sequences (same length)."""
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),
            SeqRecord(Seq("ATGCCC"), id="seq2"),
        ]
        # Should not raise
        util.stop_if_not_aligned(records)

    def test_unaligned_sequences(self):
        """Test with sequences of different lengths."""
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1"),
            SeqRecord(Seq("ATG"), id="seq2"),
        ]
        with pytest.raises(Exception) as exc_info:
            util.stop_if_not_aligned(records)
        assert "not identical" in str(exc_info.value)

    def test_single_sequence(self):
        """Test with single sequence (always aligned)."""
        records = [SeqRecord(Seq("ATGAAA"), id="seq1")]
        util.stop_if_not_aligned(records)


class TestTranslateRecords:
    """Tests for translate_records function."""

    def test_basic_translation(self):
        """Test basic protein translation."""
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1"),  # M K *
        ]
        result = util.translate_records(records, 1)
        assert str(result[0].seq) == "MK*"

    def test_translation_with_gaps(self):
        """Test translation with gap characters."""
        records = [
            SeqRecord(Seq("ATG---TGA"), id="seq1"),  # M - *
        ]
        result = util.translate_records(records, 1)
        assert str(result[0].seq) == "M-*"

    def test_translation_different_codon_tables(self):
        """Test translation with different codon tables."""
        records = [
            SeqRecord(Seq("ATGTTGTGA"), id="seq1"),  # Standard: M L *, Mitochondrial: M L W
        ]
        # Standard code
        result1 = util.translate_records(records, 1)
        # Vertebrate mitochondrial
        result2 = util.translate_records(records, 2)
        assert str(result1[0].seq) == "ML*"
        assert str(result2[0].seq) == "MLW"


class TestRecords2Array:
    """Tests for records2array function."""

    def test_basic_conversion(self):
        """Test conversion of records to numpy array."""
        records = [
            SeqRecord(Seq("ATGC"), id="seq1"),
            SeqRecord(Seq("GCTA"), id="seq2"),
        ]
        result = util.records2array(records)
        assert result.shape == (2, 4)
        assert list(result[0]) == ['A', 'T', 'G', 'C']
        assert list(result[1]) == ['G', 'C', 'T', 'A']


class TestReadGff:
    """Tests for read_gff function."""

    def test_read_gff_file(self, gff_file):
        """Test reading GFF file."""
        result = util.read_gff(str(gff_file))
        assert 'header' in result
        assert 'data' in result
        assert len(result['header']) == 1  # ##gff-version 3
        assert len(result['data']) == 3  # 3 features

    def test_gff_data_structure(self, gff_file):
        """Test GFF data has correct structure."""
        result = util.read_gff(str(gff_file))
        data = result['data']
        # Check first record
        assert data[0]['seqid'] == 'seq1'
        assert data[0]['type'] == 'gene'
        assert data[0]['start'] == 1
        assert data[0]['end'] == 100


class TestCoordinates2Ranges:
    """Tests for coordinates2ranges function."""

    def test_consecutive_coordinates(self):
        """Test with consecutive coordinates."""
        coords = [1, 2, 3, 4, 5]
        result = util.coordinates2ranges(coords)
        assert result == [(1, 5)]

    def test_non_consecutive_coordinates(self):
        """Test with gaps in coordinates."""
        coords = [1, 2, 3, 10, 11, 12]
        result = util.coordinates2ranges(coords)
        assert result == [(1, 3), (10, 12)]

    def test_single_coordinate(self):
        """Test with single coordinate."""
        coords = [5]
        result = util.coordinates2ranges(coords)
        assert result == [(5, 5)]

    def test_empty_coordinates(self):
        """Test with empty list."""
        coords = []
        result = util.coordinates2ranges(coords)
        assert result == []

    def test_multiple_ranges(self):
        """Test with multiple separate ranges."""
        coords = [1, 5, 6, 7, 20]
        result = util.coordinates2ranges(coords)
        assert result == [(1, 1), (5, 7), (20, 20)]
