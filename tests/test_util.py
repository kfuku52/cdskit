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
from Bio.SeqFeature import FeatureLocation, SeqFeature

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


class TestThreadHelpers:
    """Tests for thread-related utility helpers."""

    def test_resolve_threads_default_and_auto(self, monkeypatch):
        assert util.resolve_threads(None) == 1
        monkeypatch.setattr(util.os, "cpu_count", lambda: 7)
        assert util.resolve_threads(0) == 7

    def test_resolve_threads_rejects_negative(self):
        with pytest.raises(Exception) as exc_info:
            util.resolve_threads(-1)
        assert "--threads should be >= 0" in str(exc_info.value)

    def test_parallel_map_ordered_keeps_input_order(self):
        items = [5, 3, 1, 4, 2]
        result = util.parallel_map_ordered(items=items, worker=lambda x: x * 2, threads=3)
        assert result == [10, 6, 2, 8, 4]


class TestReadItemPerLineFile:
    """Tests for read_item_per_line_file function."""

    def test_reads_non_empty_lines_only(self, temp_dir):
        path = temp_dir / "items.txt"
        path.write_text("alpha\n\nbeta\n\ngamma\n")
        assert util.read_item_per_line_file(str(path)) == ["alpha", "beta", "gamma"]

    def test_strips_whitespace_around_each_item(self, temp_dir):
        path = temp_dir / "items_whitespace.txt"
        path.write_text(" alpha \n\tbeta\t\n\n gamma\n")
        assert util.read_item_per_line_file(str(path)) == ["alpha", "beta", "gamma"]


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


class TestStopIfNotDna:
    """Tests for stop_if_not_dna function."""

    def test_accepts_dna_sequences(self):
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1"),
            SeqRecord(Seq("ATGN--TGA"), id="seq2"),
        ]
        util.stop_if_not_dna(records)

    def test_rejects_rna_sequences(self):
        records = [
            SeqRecord(Seq("AUGAAATGA"), id="seq1"),
            SeqRecord(Seq("ATGaaauaa"), id="seq2"),
        ]
        with pytest.raises(Exception) as exc_info:
            util.stop_if_not_dna(records, label="--seqfile")
        assert "DNA-only input is required" in str(exc_info.value)
        assert "seq1,seq2" in str(exc_info.value)

    def test_rejects_non_dna_letters(self):
        records = [
            SeqRecord(Seq("ATGPPP"), id="seq_bad"),
            SeqRecord(Seq("ATGAAA"), id="seq_ok"),
        ]
        with pytest.raises(Exception) as exc_info:
            util.stop_if_not_dna(records, label="--seqfile")
        assert "DNA-only input is required" in str(exc_info.value)
        assert "seq_bad" in str(exc_info.value)
        assert "P" in str(exc_info.value)


class TestStopIfInvalidCodontable:
    def test_accepts_valid_codontable(self):
        util.stop_if_invalid_codontable(1)

    def test_rejects_invalid_codontable(self):
        with pytest.raises(Exception) as exc_info:
            util.stop_if_invalid_codontable(999)
        assert "Invalid --codontable" in str(exc_info.value)


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

    def test_translation_handles_question_and_dot_as_missing(self):
        records = [
            SeqRecord(Seq("ATG???CCC"), id="q1"),
            SeqRecord(Seq("ATG...CCC"), id="d1"),
        ]
        result = util.translate_records(records, 1)
        assert str(result[0].seq) == "MXP"
        assert str(result[1].seq) == "M-P"


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


class TestGetSeqname:
    """Tests for get_seqname function."""

    def test_builds_name_from_multiple_annotation_fields(self):
        record = SeqRecord(Seq("ATG"), id="r1")
        record.annotations["organism"] = "Homo sapiens"
        record.annotations["accessions"] = ["ABC123", "DEF456"]
        result = util.get_seqname(record, "organism_accessions")
        assert result == "Homo_sapiens_ABC123"

    def test_raises_for_unknown_annotation_key(self):
        record = SeqRecord(Seq("ATG"), id="r2")
        record.annotations["organism"] = "Homo sapiens"
        with pytest.raises(Exception) as exc_info:
            util.get_seqname(record, "organism_unknown")
        assert "Invalid --seqnamefmt element (unknown)" in str(exc_info.value)


class TestReplaceSeq2Cds:
    """Tests for replace_seq2cds function."""

    def test_replaces_sequence_with_cds_feature(self):
        record = SeqRecord(Seq("AAATGCCCCTTT"), id="cds_record")
        record.features = [
            SeqFeature(FeatureLocation(3, 9), type="CDS"),
        ]
        result = util.replace_seq2cds(record)
        assert result is not None
        assert str(result.seq) == "TGCCCC"

    def test_returns_none_when_no_cds_feature(self):
        record = SeqRecord(Seq("AAATGCCCCTTT"), id="no_cds_record")
        record.features = []
        result = util.replace_seq2cds(record)
        assert result is None


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

    def test_single_record_gff_is_returned_as_1d_array(self, temp_dir):
        """Single non-header line should still produce length-1 structured array."""
        path = temp_dir / "single.gff"
        path.write_text("##gff-version 3\nseq1\tsource\tgene\t1\t10\t.\t+\t.\tID=g1\n")
        result = util.read_gff(str(path))
        assert len(result["data"]) == 1
        assert result["data"][0]["seqid"] == "seq1"

    def test_read_gff_preserves_long_attributes(self, temp_dir):
        path = temp_dir / "long_attr.gff"
        long_attr = "ID=" + ("A" * 700)
        path.write_text(f"##gff-version 3\nseq1\tsource\tgene\t1\t10\t.\t+\t.\t{long_attr}\n")
        result = util.read_gff(str(path))
        assert len(result["data"]) == 1
        assert result["data"][0]["attributes"] == long_attr


class TestWriteGff:
    """Tests for write_gff function."""

    def test_write_and_read_roundtrip(self, temp_dir):
        out_path = temp_dir / "roundtrip.gff"
        dtype = [
            ('seqid', 'U100'),
            ('source', 'U100'),
            ('type', 'U100'),
            ('start', 'i4'),
            ('end', 'i4'),
            ('score', 'U100'),
            ('strand', 'U10'),
            ('phase', 'U10'),
            ('attributes', 'U500')
        ]
        data = np.array(
            [
                ("seq1", "src", "gene", 1, 100, ".", "+", ".", "ID=g1"),
                ("seq1", "src", "CDS", 10, 90, ".", "+", "0", "ID=c1"),
            ],
            dtype=dtype,
        )
        gff = {"header": ["##gff-version 3"], "data": data}

        util.write_gff(gff, str(out_path))
        reread = util.read_gff(str(out_path))
        assert reread["header"] == ["##gff-version 3"]
        assert len(reread["data"]) == 2
        assert reread["data"][1]["type"] == "CDS"


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
