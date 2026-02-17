"""
Tests for cdskit label command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.label import (
    apply_char_replacement,
    clip_label_ids,
    label_main,
    parse_replace_chars,
    uniquify_label_ids,
)


class TestLabelHelpers:
    """Tests for label helper functions."""

    def test_parse_replace_chars(self):
        from_chars, to_char = parse_replace_chars(":|--_")
        assert from_chars == [":", "|"]
        assert to_char == "_"

    def test_apply_char_replacement_counts_records_once(self):
        records = [
            SeqRecord(Seq("ATG"), id="a:b:c", description=""),
            SeqRecord(Seq("ATG"), id="plain", description=""),
        ]
        replaced = apply_char_replacement(records, [":"], "_")
        assert replaced == 1
        assert records[0].id == "a_b_c"
        assert records[1].id == "plain"

    def test_clip_label_ids(self):
        records = [
            SeqRecord(Seq("ATG"), id="long_name_here", description=""),
            SeqRecord(Seq("ATG"), id="short", description=""),
        ]
        clipped = clip_label_ids(records, 5)
        assert clipped == 1
        assert records[0].id == "long_"
        assert records[1].id == "short"

    def test_uniquify_label_ids(self):
        records = [
            SeqRecord(Seq("ATG"), id="dup", description="d1"),
            SeqRecord(Seq("ATG"), id="dup", description="d2"),
            SeqRecord(Seq("ATG"), id="uniq", description="d3"),
        ]
        resolved_count, nonunique_names = uniquify_label_ids(records)
        assert resolved_count == 2
        assert nonunique_names == ["dup"]
        assert [r.id for r in records] == ["dup_1", "dup_2", "uniq"]
        assert records[0].description == ""
        assert records[1].description == ""


class TestLabelMain:
    """Tests for label_main function."""

    def test_label_replace_chars(self, temp_dir, mock_args):
        """Test replacing characters in sequence labels."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq:1:name", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            replace_chars=':--_',  # Replace : with _
            clip_len=0,
            unique=False,
        )

        label_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert result[0].id == "seq_1_name"

    def test_label_clip_length(self, temp_dir, mock_args):
        """Test clipping sequence labels to max length."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="very_long_sequence_name", description=""),
            SeqRecord(Seq("ATGCCC"), id="short", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            replace_chars='',
            clip_len=10,
            unique=False,
        )

        label_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert result[0].id == "very_long_"  # Clipped to 10 chars
        assert result[1].id == "short"  # Unchanged (< 10 chars)

    def test_label_make_unique(self, temp_dir, mock_args):
        """Test making duplicate labels unique."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="duplicate", description=""),
            SeqRecord(Seq("ATGCCC"), id="duplicate", description=""),
            SeqRecord(Seq("ATGGGG"), id="duplicate", description=""),
            SeqRecord(Seq("ATGTTT"), id="unique_one", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            replace_chars='',
            clip_len=0,
            unique=True,
        )

        label_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        ids = [r.id for r in result]
        # All IDs should be unique
        assert len(ids) == len(set(ids))
        # Duplicates should have suffixes
        assert "duplicate_1" in ids
        assert "duplicate_2" in ids
        assert "duplicate_3" in ids

    def test_label_combined_operations(self, temp_dir, mock_args):
        """Test combining clip and unique operations."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="long_sequence_name_here", description=""),
            SeqRecord(Seq("ATGCCC"), id="long_sequence_name_here", description=""),  # Duplicate
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            replace_chars='',
            clip_len=15,
            unique=True,
        )

        label_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        ids = [r.id for r in result]
        # After clipping both become "long_sequence_na", then uniqueness adds suffixes
        assert len(ids) == len(set(ids))

    def test_label_no_changes(self, temp_dir, mock_args):
        """Test when no modifications are needed."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            replace_chars='',
            clip_len=0,
            unique=False,
        )

        label_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert result[0].id == "seq1"
        assert result[1].id == "seq2"

    def test_label_replace_single_char(self, temp_dir, mock_args):
        """Test replacing a single character type."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="a|b|c|d", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            replace_chars='|--_',  # Replace | with _
            clip_len=0,
            unique=False,
        )

        label_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert result[0].id == "a_b_c_d"
