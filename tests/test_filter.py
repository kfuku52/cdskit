"""
Tests for cdskit filter command.
"""

import json
from pathlib import Path

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.filter import filter_main


class TestFilterMain:
    """Tests for filter_main function."""

    def test_filter_drops_requested_sequences_and_keeps_longest_duplicate(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="keep", description=""),
            SeqRecord(Seq("ATGAA"), id="nontrip", description=""),
            SeqRecord(Seq("---------"), id="gap", description=""),
            SeqRecord(Seq("ATGTGACCC"), id="internal_stop", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="ambiguous", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="dup", description=""),
            SeqRecord(Seq("ATGAAACCCAAA"), id="dup", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=True,
            drop_gap_only=True,
            drop_internal_stop=True,
            max_ambiguous_codon_rate=0.2,
            dedup='keep-longest',
            report='',
        )

        filter_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [record.id for record in result] == ["keep", "dup"]
        assert str(result[1].seq) == "ATGAAACCCAAA"

    def test_filter_applies_dedup_after_quality_filtering(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("---------"), id="dup", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="dup", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=False,
            drop_gap_only=True,
            drop_internal_stop=False,
            max_ambiguous_codon_rate=1.0,
            dedup='keep-first',
            report='',
        )

        filter_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert result[0].id == "dup"
        assert str(result[0].seq) == "ATGAAACCC"

    def test_filter_writes_json_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.json"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="keep", description=""),
            SeqRecord(Seq("ATGAA"), id="nontrip", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=True,
            drop_gap_only=False,
            drop_internal_stop=False,
            max_ambiguous_codon_rate=1.0,
            dedup='no',
            report=str(report_path),
        )

        filter_main(args)

        report = json.loads(report_path.read_text())
        assert report["num_dropped_sequences"] == 1
        assert report["dropped_ids_by_reason"]["non_triplet"] == ["nontrip"]

    def test_filter_rejects_invalid_ambiguous_rate(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAACCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=False,
            drop_gap_only=False,
            drop_internal_stop=False,
            max_ambiguous_codon_rate=1.5,
            dedup='no',
            report='',
        )

        with pytest.raises(Exception) as exc_info:
            filter_main(args)
        assert "--max_ambiguous_codon_rate" in str(exc_info.value)
