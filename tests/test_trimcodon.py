"""
Tests for cdskit trimcodon command.
"""

import json
from pathlib import Path

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.trimcodon import trimcodon_main


class TestTrimcodonMain:
    """Tests for trimcodon_main function."""

    def test_trimcodon_filters_by_clean_fraction(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCCGGG"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC---"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCC---"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            min_clean_fraction=0.67,
            report='',
        )

        trimcodon_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(record.seq) for record in result] == ["ATGCCC", "ATGCCC", "ATGCCC"]

    def test_trimcodon_treats_stop_codon_columns_as_unclean(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGTAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            min_clean_fraction=1.0,
            report='',
        )

        trimcodon_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(record.seq) for record in result] == ["ATGCCC", "ATGCCC"]

    def test_trimcodon_writes_json_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.json"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            min_clean_fraction=1.0,
            report=str(report_path),
        )

        trimcodon_main(args)

        report = json.loads(report_path.read_text())
        assert report["num_removed_codon_sites"] == 1
        assert report["removed_codon_sites_1based"] == [2]

    def test_trimcodon_rejects_unaligned_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(temp_dir / "output.fasta"),
            min_clean_fraction=0.5,
            report='',
        )

        with pytest.raises(Exception) as exc_info:
            trimcodon_main(args)
        assert "correctly aligned" in str(exc_info.value)
