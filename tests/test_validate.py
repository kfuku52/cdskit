"""
Tests for cdskit validate command.
"""

import json
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.validate import summarize_records, validate_main


class TestValidateHelpers:
    """Tests for validate helper logic."""

    def test_summarize_records_mixed_issues(self):
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seqA", description=""),   # terminal stop only
            SeqRecord(Seq("ATGTGACCC"), id="seqB", description=""),   # internal stop
            SeqRecord(Seq("ATGAAACCC"), id="seqB", description=""),   # duplicate id
            SeqRecord(Seq("ATGNNNCCC"), id="seqC", description=""),   # ambiguous codon
            SeqRecord(Seq("---------"), id="seqD", description=""),   # gap-only
            SeqRecord(Seq("ATGAA"), id="seqE", description=""),       # non-triplet
        ]
        summary = summarize_records(records=records, codontable=1)

        assert summary["num_sequences"] == 6
        assert summary["aligned"] is False
        assert summary["non_triplet_ids"] == ["seqE"]
        assert summary["duplicate_ids"] == ["seqB"]
        assert summary["gap_only_ids"] == ["seqD"]
        assert summary["internal_stop_ids"] == ["seqB"]
        assert summary["ambiguous_codons"] == 1
        assert summary["evaluable_codons"] > 0
        assert summary["ambiguous_codon_rate"] > 0
        assert summary["num_sequences_with_issues"] >= 4


class TestValidateMain:
    """Tests for validate_main function."""

    def test_validate_prints_summary(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "input.fasta"
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGTGACCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            codontable=1,
            report='',
        )
        validate_main(args)
        captured = capsys.readouterr()

        assert "Validation summary" in captured.out
        assert "num_sequences\t2" in captured.out
        assert "num_internal_stop_sequences\t1" in captured.out

    def test_validate_writes_json_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        report_path = temp_dir / "report.json"
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            codontable=1,
            report=str(report_path),
        )
        validate_main(args)

        report = json.loads(report_path.read_text())
        assert report["num_sequences"] == 2
        assert report["ambiguous_codons"] == 1
        assert "sequence_ids_with_issues" in report

    def test_validate_writes_tsv_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        report_path = temp_dir / "report.tsv"
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAATGA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            codontable=1,
            report=str(report_path),
        )
        validate_main(args)

        txt = report_path.read_text()
        assert txt.startswith("metric\tvalue")
        assert "num_sequences\t2" in txt

    def test_validate_empty_input(self, temp_dir, mock_args, capsys):
        input_path = temp_dir / "empty.fasta"
        input_path.write_text("")
        args = mock_args(
            seqfile=str(input_path),
            codontable=1,
            report='',
        )
        validate_main(args)
        captured = capsys.readouterr()
        assert "num_sequences\t0" in captured.out

    def test_validate_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        report_single = temp_dir / "single.json"
        report_threaded = temp_dir / "threaded.json"
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGTGACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq3", description=""),
            SeqRecord(Seq("---------"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            codontable=1,
            report=str(report_single),
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            codontable=1,
            report=str(report_threaded),
            threads=3,
        )

        validate_main(args_single)
        validate_main(args_threaded)

        single = json.loads(report_single.read_text())
        threaded = json.loads(report_threaded.read_text())
        assert single == threaded
