"""
Tests for cdskit filter command.
"""

import json

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.filter import filter_main
from cdskit.tsvio import read_tsv


class TestFilterMain:
    """Tests for filter_main function."""

    def test_filter_drops_requested_sequences_and_keeps_longest_duplicate(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="keep", description=""),
            SeqRecord(Seq("ATGAA"), id="nontrip", description=""),
            SeqRecord(Seq("---------"), id="gap", description=""),
            SeqRecord(Seq("ATGTGACCC"), id="internal_stop", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="ambiguous", description=""),
            SeqRecord(Seq("ATG---CCC"), id="gappy", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="dup", description=""),
            SeqRecord(Seq("ATGAAACCCAAA"), id="dup", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=True,
            drop_internal_stop=True,
            min_clean_codon_fraction=0.8,
            dedup="keep-longest",
            report="",
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
            drop_internal_stop=False,
            min_clean_codon_fraction=0.5,
            dedup="keep-first",
            report="",
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
            drop_internal_stop=False,
            min_clean_codon_fraction=0.0,
            dedup="no",
            report=str(report_path),
        )

        filter_main(args)

        report = json.loads(report_path.read_text())
        assert report["num_dropped_sequences"] == 1
        assert report["drop_counts_by_reason"]["non_triplet"] == 1
        assert report["dropped_ids_by_reason"]["non_triplet"] == ["nontrip"]
        assert report["sequence_reports"][0]["id"] == "keep"
        assert report["sequence_reports"][0]["kept"] is True
        assert report["sequence_reports"][1]["id"] == "nontrip"
        assert report["sequence_reports"][1]["drop_reasons"] == ["non_triplet"]
        assert report["sequence_reports"][1]["length_nt"] == 5
        assert report["sequence_reports"][1]["tail_nt"] == 2

    def test_filter_drops_sequences_below_clean_codon_fraction(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.json"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="clean", description=""),
            SeqRecord(Seq("ATG---CCC"), id="gappy", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="ambiguous", description=""),
            SeqRecord(Seq("ATGTGACCC"), id="stop", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=False,
            drop_internal_stop=False,
            min_clean_codon_fraction=0.8,
            dedup="no",
            report=str(report_path),
        )

        filter_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [record.id for record in result] == ["clean"]
        report = json.loads(report_path.read_text())
        assert report["dropped_ids_by_reason"]["clean_codon_fraction"] == [
            "gappy",
            "ambiguous",
            "stop",
        ]

    def test_filter_writes_richer_tsv_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.tsv"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="keep", description=""),
            SeqRecord(Seq("ATG---CCC"), id="gappy", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=False,
            drop_internal_stop=False,
            min_clean_codon_fraction=0.8,
            dedup="no",
            report=str(report_path),
        )

        filter_main(args)

        rows, fieldnames = read_tsv(str(report_path), return_fieldnames=True)
        assert fieldnames[:2] == ["schema_version", "section"]
        assert {row["schema_version"] for row in rows} == {"2"}
        reason = next(
            row
            for row in rows
            if row["section"] == "drop_reason"
            and row["drop_reason"] == "clean_codon_fraction"
        )
        assert reason["drop_reason"] == "clean_codon_fraction"
        assert reason["count"] == "1"
        sequence_rows = {row["id"]: row for row in rows if row["section"] == "sequence"}
        assert sequence_rows["keep"]["kept"] == "yes"
        assert sequence_rows["gappy"]["kept"] == "no"
        assert json.loads(sequence_rows["gappy"]["drop_reasons"]) == [
            "clean_codon_fraction"
        ]
        assert sequence_rows["gappy"]["clean_codon_fraction"] == "0.666667"

    @pytest.mark.parametrize("fraction", [1.5, float("nan")])
    def test_filter_rejects_invalid_clean_codon_fraction(
        self, temp_dir, mock_args, fraction
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAACCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            drop_non_triplet=False,
            drop_internal_stop=False,
            min_clean_codon_fraction=fraction,
            dedup="no",
            report="",
        )

        with pytest.raises(ValueError) as exc_info:
            filter_main(args)
        assert "--min_clean_codon_fraction" in str(exc_info.value)
