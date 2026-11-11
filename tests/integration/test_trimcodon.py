"""
Tests for cdskit trimcodon command.
"""

import json
import math

import Bio.SeqIO
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit import trimcodon
from cdskit.tsvio import read_tsv
from cdskit.trimcodon import trimcodon_main


class TestTrimcodonMain:
    """Tests for trimcodon_main function."""

    @pytest.mark.parametrize("extension", ["json", "tsv"])
    def test_report_failure_rolls_back_sequence_and_report(
        self, temp_dir, mock_args, monkeypatch, extension
    ):
        source = temp_dir / "input.fa"
        output, report = temp_dir / "output.fa", temp_dir / f"sites.{extension}"
        source.write_text(">x\nATGGCTCCC\n")
        output.write_bytes(b"previous sequence\n")
        report.write_bytes(b"previous report\n")
        build_rows = trimcodon._site_summaries_from_counts

        def fail_after_one_row(counts):
            yield next(build_rows(counts))
            raise RuntimeError("site report generation failed")

        monkeypatch.setattr(
            trimcodon, "_site_summaries_from_counts", fail_after_one_row
        )
        args = mock_args(
            seqfile=str(source),
            outfile=str(output),
            report=str(report),
            min_clean_fraction=0.5,
        )
        with pytest.raises(RuntimeError, match="site report generation failed"):
            trimcodon_main(args)
        assert output.read_bytes() == b"previous sequence\n"
        assert report.read_bytes() == b"previous report\n"

    @pytest.mark.parametrize("threshold", [1 / 3, math.nextafter(1 / 3, 1)])
    def test_trimcodon_without_report_matches_site_decisions(
        self, temp_dir, mock_args, threshold
    ):
        source = temp_dir / "input.fa"
        before, after = temp_dir / "with_report.fa", temp_dir / "without_report.fa"
        source.write_text(">a\nATGTGAGCTGCN\n>b\nATGTGA---TAR\n>c\nATGNNNTAN?AG\n")
        for output, report in ((before, str(temp_dir / "sites.json")), (after, "")):
            trimcodon_main(
                mock_args(
                    seqfile=str(source),
                    outfile=str(output),
                    codontable=27,
                    min_clean_fraction=threshold,
                    report=report,
                )
            )
        assert before.read_bytes() == after.read_bytes()

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
            report="",
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
            report="",
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

    def test_trimcodon_writes_versioned_tsv_report(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.tsv"
        Bio.SeqIO.write(
            [
                SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
                SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
            ],
            str(input_path),
            "fasta",
        )
        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            min_clean_fraction=1.0,
            report=str(report_path),
        )

        trimcodon_main(args)

        rows, fieldnames = read_tsv(str(report_path), return_fieldnames=True)
        assert fieldnames[:2] == ["schema_version", "section"]
        assert {row["schema_version"] for row in rows} == {"2"}
        assert any(row["section"] == "summary" for row in rows)
        assert any(row["section"] == "site" for row in rows)

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
            report="",
        )

        with pytest.raises(ValueError) as exc_info:
            trimcodon_main(args)
        assert "correctly aligned" in str(exc_info.value)
