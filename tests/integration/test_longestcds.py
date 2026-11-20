"""
Tests for cdskit longestcds command.
"""

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.cli import main as cli_main
from cdskit.longestcds import longestcds_main


def test_longestorf_exports_distinct_same_transcript_alternatives(tmp_path):
    source, output, alternatives = (
        tmp_path / name for name in ("transcripts.fa", "baseline.fa", "alt.fa")
    )
    source.write_text(">seq1\nATGCATTTTGACTTAAATACC\n")
    cli_main(
        [
            "longestorf",
            "--seq_file",
            str(source),
            "--out_file",
            str(output),
            "--selection",
            "longest",
            "--alternate_orfs",
            str(alternatives),
            "--strand",
            "forward",
        ]
    )
    baseline = Bio.SeqIO.read(output, "fasta")
    candidates = list(Bio.SeqIO.parse(alternatives, "fasta"))
    assert str(baseline.seq) == "ATGCATTTTGACTTAAATACC"
    assert 1 <= len(candidates) <= 6
    assert len({str(candidate.seq) for candidate in candidates}) == len(candidates)
    assert all(
        candidate.id.startswith("seq1.cdskit-orf-") and candidate.seq != baseline.seq
        for candidate in candidates
    )
    assert any(
        str(candidate.seq) == "TTGACTTAA" and "strand=+" in candidate.description
        for candidate in candidates
    )


class TestLongestCdsMain:
    """Tests for longestcds_main function."""

    def test_longestcds_accepts_codontable_name_when_called_programmatically(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAATAG"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path), outfile=str(output_path), codontable="Standard"
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert str(result[0].seq) == "ATGAAATAG"

    def test_longestcds_plus_strand_frame1_complete_orf(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("GGGATGAAATAGCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path), outfile=str(output_path), codontable=1
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert str(result[0].seq) == "ATGAAATAG"
        assert result[0].description == "seq1"

    def test_longestcds_with_annotation_outputs_metadata_in_header(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("GGGATGAAATAGCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert "strand=+" in result[0].description
        assert "frame=1" in result[0].description
        assert "category=complete" in result[0].description

    def test_longestcds_plus_strand_frame2_complete_orf(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("CATGAAATAG"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAATAG"
        assert "strand=+" in result[0].description
        assert "frame=2" in result[0].description
        assert "category=complete" in result[0].description

    def test_longestcds_minus_strand_complete_orf(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("GGGCTATTTCATCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAATAG"
        assert "strand=-" in result[0].description
        assert "category=complete" in result[0].description

    def test_longestcds_selects_longest_complete_orf(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAATAGATGAAAAAATGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAAAAATGA"
        assert "category=complete" in result[0].description

    def test_longestcds_falls_back_to_partial_start_orf(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("CCCATGAAACCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAACCC"
        assert "category=partial" in result[0].description

    def test_longestcds_falls_back_to_no_start_segment(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("CCCCCCCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "CCCCCCCCC"
        assert "category=no_start" in result[0].description

    def test_longestcds_short_sequence_outputs_empty(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("AT"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            annotate_seqname=True,
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == ""
        assert "category=none" in result[0].description

    def test_longestcds_preserves_record_order(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [
            SeqRecord(Seq("GGGATGAAATAGCCC"), id="zebra", description=""),
            SeqRecord(Seq("CCCAAACCC"), id="apple", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path), outfile=str(output_path), codontable=1
        )
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["zebra", "apple"]


class TestLongestCommandAliases:
    """Tests for longestorf canonical command and longestcds alias behavior."""

    def test_longestcds_cli_warns_deprecated_alias(self, temp_dir, capsys):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("GGGATGAAATAGCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        return_code = cli_main(
            [
                "longestcds",
                "--seq_file",
                str(input_path),
                "--out_file",
                str(output_path),
            ]
        )
        captured = capsys.readouterr()

        assert return_code == 0
        assert "deprecated" in captured.err.lower()
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAATAG"

    def test_longestorf_cli_is_canonical_without_deprecation_warning(
        self, temp_dir, capsys
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("GGGATGAAATAGCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        return_code = cli_main(
            [
                "longestorf",
                "--seq_file",
                str(input_path),
                "--out_file",
                str(output_path),
            ]
        )
        captured = capsys.readouterr()

        assert return_code == 0
        assert "deprecated" not in captured.err.lower()
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "ATGAAATAG"
