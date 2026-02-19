"""
Tests for cdskit longestcds command.
"""

from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.longestcds import longestcds_main


class TestLongestCdsMain:
    """Tests for longestcds_main function."""

    def test_longestcds_plus_strand_frame1_complete_orf(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("GGGATGAAATAGCCC"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(seqfile=str(input_path), outfile=str(output_path), codontable=1)
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert str(result[0].seq) == "ATGAAATAG"
        assert result[0].description == "seq1"

    def test_longestcds_with_annotation_outputs_metadata_in_header(self, temp_dir, mock_args):
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

        args = mock_args(seqfile=str(input_path), outfile=str(output_path), codontable=1)
        longestcds_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["zebra", "apple"]
