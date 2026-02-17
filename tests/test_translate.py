"""
Tests for cdskit translate command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.translate import translate_main


class TestTranslateMain:
    """Tests for translate_main function."""

    def test_translate_basic(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATG---TGA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            to_stop=False,
        )
        translate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1", "seq2"]
        assert str(result[0].seq) == "MK*"
        assert str(result[1].seq) == "M-*"

    def test_translate_to_stop(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [SeqRecord(Seq("ATGAAATGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            to_stop=True,
        )
        translate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "MK"

    def test_translate_respects_codon_table(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [SeqRecord(Seq("ATGTGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=2,  # Vertebrate mitochondrial: TGA -> W
            to_stop=False,
        )
        translate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == "MW"

    def test_translate_rejects_non_triplet(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            to_stop=False,
        )
        with pytest.raises(Exception) as exc_info:
            translate_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_translate_empty_input(self, temp_dir, mock_args):
        input_path = temp_dir / "empty.fasta"
        output_path = temp_dir / "output.fasta"
        input_path.write_text("")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            to_stop=False,
        )
        translate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 0
