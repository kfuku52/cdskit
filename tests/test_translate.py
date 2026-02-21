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

    @pytest.mark.parametrize(
        "seq,expected",
        [
            ("ATG???TGA", "MX*"),
            ("ATG...TGA", "M-*"),
        ],
    )
    def test_translate_handles_missing_question_and_dot_codons(self, temp_dir, mock_args, seq, expected):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [SeqRecord(Seq(seq), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            to_stop=False,
        )
        translate_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert str(result[0].seq) == expected

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

    def test_translate_rejects_rna_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [SeqRecord(Seq("AUGAAATGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            to_stop=False,
        )
        with pytest.raises(Exception) as exc_info:
            translate_main(args)
        assert "DNA-only input is required" in str(exc_info.value)

    def test_translate_rejects_invalid_codontable(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [SeqRecord(Seq("ATGAAATGA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=999,
            to_stop=False,
        )
        with pytest.raises(Exception) as exc_info:
            translate_main(args)
        assert "Invalid --codontable" in str(exc_info.value)

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

    def test_translate_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq2", description=""),
            SeqRecord(Seq("ATGTTTTAA"), id="seq3", description=""),
            SeqRecord(Seq("ATG---TGA"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            codontable=1,
            to_stop=False,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            codontable=1,
            to_stop=False,
            threads=4,
        )

        translate_main(args_single)
        translate_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))

        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]
