"""
Tests for cdskit hammer command.
"""

import pytest

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.hammer import hammer_main


class TestHammerMain:
    """Tests for hammer_main function."""

    def test_hammer_handles_question_codon_as_missing_not_error(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATG???CCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="3",
            prevent_gap_only=True,
        )
        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(r.seq) for r in result] == ["ATGCCC", "ATGCCC", "ATGCCC"]

    def test_hammer_prevent_gap_only_relaxes_for_question_missing_codons(
        self, temp_dir, mock_args
    ):
        """prevent_gap_only should treat ?/. as missing in codon-level gap-only checks."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("??????ATG"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAA---"), id="seq3", description=""),
            SeqRecord(Seq("ATGAAA---"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="3",
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # With correct missing handling, --nail should relax from 3 to 2
        # so seq2 is not all-missing in output.
        assert [str(r.seq) for r in result] == [
            "ATGAAACCC",
            "??????ATG",
            "ATGAAA---",
            "ATGAAA---",
        ]

    def test_hammer_nail_all(self, temp_dir, mock_args):
        """Test hammer with --nail all."""
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
            nail="all",
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [(r.id, str(r.seq)) for r in result] == [
            ("seq1", "ATG"),
            ("seq2", "ATG"),
        ]

    def test_hammer_relaxed_nail(self, temp_dir, mock_args):
        """Test hammer with relaxed nail threshold."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATG---TGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq2", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="2",  # Only require 2/3 sequences
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [(r.id, str(r.seq)) for r in result] == [
            ("seq1", "ATG---"),
            ("seq2", "ATGCCC"),
            ("seq3", "ATGCCC"),
        ]

    def test_hammer_prevent_gap_only(self, temp_dir, mock_args):
        """Test hammer prevents gap-only sequences."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Design sequences where strict nail would create gap-only
        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("------"), id="seq2", description=""),  # All gaps
            SeqRecord(Seq("ATGAAA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="3",  # Would remove all columns
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(r.seq) for r in result] == ["ATGAAA", "------", "ATGAAA"]

    def test_hammer_rejects_non_aligned(self, temp_dir, mock_args):
        """Test hammer rejects non-aligned sequences."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATG"), id="seq2", description=""),  # Different length
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="all",
            prevent_gap_only=True,
        )

        with pytest.raises(ValueError) as exc_info:
            hammer_main(args)
        assert "not identical" in str(exc_info.value)

    def test_hammer_nail_1_gap_only(self, temp_dir, mock_args):
        """Test hammer with --nail 1: removes columns where fewer than 1 have data."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Create alignment where positions 4-6 are ALL gaps
        records = [
            SeqRecord(Seq("ATG---TGA"), id="seq1", description=""),
            SeqRecord(Seq("ATG---TGA"), id="seq2", description=""),
            SeqRecord(Seq("ATG---TGA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="1",  # Require at least 1 sequence to have non-gap
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(r.seq) for r in result] == ["ATG", "ATG", "ATG"]

    def test_hammer_empty_input_writes_empty_output(self, temp_dir, mock_args):
        input_path = temp_dir / "empty.fasta"
        output_path = temp_dir / "output.fasta"
        input_path.write_text("")

        hammer_main(
            mock_args(
                seqfile=str(input_path),
                outfile=str(output_path),
                codontable=1,
                nail="all",
                prevent_gap_only=True,
            )
        )

        assert output_path.read_text() == ""

    def test_hammer_all_gap_sequences_with_prevent_gap_only(self, temp_dir, mock_args):
        """Test hammer with sequences that become all-gaps after filtering.

        Issue #3 related: When --prevent_gap_only is used and filtering produces
        gap-only sequences, hammer should handle this gracefully.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Create sequences where one is sparse - after strict filtering might become gap-only
        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(
                Seq("---AAA---"), id="seq2", description=""
            ),  # Only middle codon has data
            SeqRecord(Seq("ATGAAATGA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail="all",  # Requires all 3 to have non-gap - middle codon only
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(r.seq) for r in result] == ["AAA", "AAA", "AAA"]

        # The all_gaps sequence should still be all gaps, but that's allowed when nail is reduced
        # The stderr should show nail adjustment message
        # (Note: This depends on actual implementation behavior)

    @pytest.mark.parametrize("nail", ["0", "-1"])
    def test_hammer_rejects_non_positive_nail(self, temp_dir, mock_args, nail):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail=nail,
            prevent_gap_only=True,
        )

        with pytest.raises(ValueError) as exc_info:
            hammer_main(args)
        assert "--nail should be a positive integer" in str(exc_info.value)
