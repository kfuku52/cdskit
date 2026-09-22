"""
Tests for cdskit rmseq command.
"""

import pytest

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.rmseq import problematic_rate, rmseq_main, should_remove_record


class TestRmseqHelpers:
    """Tests for rmseq helper functions."""

    def test_problematic_rate_deduplicates_problematic_chars(self):
        rate = problematic_rate("NNAA", "NN")
        assert rate == pytest.approx(2 / 4)

    def test_should_remove_record_matches_id_not_name(self):
        record = SeqRecord(
            Seq("ATGAAA"), id="remove_me", name="other_name", description=""
        )
        remove = should_remove_record(
            record=record,
            seqname_pattern="remove.*",
            problematic_percent=0,
            problematic_chars=["N"],
        )
        assert remove is True


class TestRmseqMain:
    """Tests for rmseq_main function."""

    def test_rmseq_by_name_regex(self, temp_dir, mock_args):
        """Test removing sequences by name regex."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="keep_this", description=""),
            SeqRecord(Seq("ATGCCC"), id="remove_me", description=""),
            SeqRecord(Seq("ATGGGG"), id="keep_also", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="remove.*",  # Regex to match sequences to remove
            problematic_percent=0,
            problematic_char=["N", "X", "-", "?"],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 2
        ids = [r.id for r in result]
        assert "keep_this" in ids
        assert "keep_also" in ids
        assert "remove_me" not in ids

    def test_rmseq_handles_empty_sequence_without_crash(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq(""), id="empty", description=""),
            SeqRecord(Seq("ATGAAA"), id="clean", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=10,
            problematic_char=["N"],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["empty", "clean"]

    def test_rmseq_multiple_problematic_chars(self, temp_dir, mock_args):
        """Test counting multiple problematic character types."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="clean", description=""),
            SeqRecord(Seq("ATN-X?"), id="mixed_problems", description=""),  # 4/6 = 67%
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=50,
            problematic_char=["N", "X", "-", "?"],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert result[0].id == "clean"

    def test_rmseq_combined_filters(self, temp_dir, mock_args):
        """Test combining name and character filters."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="good_seq", description=""),
            SeqRecord(Seq("ATGCCC"), id="bad_name", description=""),  # Matches regex
            SeqRecord(Seq("NNNAAA"), id="good_name", description=""),  # 50% N
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="bad.*",
            problematic_percent=50,
            problematic_char=["N"],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert result[0].id == "good_seq"

    def test_rmseq_exact_match_data(self, data_dir, temp_dir, mock_args):
        """Test rmseq with rmseq_01 data comparing to expected output."""
        input_path = data_dir / "rmseq_01" / "input.fasta"
        expected_path = data_dir / "rmseq_01" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        assert input_path.exists(), "required tracked fixture rmseq_01 input is missing"
        assert expected_path.exists(), (
            "required tracked fixture rmseq_01 output is missing"
        )

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="Arabidopsis_thaliana.*",
            problematic_percent=50,
            problematic_char=["N"],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        expected = list(Bio.SeqIO.parse(str(expected_path), "fasta"))

        # Compare IDs
        result_ids = set(r.id for r in result)
        expected_ids = set(e.id for e in expected)
        assert result_ids == expected_ids

    def test_rmseq_gaps_as_problematic(self, temp_dir, mock_args):
        """Test removing sequences with too many gaps."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="no_gaps", description=""),
            SeqRecord(Seq("ATG---CCC"), id="some_gaps", description=""),  # 33% gaps
            SeqRecord(Seq("------CCC"), id="many_gaps", description=""),  # 67% gaps
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=50,
            problematic_char=["-"],  # Only count gaps
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        result_ids = [r.id for r in result]
        assert "no_gaps" in result_ids
        assert "some_gaps" in result_ids  # 33% < 50%
        assert "many_gaps" not in result_ids  # 67% >= 50%

    def test_rmseq_boundary_percent(self, temp_dir, mock_args):
        """Test behavior at exactly the boundary percent."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGNNN"), id="exactly_50", description=""),  # Exactly 50%
            SeqRecord(Seq("ATGNNA"), id="just_under_50", description=""),  # 33% (2/6 N)
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=50,
            problematic_char=["N"],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # exactly_50 should be removed (>= 50%)
        # just_under_50 should be kept
        assert len(result) == 1
        assert result[0].id == "just_under_50"

    def test_rmseq_counts_lowercase_problematic_chars(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGnnn"), id="lower_n", description=""),
            SeqRecord(Seq("ATGAAA"), id="clean", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=50,
            problematic_char="N",
        )

        rmseq_main(args)
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["clean"]

    def test_rmseq_rejects_invalid_regex(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="[",
            problematic_percent=0,
            problematic_char=["N"],
        )
        with pytest.raises(ValueError) as exc_info:
            rmseq_main(args)
        assert "Invalid regex in --seq_name_regex" in str(exc_info.value)

    @pytest.mark.parametrize(
        "problematic_percent", [-1, 101, float("nan"), float("inf")]
    )
    def test_rmseq_rejects_invalid_problematic_percent(
        self, temp_dir, mock_args, problematic_percent
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=problematic_percent,
            problematic_char=["N"],
        )
        with pytest.raises(ValueError) as exc_info:
            rmseq_main(args)
        assert "--problematic_percent should be" in str(exc_info.value)

    def test_rmseq_rejects_empty_problematic_char_when_percent_positive(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="$^",
            problematic_percent=1,
            problematic_char="",
        )
        with pytest.raises(ValueError) as exc_info:
            rmseq_main(args)
        assert "--problematic_chars must contain at least one character" in str(
            exc_info.value
        )

    def test_rmseq_accepts_protein_input_when_seqtype_protein(
        self, temp_dir, mock_args
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [
            SeqRecord(Seq("MKT"), id="prot_keep", description=""),
            SeqRecord(Seq("QQQ"), id="prot_remove", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname="prot_remove",
            problematic_percent=0,
            problematic_char=["X"],
            seqtype="protein",
        )
        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert result[0].id == "prot_keep"
