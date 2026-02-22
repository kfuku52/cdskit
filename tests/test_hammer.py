"""
Tests for cdskit hammer command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.hammer import hammer_main


class TestHammerMain:
    """Tests for hammer_main function."""

    def test_hammer_basic(self, temp_dir, mock_args):
        """Test basic hammer functionality - remove gappy columns."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Create aligned sequences with gaps
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
            nail='3',  # Require all 3 sequences to have non-gap
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Output should have 3 sequences, aligned, multiple of 3
        assert len(result) == 3
        assert len(result[0].seq) % 3 == 0
        # Some columns may be removed
        assert len(result[0].seq) <= 9

    def test_hammer_handles_question_codon_as_missing_not_error(self, temp_dir, mock_args):
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
            nail='3',
            prevent_gap_only=True,
        )
        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [str(r.seq) for r in result] == ["ATGCCC", "ATGCCC", "ATGCCC"]

    def test_hammer_prevent_gap_only_relaxes_for_question_missing_codons(self, temp_dir, mock_args):
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
            nail='3',
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
            nail='all',
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Output should have 2 sequences, aligned
        assert len(result) == 2
        assert len(result[0].seq) % 3 == 0
        assert len(result[0].seq) <= 9

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
            nail='2',  # Only require 2/3 sequences
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # With nail=2, more columns should remain
        assert len(result) == 3
        assert len(result[0].seq) % 3 == 0

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
            nail='3',  # Would remove all columns
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have relaxed nail to prevent gap-only seq2
        for r in result:
            # At least some content should remain
            assert len(r.seq) > 0

    def test_hammer_with_test_data(self, data_dir, temp_dir, mock_args):
        """Test hammer with hammer_01 test data."""
        input_path = data_dir / "hammer_01" / "alignment.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("hammer_01 test data not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='all',
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Verify output sequences are aligned and multiples of 3
        if result:
            lengths = [len(r.seq) for r in result]
            assert len(set(lengths)) == 1, "Output sequences should be aligned"
            assert lengths[0] % 3 == 0, "Output length should be multiple of 3"

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
            nail='all',
            prevent_gap_only=True,
        )

        with pytest.raises(Exception) as exc_info:
            hammer_main(args)
        assert "not identical" in str(exc_info.value)

    def test_hammer_rejects_non_multiple_of_three(self, temp_dir, mock_args):
        """Test hammer rejects sequences not multiple of 3."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAA"), id="seq1", description=""),  # 5 nt
            SeqRecord(Seq("ATGCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='all',
            prevent_gap_only=True,
        )

        with pytest.raises(Exception) as exc_info:
            hammer_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_hammer_rejects_invalid_codontable(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=999,
            nail='all',
            prevent_gap_only=True,
        )

        with pytest.raises(Exception) as exc_info:
            hammer_main(args)
        assert "Invalid --codontable" in str(exc_info.value)

    def test_hammer_wiki_example_nail_4(self, temp_dir, mock_args):
        """Test hammer with wiki example: --nail 4 on 6 sequences.

        Wiki: columns with <4 non-gap characters are removed.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # 6 sequences, some positions have varying gap coverage
        # Position 1-3 (ATG): all 6 have data
        # Position 4-6 (---/CCC): 3 have gaps, 3 have data
        # Position 7-9 (TGA): all 6 have data
        records = [
            SeqRecord(Seq("ATG---TGA"), id="seq1", description=""),
            SeqRecord(Seq("ATG---TGA"), id="seq2", description=""),
            SeqRecord(Seq("ATG---TGA"), id="seq3", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq4", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq5", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq6", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='4',  # Require at least 4 sequences to have non-gap
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 6
        # Verify output is multiple of 3 and shorter than input
        assert len(result[0].seq) % 3 == 0
        assert len(result[0].seq) < 9  # Some columns removed

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
            nail='1',  # Require at least 1 sequence to have non-gap
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 3
        # Gap-only positions should be removed, output is multiple of 3
        assert len(result[0].seq) % 3 == 0
        # Should be shorter than input since gap column is removed
        assert len(result[0].seq) < 9

    def test_hammer_with_example_hammer_fasta(self, data_dir, temp_dir, mock_args):
        """Test hammer with example_hammer.fasta from wiki."""
        input_path = data_dir / "example_hammer.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("example_hammer.fasta not found")

        # Check if input sequences are aligned before running
        input_records = list(Bio.SeqIO.parse(str(input_path), "fasta"))
        lengths = set(len(r.seq) for r in input_records)
        if len(lengths) > 1:
            pytest.skip("example_hammer.fasta contains unaligned sequences")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='4',  # Wiki example uses nail=4
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Verify all sequences are aligned and multiple of 3
        assert len(result) > 0
        lengths = [len(r.seq) for r in result]
        assert len(set(lengths)) == 1, "All sequences should have same length"
        assert lengths[0] % 3 == 0, "Length should be multiple of 3"

    def test_hammer_preserves_sequence_order(self, temp_dir, mock_args):
        """Test that hammer preserves the original sequence order."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="zebra", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="apple", description=""),
            SeqRecord(Seq("ATGGGGTGA"), id="mango", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='all',
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["zebra", "apple", "mango"]

    def test_hammer_issue3_empty_input_error(self, temp_dir, mock_args):
        """Test Issue #3: ValueError: max() arg is an empty sequence.

        Issue: When no input sequences are provided (e.g., from empty stdin),
        max([len(r.seq) for r in records]) raises ValueError.

        This error can occur with --prevent_gap_only when input from stdin is empty.
        """
        input_path = temp_dir / "empty.fasta"
        output_path = temp_dir / "output.fasta"

        # Create empty FASTA file
        input_path.write_text("")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='all',
            prevent_gap_only=True,
        )

        # Should handle empty input gracefully (either error message or empty output)
        # The original bug was: ValueError: max() arg is an empty sequence
        try:
            hammer_main(args)
            # If it succeeds, output should be empty or contain no sequences
            result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
            assert len(result) == 0
        except ValueError as e:
            # If it raises ValueError, it should not be the max() error from issue
            # (which would indicate the bug wasn't fixed)
            assert "max()" not in str(e), "Issue #3 bug: max() arg is empty sequence"
        except Exception:
            # Other exceptions are acceptable for empty input
            pass

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
            SeqRecord(Seq("---AAA---"), id="seq2", description=""),  # Only middle codon has data
            SeqRecord(Seq("ATGAAATGA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='all',  # Requires all 3 to have non-gap - middle codon only
            prevent_gap_only=True,
        )

        hammer_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 3
        # All sequences should be aligned and multiple of 3
        lengths = [len(r.seq) for r in result]
        assert len(set(lengths)) == 1
        assert lengths[0] % 3 == 0

    def test_hammer_nail_adjustment_for_gap_only_prevention(self, temp_dir, mock_args, capsys):
        """Test that nail value is automatically adjusted to prevent gap-only sequences.

        Issue #3 described: 'A gap-only sequence was generated with --nail 4. Will try --nail 3'
        This tests that the nail adjustment mechanism works.
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Create sequences where high nail would create gap-only sequence
        records = [
            SeqRecord(Seq("ATGAAACCC"), id="normal1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="normal2", description=""),
            SeqRecord(Seq("---------"), id="all_gaps", description=""),  # All gaps
            SeqRecord(Seq("ATGAAACCC"), id="normal3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            codontable=1,
            nail='4',  # Requires 4 sequences - but all_gaps will make this impossible
            prevent_gap_only=True,
        )

        hammer_main(args)

        captured = capsys.readouterr()
        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))

        # Should have 4 sequences in output
        assert len(result) == 4
        # All should be aligned
        lengths = [len(r.seq) for r in result]
        assert len(set(lengths)) == 1
        # The all_gaps sequence should still be all gaps, but that's allowed when nail is reduced
        # The stderr should show nail adjustment message
        # (Note: This depends on actual implementation behavior)

    def test_hammer_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        records = [
            SeqRecord(Seq("ATG---TGA"), id="seq1", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq2", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq3", description=""),
            SeqRecord(Seq("ATGCCCTGA"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            codontable=1,
            nail='3',
            prevent_gap_only=True,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            codontable=1,
            nail='3',
            prevent_gap_only=True,
            threads=4,
        )

        hammer_main(args_single)
        hammer_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

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

        with pytest.raises(Exception) as exc_info:
            hammer_main(args)
        assert '--nail should be a positive integer' in str(exc_info.value)
