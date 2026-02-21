"""
Tests for cdskit rmseq command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.rmseq import problematic_rate, rmseq_main, should_remove_record


class TestRmseqHelpers:
    """Tests for rmseq helper functions."""

    def test_problematic_rate_multiple_char_sets(self):
        seq = "ATN-X?"
        rate = problematic_rate(seq, ['N', '-', 'X', '?'])
        assert rate == pytest.approx(4 / 6)

    def test_problematic_rate_empty_sequence(self):
        rate = problematic_rate("", ['N', '-', 'X', '?'])
        assert rate == 0.0

    def test_problematic_rate_deduplicates_problematic_chars(self):
        rate = problematic_rate("NNAA", "NN")
        assert rate == pytest.approx(2 / 4)

    def test_problematic_rate_is_case_insensitive(self):
        rate = problematic_rate("ATGnnn", ['N'])
        assert rate == pytest.approx(3 / 6)

    def test_should_remove_record_by_name_pattern(self):
        record = SeqRecord(Seq("ATGAAA"), id="remove_me", name="remove_me", description="")
        remove = should_remove_record(
            record=record,
            seqname_pattern="remove.*",
            problematic_percent=0,
            problematic_chars=['N'],
        )
        assert remove is True

    def test_should_remove_record_matches_id_not_name(self):
        record = SeqRecord(Seq("ATGAAA"), id="remove_me", name="other_name", description="")
        remove = should_remove_record(
            record=record,
            seqname_pattern="remove.*",
            problematic_percent=0,
            problematic_chars=['N'],
        )
        assert remove is True

    def test_should_remove_record_by_problematic_threshold(self):
        record = SeqRecord(Seq("ATGNNN"), id="seq1", name="seq1", description="")
        remove = should_remove_record(
            record=record,
            seqname_pattern="$^",
            problematic_percent=50,
            problematic_chars=['N'],
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
            seqname='remove.*',  # Regex to match sequences to remove
            problematic_percent=0,
            problematic_char=['N', 'X', '-', '?'],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 2
        ids = [r.id for r in result]
        assert "keep_this" in ids
        assert "keep_also" in ids
        assert "remove_me" not in ids

    def test_rmseq_by_problematic_chars(self, temp_dir, mock_args):
        """Test removing sequences with too many problematic characters."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="clean", description=""),  # 0% N
            SeqRecord(Seq("ATGNNN"), id="half_n", description=""),  # 50% N
            SeqRecord(Seq("NNNNNN"), id="all_n", description=""),  # 100% N
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='$^',  # Regex that matches nothing
            problematic_percent=50,  # Remove if >= 50% problematic
            problematic_char=['N'],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert result[0].id == "clean"

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
            seqname='$^',
            problematic_percent=10,
            problematic_char=['N'],
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
            seqname='$^',
            problematic_percent=50,
            problematic_char=['N', 'X', '-', '?'],
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
            seqname='bad.*',
            problematic_percent=50,
            problematic_char=['N'],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 1
        assert result[0].id == "good_seq"

    def test_rmseq_no_removal(self, temp_dir, mock_args):
        """Test when no sequences are removed."""
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
            seqname='$^',  # Matches nothing
            problematic_percent=0,  # No character filtering
            problematic_char=['N'],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 2

    def test_rmseq_with_test_data(self, data_dir, temp_dir, mock_args):
        """Test rmseq with rmseq_01 test data."""
        input_path = data_dir / "rmseq_01" / "input.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists():
            pytest.skip("rmseq_01 test data not found")

        input_records = list(Bio.SeqIO.parse(str(input_path), "fasta"))

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='$^',
            problematic_percent=50,
            problematic_char=['N', '-'],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Result should have same or fewer sequences than input
        assert len(result) <= len(input_records)

    def test_rmseq_wiki_example_species_removal(self, temp_dir, mock_args):
        """Test wiki example: remove Arabidopsis sequences and high-N sequences.

        Wiki command: cdskit rmseq --seqname "Arabidopsis_thaliana.*" --problematic_percent 50
        This removes:
        - All Arabidopsis_thaliana sequences (by name regex)
        - Sequences with >=50% N characters
        """
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Simulate wiki example data
        records = [
            SeqRecord(Seq("AGAGTTCAATATGCTTTGAGTCGAATTCGTAACAATGCTAGAAATCTTCTTACTCTTGAT"),
                      id="Aquilegia_coerulea_1", description=""),
            SeqRecord(Seq("AGAGTTCAATATGCTTTAAGTCGAATTCGAAACAATGCTAGAAATCTTCTCACTCTGGAT"),
                      id="Aquilegia_coerulea_2", description=""),
            SeqRecord(Seq("AGAGTTCAATATGCTTTAAGTCGAATTCGTAACAATGCAAGAAATCTTCTTACACTTGAT"),
                      id="Aquilegia_coerulea_3", description=""),
            # This should be removed - over 50% N
            SeqRecord(Seq("AGGGTCCAATATGTTCTGAGCCGTATCCNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"),
                      id="Hylocereus_undatus_1", description=""),
            SeqRecord(Seq("AGGGTTCAATACGTTCTGAGCCGTATCCGTAATGCTGCAAGGCATCTTCTTACCCTGGAT"),
                      id="Hylocereus_undatus_2", description=""),
            # This should be removed - over 50% N
            SeqRecord(Seq("NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNTGCGGCAAGGCACCTTCTCACTCTGGAT"),
                      id="Hylocereus_undatus_3", description=""),
            # These should be removed - match Arabidopsis regex
            SeqRecord(Seq("AGAGTTCAATATACACTTAGCAGAATCCGTAATGCTGCAAGAGAACTCTTAACTCTTGAT"),
                      id="Arabidopsis_thaliana_1", description=""),
            SeqRecord(Seq("AGAGTGCAGTACTCTCTTAGCCGTATCCGTAATGCTGCTAGAGATCTTTTGACTCTTGAT"),
                      id="Arabidopsis_thaliana_2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='Arabidopsis_thaliana.*',  # Wiki example regex
            problematic_percent=50,
            problematic_char=['N'],
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        result_ids = [r.id for r in result]

        # Should have 4 sequences remaining:
        # - Aquilegia_coerulea_1, 2, 3 (kept)
        # - Hylocereus_undatus_2 (kept - low N%)
        assert len(result) == 4
        assert "Aquilegia_coerulea_1" in result_ids
        assert "Aquilegia_coerulea_2" in result_ids
        assert "Aquilegia_coerulea_3" in result_ids
        assert "Hylocereus_undatus_2" in result_ids

        # Should NOT have:
        assert "Arabidopsis_thaliana_1" not in result_ids
        assert "Arabidopsis_thaliana_2" not in result_ids
        assert "Hylocereus_undatus_1" not in result_ids  # High N%
        assert "Hylocereus_undatus_3" not in result_ids  # High N%

    def test_rmseq_exact_match_data(self, data_dir, temp_dir, mock_args):
        """Test rmseq with rmseq_01 data comparing to expected output."""
        input_path = data_dir / "rmseq_01" / "input.fasta"
        expected_path = data_dir / "rmseq_01" / "output.fasta"
        output_path = temp_dir / "output.fasta"

        if not input_path.exists() or not expected_path.exists():
            pytest.skip("rmseq_01 test data not found")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='Arabidopsis_thaliana.*',
            problematic_percent=50,
            problematic_char=['N'],
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
            seqname='$^',
            problematic_percent=50,
            problematic_char=['-'],  # Only count gaps
        )

        rmseq_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        result_ids = [r.id for r in result]
        assert "no_gaps" in result_ids
        assert "some_gaps" in result_ids  # 33% < 50%
        assert "many_gaps" not in result_ids  # 67% >= 50%

    def test_rmseq_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="keep_this", description=""),
            SeqRecord(Seq("ATGCCC"), id="remove_me", description=""),
            SeqRecord(Seq("NNNNNN"), id="all_n", description=""),
            SeqRecord(Seq("ATGNNN"), id="half_n", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            seqname='remove.*',
            problematic_percent=50,
            problematic_char=['N'],
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            seqname='remove.*',
            problematic_percent=50,
            problematic_char=['N'],
            threads=4,
        )

        rmseq_main(args_single)
        rmseq_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

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
            seqname='$^',
            problematic_percent=50,
            problematic_char=['N'],
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
            seqname='$^',
            problematic_percent=50,
            problematic_char='N',
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
            seqname='[',
            problematic_percent=0,
            problematic_char=['N'],
        )
        with pytest.raises(Exception) as exc_info:
            rmseq_main(args)
        assert 'Invalid regex in --seqname' in str(exc_info.value)

    @pytest.mark.parametrize('problematic_percent', [-1, 101, float('nan'), float('inf')])
    def test_rmseq_rejects_invalid_problematic_percent(self, temp_dir, mock_args, problematic_percent):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='$^',
            problematic_percent=problematic_percent,
            problematic_char=['N'],
        )
        with pytest.raises(Exception) as exc_info:
            rmseq_main(args)
        assert '--problematic_percent should be' in str(exc_info.value)

    def test_rmseq_rejects_empty_problematic_char_when_percent_positive(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("ATGAAA"), id="seq1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='$^',
            problematic_percent=1,
            problematic_char='',
        )
        with pytest.raises(Exception) as exc_info:
            rmseq_main(args)
        assert '--problematic_char must contain at least one character' in str(exc_info.value)

    def test_rmseq_rejects_non_dna_input(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        records = [SeqRecord(Seq("PPP"), id="prot1", description="")]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            seqname='$^',
            problematic_percent=0,
            problematic_char=['N'],
        )
        with pytest.raises(Exception) as exc_info:
            rmseq_main(args)
        assert 'DNA-only input is required' in str(exc_info.value)
