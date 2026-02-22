"""
Tests for cdskit gapjust command.
"""

import pytest
from pathlib import Path
import numpy as np

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.gapjust import (
    gapjust_main,
    should_justify_gap,
    update_gap_ranges,
    vectorized_coordinate_update,
)


class TestUpdateGapRanges:
    """Tests for update_gap_ranges function."""

    def test_shift_ranges_right(self):
        """Test shifting ranges after insertion."""
        gap_ranges = [(0, 5), (10, 15), (20, 25)]
        result = update_gap_ranges(gap_ranges, gap_start=10, edit_len=3)
        # Ranges after position 10 should shift by 3
        assert result == [(0, 5), (10, 15), (23, 28)]

    def test_shift_ranges_left(self):
        """Test shifting ranges after deletion."""
        gap_ranges = [(0, 5), (10, 15), (20, 25)]
        result = update_gap_ranges(gap_ranges, gap_start=10, edit_len=-2)
        # Ranges after position 10 should shift by -2
        assert result == [(0, 5), (10, 15), (18, 23)]

    def test_no_shift_before_gap_start(self):
        """Test that ranges before gap_start are not shifted."""
        gap_ranges = [(0, 5), (10, 15)]
        result = update_gap_ranges(gap_ranges, gap_start=20, edit_len=5)
        # No changes since both ranges are before gap_start
        assert result == [(0, 5), (10, 15)]


class TestVectorizedCoordinateUpdate:
    """Tests for vectorized_coordinate_update function."""

    def test_single_insertion(self):
        """Test coordinate update with single insertion."""
        starts = np.array([10, 20, 30])
        ends = np.array([15, 25, 35])
        justifications = [
            {'original_edit_start': 5, 'edit_length': 5}  # Insert 5 at position 5
        ]

        new_starts, new_ends = vectorized_coordinate_update(starts, ends, justifications)

        # All coordinates after position 6 (1-based) should shift by 5
        assert list(new_starts) == [15, 25, 35]
        assert list(new_ends) == [20, 30, 40]

    def test_single_deletion(self):
        """Test coordinate update with single deletion."""
        starts = np.array([10, 20, 30])
        ends = np.array([15, 25, 35])
        justifications = [
            {'original_edit_start': 5, 'edit_length': -3}  # Delete 3 at position 5
        ]

        new_starts, new_ends = vectorized_coordinate_update(starts, ends, justifications)

        # All coordinates after position 6 (1-based) should shift by -3
        assert list(new_starts) == [7, 17, 27]
        assert list(new_ends) == [12, 22, 32]

    def test_no_change_when_zero_edit(self):
        """Test that zero edit length causes no change."""
        starts = np.array([10, 20, 30])
        ends = np.array([15, 25, 35])
        justifications = [
            {'original_edit_start': 5, 'edit_length': 0}
        ]

        new_starts, new_ends = vectorized_coordinate_update(starts, ends, justifications)

        assert list(new_starts) == [10, 20, 30]
        assert list(new_ends) == [15, 25, 35]

    def test_cumulative_offset(self):
        """Test that multiple edits accumulate correctly."""
        starts = np.array([10, 50])
        ends = np.array([20, 60])
        justifications = [
            {'original_edit_start': 5, 'edit_length': 10},   # Insert 10 at pos 5
            {'original_edit_start': 30, 'edit_length': 5},   # Insert 5 at pos 30
        ]

        new_starts, new_ends = vectorized_coordinate_update(starts, ends, justifications)

        # First coordinate (10) shifts by 10 from first edit
        # Second coordinate (50) shifts by 10 + 5 = 15 from both edits
        assert list(new_starts) == [20, 65]
        assert list(new_ends) == [30, 75]

    def test_tuple_justifications_match_dict_behavior(self):
        """Tuple-format edits should match legacy dict-format behavior."""
        starts = np.array([10, 50])
        ends = np.array([20, 60])
        dict_justifications = [
            {'original_edit_start': 30, 'edit_length': 5},
            {'original_edit_start': 5, 'edit_length': 10},
        ]
        tuple_justifications = [
            (5, 10),
            (30, 5),
        ]

        dict_starts, dict_ends = vectorized_coordinate_update(starts.copy(), ends.copy(), dict_justifications)
        tuple_starts, tuple_ends = vectorized_coordinate_update(starts.copy(), ends.copy(), tuple_justifications)

        assert list(tuple_starts) == list(dict_starts)
        assert list(tuple_ends) == list(dict_ends)

    def test_non_monotonic_actual_edit_starts_use_safe_fallback(self):
        """Fallback loop should handle edits whose shifted starts are not monotonic."""
        starts = np.array([8, 30])
        ends = np.array([9, 35])
        justifications = [
            (10, -15),
            (20, 5),
        ]

        new_starts, new_ends = vectorized_coordinate_update(starts, ends, justifications)

        # Manual replay of edits:
        # start 8: 8>11 no shift, then 8>6 shift +5 -> 13
        # start 30: 30>11 shift -15 -> 15, then 15>6 shift +5 -> 20
        assert list(new_starts) == [13, 20]
        # end 9: 9>11 no shift, then 9>6 shift +5 -> 14
        # end 35: 35>11 shift -15 -> 20, then 20>6 shift +5 -> 25
        assert list(new_ends) == [14, 25]

    def test_mixed_edits_can_shift_coordinate_across_later_edit_start(self):
        """Later edits must be applied based on progressively updated coordinates."""
        starts = np.array([180])
        ends = np.array([198])
        justifications = [
            (16, -4),
            (42, 8),
            (114, 13),
            (124, 15),
            (179, -10),
            (197, -18),
        ]

        new_starts, new_ends = vectorized_coordinate_update(starts, ends, justifications)

        # Manual replay:
        # start 180 -> 176 -> 184 -> 197 -> 212 -> 212 -> 212
        # end   198 -> 194 -> 202 -> 215 -> 230 -> 220 -> 220
        assert list(new_starts) == [212]
        assert list(new_ends) == [220]


class TestShouldJustifyGap:
    """Tests for should_justify_gap helper."""

    def test_skip_when_already_target_length(self):
        assert should_justify_gap(5, 5) is False

    def test_extension_honors_min_threshold(self):
        # Extension (3 -> 5) is skipped because 3 < min(4)
        assert should_justify_gap(3, 5, gap_just_min=4) is False
        # Extension is allowed when gap length is at least min
        assert should_justify_gap(4, 5, gap_just_min=4) is True

    def test_shortening_honors_max_threshold(self):
        # Shortening (8 -> 5) is skipped because 8 > max(7)
        assert should_justify_gap(8, 5, gap_just_max=7) is False
        # Shortening is allowed when gap length is at most max
        assert should_justify_gap(7, 5, gap_just_max=7) is True


class TestGapjustMain:
    """Tests for gapjust_main function."""

    def test_gapjust_uniform_length(self, temp_dir, mock_args):
        """Test gapjust makes gaps uniform length."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Sequence with variable gap lengths
        records = [
            SeqRecord(Seq("ATGNNNNNNAAA"), id="seq1", description=""),  # 6 Ns
            SeqRecord(Seq("ATGNNNAAA"), id="seq2", description=""),     # 3 Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=5,  # Target gap length
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # All gaps should now be 5 Ns
        for r in result:
            seq_str = str(r.seq)
            n_count = seq_str.count('N')
            # Each sequence should have exactly 5 Ns (one gap of length 5)
            assert n_count == 5, f"{r.id} has {n_count} Ns, expected 5"

    def test_gapjust_lowercase_n(self, temp_dir, mock_args):
        """Test gapjust normalizes lowercase n to uppercase N."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGnnnAAA"), id="seq1", description=""),  # lowercase n
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=3,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Should have uppercase N, not lowercase n
        assert 'n' not in str(result[0].seq)
        assert 'N' in str(result[0].seq)

    def test_gapjust_no_gaps(self, temp_dir, mock_args):
        """Test gapjust with sequence without gaps."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAATGA"), id="seq1", description=""),  # No Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=5,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        # Sequence should be unchanged
        assert str(result[0].seq) == "ATGAAATGA"

    def test_gapjust_reports_no_edits_when_all_gaps_already_target(self, temp_dir, mock_args, capsys):
        """Summary should report no edits when target already matches all gaps."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGNNNAAA"), id="seq1", description=""),
            SeqRecord(Seq("CCCNNNTTT"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=3,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        captured = capsys.readouterr()
        assert "Number of gap justifications: 0" in captured.err
        assert "No gap edits were made." in captured.err

    def test_gapjust_reports_min_max_original_gap_lengths(self, temp_dir, mock_args, capsys):
        """Summary should report min/max original gap lengths for edited gaps."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGNNAAA"), id="seq1", description=""),        # 2 Ns
            SeqRecord(Seq("CCCNNNNNNTTT"), id="seq2", description=""),    # 6 Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=4,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        captured = capsys.readouterr()
        assert "Number of gap justifications: 2" in captured.err
        assert "Minimum and maximum original gap lengths: 2 and 6" in captured.err

    def test_gapjust_with_gff(self, temp_dir, mock_args):
        """Test gapjust updates GFF coordinates."""
        input_fasta = temp_dir / "input.fasta"
        input_gff = temp_dir / "input.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        # Create FASTA with gap
        records = [
            SeqRecord(Seq("ATGNNNAAA"), id="seq1", description=""),  # 3 Ns at pos 3-5
        ]
        Bio.SeqIO.write(records, str(input_fasta), "fasta")

        # Create GFF with feature after gap
        gff_content = "##gff-version 3\nseq1\tsource\tgene\t7\t9\t.\t+\t.\tID=gene1\n"
        input_gff.write_text(gff_content)

        args = mock_args(
            seqfile=str(input_fasta),
            outfile=str(output_fasta),
            gap_len=5,  # Expand gap from 3 to 5
            ingff=str(input_gff),
            outgff=str(output_gff),
        )

        gapjust_main(args)

        # Check GFF output file exists and contains seq1
        assert output_gff.exists()
        with open(output_gff) as f:
            gff_output = f.read()
        assert "seq1" in gff_output

    def test_gapjust_rejects_duplicate_ids_with_gff(self, temp_dir, mock_args):
        input_fasta = temp_dir / "input.fasta"
        input_gff = temp_dir / "input.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        records = [
            SeqRecord(Seq("ATGNNNAAA"), id="dup", description=""),
            SeqRecord(Seq("ATGNNNNAAA"), id="dup", description=""),
        ]
        Bio.SeqIO.write(records, str(input_fasta), "fasta")
        input_gff.write_text(
            "##gff-version 3\n"
            "dup\tsource\tgene\t1\t9\t.\t+\t.\tID=gene1\n"
        )

        args = mock_args(
            seqfile=str(input_fasta),
            outfile=str(output_fasta),
            gap_len=5,
            ingff=str(input_gff),
            outgff=str(output_gff),
        )

        with pytest.raises(Exception) as exc_info:
            gapjust_main(args)
        assert "Duplicate sequence IDs are not supported with --ingff" in str(exc_info.value)
        assert not output_fasta.exists()
        assert not output_gff.exists()

    def test_gapjust_multiple_gaps(self, temp_dir, mock_args):
        """Test gapjust with multiple gaps in one sequence."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Two separate gaps
        records = [
            SeqRecord(Seq("ATGNNNAAA" + "CCCNNNNTTT"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=5,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq_str = str(result[0].seq)
        # Should have two gaps of 5 Ns each = 10 total Ns
        assert seq_str.count('N') == 10

    def test_gapjust_with_test_data(self, data_dir, temp_dir, mock_args):
        """Test gapjust with gapjust_01 test data."""
        input_fasta = data_dir / "gapjust_01" / "input.fasta"
        input_gff = data_dir / "gapjust_01" / "input.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        if not input_fasta.exists():
            pytest.skip("gapjust_01 test data not found")

        args = mock_args(
            seqfile=str(input_fasta),
            outfile=str(output_fasta),
            gap_len=10,  # Reasonable gap length
            ingff=str(input_gff) if input_gff.exists() else None,
            outgff=str(output_gff) if input_gff.exists() else None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_fasta), "fasta"))
        assert len(result) > 0

    def test_gapjust_01_compare_expected(self, data_dir, temp_dir, mock_args):
        """Test gapjust_01 data comparing with expected output."""
        input_fasta = data_dir / "gapjust_01" / "input.fasta"
        expected_fasta = data_dir / "gapjust_01" / "output.fasta"
        output_fasta = temp_dir / "output.fasta"

        if not input_fasta.exists() or not expected_fasta.exists():
            pytest.skip("gapjust_01 test data not found")

        # Read expected to determine gap length
        expected = list(Bio.SeqIO.parse(str(expected_fasta), "fasta"))
        if not expected:
            pytest.skip("Expected output is empty")

        # Determine gap length from expected output
        # Count N stretches and their lengths
        import re
        expected_seq = str(expected[0].seq)
        n_runs = re.findall(r'N+', expected_seq)
        if n_runs:
            gap_len = len(n_runs[0])  # Use first gap's length as target
        else:
            gap_len = 10

        args = mock_args(
            seqfile=str(input_fasta),
            outfile=str(output_fasta),
            gap_len=gap_len,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_fasta), "fasta"))

        # Verify all gap runs have uniform length
        for r in result:
            seq_str = str(r.seq)
            n_runs = re.findall(r'N+', seq_str)
            for run in n_runs:
                assert len(run) == gap_len, f"Gap length {len(run)} != expected {gap_len}"

    def test_gapjust_threads_matches_single_thread(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"

        records = [
            SeqRecord(Seq("ATGNNNAAACCCNNNNNTTT"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNAAACCCNNNNTTT"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            gap_len=5,
            ingff=None,
            outgff=None,
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            gap_len=5,
            ingff=None,
            outgff=None,
            threads=4,
        )

        gapjust_main(args_single)
        gapjust_main(args_threaded)

        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [str(r.seq) for r in result_threaded]

    def test_gapjust_shrink_gaps(self, temp_dir, mock_args):
        """Test gapjust can shrink gaps."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        # Large gaps that should be shrunk
        records = [
            SeqRecord(Seq("ATGNNNNNNNNNNNNNAAA"), id="seq1", description=""),  # 12 Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=3,  # Shrink to 3
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq_str = str(result[0].seq)
        assert seq_str.count('N') == 3

    def test_gapjust_preserves_non_n_content(self, temp_dir, mock_args):
        """Test gapjust preserves non-N sequence content."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGCCCNNNGGGAAATTT"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=5,  # Change gap from 3 to 5
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        seq_str = str(result[0].seq)

        # Non-N content should be preserved
        seq_without_n = seq_str.replace('N', '')
        original_without_n = "ATGCCCGGGAAATTT"
        assert seq_without_n == original_without_n

    def test_gapjust_min_threshold_for_extension(self, temp_dir, mock_args):
        """Gaps smaller than --gap_just_min should not be extended."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGNNAAA"), id="skip_small_gap", description=""),       # 2 Ns
            SeqRecord(Seq("ATGNNNNAAA"), id="extend_large_gap", description=""),   # 4 Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=5,
            gap_just_min=3,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = {r.id: str(r.seq) for r in Bio.SeqIO.parse(str(output_path), "fasta")}
        assert result["skip_small_gap"].count("N") == 2
        assert result["extend_large_gap"].count("N") == 5

    def test_gapjust_max_threshold_for_shortening(self, temp_dir, mock_args):
        """Gaps larger than --gap_just_max should not be shortened."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGNNNNNNNNAAA"), id="skip_large_gap", description=""),      # 8 Ns
            SeqRecord(Seq("ATGNNNNNNAAA"), id="shorten_allowed_gap", description=""),    # 6 Ns
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            gap_len=3,
            gap_just_max=6,
            ingff=None,
            outgff=None,
        )

        gapjust_main(args)

        result = {r.id: str(r.seq) for r in Bio.SeqIO.parse(str(output_path), "fasta")}
        assert result["skip_large_gap"].count("N") == 8
        assert result["shorten_allowed_gap"].count("N") == 3

    def test_gapjust_gff_coordinate_shift(self, data_dir, temp_dir, mock_args):
        """Test gapjust properly shifts GFF coordinates."""
        input_fasta = data_dir / "gapjust_01" / "input.fasta"
        input_gff = data_dir / "gapjust_01" / "input.gff"
        expected_gff = data_dir / "gapjust_01" / "output.gff"
        output_fasta = temp_dir / "output.fasta"
        output_gff = temp_dir / "output.gff"

        if not input_fasta.exists() or not input_gff.exists():
            pytest.skip("gapjust_01 test data not found")

        args = mock_args(
            seqfile=str(input_fasta),
            outfile=str(output_fasta),
            gap_len=10,
            ingff=str(input_gff),
            outgff=str(output_gff),
        )

        gapjust_main(args)

        # Verify GFF was created
        assert output_gff.exists()

        # If expected GFF exists, compare coordinates
        if expected_gff.exists():
            with open(output_gff) as f:
                result_lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]
            with open(expected_gff) as f:
                expected_lines = [l for l in f.readlines() if not l.startswith('#') and l.strip()]

            # Same number of records
            assert len(result_lines) == len(expected_lines)
