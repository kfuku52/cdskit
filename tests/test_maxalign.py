"""
Tests for cdskit maxalign command.
"""

import pytest
from pathlib import Path

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cdskit.maxalign import (
    alignment_area,
    extract_complete_codon_indices,
    maxalign_main,
    parse_missing_chars,
    pick_solver_mode,
    solve_exact,
    solve_greedy,
)


class TestMaxalignMain:
    """Tests for maxalign_main function."""

    def test_maxalign_removes_gap_heavy_sequence(self, temp_dir, mock_args):
        """A gap-heavy sequence should be dropped when it reduces codon-area."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCCGGG"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCCGGG"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCCGGG"), id="seq3", description=""),
            SeqRecord(Seq("---AAA---GGG"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='exact',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1", "seq2", "seq3"]
        assert all(str(r.seq) == "ATGAAACCCGGG" for r in result)

    def test_maxalign_counts_area_by_codon_units(self, temp_dir, mock_args):
        """A codon containing one gap should remove that full codon site."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGA-ACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGA-ACCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='exact',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 3
        # The middle codon (A-A) is excluded as one codon unit.
        assert all(str(r.seq) == "ATGCCC" for r in result)

    def test_maxalign_exact_mode_respects_limit(self, temp_dir, mock_args):
        """Exact mode should reject inputs above --max_exact_sequences."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='exact',
            max_exact_sequences=2,
            missing_char='-?.',
        )

        with pytest.raises(Exception) as exc_info:
            maxalign_main(args)
        assert "max_exact_sequences" in str(exc_info.value)

    def test_maxalign_empty_input(self, temp_dir, mock_args):
        """Empty input should produce empty output without crashing."""
        input_path = temp_dir / "empty.fasta"
        output_path = temp_dir / "output.fasta"
        input_path.write_text("")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='auto',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 0

    def test_maxalign_auto_mode_uses_exact_when_within_limit(self, temp_dir, mock_args, capsys):
        """Auto mode should choose exact when sequence count is small enough."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
            SeqRecord(Seq("---AAA---"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='auto',
            max_exact_sequences=3,
            missing_char='-?.',
        )

        maxalign_main(args)
        captured = capsys.readouterr()
        assert "maxalign mode: exact" in captured.err

    def test_maxalign_auto_mode_uses_greedy_when_above_limit(self, temp_dir, mock_args, capsys):
        """Auto mode should choose greedy when sequence count exceeds limit."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='auto',
            max_exact_sequences=2,
            missing_char='-?.',
        )

        maxalign_main(args)
        captured = capsys.readouterr()
        assert "maxalign mode: greedy" in captured.err

    def test_maxalign_greedy_can_remove_multiple_sequences(self, temp_dir, mock_args):
        """Greedy search should remove sequences repeatedly if area keeps improving."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="good1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="good2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="good3", description=""),
            SeqRecord(Seq("---------"), id="allgap", description=""),
            SeqRecord(Seq("---AAA---"), id="sparse", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='greedy',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["good1", "good2", "good3"]
        assert all(str(r.seq) == "ATGAAACCC" for r in result)

    def test_maxalign_exact_tie_break_prefers_more_sequences_then_earlier_indices(self, temp_dir, mock_args):
        """Exact mode tie-break: area, then num_kept, then lexicographic indices."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAA"), id="seq1", description=""),
            SeqRecord(Seq("ATG---"), id="seq2", description=""),
            SeqRecord(Seq("---AAA"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='exact',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1", "seq2"]
        assert [str(r.seq) for r in result] == ["ATG", "ATG"]

    def test_maxalign_missing_char_option_changes_codon_presence(self, temp_dir, mock_args):
        """Adding N to missing chars should treat N-containing codons as missing."""
        input_path = temp_dir / "input.fasta"
        output_default = temp_dir / "output_default.fasta"
        output_with_n = temp_dir / "output_with_n.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGNNNCCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_default = mock_args(
            seqfile=str(input_path),
            outfile=str(output_default),
            mode='exact',
            max_exact_sequences=16,
            missing_char='-?.',
        )
        maxalign_main(args_default)
        result_default = list(Bio.SeqIO.parse(str(output_default), "fasta"))
        assert [str(r.seq) for r in result_default] == ["ATGAAACCC", "ATGNNNCCC"]

        args_with_n = mock_args(
            seqfile=str(input_path),
            outfile=str(output_with_n),
            mode='exact',
            max_exact_sequences=16,
            missing_char='-?.N',
        )
        maxalign_main(args_with_n)
        result_with_n = list(Bio.SeqIO.parse(str(output_with_n), "fasta"))
        assert [str(r.seq) for r in result_with_n] == ["ATGCCC", "ATGCCC"]

    def test_maxalign_rejects_unaligned_input(self, temp_dir, mock_args):
        """Unaligned input should raise an informative error."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAA"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='auto',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        with pytest.raises(Exception) as exc_info:
            maxalign_main(args)
        assert "not identical" in str(exc_info.value)

    def test_maxalign_rejects_non_multiple_of_three(self, temp_dir, mock_args):
        """Input lengths must be multiples of 3."""
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
            mode='auto',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        with pytest.raises(Exception) as exc_info:
            maxalign_main(args)
        assert "multiple of three" in str(exc_info.value)

    def test_maxalign_single_sequence_still_drops_missing_codons(self, temp_dir, mock_args):
        """Single sequence should keep only codons without missing characters."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATG---CCC"), id="seq1", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode='auto',
            max_exact_sequences=16,
            missing_char='-?.',
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1"]
        assert str(result[0].seq) == "ATGCCC"


class TestMaxalignHelpers:
    """Tests for helper functions used by maxalign."""

    def test_parse_missing_chars_default(self):
        assert parse_missing_chars('') == {'-', '?', '.'}

    def test_pick_solver_mode_auto_switch(self):
        assert pick_solver_mode(num_records=4, mode='auto', max_exact_sequences=4) == 'exact'
        assert pick_solver_mode(num_records=5, mode='auto', max_exact_sequences=4) == 'greedy'
        assert pick_solver_mode(num_records=5, mode='exact', max_exact_sequences=4) == 'exact'

    def test_solve_exact_and_greedy_on_same_matrix(self):
        """Both solvers should agree on this simple matrix."""
        # Rows: sequences, columns: codon presence flags.
        # Best subset is indices [0, 1] with area 4 (2 seqs x 2 complete columns).
        matrix = [
            [True, True, False],
            [True, True, False],
            [False, True, True],
        ]
        exact = solve_exact(matrix)
        greedy = solve_greedy(matrix)
        assert exact['kept_indices'] == [0, 1]
        assert exact['area'] == 4
        assert greedy['kept_indices'] == [0, 1]
        assert greedy['area'] == 4

    def test_alignment_area_and_complete_indices(self):
        matrix = [
            [True, False, True],
            [True, True, True],
        ]
        area, complete = alignment_area(matrix, kept_indices=[0, 1])
        complete_indices = extract_complete_codon_indices(matrix, kept_indices=[0, 1])
        assert complete == 2
        assert area == 4
        assert complete_indices == [0, 2]
