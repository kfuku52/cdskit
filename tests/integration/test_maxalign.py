"""
Tests for cdskit maxalign command.
"""

from itertools import combinations
import json
import pytest

import Bio.SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.tsvio import read_tsv

from cdskit import maxalign
from cdskit.maxalign import (
    _solve_branch_and_bound,
    indices_to_bitmask,
    maxalign_main,
    parse_missing_chars,
    select_indices_by_patterns,
    solve_exact,
    solve_greedy,
)


def test_branch_search_respects_protection_and_reports_work_exhaustion():
    active = list(range(5))
    all_present = indices_to_bitmask(active)
    counts = {all_present: 2, all_present ^ 1: 9, all_present ^ 2: 4}
    result = _solve_branch_and_bound(counts, active, [0], 1)
    expected = solve_exact(
        None,
        active,
        [0],
        1,
        len(active),
        support_counts=counts,
        num_sequences=len(active),
    )
    assert result == expected
    with pytest.raises(ValueError, match="work limit"):
        _solve_branch_and_bound(counts, active, [0], 1, work_limit=0)


def test_branch_search_bounds_column_recovery_by_number_of_rows_removed():
    active = list(range(12))
    all_present = indices_to_bitmask(active)
    gapped = {
        all_present ^ ((1 << first) | (1 << second)): 1
        for first, second in combinations(active, 2)
    }
    counts = {all_present: 200, **gapped}
    # This budget permits processing the root's patterns exactly once.
    result = _solve_branch_and_bound(counts, active, [], 6, work_limit=3 * len(gapped))
    assert result["kept_indices"] == active
    assert result["complete_codon_columns"] == 200 and result["area"] == 2400


@pytest.mark.parametrize("protected,budget", [([], 3), ([], 2), ([1], 3)])
def test_branch_search_groups_rows_with_the_same_recoverable_gaps(protected, budget):
    active = list(range(8))
    all_present = indices_to_bitmask(active)
    group = indices_to_bitmask([1, 3, 5])
    counts = {
        all_present: 30,
        all_present ^ group: 65,
        all_present ^ (group | (1 << 7)): 25,
        all_present ^ (1 << 7): 14,
    }
    result = _solve_branch_and_bound(counts, active, protected, budget)
    expected = solve_exact(
        None,
        active,
        protected,
        budget,
        len(active),
        support_counts=counts,
        num_sequences=len(active),
    )
    assert result == expected


def test_packed_support_masks_match_scalar_across_byte_blocks(monkeypatch):
    records = [
        SeqRecord(Seq(("AcT?NNG.GT-N"[i % 4 :] + "ACGT?-.N") * 96)[:768])
        for i in range(25)
    ]
    missing = "N?.-"
    expected = maxalign.subset_support_bitmasks(
        maxalign.build_codon_presence_matrix(records, set(missing))
    )
    monkeypatch.setattr(maxalign, "_SUPPORT_BLOCK_BYTES", 2048)
    assert maxalign.build_support_masks(records, missing) == expected


def test_exact_with_small_removal_budget_accounts_for_excluded_and_protected_rows(
    monkeypatch,
):
    candidates = list(range(1, 25))
    active = indices_to_bitmask(candidates)
    counts = {active: 5, active ^ (1 << 3): 6, active ^ (1 << 4): 100}

    def unexpected_dense_search(*args):
        pytest.fail(
            "The removal-constrained problem does not need a dense subset table"
        )

    monkeypatch.setattr(maxalign, "_dense_complete_counts", unexpected_dense_search)
    result = solve_exact(
        None, candidates, [4], 2, 25, support_counts=counts, num_sequences=25
    )
    assert result["kept_indices"] == [i for i in candidates if i != 3]
    assert result["complete_codon_columns"] == 11 and result["area"] == 23 * 11


class TestMaxalignMain:
    """Tests for maxalign_main function."""

    def test_branch_and_bound_recovers_jointly_gapped_columns(
        self, temp_dir, mock_args
    ):
        source = temp_dir / "input.fa"
        output, report = temp_dir / "output.fa", temp_dir / "report.json"
        records = [
            SeqRecord(
                Seq(
                    "GCT" * 220
                    + "".join(
                        ("---" if i in {0, j} else "GCT") * 10 for j in range(1, 10)
                    )
                ),
                id=f"seq{i}",
                description="",
            )
            for i in range(36)
        ]
        Bio.SeqIO.write(records, source, "fasta")
        args = mock_args(
            seqfile=str(source),
            outfile=str(output),
            report=str(report),
            max_removed=8,
            mode="branch-and-bound",
        )
        maxalign_main(args)
        result = json.loads(report.read_text())
        assert result["mode"] == "branch-and-bound"
        assert result["initial_area"] == 7920 and result["final_area"] == 8120
        assert result["removed_ids"] == ["seq0", *(f"seq{i}" for i in range(4, 10))]
        saved = list(Bio.SeqIO.parse(output, "fasta"))
        assert len(saved) == 29 and all(
            str(record.seq) == "GCT" * 280 for record in saved
        )

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
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
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
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
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
            mode="exact",
            max_exact_sequences=2,
            missing_char="-?.",
        )

        with pytest.raises(ValueError) as exc_info:
            maxalign_main(args)
        assert "max_exact_sequences" in str(exc_info.value)

    @pytest.mark.parametrize("mode,max_exact_sequences", [("auto", 0), ("exact", -1)])
    def test_maxalign_rejects_non_positive_max_exact_sequences(
        self,
        temp_dir,
        mock_args,
        mode,
        max_exact_sequences,
    ):
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode=mode,
            max_exact_sequences=max_exact_sequences,
            missing_char="-?.",
        )

        with pytest.raises(ValueError) as exc_info:
            maxalign_main(args)
        assert "--max_exact_sequences should be >= 1" in str(exc_info.value)

    def test_maxalign_empty_input(self, temp_dir, mock_args):
        """Empty input should produce empty output without crashing."""
        input_path = temp_dir / "empty.fasta"
        output_path = temp_dir / "output.fasta"
        input_path.write_text("")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode="auto",
            max_exact_sequences=16,
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert len(result) == 0

    @pytest.mark.parametrize("suffix", ["json", "tsv"])
    def test_maxalign_empty_input_writes_requested_report(
        self, tmp_path, mock_args, suffix
    ):
        source, output = tmp_path / "empty.fa", tmp_path / "out.fa"
        report = tmp_path / f"report.{suffix}"
        source.write_text("", encoding="utf-8")
        maxalign_main(
            mock_args(
                seqfile=str(source),
                outfile=str(output),
                report=str(report),
                mode="auto",
                max_exact_sequences=16,
                missing_char="-?.",
            )
        )
        assert output.read_text(encoding="utf-8") == ""
        if suffix == "json":
            data = json.loads(report.read_text(encoding="utf-8"))
            assert data["num_input_sequences"] == 0
            assert data["steps"] == []
        else:
            rows = read_tsv(report, required_columns=["section", "metric", "value"])
            assert any(
                row["metric"] == "num_input_sequences" and row["value"] == "0"
                for row in rows
            )

    def test_maxalign_auto_mode_uses_exact_when_within_limit(
        self, temp_dir, mock_args, capsys
    ):
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
            mode="auto",
            max_exact_sequences=3,
            missing_char="-?.",
        )

        maxalign_main(args)
        captured = capsys.readouterr()
        assert "maxalign mode: exact" in captured.err

    def test_maxalign_auto_mode_uses_greedy_when_above_limit(
        self, temp_dir, mock_args, capsys
    ):
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
            mode="auto",
            max_exact_sequences=2,
            missing_char="-?.",
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
            mode="greedy",
            max_exact_sequences=16,
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["good1", "good2", "good3"]
        assert all(str(r.seq) == "ATGAAACCC" for r in result)

    def test_maxalign_greedy_can_remove_gap_pattern_set_from_zero_area(
        self, temp_dir, mock_args
    ):
        """Greedy search should remove a multi-sequence gap-pattern set when single removals do not help."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"

        records = [
            SeqRecord(Seq("ATG---"), id="seq1", description=""),
            SeqRecord(Seq("ATG---"), id="seq2", description=""),
            SeqRecord(Seq("---AAA"), id="seq3", description=""),
            SeqRecord(Seq("---AAA"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            mode="greedy",
            max_exact_sequences=2,
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq3", "seq4"]
        assert all(str(r.seq) == "AAA" for r in result)

    def test_maxalign_exact_tie_break_prefers_more_sequences_then_earlier_indices(
        self, temp_dir, mock_args
    ):
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
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1", "seq2"]
        assert [str(r.seq) for r in result] == ["ATG", "ATG"]

    def test_maxalign_missing_char_option_changes_codon_presence(
        self, temp_dir, mock_args
    ):
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
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
        )
        maxalign_main(args_default)
        result_default = list(Bio.SeqIO.parse(str(output_default), "fasta"))
        assert [str(r.seq) for r in result_default] == ["ATGAAACCC", "ATGNNNCCC"]

        args_with_n = mock_args(
            seqfile=str(input_path),
            outfile=str(output_with_n),
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.N",
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
            mode="auto",
            max_exact_sequences=16,
            missing_char="-?.",
        )

        with pytest.raises(ValueError) as exc_info:
            maxalign_main(args)
        assert "not identical" in str(exc_info.value)

    def test_maxalign_single_sequence_still_drops_missing_codons(
        self, temp_dir, mock_args
    ):
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
            mode="auto",
            max_exact_sequences=16,
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1"]
        assert str(result[0].seq) == "ATGCCC"

    def test_maxalign_keep_regex_protects_sequence_from_dropping(
        self, temp_dir, mock_args
    ):
        """--keep should protect matching sequences even if area decreases."""
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
            mode="exact",
            max_exact_sequences=16,
            keep="seq3",
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1", "seq2", "seq3"]
        assert all(str(r.seq) == "AAA" for r in result)

    def test_maxalign_max_removed_limits_drops(self, temp_dir, mock_args):
        """--max_removed should cap optimization-driven removals."""
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
            mode="exact",
            max_exact_sequences=16,
            max_removed=0,
            missing_char="-?.",
        )

        maxalign_main(args)

        result = list(Bio.SeqIO.parse(str(output_path), "fasta"))
        assert [r.id for r in result] == ["seq1", "seq2", "seq3"]
        assert all(str(r.seq) == "AAA" for r in result)

    def test_maxalign_writes_optional_json_report(self, temp_dir, mock_args):
        """--report should write JSON report when path ends with .json."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.json"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
            SeqRecord(Seq("---AAA---"), id="seq3", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            report=str(report_path),
            mode="greedy",
            max_exact_sequences=2,
            missing_char="-?.",
        )

        maxalign_main(args)

        report = json.loads(report_path.read_text())
        assert report["mode"] == "greedy"
        assert report["num_input_sequences"] == 3
        assert "steps" in report
        assert report["final_area"] >= report["initial_area"]

    def test_maxalign_writes_optional_tsv_report(self, temp_dir, mock_args):
        """--report should write TSV-style output for non-.json paths."""
        input_path = temp_dir / "input.fasta"
        output_path = temp_dir / "output.fasta"
        report_path = temp_dir / "report.tsv"

        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args = mock_args(
            seqfile=str(input_path),
            outfile=str(output_path),
            report=str(report_path),
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
        )

        maxalign_main(args)

        rows, fieldnames = read_tsv(str(report_path), return_fieldnames=True)
        assert fieldnames[:2] == ["schema_version", "section"]
        assert {row["schema_version"] for row in rows} == {"2"}
        assert any(
            row["section"] == "summary" and row["metric"] == "mode" for row in rows
        )
        assert any(
            row["section"] == "step" and row["label"] == "initial" for row in rows
        )

    def test_maxalign_threads_matches_single_thread_exact(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"
        records = [
            SeqRecord(Seq("ATGAAACCC"), id="seq1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="seq2", description=""),
            SeqRecord(Seq("ATGA-ACCC"), id="seq3", description=""),
            SeqRecord(Seq("---AAA---"), id="seq4", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            mode="exact",
            max_exact_sequences=16,
            missing_char="-?.",
            threads=4,
        )

        maxalign_main(args_single)
        maxalign_main(args_threaded)
        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [
            str(r.seq) for r in result_threaded
        ]

    def test_maxalign_threads_matches_single_thread_greedy(self, temp_dir, mock_args):
        input_path = temp_dir / "input.fasta"
        out_single = temp_dir / "single.fasta"
        out_threaded = temp_dir / "threaded.fasta"
        records = [
            SeqRecord(Seq("ATGAAACCC"), id="good1", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="good2", description=""),
            SeqRecord(Seq("ATGAAACCC"), id="good3", description=""),
            SeqRecord(Seq("---AAA---"), id="sparse", description=""),
            SeqRecord(Seq("---------"), id="allgap", description=""),
        ]
        Bio.SeqIO.write(records, str(input_path), "fasta")

        args_single = mock_args(
            seqfile=str(input_path),
            outfile=str(out_single),
            mode="greedy",
            max_exact_sequences=2,
            missing_char="-?.",
            threads=1,
        )
        args_threaded = mock_args(
            seqfile=str(input_path),
            outfile=str(out_threaded),
            mode="greedy",
            max_exact_sequences=2,
            missing_char="-?.",
            threads=4,
        )

        maxalign_main(args_single)
        maxalign_main(args_threaded)
        result_single = list(Bio.SeqIO.parse(str(out_single), "fasta"))
        result_threaded = list(Bio.SeqIO.parse(str(out_threaded), "fasta"))
        assert [r.id for r in result_single] == [r.id for r in result_threaded]
        assert [str(r.seq) for r in result_single] == [
            str(r.seq) for r in result_threaded
        ]


class TestMaxalignHelpers:
    """Tests for helper functions used by maxalign."""

    def test_parse_missing_chars_default(self):
        assert parse_missing_chars("") == {"-", "?", "."}

    def test_solve_exact_with_required_indices(self):
        matrix = [
            [True, True, False],
            [True, True, False],
            [False, True, True],
        ]
        exact = solve_exact(
            matrix,
            candidate_indices=[0, 1, 2],
            required_indices=[2],
            max_removed=None,
            total_sequences=3,
        )
        assert exact["kept_indices"] == [0, 1, 2]

    def test_solve_greedy_tie_break_matches_original_scan_order(self):
        matrix = [
            [True, False],
            [False, True],
            [True, True],
        ]
        greedy = solve_greedy(matrix, threads=1)
        assert greedy["kept_indices"] == [1, 2]
        assert greedy["area"] == 2

    def test_solve_exact_process_fallback_to_threads(self, monkeypatch):
        attempted = []

        class FailingProcessPool:
            def __init__(self, *args, **kwargs):
                attempted.append(True)
                raise PermissionError("no process pool")

        monkeypatch.setattr("cdskit.maxalign.ProcessPoolExecutor", FailingProcessPool)
        # Nonintegral direct-API weights use the scalar subset scorer.
        result = solve_exact(
            None, num_sequences=17, support_counts={(1 << 17) - 1: 100.5}, threads=4
        )
        assert attempted
        assert result is not None
        assert result["area"] == 17 * 100.5

    def test_select_indices_by_patterns_matches_record_id(self):
        records = [
            SeqRecord(Seq("ATG"), id="keep_me", name="other_name", description=""),
            SeqRecord(Seq("ATG"), id="drop_me", name="keep_me", description=""),
        ]
        selected = select_indices_by_patterns(records, patterns=[r"keep_me"])
        assert selected == [0]
