"""Biological safety and coordinate provenance for scaffold gap normalization."""

import json

import numpy as np
import pytest
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.cli import main as cli_main
from cdskit.gapjust import (
    apply_gap_justifications_to_gff,
    gapjust_main,
    normalize_record_gap_lengths,
    plan_record_gap_lengths,
    vectorized_coordinate_update,
)
from cdskit.gapjust_gff import (
    feature_identity,
    gff_diagnostics,
    select_gap_edits,
    update_sequence_regions,
)
from cdskit.util import GFF_DTYPE, read_gff


def annotation(rows, header=None):
    return {"data": np.array(rows, dtype=GFF_DTYPE), "header": header or []}


def cds(start, end, strand="+", phase="0", attributes="ID=c;Parent=t", kind="CDS"):
    return ("s", ".", kind, start, end, ".", strand, phase, attributes)


@pytest.mark.parametrize("target", [0, 1, 2, 4, 5, 6])
@pytest.mark.parametrize("strand", ["+", "-"])
def test_cds_overlap_is_rejected_even_for_in_frame_changes(target, strand):
    record = SeqRecord(Seq("ATGNNNAAACCCGGGTTTAAA"), id="s")
    gff = annotation([cds(1, 9, strand), cds(13, 21, strand)])
    original = gff["data"].copy()
    with pytest.raises(ValueError, match="CDS overlap"):
        normalize_record_gap_lengths(record, target, gff=gff)
    assert str(record.seq) == "ATGNNNAAACCCGGGTTTAAA"
    edits = plan_record_gap_lengths(record, target)
    with pytest.raises(ValueError, match="CDS overlap"):
        apply_gap_justifications_to_gff(gff, {"s": edits})
    np.testing.assert_array_equal(gff["data"], original)


@pytest.mark.parametrize("start,end", [(1, 4), (6, 9), (4, 6), (5, 5)])
@pytest.mark.parametrize("kind", ["CDS", "SO:0000316"])
def test_boundaries_and_missing_parent_or_strand_still_protected(start, end, kind):
    record = SeqRecord(Seq("AAANNNAAA"), id="s")
    gff = annotation([cds(start, end, "?", ".", ".", kind)])
    with pytest.raises(ValueError, match="CDS overlap"):
        normalize_record_gap_lengths(record, 4, gff=gff)


def test_no_length_change_is_allowed_inside_cds():
    record = SeqRecord(Seq("AAAnnnAAA"), id="s")
    edits, count, *_ = normalize_record_gap_lengths(
        record, 3, gff=annotation([cds(1, 9)])
    )
    assert count == 0 and edits == []
    assert str(record.seq) == "AAANNNAAA"


def test_skip_entire_run_crossing_multiple_cds_and_parents():
    record = SeqRecord(Seq("AAANNNNNNNAAANNNAAA"), id="s")
    gff = annotation([cds(4, 5, attributes="ID=c;Parent=t1,t2"), cds(8, 9)])
    plans = plan_record_gap_lengths(record, 1)
    accepted, audit = select_gap_edits(gff, {"s": plans}, cds_overlap="skip")
    assert len(accepted["s"]) == 1
    assert audit[0]["action"] == "skip"
    assert audit[0]["features"][0]["parents"] == ["t1", "t2"]
    assert len(audit[0]["features"]) == 2
    normalize_record_gap_lengths(record, 1, gff=gff, cds_overlap="skip")
    assert str(record.seq) == "AAANNNNNNNAAANAAA"


@pytest.mark.parametrize("target", [0, 2])
@pytest.mark.parametrize("bounds", [(6, 7), (4, 6), (4, 4)])
def test_deleted_non_cds_endpoints_are_never_clamped(target, bounds):
    gff = annotation([cds(*bounds, kind="exon")])
    edits = [
        {"original_gap_start": 3, "original_gap_length": 4, "target_gap_length": target}
    ]
    deleted = any(4 + target <= endpoint <= 7 for endpoint in bounds)
    if deleted:
        with pytest.raises(ValueError, match="Deleted feature endpoint"):
            apply_gap_justifications_to_gff(gff, {"s": edits})
        with pytest.raises(ValueError, match="Deleted feature endpoint"):
            vectorized_coordinate_update(
                np.array([bounds[0]]), np.array([bounds[1]]), edits
            )
    else:
        apply_gap_justifications_to_gff(gff, {"s": edits})
        assert int(gff["data"][0]["start"]) == bounds[0]


@pytest.mark.parametrize(
    "edits",
    [
        [(3, -1)],
        [{"original_edit_start": 3, "edit_length": 1}],
        [{"original_gap_start": 3, "original_gap_length": 3, "target_gap_length": -1}],
        [{"original_gap_start": 3.5, "original_gap_length": 3, "target_gap_length": 1}],
        [
            {
                "original_gap_start": 3,
                "original_gap_length": 3,
                "target_gap_length": 1,
                "edit_length": 1,
            }
        ],
        [
            {"original_gap_start": 3, "original_gap_length": 3, "target_gap_length": 1},
            {"original_gap_start": 5, "original_gap_length": 3, "target_gap_length": 1},
        ],
    ],
)
def test_gff_api_rejects_insufficient_or_invalid_edit_ranges(edits):
    with pytest.raises(ValueError):
        apply_gap_justifications_to_gff(annotation([]), {"s": edits})


def test_feature_attributes_decode_after_splitting():
    row = annotation([cds(1, 3, attributes="ID=c%3B1;Parent=t%2C1,t2")])["data"][0]
    assert feature_identity(row)["ids"] == ["c;1"]
    assert feature_identity(row)["parents"] == ["t,1", "t2"]


def test_input_phase_diagnostics_respect_transcription_order_and_initial_phase():
    # In transcription order: length 5, phase 1 -> next phase 2.
    good = annotation([cds(1, 5, "-", "2"), cds(9, 13, "-", "1")])
    assert not any("inconsistent" in item for item in gff_diagnostics(good))
    good["data"][0]["phase"] = "0"
    assert any("inconsistent" in item for item in gff_diagnostics(good))


def run_files(tmp_path, mock_args, sequence, rows, *, policy="error", header=""):
    fasta, gff = tmp_path / "in.fa", tmp_path / "in.gff"
    fasta.write_text(sequence)
    gff.write_text(
        "##gff-version 3\n"
        + header
        + "".join("\t".join(map(str, row)) + "\n" for row in rows)
    )
    return mock_args(
        seqfile=str(fasta),
        ingff=str(gff),
        outfile=str(tmp_path / "out.fa"),
        outgff=str(tmp_path / "out.gff"),
        gap_len=4,
        cds_overlap=policy,
        edit_report=str(tmp_path / "edits.json"),
    )


def test_late_rejection_preserves_all_existing_outputs(tmp_path, mock_args, capsys):
    args = run_files(
        tmp_path, mock_args, ">safe\nAAANNNAAA\n>s\nAAANNNAAA\n", [cds(1, 9)]
    )
    for filename in ("out.fa", "out.gff", "edits.json"):
        (tmp_path / filename).write_text("sentinel")
    with pytest.raises(ValueError, match="CDS overlap"):
        gapjust_main(args)
    for filename in ("out.fa", "out.gff", "edits.json"):
        assert (tmp_path / filename).read_text() == "sentinel"
    assert capsys.readouterr().out == ""
    args.outfile = "-"
    with pytest.raises(ValueError, match="CDS overlap"):
        gapjust_main(args)
    assert capsys.readouterr().out == ""


def test_cli_skip_report_and_region_directive(tmp_path, mock_args):
    args = run_files(
        tmp_path,
        mock_args,
        ">s\nAAANNNAAANNNAAA\n",
        [cds(1, 9)],
        policy="skip",
        header="##sequence-region s 1 15\n",
    )
    cli_main(
        [
            "gapjust",
            "--seq_file",
            args.seqfile,
            "--in_gff",
            args.ingff,
            "--out_file",
            args.outfile,
            "--out_gff",
            args.outgff,
            "--gap_len",
            "4",
            "--cds_overlap",
            "skip",
            "--edit_report",
            args.edit_report,
        ]
    )
    assert "AAANNNAAANNNNAAA" in (tmp_path / "out.fa").read_text()
    assert "##sequence-region s 1 16" in (tmp_path / "out.gff").read_text()
    report = json.loads((tmp_path / "edits.json").read_text())
    assert report["schema_version"] == 1 and report["cds_checked"] is True
    assert [entry["action"] for entry in report["edits"]] == ["skip", "apply"]
    assert report["edits"][0]["original_start"] == 4


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("target", [0, 2, 4, 6])
def test_safe_edits_preserve_spliced_cds_with_independent_base_map(strand, target):
    sequence = "NNNATGAAANNNCCCTTTNNN"
    record = SeqRecord(Seq(sequence), id="s")
    gff = annotation([cds(4, 9, strand), cds(13, 18, strand)])
    old = gff["data"].copy()
    edits, *_ = normalize_record_gap_lengths(record, target, gff=gff)
    apply_gap_justifications_to_gff(gff, {"s": edits})
    # Independent oracle: retain base identities and append anonymous inserted bases.
    tokens = list(range(1, len(sequence) + 1))
    for start in (18, 9, 0):
        tokens[start : start + 3] = tokens[start : start + min(3, target)] + [
            None
        ] * max(0, target - 3)
    for before, after in zip(old, gff["data"], strict=True):
        assert int(after["start"]) == tokens.index(int(before["start"])) + 1
        assert int(after["end"]) == tokens.index(int(before["end"])) + 1

    def extract(seq, rows):
        parts = [Seq(seq[int(row["start"]) - 1 : int(row["end"])]) for row in rows]
        return (
            "".join(str(part) for part in parts)
            if strand == "+"
            else "".join(str(part.reverse_complement()) for part in parts[::-1])
        )

    assert extract(sequence, old) == extract(str(record.seq), gff["data"])
    assert list(gff["data"]["phase"]) == list(old["phase"])


def test_sequence_region_terminal_deletion():
    edits = {
        "s": [
            {"original_gap_start": 6, "original_gap_length": 3, "target_gap_length": 0}
        ]
    }
    assert update_sequence_regions(["##sequence-region s 1 9"], edits) == [
        "##sequence-region s 1 6"
    ]
    with pytest.raises(ValueError, match="entire"):
        update_sequence_regions(["##sequence-region s 7 9"], edits)
    with pytest.raises(ValueError, match="boundary"):
        update_sequence_regions(["##sequence-region s 1 8"], edits)


def test_api_checks_all_sequences_before_mutation():
    gff = annotation(
        [cds(7, 9, kind="gene"), ("z", ".", "CDS", 1, 9, ".", "+", "0", "ID=z")]
    )
    original = gff["data"].copy()
    edits = [
        {"original_gap_start": 3, "original_gap_length": 3, "target_gap_length": 4}
    ]
    with pytest.raises(ValueError, match="CDS overlap"):
        apply_gap_justifications_to_gff(gff, {"s": edits, "z": edits})
    np.testing.assert_array_equal(gff["data"], original)


def test_report_collision_rejected_before_output(tmp_path, mock_args):
    args = run_files(tmp_path, mock_args, ">s\nAAANNNAAA\n", [])
    args.edit_report = args.seqfile
    with pytest.raises(ValueError, match="different"):
        gapjust_main(args)
    assert not (tmp_path / "out.fa").exists()


def test_duplicate_fasta_without_gff_preserves_each_plan(tmp_path, mock_args):
    args = run_files(tmp_path, mock_args, ">s\nAAANNNAAA\n>s\nNNNNN\n", [])
    args.ingff = None
    gapjust_main(args)
    text = (tmp_path / "out.fa").read_text()
    assert "AAANNNNAAA" in text and "\nNNNN\n" in text
    report = json.loads((tmp_path / "edits.json").read_text())
    assert report["cds_checked"] is False
    assert len(report["edits"]) == 2


def test_safe_fixture_gff_is_preserved(tmp_path, mock_args, data_dir):
    args = mock_args(
        seqfile=str(data_dir / "gapjust_01/input.fasta"),
        ingff=str(data_dir / "gapjust_01/input.gff"),
        outfile=str(tmp_path / "out.fa"),
        outgff=str(tmp_path / "out.gff"),
        gap_len=100,
    )
    gapjust_main(args)
    expected = read_gff(data_dir / "gapjust_01/output.gff")
    actual = read_gff(tmp_path / "out.gff")
    np.testing.assert_array_equal(actual["data"], expected["data"])


def test_input_relationship_diagnostics_do_not_disable_cds_protection():
    gff = annotation(
        [
            cds(1, 9, "?", ".", "ID=c1;Parent=c2"),
            cds(13, 21, "+", "0", "ID=c2;Parent=c1"),
        ]
    )
    diagnostics = gff_diagnostics(gff)
    assert any("cycle" in message for message in diagnostics)
    assert any("unknown strand" in message for message in diagnostics)
    record = SeqRecord(Seq("ATGNNNAAACCCGGGTTTAAA"), id="s")
    with pytest.raises(ValueError, match="CDS overlap"):
        normalize_record_gap_lengths(record, 4, gff=gff)


def test_shared_repeated_id_cds_and_row_order_do_not_hide_overlap():
    rows = [
        cds(1, 3, attributes="ID=shared;Parent=t1,t2"),
        cds(7, 9, attributes="ID=shared;Parent=t1,t2"),
        cds(1, 20, kind="gene", attributes="ID=g"),
        cds(1, 20, kind="mRNA", attributes="ID=t1;Parent=g"),
        cds(1, 20, kind="mRNA", attributes="ID=t2;Parent=g"),
    ]
    record = SeqRecord(Seq("AAANNNNNNAAANNNAAAAAA"), id="s")
    for ordered in (rows, rows[::-1]):
        gff = annotation(ordered)
        edits = plan_record_gap_lengths(record, 4)
        accepted, audit = select_gap_edits(gff, {"s": edits}, cds_overlap="skip")
        assert [entry["action"] for entry in audit] == ["skip", "apply"]
        assert len(accepted["s"]) == 1
        assert not any("cycle" in message for message in gff_diagnostics(gff))


def test_net_zero_length_change_does_not_allow_cds_edits():
    record = SeqRecord(Seq("AAANNAAANNNNAAA"), id="s")
    edits = plan_record_gap_lengths(record, 3)
    assert sum(edit["edit_length"] for edit in edits) == 0
    with pytest.raises(ValueError, match="CDS overlap"):
        normalize_record_gap_lengths(record, 3, gff=annotation([cds(1, 15)]))


def test_threshold_excluded_cds_gap_is_allowed():
    record = SeqRecord(Seq("AAANNNAAA"), id="s")
    edits, *_ = normalize_record_gap_lengths(
        record, 4, gap_just_min=5, gff=annotation([cds(1, 9)])
    )
    assert edits == []
    assert str(record.seq) == "AAANNNAAA"


@pytest.mark.parametrize("policy", ["error", "skip"])
def test_non_cds_deletion_remains_error_in_skip_mode(tmp_path, mock_args, policy):
    args = run_files(
        tmp_path, mock_args, ">s\nAAANNNAAA\n", [cds(4, 6, kind="exon")], policy=policy
    )
    args.gap_len = 0
    with pytest.raises(ValueError, match="Deleted feature endpoint"):
        gapjust_main(args)
    assert not (tmp_path / "out.fa").exists()
    assert not (tmp_path / "edits.json").exists()


@pytest.mark.parametrize(
    "header", ["##sequence-region missing 1 9\n", "##sequence-region s 1 99\n"]
)
def test_invalid_region_is_rejected_before_output(tmp_path, mock_args, header):
    args = run_files(tmp_path, mock_args, ">s\nAAANNNAAA\n", [], header=header)
    with pytest.raises(ValueError, match="sequence-region"):
        gapjust_main(args)
    assert not (tmp_path / "out.fa").exists()


def test_threads_produce_identical_safe_paired_output(tmp_path, mock_args):
    args = run_files(
        tmp_path, mock_args, ">s\nAAANNNAAANNNAAA\n", [cds(1, 9)], policy="skip"
    )
    args.threads = 1
    gapjust_main(args)
    original = [
        (tmp_path / name).read_bytes() for name in ("out.fa", "out.gff", "edits.json")
    ]
    args.threads = 4
    gapjust_main(args)
    assert original == [
        (tmp_path / name).read_bytes() for name in ("out.fa", "out.gff", "edits.json")
    ]


def test_api_invalid_policy_is_rejected():
    with pytest.raises(ValueError, match="cds_overlap"):
        select_gap_edits(None, {}, cds_overlap="allow")


@pytest.mark.parametrize(
    "edit",
    [
        {"original_gap_start": 0, "original_gap_length": 3, "target_gap_length": 1},
        {"original_gap_start": 3, "original_gap_length": 30, "target_gap_length": 1},
        {"original_gap_start": 3, "original_gap_length": 3, "target_gap_length": True},
        {
            "original_gap_start": 3,
            "original_gap_length": 3,
            "target_gap_length": 1,
            "original_edit_start": 0,
        },
    ],
)
def test_sequence_api_rejects_invalid_or_stale_plan_without_mutating(edit):
    from cdskit.gapjust import apply_record_gap_edits

    record = SeqRecord(Seq("AAANNNAAA"), id="s")
    with pytest.raises(ValueError):
        apply_record_gap_edits(record, [edit])
    assert str(record.seq) == "AAANNNAAA"


def test_sequence_api_checks_later_edits_before_mutating():
    from cdskit.gapjust import apply_record_gap_edits

    record = SeqRecord(Seq("NNNAAANNN"), id="s")
    edits = [
        {"original_gap_start": 0, "original_gap_length": 3, "target_gap_length": 1},
        {"original_gap_start": 3, "original_gap_length": 3, "target_gap_length": 1},
    ]
    with pytest.raises(ValueError, match="N interval"):
        apply_record_gap_edits(record, edits)
    assert str(record.seq) == "NNNAAANNN"


def test_sequence_api_sorts_original_coordinate_plan():
    from cdskit.gapjust import apply_record_gap_edits

    record = SeqRecord(Seq("NNNAAANNN"), id="s")
    edits = [
        {"original_gap_start": 6, "original_gap_length": 3, "target_gap_length": 1},
        {"original_gap_start": 0, "original_gap_length": 3, "target_gap_length": 2},
    ]
    apply_record_gap_edits(record, edits)
    assert str(record.seq) == "NNAAAN"
    assert edits[0]["original_gap_start"] == 6  # Caller-owned plan is unchanged.


def test_two_stdout_outputs_are_rejected(tmp_path, mock_args, capsys):
    args = run_files(tmp_path, mock_args, ">s\nAAANNNAAA\n", [])
    args.outfile = args.outgff = "-"
    with pytest.raises(ValueError, match="cannot share standard output"):
        gapjust_main(args)
    assert capsys.readouterr().out == ""
    assert not (tmp_path / "edits.json").exists()


def test_missing_gff_output_is_rejected_before_stdout(tmp_path, mock_args, capsys):
    args = run_files(tmp_path, mock_args, ">s\nAAANNNAAA\n", [])
    args.outfile, args.outgff = "-", None
    with pytest.raises(ValueError, match="out_gff"):
        gapjust_main(args)
    assert capsys.readouterr().out == ""


def test_unknown_phase_does_not_bridge_nonadjacent_cds():
    gff = annotation(
        [cds(1, 4, phase="0"), cds(8, 9, phase="."), cds(13, 15, phase="0")]
    )
    messages = gff_diagnostics(gff)
    assert any("missing/invalid" in message for message in messages)
    assert not any("inconsistent" in message for message in messages)


def test_gff_api_preserves_all_rows_on_late_mapping_error(monkeypatch):
    from cdskit import gapjust

    gff = annotation(
        [cds(7, 9, kind="gene"), ("z", ".", "gene", 7, 9, ".", "+", ".", "ID=z")],
        ["##sequence-region s 1 9", "##sequence-region z 1 9"],
    )
    original = gff["data"].copy()
    headers = list(gff["header"])
    edit = {"original_gap_start": 3, "original_gap_length": 3, "target_gap_length": 4}
    original_map = gapjust.vectorized_coordinate_update
    calls = []

    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("late mapping error")
        return original_map(*args)

    monkeypatch.setattr(gapjust, "vectorized_coordinate_update", fail_second)
    with pytest.raises(ValueError, match="late mapping error"):
        apply_gap_justifications_to_gff(gff, {"s": [edit], "z": [edit]})
    np.testing.assert_array_equal(gff["data"], original)
    assert gff["header"] == headers


def test_randomized_endpoint_mapping_against_base_identity_oracle():
    import random

    rng = random.Random(207)
    for _ in range(200):
        sequence = (
            "".join("A" * rng.randint(1, 5) + "N" * rng.randint(1, 7) for _ in range(5))
            + "AAA"
        )
        record = SeqRecord(Seq(sequence), id="s")
        edits = plan_record_gap_lengths(record, rng.randint(0, 8))
        tokens = list(range(1, len(sequence) + 1))
        for edit in reversed(edits):
            start, length, target = (
                edit[key]
                for key in (
                    "original_gap_start",
                    "original_gap_length",
                    "target_gap_length",
                )
            )
            tokens[start : start + length] = tokens[
                start : start + min(length, target)
            ] + [None] * max(0, target - length)
        expected = {
            old: new for new, old in enumerate(tokens, start=1) if old is not None
        }
        original = np.array(sorted(expected))
        mapped, _ = vectorized_coordinate_update(original, original, edits)
        assert list(mapped) == [expected[old] for old in original]
        deleted = sorted(set(range(1, len(sequence) + 1)) - expected.keys())
        if deleted:
            with pytest.raises(ValueError, match="Deleted"):
                vectorized_coordinate_update(
                    np.array(deleted), np.array(deleted), edits
                )


def test_compensating_edits_clear_stale_record_metadata():
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from cdskit.gapjust import apply_record_gap_edits

    record = SeqRecord(Seq("NNAAANNNN"), id="s")
    record.letter_annotations["phred_quality"] = list(range(9))
    record.features = [SeqFeature(SimpleLocation(2, 5), type="gene")]
    plan = plan_record_gap_lengths(record, 3)
    apply_record_gap_edits(record, plan)
    assert str(record.seq) == "NNNAAANNN"
    assert len(record) == 9
    assert record.letter_annotations == {}
    assert record.features == []


def test_noop_plan_retains_record_metadata():
    from Bio.SeqFeature import SeqFeature, SimpleLocation
    from cdskit.gapjust import apply_record_gap_edits

    record = SeqRecord(Seq("NNNAAA"), id="s")
    record.letter_annotations["phred_quality"] = list(range(6))
    feature = SeqFeature(SimpleLocation(3, 6), type="gene")
    record.features = [feature]
    apply_record_gap_edits(record, plan_record_gap_lengths(record, 3))
    assert record.letter_annotations["phred_quality"] == list(range(6))
    assert record.features == [feature]


@pytest.mark.parametrize("with_gff", [True, False])
def test_edit_report_is_registered_as_command_output(with_gff):
    from argparse import Namespace
    from cdskit.command_paths import command_paths

    args = Namespace(
        command="gapjust",
        seqfile="in.fa",
        ingff="in.gff" if with_gff else None,
        outfile="out.fa",
        outgff="out.gff",
        edit_report="edits.json",
    )
    inputs, outputs = command_paths(args)
    assert "edits.json" in outputs
    assert ("out.gff" in outputs) == with_gff
    assert ("in.gff" in inputs) == with_gff
