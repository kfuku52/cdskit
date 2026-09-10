import numpy as np
import os
import sys
import re
from collections import Counter
from functools import partial

from cdskit.atomicio import (
    atomic_output_paths,
    atomic_write_json,
    validate_distinct_paths,
)
from cdskit.gapjust_gff import (
    select_gap_edits,
    validate_edits,
    validate_gff_bounds,
    update_sequence_regions,
    gff_diagnostics,
)
from cdskit.util import (
    parallel_map_ordered,
    read_gff,
    read_seqs,
    replace_record_sequence,
    resolve_threads,
    stop_if_not_dna,
    write_gff,
    write_seqs,
)


def update_gap_ranges(gap_ranges, gap_start, edit_len):
    """
    Shifts all gap ranges to the right of `gap_start` by `edit_len`.
    This is used to keep track of future gap coordinates as we iteratively
    insert or delete bases in the FASTA.
    """
    for i in range(len(gap_ranges)):
        start, end = gap_ranges[i]
        if start > gap_start:
            gap_ranges[i] = (start + edit_len, end + edit_len)
    return gap_ranges


def vectorized_coordinate_update(
    seq_gff_start_coordinates, seq_gff_end_coordinates, justifications
):
    """Map feature endpoints; phase and annotation relationships are not changed.

    Range-aware dictionaries use 0-based original gap starts and reject deleted
    endpoints. Legacy point shifts are arithmetic only and cannot verify CDS
    safety; paired FASTA/GFF callers must use the validated interval API.
    """
    if len(justifications) == 0:
        return seq_gff_start_coordinates, seq_gff_end_coordinates

    # Range-aware edits preserve coordinates in the retained part of a gap.
    if (
        isinstance(justifications[0], dict)
        and "original_gap_length" in justifications[0]
    ):
        edits = sorted(
            (
                int(just["original_gap_start"]) + 1,
                int(just["original_gap_length"]),
                int(just["target_gap_length"]),
            )
            for just in justifications
        )

        def apply_range_aware(coords):
            original = np.asarray(coords)
            starts = np.asarray([edit[0] for edit in edits], dtype=np.int64)
            old_lengths = np.asarray([edit[1] for edit in edits], dtype=np.int64)
            targets = np.asarray([edit[2] for edit in edits], dtype=np.int64)
            old_ends = starts + old_lengths - 1
            deltas = targets - old_lengths
            prefix_delta = np.concatenate(
                (
                    np.zeros((1,), dtype=np.int64),
                    np.cumsum(deltas, dtype=np.int64),
                )
            )
            completed = np.searchsorted(old_ends, original, side="left")
            updated = original + prefix_delta[completed]

            containing = np.searchsorted(starts, original, side="right") - 1
            valid = containing >= 0
            clipped = np.maximum(containing, 0)
            inside = valid & (original <= old_ends[clipped])
            deleted = (
                inside
                & (original >= starts[clipped] + targets[clipped])
                & (deltas[clipped] < 0)
            )
            if np.any(deleted):
                raise ValueError(
                    "Deleted feature endpoint has no corresponding output coordinate."
                )
            return updated

        return (
            apply_range_aware(seq_gff_start_coordinates),
            apply_range_aware(seq_gff_end_coordinates),
        )

    # Accept both legacy dict format and internal compact tuple format.
    # dict: {'original_edit_start': int, 'edit_length': int}
    # tuple: (original_edit_start, edit_length)
    if isinstance(justifications[0], dict):
        legacy_edits = [
            (int(just["original_edit_start"]) + 1, int(just["edit_length"]))
            for just in justifications
        ]
    else:
        legacy_edits = [(int(just[0]) + 1, int(just[1])) for just in justifications]
    # Preserve the legacy tie-break on edit length for edits at the same site.
    legacy_edits.sort()

    def apply_coordinate_shift(coords):
        if len(legacy_edits) == 0:
            return coords
        starts = np.asarray([edit[0] for edit in legacy_edits], dtype=np.int64)
        deltas = np.asarray([edit[1] for edit in legacy_edits], dtype=np.int64)
        prefix_delta = np.concatenate(
            (
                np.zeros((1,), dtype=np.int64),
                np.cumsum(deltas, dtype=np.int64),
            )
        )
        actual_starts = starts + prefix_delta[:-1]
        if np.any(actual_starts[1:] < actual_starts[:-1]):
            updated = np.asarray(coords).copy()
            for actual_start, edit_len in zip(actual_starts, deltas, strict=False):
                updated[updated > actual_start] += edit_len
            return updated
        original = np.asarray(coords)
        completed = np.searchsorted(starts, original, side="left")
        return original + prefix_delta[completed]

    updated_starts = apply_coordinate_shift(seq_gff_start_coordinates)
    updated_ends = apply_coordinate_shift(seq_gff_end_coordinates)
    return updated_starts, updated_ends


def should_justify_gap(
    gap_length, target_gap_length, gap_just_min=None, gap_just_max=None
):
    """
    Returns True when a gap should be justified to `target_gap_length`.

    Rules:
      - If `gap_length` already equals target, skip.
      - Gap extension (gap_length < target) follows `gap_just_min` when set.
      - Gap shortening (gap_length > target) follows `gap_just_max` when set.
    """
    if gap_length == target_gap_length:
        return False

    if gap_length < target_gap_length:
        if gap_just_min is not None and gap_length < gap_just_min:
            return False
        return True

    if gap_just_max is not None and gap_length > gap_just_max:
        return False
    return True


def validate_gapjust_args(gap_len, gap_just_min, gap_just_max):
    for value, label in [
        (gap_len, "--gap_len"),
        (gap_just_min, "--gap_just_min"),
        (gap_just_max, "--gap_just_max"),
    ]:
        if value is not None and value < 0:
            raise ValueError(f"{label} must be >= 0. Got {value}.")
    maximum = int(os.environ.get("CDSKIT_MAX_GAP_LENGTH", "1000000"))
    if gap_len > maximum:
        raise ValueError(
            "--gap_len exceeds the {} base safety limit. Set "
            "CDSKIT_MAX_GAP_LENGTH to change it.".format(maximum)
        )


def plan_record_gap_lengths(
    record, target_gap_length, gap_just_min=None, gap_just_max=None
):
    """Enumerate edits in original coordinates without changing the sequence."""
    validate_gapjust_args(target_gap_length, gap_just_min, gap_just_max)
    edits = []
    for match in re.finditer("N+", str(record.seq).upper()):
        start, end = match.span()
        old = end - start
        if should_justify_gap(old, target_gap_length, gap_just_min, gap_just_max):
            edits.append(
                {
                    "original_gap_start": start,
                    "original_gap_length": old,
                    "target_gap_length": target_gap_length,
                    "original_edit_start": start,
                    "edit_length": target_gap_length - old,
                }
            )
    return edits


def apply_record_gap_edits(record, edits):
    """Apply a previously accepted plan; preserved gap bases retain their positions."""
    sequence = str(record.seq).replace("n", "N")
    edits = validate_edits(edits)
    for edit in edits:
        start = edit["original_gap_start"]
        end = start + edit["original_gap_length"]
        validate_gapjust_args(edit["target_gap_length"], None, None)
        if end > len(sequence) or sequence[start:end] != "N" * (end - start):
            raise ValueError(
                "Gap edit does not match an N interval in the original sequence."
            )
    rebuilt: list[str] = []
    cursor = 0
    for edit in edits:
        start = edit["original_gap_start"]
        end = start + edit["original_gap_length"]
        rebuilt.extend((sequence[cursor:start], "N" * edit["target_gap_length"]))
        cursor = end
    rebuilt.append(sequence[cursor:])
    if any(edit["target_gap_length"] != edit["original_gap_length"] for edit in edits):
        # Local coordinate changes invalidate metadata even when deltas sum to zero.
        record.letter_annotations = {}
        record.features = []
    replace_record_sequence(record, "".join(rebuilt))


def normalize_record_gap_lengths(
    record,
    target_gap_length,
    gap_just_min=None,
    gap_just_max=None,
    *,
    gff=None,
    cds_overlap="error",
):
    """Normalize one record, optionally protecting CDS using original-coordinate GFF.

    This changes only the sequence. For paired FASTA/GFF output use gapjust_main,
    or plan/select once and apply the accepted edits to both data structures.
    """
    edits = plan_record_gap_lengths(
        record, target_gap_length, gap_just_min, gap_just_max
    )
    accepted, _ = select_gap_edits(gff, {record.id: edits}, cds_overlap=cds_overlap)
    edits = accepted[record.id]
    apply_record_gap_edits(record, edits)
    lengths = [edit["original_gap_length"] for edit in edits]
    return edits, len(edits), min(lengths, default=None), max(lengths, default=0)


def normalize_record_gap_lengths_entry(
    record,
    target_gap_length,
    gap_just_min=None,
    gap_just_max=None,
    *,
    gff=None,
    cds_overlap="error",
):
    return normalize_record_gap_lengths(
        record,
        target_gap_length,
        gap_just_min,
        gap_just_max,
        gff=gff,
        cds_overlap=cds_overlap,
    )


def summarize_gap_justifications(
    num_justifications, min_original_gap_length, max_original_gap_length
):
    sys.stderr.write(f"Number of gap justifications: {num_justifications}\n")
    if num_justifications > 0:
        sys.stderr.write(
            f"Minimum and maximum original gap lengths: {min_original_gap_length} and {max_original_gap_length}\n"
        )
    else:
        sys.stderr.write("No gap edits were made.\n")


def build_seqid_to_gff_indices(gff_data):
    seqid_to_gff_indices: dict[str, list[int]] = {}
    for ix, seqid in enumerate(gff_data["seqid"]):
        if seqid not in seqid_to_gff_indices:
            seqid_to_gff_indices[seqid] = []
        seqid_to_gff_indices[seqid].append(ix)
    return seqid_to_gff_indices


def apply_gap_justifications_to_gff(gff, justifications_by_seq):
    """Apply safe interval edits, rejecting CDS overlap before mutating any row.

    To skip CDS edits, call select_gap_edits first and use its accepted plan for
    BOTH FASTA and GFF. Legacy coordinate-only edits cannot establish safety.
    """
    justifications_by_seq, _ = select_gap_edits(gff, justifications_by_seq)
    headers = update_sequence_regions(gff.get("header", []), justifications_by_seq)
    seqid_to_gff_indices = build_seqid_to_gff_indices(gff["data"])
    updated_starts = gff["data"]["start"].copy()
    updated_ends = gff["data"]["end"].copy()
    num_justified_start_coordinate = 0
    num_justified_end_coordinate = 0
    num_justified_gff_gene = 0

    for seqid, seqid_justs in justifications_by_seq.items():
        if seqid not in seqid_to_gff_indices:
            continue

        index_gff_seq = np.array(seqid_to_gff_indices[seqid], dtype=int)
        seq_gff_start_original = gff["data"]["start"][index_gff_seq].copy()
        seq_gff_end_original = gff["data"]["end"][index_gff_seq].copy()
        seq_gff_start_updated = seq_gff_start_original.copy()
        seq_gff_end_updated = seq_gff_end_original.copy()

        seq_gff_start_updated, seq_gff_end_updated = vectorized_coordinate_update(
            seq_gff_start_updated,
            seq_gff_end_updated,
            seqid_justs,
        )

        updated_starts[index_gff_seq] = seq_gff_start_updated
        updated_ends[index_gff_seq] = seq_gff_end_updated

        is_gene = gff["data"]["type"][index_gff_seq] == "gene"
        changed_start = seq_gff_start_original != seq_gff_start_updated
        changed_end = seq_gff_end_original != seq_gff_end_updated

        num_justified_start_coordinate += changed_start.sum()
        num_justified_end_coordinate += changed_end.sum()
        justified_changes = np.logical_and(
            is_gene, np.logical_or(changed_start, changed_end)
        )
        num_justified_gff_gene += justified_changes.sum()

    gff["data"]["start"] = updated_starts
    gff["data"]["end"] = updated_ends
    if "header" in gff:
        gff["header"] = headers

    return (
        num_justified_start_coordinate,
        num_justified_end_coordinate,
        num_justified_gff_gene,
    )


def summarize_gff_justifications(
    num_justified_start_coordinate,
    num_justified_end_coordinate,
    num_justified_gff_gene,
):
    sys.stderr.write(
        f"Number of justified GFF start coordinates: {num_justified_start_coordinate}\n"
    )
    sys.stderr.write(
        f"Number of justified GFF end coordinates: {num_justified_end_coordinate}\n"
    )
    sys.stderr.write(
        f"Number of justified GFF gene features: {num_justified_gff_gene}\n"
    )


def stop_if_duplicate_sequence_ids(records):
    counts = Counter(record.id for record in records)
    duplicated = [seq_id for seq_id, count in counts.items() if count > 1]
    if len(duplicated) == 0:
        return
    shown = ",".join(sorted(duplicated)[:10])
    if len(duplicated) > 10:
        shown += ",..."
    txt = (
        "Duplicate sequence IDs are not supported with --in_gff because "
        "GFF seqid mapping becomes ambiguous. Duplicate IDs: {}. Exiting.\n"
    )
    raise ValueError(txt.format(shown))


def gapjust_main(args):
    """Plan and validate all edits before writing paired sequence/annotation output."""
    gap_just_min = getattr(args, "gap_just_min", None)
    gap_just_max = getattr(args, "gap_just_max", None)
    cds_overlap = getattr(args, "cds_overlap", "error")
    report_path = getattr(args, "edit_report", None)
    if args.ingff is not None:
        if args.outgff in (None, ""):
            raise ValueError("--out_gff is required with --in_gff.")
        if args.outfile == "-" and args.outgff == "-":
            raise ValueError(
                "FASTA and GFF cannot share standard output; choose a file for one output."
            )
    if report_path == "-":
        raise ValueError("--edit_report requires a file path, not standard output.")
    validate_gapjust_args(args.gap_len, gap_just_min, gap_just_max)
    validate_distinct_paths(
        inputs=[args.seqfile, args.ingff],
        outputs=[
            args.outfile,
            args.outgff if args.ingff is not None else None,
            report_path,
        ],
    )
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    threads = resolve_threads(getattr(args, "threads", 1))
    gff = None
    diagnostics = []
    if args.ingff is not None:
        stop_if_duplicate_sequence_ids(records)
        gff = read_gff(args.ingff)
        validate_gff_bounds(gff, {record.id: len(record) for record in records})
        diagnostics = gff_diagnostics(gff)
    else:
        diagnostics = [
            "No GFF supplied: CDS overlap is not checked; gapjust is not CDS repair."
        ]
    for diagnostic in diagnostics:
        sys.stderr.write(diagnostic + "\n")

    worker = partial(
        plan_record_gap_lengths,
        target_gap_length=args.gap_len,
        gap_just_min=gap_just_min,
        gap_just_max=gap_just_max,
    )
    plans = parallel_map_ordered(items=records, worker=worker, threads=threads)
    if gff is not None:
        accepted, audit = select_gap_edits(
            gff,
            {record.id: plan for record, plan in zip(records, plans, strict=True)},
            cds_overlap=cds_overlap,
        )
        accepted_plans = [accepted[record.id] for record in records]
        record_indices = {
            record.id: index for index, record in enumerate(records, start=1)
        }
        for entry in audit:
            entry["record_index"] = record_indices[entry["seqid"]]
    else:
        accepted_plans, audit = [], []
        # Duplicate FASTA IDs remain supported without GFF.
        for record_index, (record, plan) in enumerate(
            zip(records, plans, strict=True), start=1
        ):
            accepted, entries = select_gap_edits(
                None, {record.id: plan}, cds_overlap=cds_overlap
            )
            accepted_plans.append(accepted[record.id])
            for entry in entries:
                entry["record_index"] = record_index
            audit.extend(entries)
    justifications_by_seq = {
        record.id: plan for record, plan in zip(records, accepted_plans, strict=True)
    }
    if gff is not None:
        counts = apply_gap_justifications_to_gff(gff, justifications_by_seq)
        summarize_gff_justifications(*counts)
    for record, plan in zip(records, accepted_plans, strict=True):
        apply_record_gap_edits(record, plan)
    if gff is not None:
        validate_gff_bounds(gff, {record.id: len(record) for record in records})
    lengths = [edit["original_gap_length"] for plan in accepted_plans for edit in plan]
    summarize_gap_justifications(
        len(lengths), min(lengths, default=None), max(lengths, default=0)
    )
    skipped = sum(entry["action"] == "skip" for entry in audit)
    sys.stderr.write(f"Number of skipped CDS-overlapping gaps: {skipped}\n")
    output_paths = [
        path
        for path in (
            args.outfile,
            args.outgff if gff is not None else None,
            report_path,
        )
        if path not in (None, "", "-")
    ]
    with atomic_output_paths(output_paths) as staged_paths:
        staged = dict(zip(output_paths, staged_paths, strict=True))
        if report_path:
            atomic_write_json(
                staged[report_path],
                {
                    "schema_version": 1,
                    "coordinate_system": "1-based-inclusive",
                    "cds_overlap": cds_overlap,
                    "cds_checked": gff is not None,
                    "input_diagnostics": diagnostics,
                    "edits": audit,
                },
                indent=2,
            )
        write_seqs(
            records=records,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        if gff is not None:
            write_gff(gff, staged.get(args.outgff, args.outgff))
