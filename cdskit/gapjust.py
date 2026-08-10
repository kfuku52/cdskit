import numpy as np
import os
import sys
import re
from collections import Counter
from functools import partial

from cdskit.atomicio import atomic_output_paths
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
    """
    Updates GFF feature coordinates in place for a list of gap
    justifications. Does NOT touch phase.

    We assume that `original_edit_start` in each justification
    is a 0-based index (as used in Python strings), whereas
    GFF is 1-based. Hence we add +1 when applying the shift.

    We also apply a 'cumulative_offset' so that each gap edit
    is replayed in ascending order, just like the iterative
    edits to the FASTA.
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
            for edit_index in np.unique(containing[inside]):
                mask = inside & (containing == edit_index)
                target_length = int(targets[edit_index])
                mapped_start = int(starts[edit_index] + prefix_delta[edit_index])
                if target_length == 0:
                    updated[mask] = max(1, mapped_start - 1)
                elif deltas[edit_index] < 0:
                    updated[mask] = np.minimum(
                        updated[mask],
                        mapped_start + target_length - 1,
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


def normalize_record_gap_lengths(
    record, target_gap_length, gap_just_min=None, gap_just_max=None
):
    seq_str = str(record.seq).replace("n", "N")
    seq_justifications = []
    rebuilt = []
    cursor = 0
    num_justifications = 0
    min_original_gap_length = None
    max_original_gap_length = 0

    for match in re.finditer("N+", seq_str):
        start = match.start()
        end = match.end()
        gap_length = end - start
        rebuilt.append(seq_str[cursor:start])

        justify_gap = should_justify_gap(
            gap_length=gap_length,
            target_gap_length=target_gap_length,
            gap_just_min=gap_just_min,
            gap_just_max=gap_just_max,
        )

        if justify_gap:
            rebuilt.append("N" * target_gap_length)
            edit_len = target_gap_length - gap_length
            seq_justifications.append(
                {
                    "original_gap_start": start,
                    "original_gap_length": gap_length,
                    "target_gap_length": target_gap_length,
                    "original_edit_start": start,
                    "edit_length": edit_len,
                }
            )
            num_justifications += 1
            if (min_original_gap_length is None) or (
                gap_length < min_original_gap_length
            ):
                min_original_gap_length = gap_length
            if gap_length > max_original_gap_length:
                max_original_gap_length = gap_length
        else:
            rebuilt.append(seq_str[start:end])

        cursor = end

    rebuilt.append(seq_str[cursor:])
    replace_record_sequence(record, "".join(rebuilt))

    return (
        seq_justifications,
        num_justifications,
        min_original_gap_length,
        max_original_gap_length,
    )


def normalize_record_gap_lengths_entry(
    record, target_gap_length, gap_just_min=None, gap_just_max=None
):
    return normalize_record_gap_lengths(
        record=record,
        target_gap_length=target_gap_length,
        gap_just_min=gap_just_min,
        gap_just_max=gap_just_max,
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
    seqid_to_gff_indices = build_seqid_to_gff_indices(gff["data"])
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

        gff["data"]["start"][index_gff_seq] = seq_gff_start_updated
        gff["data"]["end"][index_gff_seq] = seq_gff_end_updated

        is_gene = gff["data"]["type"][index_gff_seq] == "gene"
        changed_start = seq_gff_start_original != seq_gff_start_updated
        changed_end = seq_gff_end_original != seq_gff_end_updated

        num_justified_start_coordinate += changed_start.sum()
        num_justified_end_coordinate += changed_end.sum()
        justified_changes = np.logical_and(
            is_gene, np.logical_or(changed_start, changed_end)
        )
        num_justified_gff_gene += justified_changes.sum()

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
    """
    Main routine for:
      1) Reading FASTA and replacing all gap lengths with a uniform length (args.gap_len).
      2) Tracking each insertion/deletion in 'justifications'.
      3) Writing the updated FASTA.
      4) Updating GFF coordinates accordingly (without modifying phase),
         then writing the updated GFF.
    """

    gap_just_min = getattr(args, "gap_just_min", None)
    gap_just_max = getattr(args, "gap_just_max", None)
    validate_gapjust_args(args.gap_len, gap_just_min, gap_just_max)

    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    threads = resolve_threads(getattr(args, "threads", 1))
    num_justifications = 0
    min_original_gap_length = None
    max_original_gap_length = 0
    justifications_by_seq = dict()

    worker = partial(
        normalize_record_gap_lengths_entry,
        target_gap_length=args.gap_len,
        gap_just_min=gap_just_min,
        gap_just_max=gap_just_max,
    )
    normalized_results = parallel_map_ordered(
        items=records, worker=worker, threads=threads
    )

    for record, normalized_result in zip(records, normalized_results, strict=False):
        (
            seq_justifications,
            record_num_justifications,
            record_min_original_gap_length,
            record_max_original_gap_length,
        ) = normalized_result

        num_justifications += record_num_justifications
        if record_min_original_gap_length is not None:
            if (min_original_gap_length is None) or (
                record_min_original_gap_length < min_original_gap_length
            ):
                min_original_gap_length = record_min_original_gap_length
            max_original_gap_length = max(
                max_original_gap_length, record_max_original_gap_length
            )
        if seq_justifications:
            justifications_by_seq[record.id] = seq_justifications

    if args.ingff is not None:
        stop_if_duplicate_sequence_ids(records=records)

    gff = None
    if args.ingff is not None:
        gff = read_gff(args.ingff)
        (
            num_justified_start_coordinate,
            num_justified_end_coordinate,
            num_justified_gff_gene,
        ) = apply_gap_justifications_to_gff(gff, justifications_by_seq)
        summarize_gff_justifications(
            num_justified_start_coordinate,
            num_justified_end_coordinate,
            num_justified_gff_gene,
        )
        sequence_lengths = {record.id: len(record) for record in records}
        for row in gff["data"]:
            seq_len = sequence_lengths.get(str(row["seqid"]))
            if seq_len is None:
                continue
            start = int(row["start"])
            end = int(row["end"])
            if not (1 <= start <= end <= seq_len):
                raise ValueError(
                    "Gap-adjusted GFF coordinates are invalid for {}: {}-{} "
                    "(sequence length {}).".format(
                        row["seqid"],
                        start,
                        end,
                        seq_len,
                    )
                )

    summarize_gap_justifications(
        num_justifications,
        min_original_gap_length,
        max_original_gap_length,
    )
    output_paths = [
        path
        for path in (args.outfile, args.outgff if gff is not None else None)
        if path not in (None, "", "-")
    ]
    with atomic_output_paths(output_paths) as staged_paths:
        staged = dict(zip(output_paths, staged_paths, strict=False))
        write_seqs(
            records=records,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        if gff is not None:
            write_gff(gff, staged.get(args.outgff, args.outgff))
