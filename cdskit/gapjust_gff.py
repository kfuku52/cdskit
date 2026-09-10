"""Conservative annotation checks for scaffold gap edits (not CDS repair)."""

from bisect import bisect_right
from itertools import accumulate, pairwise
from urllib.parse import unquote

import numpy as np

CDS_TYPES = {"CDS", "SO:0000316"}


def feature_identity(row):
    # Split structural delimiters BEFORE decoding escaped attribute values.
    attributes = {}
    for field in str(row["attributes"]).split(";"):
        key, sep, value = field.partition("=")
        if sep:
            attributes[unquote(key)] = [unquote(item) for item in value.split(",")]
    return {
        "ids": attributes.get("ID", []),
        "parents": attributes.get("Parent", []),
        "type": str(row["type"]),
        "start": int(row["start"]),
        "end": int(row["end"]),
        "strand": str(row["strand"]),
    }


def validate_edits(edits):
    """Require complete, nonoverlapping original-coordinate edits for GFF use."""
    previous_end = 0
    ordered = []
    for edit in edits:
        required = {"original_gap_start", "original_gap_length", "target_gap_length"}
        if not isinstance(edit, dict) or not required <= edit.keys():
            raise ValueError(
                "GFF edits require original gap start/length and target length; legacy shifts are insufficient."
            )
        start, old, target = (
            edit[key]
            for key in (
                "original_gap_start",
                "original_gap_length",
                "target_gap_length",
            )
        )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in (start, old, target)
        ):
            raise ValueError("Gap edit coordinates and lengths must be integers.")
        if start < 0 or old <= 0 or target < 0:
            raise ValueError("Invalid gap edit range or target length.")
        if "edit_length" in edit and edit["edit_length"] != target - old:
            raise ValueError(
                "Gap edit length disagrees with original and target lengths."
            )
        if "original_edit_start" in edit and edit["original_edit_start"] != start:
            raise ValueError("Gap edit start disagrees with the original gap start.")
        ordered.append(
            {
                **edit,
                "original_gap_start": int(start),
                "original_gap_length": int(old),
                "target_gap_length": int(target),
            }
        )
    ordered.sort(key=lambda edit: edit["original_gap_start"])
    for edit in ordered:
        start = edit["original_gap_start"]
        if start < previous_end:
            raise ValueError("Gap edits overlap in original coordinates.")
        previous_end = start + edit["original_gap_length"]
    return ordered


def feature_index(gff):
    rows_by_seq: dict[str, list[np.void]] = {}
    if gff is not None:
        for row in gff["data"]:
            if not 1 <= int(row["start"]) <= int(row["end"]):
                raise ValueError("Invalid input GFF feature coordinates.")
            rows_by_seq.setdefault(str(row["seqid"]), []).append(row)
    index = {}
    for seqid, rows in rows_by_seq.items():
        rows.sort(key=lambda row: int(row["start"]))
        starts = [int(row["start"]) for row in rows]
        max_ends = list(accumulate((int(row["end"]) for row in rows), max))
        index[seqid] = (rows, starts, max_ends)
    return index


def overlapping_features(index, seqid, start, end):
    """Query a 0-based half-open interval; GFF bounds remain 1-based inclusive."""
    if seqid not in index:
        return []
    rows, starts, max_ends = index[seqid]
    left, right = bisect_right(max_ends, start), bisect_right(starts, end)
    return [row for row in rows[left:right] if int(row["end"]) > start]


def select_gap_edits(gff, justifications_by_seq, *, cds_overlap="error"):
    """Return accepted edits and JSON-ready audit entries without mutating inputs."""
    if cds_overlap not in {"error", "skip"}:
        raise ValueError("cds_overlap must be 'error' or 'skip'.")
    index = feature_index(gff)
    accepted, audit = {}, []
    for seqid, edits in justifications_by_seq.items():
        selected = []
        for edit in validate_edits(edits):
            start = int(edit["original_gap_start"])
            old = int(edit["original_gap_length"])
            target = int(edit["target_gap_length"])
            if old == target:
                continue
            overlaps = overlapping_features(index, seqid, start, start + old)
            cds = [row for row in overlaps if str(row["type"]) in CDS_TYPES]
            entry = {
                "seqid": seqid,
                "original_start": start + 1,
                "original_end": start + old,
                "target_length": target,
                "length_delta": target - old,
                "action": "apply",
                "reason": "non_cds_edit" if gff is not None else "cds_not_checked",
                "features": [feature_identity(row) for row in overlaps],
            }
            if cds:
                entry.update(action="skip", reason="cds_overlap")
                if cds_overlap == "error":
                    raise ValueError(
                        f"CDS overlap at {seqid}:{start + 1}-{start + old}: {entry['features']}. Use --cds_overlap skip to preserve the entire N run."
                    )
            else:
                deleted_start = start + target + 1
                if target < old and any(
                    deleted_start <= int(row[column]) <= start + old
                    for row in overlaps
                    for column in ("start", "end")
                ):
                    raise ValueError(
                        f"Deleted feature endpoint at {seqid}:{deleted_start}-{start + old}; reannotation is required (coordinates cannot be clamped)."
                    )
                selected.append(edit)
            audit.append(entry)
        accepted[seqid] = selected
    return accepted, audit


def validate_gff_bounds(gff, sequence_lengths):
    for row in gff["data"]:
        seqid = str(row["seqid"])
        if seqid not in sequence_lengths:
            raise ValueError(f"GFF seqid {seqid} has no matching FASTA record.")
        if not 1 <= int(row["start"]) <= int(row["end"]) <= sequence_lengths[seqid]:
            raise ValueError(
                f"Invalid GFF coordinates for {seqid}: {row['start']}-{row['end']}."
            )

    for line in gff.get("header", []):
        fields = line.split()
        if fields and fields[0] == "##sequence-region":
            if len(fields) != 4 or fields[1] not in sequence_lengths:
                raise ValueError("Invalid or unmatched ##sequence-region directive.")
            if not 1 <= int(fields[2]) <= int(fields[3]) <= sequence_lengths[fields[1]]:
                raise ValueError("##sequence-region lies outside the FASTA record.")


def parent_graph_has_cycle(identities):
    graph: dict[str, set[str]] = {}
    for identity in identities:
        for key in identity["ids"]:
            graph.setdefault(key, set()).update(identity["parents"])
    # Kahn's algorithm handles shared children and repeated IDs without recursion.
    degree = {key: 0 for key in graph}
    for parents in graph.values():
        for parent in parents:
            if parent in degree:
                degree[parent] += 1
    pending = [key for key, count in degree.items() if count == 0]
    visited = 0
    while pending:
        key = pending.pop()
        visited += 1
        for parent in graph[key]:
            if parent in degree:
                degree[parent] -= 1
                if degree[parent] == 0:
                    pending.append(parent)
    return visited != len(graph)


def update_sequence_regions(headers, edits_by_seq):
    """Map directive intervals, including full regions whose end bases are deleted."""
    result = []
    for line in headers:
        fields = line.split()
        if not fields or fields[0] != "##sequence-region":
            result.append(line)
            continue
        if len(fields) != 4:
            raise ValueError("Invalid ##sequence-region directive.")
        seqid = fields[1]
        start, end = int(fields[2]), int(fields[3])
        if not 1 <= start <= end:
            raise ValueError("Invalid ##sequence-region bounds.")
        left, right = start - 1, end
        for edit in edits_by_seq.get(seqid, []):
            gap = edit["original_gap_start"]
            old, target = edit["original_gap_length"], edit["target_gap_length"]
            # Crossing a directive boundary leaves interval ownership ambiguous.
            if gap < start - 1 < gap + old or gap < end < gap + old:
                raise ValueError("Gap edit crosses a ##sequence-region boundary.")
            if gap + old <= start - 1:
                left += target - old
            if gap + old <= end:
                right += target - old
        if right <= left:
            raise ValueError("Gap edits delete an entire ##sequence-region.")
        result.append(f"##sequence-region {seqid} {left + 1} {right}")
    return result


def gff_diagnostics(gff):
    """Report input limitations; never infer phase or alter annotation relationships."""
    identities = [feature_identity(row) for row in gff["data"]]
    known = {key for identity in identities for key in identity["ids"]}
    messages = set()
    if parent_graph_has_cycle(identities):
        messages.add(
            "Input GFF contains a Parent cycle; relationships are not repaired."
        )
    groups: dict[str, list[np.void]] = {}
    unverified_parents = set()
    for row, identity in zip(gff["data"], identities, strict=True):
        parents = identity["parents"]
        if any(parent not in known for parent in parents):
            messages.add("Input GFF contains unresolved Parent references.")
        if str(row["type"]) not in CDS_TYPES:
            continue
        if not parents:
            messages.add(
                "Input CDS lacks Parent; transcript continuity is not verified."
            )
        if str(row["phase"]) not in {"0", "1", "2"} or str(row["strand"]) not in {
            "+",
            "-",
        }:
            messages.add(
                "Input CDS has missing/invalid phase or unknown strand; continuity is not verified."
            )
            unverified_parents.update(parents)
            continue
        for parent in parents:
            groups.setdefault(parent, []).append(row)
    for parent, rows in groups.items():
        if parent in unverified_parents:
            continue
        strands = {str(row["strand"]) for row in rows}
        seqids = {str(row["seqid"]) for row in rows}
        if len(strands) != 1 or len(seqids) != 1:
            messages.add(
                "Input CDS Parent spans multiple strands or seqids; continuity is not verified."
            )
            continue
        rows = sorted(
            rows, key=lambda row: int(row["start"]), reverse=next(iter(strands)) == "-"
        )
        for previous, current in pairwise(rows):
            expected = (
                int(previous["phase"])
                - (int(previous["end"]) - int(previous["start"]) + 1)
            ) % 3
            if int(current["phase"]) != expected:
                messages.add(
                    "Input CDS phase continuity is inconsistent under an ordinary spliced-CDS model; biological exceptions require separate review."
                )
    return sorted(messages)
