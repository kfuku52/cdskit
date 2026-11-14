import copy
import sys
from functools import lru_cache, partial
from itertools import product

import numpy as np
from Bio.Seq import Seq

from cdskit.codonutil import degeneracy_fold_by_position
from cdskit.split import resolve_output_prefix
from cdskit.atomicio import (
    atomic_output_paths,
    atomic_write_json,
    validate_distinct_paths,
)
from cdskit.util import (
    read_seqs,
    replace_record_sequence,
    stop_if_invalid_codontable,
    stop_if_not_aligned,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_transformed_seqs,
)
from cdskit.tsvio import write_tsv


def build_degeneracy_output_path(prefix, fold, outseqformat):
    return f"{prefix}_{fold}fold_positions.{outseqformat}"


_BASE_CODES = bytes(
    "ACGTacgt".index(chr(i)) % 4 if chr(i) in "ACGTacgt" else 4 for i in range(256)
)
_CLASSIFY_BLOCK_NT = 3 * 65_536
_COMPACT_SELECTION_MIN = 1_000_000


@lru_cache(maxsize=64)
def _fold_lookup(codontable):
    # The fifth base code covers ambiguity, gaps and invalid characters.
    lookup = np.full((125, 3), -1, dtype=np.int8)
    for codes in product(range(4), repeat=3):
        codon = "".join("ACGT"[code] for code in codes)
        folds = degeneracy_fold_by_position(codon, codontable)
        if folds is not None:
            lookup[codes[0] * 25 + codes[1] * 5 + codes[2]] = folds
    lookup.flags.writeable = False
    return lookup


def classify_alignment_columns(records, codontable):
    result = _classify_alignment_columns(records, codontable)
    folds = result["assignments"]
    assignments = folds.astype(object)
    assignments[folds < 0] = None
    return {**result, "assignments": assignments.tolist()}


def _classify_alignment_columns(records, codontable):
    length = len(records[0])
    minimum = np.full(length, 5, dtype=np.int8)
    maximum = np.full(length, -1, dtype=np.int8)
    lookup = _fold_lookup(codontable)
    for record in records:
        sequence = record.seq
        if type(sequence) is Seq:
            data = bytes(sequence)
            if not data.isascii():
                data.decode("ascii")  # Preserve the sequence decoding error.
        else:
            data = str(sequence).encode("ascii")
        if len(data) != length or length % 3:
            raise ValueError("Expected aligned sequences with complete codons.")
        for start in range(0, length, _CLASSIFY_BLOCK_NT):
            stop = min(start + _CLASSIFY_BLOCK_NT, length)
            encoded = data[start:stop].translate(_BASE_CODES)
            codons = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 3)
            indices = codons[:, 0] * 25 + codons[:, 1] * 5 + codons[:, 2]
            folds = lookup[indices].reshape(-1)
            low, high = minimum[start:stop], maximum[start:stop]
            np.maximum(high, folds, out=high)
            np.minimum(low, np.where(folds < 0, 5, folds), out=low)
    counts = dict.fromkeys((0, 2, 3, 4), 0)
    conflicts = unassigned = 0
    for start in range(0, length, _CLASSIFY_BLOCK_NT):
        stop = min(start + _CLASSIFY_BLOCK_NT, length)
        low, high = minimum[start:stop], maximum[start:stop]
        assigned = low == high
        low[~assigned] = -1
        for fold in counts:
            counts[fold] += int(np.count_nonzero(low == fold))
        conflicts += int(np.count_nonzero((high >= 0) & ~assigned))
        unassigned += int(np.count_nonzero(high < 0))
    return {
        "assignments": minimum,
        "counts_by_fold": counts,
        "num_conflict_sites": conflicts,
        "num_unassigned_sites": unassigned,
    }


def trim_record_to_positions(record, positions):
    trimmed = copy.copy(record)
    sequence: bytes | str
    if (
        isinstance(positions, np.ndarray)
        and positions.ndim == 1
        and positions.dtype.kind in "iu"
    ):
        bases = np.frombuffer(bytes(record.seq), dtype=np.uint8)
        sequence = bases[positions].tobytes()
    else:
        seq_str = str(record.seq)
        sequence = "".join(seq_str[pos] for pos in positions)
    replace_record_sequence(trimmed, sequence)
    return trimmed


def _positions_for_fold(assignments, fold, count):
    if (
        count < _COMPACT_SELECTION_MIN
        or np.dtype(np.intp).itemsize <= np.dtype(np.int32).itemsize
        or len(assignments) - 1 > np.iinfo(np.int32).max
    ):
        return np.flatnonzero(assignments == fold)
    # The report already gives the count. Build compact indexes without ever
    # allocating a full-width boolean mask or converting a full intp array.
    positions = np.empty(count, dtype=np.int32)
    cursor = 0
    for start in range(0, len(assignments), _CLASSIFY_BLOCK_NT):
        block = assignments[start : start + _CLASSIFY_BLOCK_NT]
        local = np.flatnonzero(block == fold)
        stop = cursor + len(local)
        positions[cursor:stop] = local + start
        cursor = stop
    return positions


def write_degeneracy_report(report_path, summary):
    if report_path == "":
        return
    if report_path.lower().endswith(".json"):
        atomic_write_json(report_path, summary, indent=2)
        return
    rows = [
        {"metric": key, "value": summary[key]}
        for key in [
            "num_sequences",
            "num_input_nt_sites",
            "num_input_codon_sites",
            "num_conflict_sites",
            "num_unassigned_sites",
        ]
    ]
    rows.extend(
        [
            {
                "metric": "num_{}fold_sites".format(fold),
                "value": summary["counts_by_fold"][str(fold)],
            }
            for fold in [0, 2, 3, 4]
        ]
    )
    write_tsv(path=report_path, rows=rows, fieldnames=["metric", "value"])


def degeneracy_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_not_aligned(records=records)
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    selected_folds = sorted(set(int(fold) for fold in args.fold))
    counts_by_fold = {0: 0, 2: 0, 3: 0, 4: 0}
    summary = {
        "num_sequences": len(records),
        "num_input_nt_sites": 0,
        "num_input_codon_sites": 0,
        "num_conflict_sites": 0,
        "num_unassigned_sites": 0,
        "counts_by_fold": {str(key): value for key, value in counts_by_fold.items()},
        "selected_folds": selected_folds,
    }
    prefix_str = resolve_output_prefix(args)
    assignments = np.empty(0, dtype=np.int8)
    if records:
        classification = _classify_alignment_columns(
            records=records, codontable=args.codontable
        )
        assignments = classification["assignments"]
        counts_by_fold = classification["counts_by_fold"]
        summary = {
            "num_sequences": len(records),
            "num_input_nt_sites": len(records[0]),
            "num_input_codon_sites": len(records[0]) // 3,
            "num_conflict_sites": classification["num_conflict_sites"],
            "num_unassigned_sites": classification["num_unassigned_sites"],
            "counts_by_fold": {
                str(key): value for key, value in counts_by_fold.items()
            },
            "selected_folds": selected_folds,
        }

    sequence_outputs = [
        build_degeneracy_output_path(
            prefix=prefix_str,
            fold=fold,
            outseqformat=args.outseqformat,
        )
        for fold in selected_folds
    ]
    outputs = sequence_outputs + ([args.report] if args.report not in ("", "-") else [])
    validate_distinct_paths(inputs=[args.seqfile], outputs=outputs)
    with atomic_output_paths(outputs) as staged_outputs:
        staged = dict(zip(outputs, staged_outputs, strict=False))
        write_degeneracy_report(
            report_path=staged.get(args.report, args.report),
            summary=summary,
        )
        for fold, outfile in zip(selected_folds, sequence_outputs, strict=False):
            positions = _positions_for_fold(
                assignments, fold, counts_by_fold.get(fold, 0)
            )
            sys.stderr.write(
                f"Writing {fold}-fold positions ({len(positions)} sites).\n"
            )
            write_transformed_seqs(
                records=records,
                transform=partial(trim_record_to_positions, positions=positions),
                outfile=staged[outfile],
                outseqformat=args.outseqformat,
            )
