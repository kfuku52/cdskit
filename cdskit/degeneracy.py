import copy
import sys
from functools import lru_cache
from itertools import product

import numpy as np

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
    write_seqs,
)
from cdskit.tsvio import write_tsv


def build_degeneracy_output_path(prefix, fold, outseqformat):
    return f"{prefix}_{fold}fold_positions.{outseqformat}"


_BASE_CODES = bytes(
    "ACGTacgt".index(chr(i)) % 4 if chr(i) in "ACGTacgt" else 4 for i in range(256)
)


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
    length = len(records[0])
    minimum = np.full(length, 5, dtype=np.int8)
    maximum = np.full(length, -1, dtype=np.int8)
    lookup = _fold_lookup(codontable)
    for record in records:
        encoded = str(record.seq).encode("ascii").translate(_BASE_CODES)
        codons = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 3)
        indices = codons[:, 0] * 25 + codons[:, 1] * 5 + codons[:, 2]
        folds = lookup[indices].reshape(-1)
        np.maximum(maximum, folds, out=maximum)
        np.minimum(minimum, np.where(folds < 0, 5, folds), out=minimum)
    assigned = minimum == maximum
    assignments = np.full(length, None, dtype=object)
    assignments[assigned] = minimum[assigned]
    return {
        "assignments": assignments.tolist(),
        "counts_by_fold": {
            fold: int(np.count_nonzero(assigned & (minimum == fold)))
            for fold in (0, 2, 3, 4)
        },
        "num_conflict_sites": int(np.count_nonzero((maximum >= 0) & ~assigned)),
        "num_unassigned_sites": int(np.count_nonzero(maximum < 0)),
    }


def trim_record_to_positions(record, positions):
    trimmed = copy.copy(record)
    seq_str = str(record.seq)
    replace_record_sequence(trimmed, "".join(seq_str[pos] for pos in positions))
    return trimmed


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
    summary = {
        "num_sequences": len(records),
        "num_input_nt_sites": 0,
        "num_input_codon_sites": 0,
        "num_conflict_sites": 0,
        "num_unassigned_sites": 0,
        "counts_by_fold": {"0": 0, "2": 0, "3": 0, "4": 0},
        "selected_folds": selected_folds,
    }
    prefix_str = resolve_output_prefix(args)
    assignments = []
    if records:
        classification = classify_alignment_columns(
            records=records, codontable=args.codontable
        )
        assignments = classification["assignments"]
        summary = {
            "num_sequences": len(records),
            "num_input_nt_sites": len(records[0]),
            "num_input_codon_sites": len(records[0]) // 3,
            "num_conflict_sites": classification["num_conflict_sites"],
            "num_unassigned_sites": classification["num_unassigned_sites"],
            "counts_by_fold": {
                str(key): value
                for key, value in classification["counts_by_fold"].items()
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
            positions = [
                idx for idx, assignment in enumerate(assignments) if assignment == fold
            ]
            sys.stderr.write(
                f"Writing {fold}-fold positions ({len(positions)} sites).\n"
            )
            out_records = [
                trim_record_to_positions(record=record, positions=positions)
                for record in records
            ]
            write_seqs(
                records=out_records,
                outfile=staged[outfile],
                outseqformat=args.outseqformat,
            )
