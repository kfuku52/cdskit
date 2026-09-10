import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from cdskit.codonutil import (
    CODON_SEMANTICS_VERSION,
    codon_matches_stop_set,
    get_codon_table_components,
    summarize_codons,
)

from cdskit.atomicio import atomic_write_json
from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    should_use_process_pool,
    stop_if_invalid_codontable,
    stop_if_not_dna,
)
from cdskit.tsvio import json_cell, write_sectioned_tsv


MISSING_CHARS = frozenset("-?.")
GAP_ONLY_CHARS = frozenset("-?.NXnx")
UNAMBIGUOUS_NT = frozenset("ACGTacgt")
_DROP_MISSING_CHARS_TABLE = str.maketrans("", "", "".join(sorted(MISSING_CHARS)))
_AMBIGUOUS_CODON_CLASS_CACHE: dict = {}


def chunk_codons(seq):
    return [seq[i : i + 3] for i in range(0, len(seq), 3)]


def get_stop_codons(codontable):
    """Return only unconditional stops; raw sets cannot express dual coding."""
    table = get_codon_table_components(codontable)
    return frozenset(table["stop_codons"] - table["forward_table"].keys())


def is_gap_only_sequence(seq):
    return len(seq) > 0 and all(ch in GAP_ONLY_CHARS for ch in seq)


def has_internal_stop_with_stop_codons(seq, stop_codons):
    seq_upper = seq.upper()
    codons = [seq_upper[i : i + 3] for i in range(0, len(seq_upper) - 2, 3)]
    evaluable_indices = [
        i
        for i, codon in enumerate(codons)
        if not any(ch in MISSING_CHARS for ch in codon)
    ]
    if not evaluable_indices:
        return False
    terminal_index = evaluable_indices[-1]
    for i in evaluable_indices:
        if i == terminal_index and len(seq) % 3 == 0:
            continue
        if codon_matches_stop_set(codons[i], stop_codons):
            return True
    return False


def has_internal_stop(seq, codontable):
    return bool(summarize_codons(seq, codontable)["internal_stop"])


def is_ambiguous_codon(codon):
    if any(ch in MISSING_CHARS for ch in codon):
        return False
    return any(ch not in UNAMBIGUOUS_NT for ch in codon)


def sequence_ambiguous_codon_counts(seq):
    ambiguous = 0
    evaluable = 0
    seq_len = len(seq)
    codon_class_cache = _AMBIGUOUS_CODON_CLASS_CACHE
    for i in range(0, seq_len - 2, 3):
        codon = seq[i : i + 3]
        codon_class = codon_class_cache.get(codon)
        if codon_class is None:
            ch0 = codon[0]
            ch1 = codon[1]
            ch2 = codon[2]
            if (
                (ch0 in MISSING_CHARS)
                or (ch1 in MISSING_CHARS)
                or (ch2 in MISSING_CHARS)
            ):
                codon_class = (0, 0)
            else:
                codon_class = (
                    1,
                    int(
                        (ch0 not in UNAMBIGUOUS_NT)
                        or (ch1 not in UNAMBIGUOUS_NT)
                        or (ch2 not in UNAMBIGUOUS_NT)
                    ),
                )
            codon_class_cache[codon] = codon_class
        evaluable += codon_class[0]
        ambiguous += codon_class[1]
    return ambiguous, evaluable


def get_duplicate_ids(records):
    counts = Counter(record.id for record in records)
    return sorted([seq_id for seq_id, count in counts.items() if count > 1])


def summarize_single_sequence(seq_id, seq, stop_codons=None, *, codontable=None):
    # Retain the old six-field API for callers with an explicit unconditional
    # set. New callers pass a code to retain uncertainty in the extra fields.
    if codontable is None and isinstance(stop_codons, (int, str)):
        codontable, stop_codons = stop_codons, None
    if codontable is None:
        if stop_codons is None:
            raise TypeError("Specify codontable or an explicit unconditional stop set.")
        ambiguous, evaluable = sequence_ambiguous_codon_counts(seq)
        return (
            seq_id,
            len(seq) % 3 != 0,
            is_gap_only_sequence(seq),
            has_internal_stop_with_stop_codons(seq, stop_codons),
            ambiguous,
            evaluable,
        )
    if stop_codons is not None:
        raise ValueError("Specify codontable or stop_codons, not both.")
    summary = summarize_codons(seq, codontable)
    return (
        seq_id,
        len(seq) % 3 != 0,
        is_gap_only_sequence(seq),
        summary["internal_stop"],
        summary["ambiguous"],
        summary["evaluable"],
        summary["possible_stop"],
        summary["context_dependent"],
    )


def summarize_single_record(record, stop_codons=None, *, codontable=None):
    return summarize_single_sequence(
        record.id, str(record.seq), stop_codons, codontable=codontable
    )


def summarize_single_payload(payload, stop_codons=None, *, codontable=None):
    return summarize_single_sequence(
        payload[0], payload[1], stop_codons, codontable=codontable
    )


def summarize_records_process_parallel(
    payloads, stop_codons=None, threads=1, *, codontable=None
):
    worker = partial(
        summarize_single_payload, stop_codons=stop_codons, codontable=codontable
    )
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def summarize_records(records, codontable, threads=1):
    if len(records) <= 1:
        aligned = True
    else:
        first_len = len(records[0].seq)
        aligned = all(len(record.seq) == first_len for record in records[1:])
    duplicate_ids = get_duplicate_ids(records)
    worker_threads = resolve_threads(threads=threads)
    per_record = None
    if should_use_process_pool(records=records, threads=worker_threads):
        try:
            payloads = [(record.id, str(record.seq)) for record in records]
            per_record = summarize_records_process_parallel(
                payloads=payloads,
                codontable=codontable,
                threads=worker_threads,
            )
        except (OSError, PermissionError):
            sys.stderr.write(
                "Process-based parallelism unavailable; falling back to threads.\n"
            )
    if per_record is None:
        worker = partial(summarize_single_record, codontable=codontable)
        per_record = parallel_map_ordered(
            items=records, worker=worker, threads=worker_threads
        )

    non_triplet_ids = [entry[0] for entry in per_record if entry[1]]
    gap_only_ids = [entry[0] for entry in per_record if entry[2]]
    internal_stop_ids = [entry[0] for entry in per_record if entry[3]]
    ambiguous_by_seq: dict[str, int] = {}
    total_ambiguous = 0
    total_evaluable = 0
    for entry in per_record:
        seq_id = entry[0]
        ambiguous_by_seq[seq_id] = ambiguous_by_seq.get(seq_id, 0) + entry[4]
        total_ambiguous += entry[4]
        total_evaluable += entry[5]
    ambiguous_rate = 0.0
    if total_evaluable > 0:
        ambiguous_rate = total_ambiguous / total_evaluable

    issue_ids = set(non_triplet_ids) | set(gap_only_ids) | set(internal_stop_ids)
    for seq_id, count in ambiguous_by_seq.items():
        if count > 0:
            issue_ids.add(seq_id)
    issue_ids |= set(duplicate_ids)

    return {
        "codon_semantics_version": CODON_SEMANTICS_VERSION,
        "codon_table": codontable,
        "terminal_policy": "last_evaluable",
        "possible_stop_codons": sum(entry[6] for entry in per_record),
        "context_dependent_codons": sum(entry[7] for entry in per_record),
        "possible_stop_ids": [entry[0] for entry in per_record if entry[6]],
        "context_dependent_ids": [entry[0] for entry in per_record if entry[7]],
        "num_sequences": len(records),
        "aligned": aligned,
        "non_triplet_ids": non_triplet_ids,
        "duplicate_ids": duplicate_ids,
        "gap_only_ids": gap_only_ids,
        "internal_stop_ids": internal_stop_ids,
        "ambiguous_codons": total_ambiguous,
        "evaluable_codons": total_evaluable,
        "ambiguous_codon_rate": ambiguous_rate,
        "ambiguous_codons_by_sequence": ambiguous_by_seq,
        "num_sequences_with_issues": len(issue_ids),
        "sequence_ids_with_issues": sorted(issue_ids),
    }


def write_validate_report(report_path, summary):
    if report_path == "":
        return
    if report_path.lower().endswith(".json"):
        atomic_write_json(report_path, summary, indent=2)
        return
    count_values = {
        "num_non_triplet_sequences": len(summary["non_triplet_ids"]),
        "num_duplicate_ids": len(summary["duplicate_ids"]),
        "num_gap_only_sequences": len(summary["gap_only_ids"]),
        "num_internal_stop_sequences": len(summary["internal_stop_ids"]),
    }
    rows = [
        {
            "section": "summary",
            "metric": key,
            "value": json_cell(count_values.get(key, summary.get(key, ""))),
        }
        for key in [
            "codon_semantics_version",
            "codon_table",
            "terminal_policy",
            "possible_stop_codons",
            "context_dependent_codons",
            "num_sequences",
            "aligned",
            "num_non_triplet_sequences",
            "num_duplicate_ids",
            "num_gap_only_sequences",
            "num_internal_stop_sequences",
            "ambiguous_codons",
            "evaluable_codons",
            "ambiguous_codon_rate",
            "num_sequences_with_issues",
        ]
    ]
    rows.extend(
        [
            {
                "section": "id_set",
                "metric": key,
                "ids": json_cell(summary[key]),
            }
            for key in [
                "possible_stop_ids",
                "context_dependent_ids",
                "non_triplet_ids",
                "duplicate_ids",
                "gap_only_ids",
                "internal_stop_ids",
                "sequence_ids_with_issues",
            ]
        ]
    )
    write_sectioned_tsv(
        path=report_path,
        fieldnames=["section", "metric", "value", "ids"],
        rows=rows,
    )


def print_validate_summary(summary):
    sys.stdout.write("Validation summary\n")
    for key in (
        "codon_semantics_version",
        "possible_stop_codons",
        "context_dependent_codons",
        "possible_stop_ids",
        "context_dependent_ids",
    ):
        sys.stdout.write(f"{key}\t{json_cell(summary[key])}\n")
    sys.stdout.write(f"num_sequences\t{summary['num_sequences']}\n")
    sys.stdout.write(f"aligned\t{summary['aligned']}\n")
    sys.stdout.write(f"num_non_triplet_sequences\t{len(summary['non_triplet_ids'])}\n")
    sys.stdout.write(f"num_duplicate_ids\t{len(summary['duplicate_ids'])}\n")
    sys.stdout.write(f"num_gap_only_sequences\t{len(summary['gap_only_ids'])}\n")
    sys.stdout.write(
        f"num_internal_stop_sequences\t{len(summary['internal_stop_ids'])}\n"
    )
    sys.stdout.write(f"ambiguous_codons\t{summary['ambiguous_codons']}\n")
    sys.stdout.write(f"evaluable_codons\t{summary['evaluable_codons']}\n")
    sys.stdout.write(f"ambiguous_codon_rate\t{summary['ambiguous_codon_rate']:.6f}\n")
    sys.stdout.write(
        f"num_sequences_with_issues\t{summary['num_sequences_with_issues']}\n"
    )

    if len(summary["non_triplet_ids"]) > 0:
        sys.stdout.write(f"non_triplet_ids\t{','.join(summary['non_triplet_ids'])}\n")
    if len(summary["duplicate_ids"]) > 0:
        sys.stdout.write(f"duplicate_ids\t{','.join(summary['duplicate_ids'])}\n")
    if len(summary["gap_only_ids"]) > 0:
        sys.stdout.write(f"gap_only_ids\t{','.join(summary['gap_only_ids'])}\n")
    if len(summary["internal_stop_ids"]) > 0:
        sys.stdout.write(
            f"internal_stop_ids\t{','.join(summary['internal_stop_ids'])}\n"
        )


def validate_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)
    summary = summarize_records(
        records=records,
        codontable=args.codontable,
        threads=getattr(args, "threads", 1),
    )
    print_validate_summary(summary=summary)
    report_path = getattr(args, "report", "")
    write_validate_report(report_path=report_path, summary=summary)
