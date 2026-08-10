import math
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from cdskit.codonutil import (
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
    write_seqs,
)
from cdskit.tsvio import json_cell, write_sectioned_tsv


def validate_fraction(name, value):
    value = float(value)
    if (not math.isfinite(value)) or (value < 0.0) or (value > 1.0):
        txt = "{} should be between 0 and 1 inclusive. Exiting.\n"
        raise ValueError(txt.format(name))
    return value


def analyze_record(record, codontable, inspect_internal_stop):
    seq_str = str(record.seq)
    length_nt = len(seq_str)
    codon_summary = summarize_codons(seq=seq_str, codontable=codontable)
    total_codons = codon_summary["total"]
    clean_codons = codon_summary["clean"]
    clean_codon_fraction = 0.0
    if total_codons > 0:
        clean_codon_fraction = clean_codons / total_codons
    return {
        "id": record.id,
        "length_nt": length_nt,
        "tail_nt": length_nt % 3,
        "non_triplet": (length_nt % 3) != 0,
        "internal_stop": inspect_internal_stop and codon_summary["internal_stop"],
        "total_codons": total_codons,
        "clean_codons": clean_codons,
        "unclean_codons": total_codons - clean_codons,
        "missing_codons": codon_summary["missing"],
        "ambiguous_codons": codon_summary["ambiguous"],
        "stop_codons": codon_summary["stop"],
        "clean_codon_fraction": clean_codon_fraction,
    }


def analyze_payload(payload, codontable, inspect_internal_stop):
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord

    seq_id, seq_str = payload
    return analyze_record(
        record=SeqRecord(Seq(seq_str), id=seq_id),
        codontable=codontable,
        inspect_internal_stop=inspect_internal_stop,
    )


def analyze_records_process_parallel(
    records, codontable, inspect_internal_stop, threads
):
    worker = partial(
        analyze_payload,
        codontable=codontable,
        inspect_internal_stop=inspect_internal_stop,
    )
    payloads = [(record.id, str(record.seq)) for record in records]
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def choose_duplicate_winners(records, candidate_indices, dedup):
    if dedup == "no":
        return set(candidate_indices), list()
    if dedup == "keep-first":
        seen = set()
        kept = set()
        dropped = list()
        for idx in candidate_indices:
            seq_id = records[idx].id
            if seq_id in seen:
                dropped.append(idx)
                continue
            seen.add(seq_id)
            kept.add(idx)
        return kept, dropped
    if dedup == "keep-longest":
        winners: dict[str, int] = {}
        for idx in candidate_indices:
            seq_id = records[idx].id
            winner_idx = winners.get(seq_id)
            if winner_idx is None:
                winners[seq_id] = idx
                continue
            if len(records[idx].seq) > len(records[winner_idx].seq):
                winners[seq_id] = idx
        kept = set(winners.values())
        dropped = [idx for idx in candidate_indices if idx not in kept]
        return kept, dropped
    raise ValueError("Invalid --dedup: {}. Exiting.\n".format(dedup))


def summarize_filter(records, analyses, args):
    reasons: dict[str, list[str]] = {
        "non_triplet": list(),
        "internal_stop": list(),
        "clean_codon_fraction": list(),
        "duplicate": list(),
    }
    index_reasons: dict[int, list[str]] = {idx: [] for idx in range(len(records))}
    surviving_indices = list()
    for idx, analysis in enumerate(analyses):
        should_drop = False
        if args.drop_non_triplet and analysis["non_triplet"]:
            reasons["non_triplet"].append(records[idx].id)
            index_reasons[idx].append("non_triplet")
            should_drop = True
        if args.drop_internal_stop and analysis["internal_stop"]:
            reasons["internal_stop"].append(records[idx].id)
            index_reasons[idx].append("internal_stop")
            should_drop = True
        if analysis["clean_codon_fraction"] < args.min_clean_codon_fraction:
            reasons["clean_codon_fraction"].append(records[idx].id)
            index_reasons[idx].append("clean_codon_fraction")
            should_drop = True
        if not should_drop:
            surviving_indices.append(idx)

    kept_index_set, duplicate_dropped_indices = choose_duplicate_winners(
        records=records,
        candidate_indices=surviving_indices,
        dedup=args.dedup,
    )
    for idx in duplicate_dropped_indices:
        reasons["duplicate"].append(records[idx].id)
        index_reasons[idx].append("duplicate")

    kept_indices = [idx for idx in surviving_indices if idx in kept_index_set]
    dropped_indices = [idx for idx in range(len(records)) if idx not in kept_index_set]
    sequence_reports = list()
    for idx, analysis in enumerate(analyses):
        sequence_reports.append(
            {
                "input_order": idx + 1,
                "id": records[idx].id,
                "kept": idx in kept_index_set,
                "drop_reasons": index_reasons[idx],
                "length_nt": analysis["length_nt"],
                "tail_nt": analysis["tail_nt"],
                "non_triplet": analysis["non_triplet"],
                "internal_stop": analysis["internal_stop"],
                "total_codons": analysis["total_codons"],
                "clean_codons": analysis["clean_codons"],
                "unclean_codons": analysis["unclean_codons"],
                "missing_codons": analysis["missing_codons"],
                "ambiguous_codons": analysis["ambiguous_codons"],
                "stop_codons": analysis["stop_codons"],
                "clean_codon_fraction": analysis["clean_codon_fraction"],
            }
        )

    return {
        "num_input_sequences": len(records),
        "num_output_sequences": len(kept_indices),
        "num_dropped_sequences": len(dropped_indices),
        "drop_non_triplet": bool(args.drop_non_triplet),
        "drop_internal_stop": bool(args.drop_internal_stop),
        "min_clean_codon_fraction": args.min_clean_codon_fraction,
        "dedup": args.dedup,
        "drop_counts_by_reason": {key: len(ids) for key, ids in reasons.items()},
        "dropped_ids_by_reason": reasons,
        "kept_ids": [records[idx].id for idx in kept_indices],
        "dropped_ids": [records[idx].id for idx in dropped_indices],
        "sequence_reports": sequence_reports,
    }, kept_indices


def write_filter_report(report_path, summary):
    if report_path == "":
        return
    if report_path.lower().endswith(".json"):
        atomic_write_json(report_path, summary, indent=2)
        return
    rows = [
        {"section": "summary", "metric": key, "value": json_cell(summary[key])}
        for key in [
            "num_input_sequences",
            "num_output_sequences",
            "num_dropped_sequences",
            "drop_non_triplet",
            "drop_internal_stop",
            "min_clean_codon_fraction",
            "dedup",
        ]
    ]
    rows.extend(
        [
            {
                "section": "id_set",
                "metric": "dropped_{}_ids".format(key),
                "ids": json_cell(summary["dropped_ids_by_reason"][key]),
            }
            for key in [
                "non_triplet",
                "internal_stop",
                "clean_codon_fraction",
                "duplicate",
            ]
        ]
    )
    rows.extend(
        [
            {
                "section": "id_set",
                "metric": "kept_ids",
                "ids": json_cell(summary["kept_ids"]),
            },
            {
                "section": "id_set",
                "metric": "dropped_ids",
                "ids": json_cell(summary["dropped_ids"]),
            },
        ]
    )
    rows.extend(
        [
            {
                "section": "drop_reason",
                "drop_reason": key,
                "count": summary["drop_counts_by_reason"][key],
            }
            for key in [
                "non_triplet",
                "internal_stop",
                "clean_codon_fraction",
                "duplicate",
            ]
        ]
    )
    for source_row in summary["sequence_reports"]:
        row = dict(source_row)
        row["section"] = "sequence"
        row["kept"] = json_cell(row["kept"])
        row["non_triplet"] = json_cell(row["non_triplet"])
        row["internal_stop"] = json_cell(row["internal_stop"])
        row["drop_reasons"] = json_cell(row["drop_reasons"])
        row["clean_codon_fraction"] = "{:.6g}".format(row["clean_codon_fraction"])
        rows.append(row)
    write_sectioned_tsv(
        path=report_path,
        fieldnames=[
            "section",
            "metric",
            "value",
            "ids",
            "drop_reason",
            "count",
            "input_order",
            "id",
            "kept",
            "drop_reasons",
            "length_nt",
            "tail_nt",
            "non_triplet",
            "internal_stop",
            "total_codons",
            "clean_codons",
            "unclean_codons",
            "missing_codons",
            "ambiguous_codons",
            "stop_codons",
            "clean_codon_fraction",
        ],
        rows=rows,
    )


def filter_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)
    args.min_clean_codon_fraction = validate_fraction(
        name="--min_clean_codon_fraction",
        value=getattr(args, "min_clean_codon_fraction", 0.5),
    )
    threads = resolve_threads(getattr(args, "threads", 1))
    worker = partial(
        analyze_record,
        codontable=args.codontable,
        inspect_internal_stop=args.drop_internal_stop,
    )
    analyses = None
    if should_use_process_pool(records=records, threads=threads):
        try:
            analyses = analyze_records_process_parallel(
                records=records,
                codontable=args.codontable,
                inspect_internal_stop=args.drop_internal_stop,
                threads=threads,
            )
        except (OSError, PermissionError):
            pass
    if analyses is None:
        analyses = parallel_map_ordered(items=records, worker=worker, threads=1)
    summary, kept_indices = summarize_filter(
        records=records, analyses=analyses, args=args
    )
    write_filter_report(report_path=args.report, summary=summary)
    sys.stderr.write(
        "Dropped sequences: {:,}\n".format(summary["num_dropped_sequences"])
    )
    out_records = [records[idx] for idx in kept_indices]
    write_seqs(
        records=out_records, outfile=args.outfile, outseqformat=args.outseqformat
    )
