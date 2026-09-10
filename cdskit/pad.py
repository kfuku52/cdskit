#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import Bio.Seq
import Bio.SeqIO
import sys

from cdskit.atomicio import atomic_output_paths
from cdskit.codonreport import validate_codon_output_paths, write_codon_report
from cdskit.codonutil import (
    definite_stop_patterns,
    get_codon_table_components,
    summarize_codons,
)

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    replace_record_sequence,
    resolve_threads,
    should_use_process_pool,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    write_seqs,
)


def get_stop_codons(codon_table):
    table = get_codon_table_components(codon_table)
    return set(table["stop_codons"] - table["forward_table"].keys())


def get_stop_codon_scan_list(codon_table):
    return definite_stop_patterns(frozenset(get_stop_codons(codon_table)))


def count_internal_stop_codons(seq, codon_table):
    sequence = str(seq).upper()
    stops = get_stop_codon_scan_list(codon_table)
    limit = len(sequence) - 3
    if not stops or limit <= 0:
        return 0
    count = 0
    for codon in stops:
        pos = sequence.find(codon)
        while pos != -1 and pos < limit:
            count += int(pos % 3 == 0)
            pos = sequence.find(codon, pos + 1)
    return count


class padseqs:
    def __init__(self, original_seq, codon_table="Standard", padchar="N"):
        self.new_seqs = list()
        self.num_stops = list()
        self.headn = list()
        self.tailn = list()
        self.original_seq = str(original_seq)
        self.codon_table = codon_table
        self.padchar = padchar

    def add(self, headn=0, tailn=0):
        new_seq = Bio.Seq.Seq(
            (self.padchar * headn) + self.original_seq + (self.padchar * tailn)
        )
        self.new_seqs.append(new_seq)
        self.num_stops.append(count_internal_stop_codons(new_seq, self.codon_table))
        self.headn.append(headn)
        self.tailn.append(tailn)

    def get_minimum_num_stop(self):
        min_index = min(range(len(self.num_stops)), key=lambda i: self.num_stops[i])
        out = {
            "new_seq": self.new_seqs[min_index],
            "num_stop": self.num_stops[min_index],
            "headn": self.headn[min_index],
            "tailn": self.tailn[min_index],
        }
        return out


def get_adjusted_length_and_tailpadded_sequence(clean_seq, padchar):
    seqlen = len(clean_seq)
    if seqlen % 3 == 0:
        return seqlen, clean_seq
    adjlen = ((seqlen // 3) + 1) * 3
    return adjlen, clean_seq.ljust(adjlen, padchar)


def get_padding_candidates(num_stop_input, num_missing, seqlen):
    candidates = []
    if num_stop_input:
        if (num_missing == 0) or (num_missing == 3):
            candidates.extend([(0, 0), (1, 2), (2, 1)])
        elif num_missing == 1:
            candidates.extend([(0, 1), (1, 0), (2, 2)])
        elif num_missing == 2:
            candidates.extend([(0, 2), (2, 0), (1, 1)])
    if (not num_stop_input) and (seqlen % 3):
        candidates.append((0, num_missing))
    return candidates


def choose_best_padding(
    clean_seq, codon_table, padchar, num_stop_input, num_missing, seqlen, tailpad_seq
):
    best = None
    for headn, tailn in get_padding_candidates(num_stop_input, num_missing, seqlen):
        if (headn == 0) and (tailn == num_missing):
            # Reuse already evaluated tail-padded sequence.
            new_seq = tailpad_seq
            num_stop = num_stop_input
        else:
            new_seq = (padchar * headn) + clean_seq + (padchar * tailn)
            num_stop = count_internal_stop_codons(new_seq, codon_table)
        if (best is None) or (num_stop < best["num_stop"]):
            best = {
                "new_seq": new_seq,
                "num_stop": num_stop,
                "headn": headn,
                "tailn": tailn,
            }
    return best


def process_record_padding(
    record_name, record_seq, codon_table, padchar, mode="min-stop", include_report=True
):
    if mode not in ("min-stop", "preserve-frame"):
        raise ValueError(f"Unknown padding mode: {mode}")
    if padchar not in ("N", "-"):
        raise ValueError("Padding character must be N or -.")
    clean_seq = record_seq if mode == "preserve-frame" else record_seq.replace("X", "N")
    seqlen = len(clean_seq)
    adjlen, tailpad_seq = get_adjusted_length_and_tailpadded_sequence(
        clean_seq, padchar
    )
    num_missing = adjlen - seqlen
    num_stop_input = count_internal_stop_codons(tailpad_seq, codon_table)
    placements = (
        [(0, num_missing)]
        if mode == "preserve-frame"
        else get_padding_candidates(num_stop_input, num_missing, seqlen)
    )
    if not placements:
        placements = [(0, 0)]
    candidates = []
    for headn, tailn in placements:
        sequence = padchar * headn + clean_seq + padchar * tailn
        summary = (
            summarize_codons(sequence, codon_table, "physical")
            if include_report
            else {
                "internal_stop_count": num_stop_input
                if headn == 0 and tailn == num_missing
                else count_internal_stop_codons(sequence, codon_table)
            }
        )
        candidates.append(
            {
                "head_padding": headn,
                "tail_padding": tailn,
                "original_start_in_output_1based": headn + 1 if seqlen else None,
                "original_end_in_output_1based": headn + seqlen if seqlen else None,
                "original_frame_offset": (-headn) % 3,
                "new_seq": sequence,
                **summary,
            }
        )
    best_index = min(
        range(len(candidates)), key=lambda i: candidates[i]["internal_stop_count"]
    )
    best = candidates[best_index]
    ties = [
        i
        for i, candidate in enumerate(candidates)
        if candidate["internal_stop_count"] == best["internal_stop_count"]
    ]
    headn, tailn = best["head_padding"], best["tail_padding"]
    was_padded = headn != 0 or tailn != 0
    output_seq = best["new_seq"] if (num_stop_input or seqlen % 3) else record_seq
    # Preserve the original spelling when no evaluation was needed.
    best["new_seq"] = output_seq
    log = ""
    if num_stop_input or seqlen % 3:
        log = (
            f"{record_name}, original_seqlen={seqlen}, head_padding={headn}, tail_padding={tailn}, "
            f"tail_padded_num_stop={num_stop_input}, new_num_stop={best['internal_stop_count']}\n"
        )
    result = {
        "new_seq": output_seq,
        "is_no_stop": best["internal_stop_count"] == 0,
        "was_padded": was_padded,
        "log": log,
    }
    if not include_report:
        return result
    return {
        **result,
        "mode": mode,
        "terminal_policy": "physical",
        "original_frame_offset": 0,
        "original": summarize_codons(record_seq, codon_table, "physical"),
        "tail_padded": summarize_codons(tailpad_seq, codon_table, "physical"),
        "candidates": candidates,
        "selected_candidate_0based": best_index,
        "tied_candidates_0based": ties,
        "selection_reason": "preserve_frame"
        if mode == "preserve-frame"
        else "minimum_definite_internal_stops_then_candidate_order",
    }


def process_record_padding_entry(
    record, codon_table, padchar, mode="min-stop", include_report=True
):
    return process_record_padding(
        record_name=record.name,
        record_seq=str(record.seq),
        codon_table=codon_table,
        padchar=padchar,
        mode=mode,
        include_report=include_report,
    )


def process_record_padding_payload(
    payload, codon_table, padchar, mode="min-stop", include_report=True
):
    record_name, record_seq = payload
    return process_record_padding(
        record_name=record_name,
        record_seq=record_seq,
        codon_table=codon_table,
        padchar=padchar,
        mode=mode,
        include_report=include_report,
    )


def process_padding_payloads_process_parallel(
    payloads, codon_table, padchar, threads, mode="min-stop", include_report=True
):
    worker = partial(
        process_record_padding_payload,
        codon_table=codon_table,
        padchar=padchar,
        mode=mode,
        include_report=include_report,
    )
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def pad_main(args):
    validate_codon_output_paths(args.seqfile, args.outfile, getattr(args, "report", ""))
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)
    threads = resolve_threads(getattr(args, "threads", 1))
    report = getattr(args, "report", "")
    results = None
    if should_use_process_pool(records=records, threads=threads):
        try:
            payloads = [(record.name, str(record.seq)) for record in records]
            results = process_padding_payloads_process_parallel(
                payloads=payloads,
                codon_table=args.codontable,
                padchar=args.padchar,
                mode=getattr(args, "mode", "min-stop"),
                include_report=bool(report),
                threads=threads,
            )
        except (OSError, PermissionError):
            pass
    if results is None:
        worker = partial(
            process_record_padding_entry,
            codon_table=args.codontable,
            padchar=args.padchar,
            mode=getattr(args, "mode", "min-stop"),
            include_report=bool(report),
        )
        results = parallel_map_ordered(items=records, worker=worker, threads=threads)
    report_records = []
    for index, (record, result) in enumerate(zip(records, results, strict=True)):
        if not report:
            break
        kept = not args.nopseudo or result["is_no_stop"]
        report_records.append(
            {
                "input_order": index + 1,
                "seq_id": record.id,
                **result,
                "kept": kept,
                "drop_reason": "" if kept else "definite_internal_stop_after_padding",
            }
        )
    is_no_stop = []
    was_padded = []
    log_lines = list()
    for i, result in enumerate(results):
        replace_record_sequence(records[i], result["new_seq"])
        if result["log"] != "":
            log_lines.append(result["log"])
        is_no_stop.append(result["is_no_stop"])
        was_padded.append(result["was_padded"])
    if len(log_lines) > 0:
        sys.stderr.write("".join(log_lines))
    if args.nopseudo:
        retained = [i for i in range(len(records)) if is_no_stop[i]]
        records = [records[i] for i in retained]
        seqnum_padded = sum(was_padded[i] for i in retained)
    else:
        seqnum_padded = sum(was_padded)
    sys.stderr.write(
        "Number of padded sequences: {:,} / {:,}\n".format(seqnum_padded, len(records))
    )
    outputs = [path for path in (args.outfile, report) if path not in ("", "-")]
    with atomic_output_paths(outputs) as temporary_paths:
        staged = dict(zip(outputs, temporary_paths, strict=True))
        write_seqs(
            records=records,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        write_codon_report(
            staged.get(report, report), "pad", args.codontable, report_records
        )
