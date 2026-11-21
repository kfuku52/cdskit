#!/usr/bin/env python

import sys
from collections import defaultdict
from functools import partial
from pathlib import Path

from Bio.SeqRecord import SeqRecord

from cdskit.atomicio import atomic_output_paths, validate_distinct_paths
from cdskit.homology import read_hits, select_representative, write_selection_report
from cdskit.tsvio import read_tsv_columns
from cdskit.util import (
    compile_safe_regex,
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_seqtype,
    write_seqs,
)


def aggregate_name(name, expressions):
    aggregated = name
    for expr in expressions:
        if not hasattr(expr, "sub"):
            expr = compile_safe_regex(expr, label="regex in --expression")
        aggregated = expr.sub("", aggregated)
    return aggregated


def select_aggregate_record(existing_record, candidate_record, mode):
    if mode == "longest":
        if len(existing_record.seq) < len(candidate_record.seq):
            return candidate_record
        return existing_record
    sys.stderr.write("different modes to be supported in future.")
    return existing_record


def validate_aggregate_expressions(expressions):
    return [
        compile_safe_regex(expr, label="regex in --expression") for expr in expressions
    ]


def read_group_table(path):
    groups = {}
    for transcript, group in read_tsv_columns(path, ("transcript_id", "group_id")):
        if not transcript or not group or transcript in groups:
            raise ValueError(f"Invalid or repeated group assignment for {transcript!r}")
        groups[transcript] = group
    return groups


def aggregate_with_hits(args, records, names, hit_table, report):
    ids = {record.id for record in records}
    if len(ids) != len(records) or "" in ids:
        raise ValueError("Hit-based aggregation requires unique, nonempty sequence IDs")
    hits = read_hits(hit_table, ids)
    groups = defaultdict(list)
    for record, name in zip(records, names, strict=True):
        groups[name].append(record)
    selections = [
        (name, *select_representative(group, hits)) for name, group in groups.items()
    ]
    if transcript_fasta := getattr(args, "orf_transcripts", ""):
        from cdskit.orf_containment import retain_nested_orfs

        retain_nested_orfs(selections, transcript_fasta, getattr(args, "codontable", 1))
    output_records = [selected for _, _, selected, _, _, _ in selections]
    outputs = [path for path in (args.outfile, report) if path not in ("", "-")]
    with atomic_output_paths(outputs) as paths:
        staged = dict(zip(outputs, paths, strict=True))
        if report:
            write_selection_report(staged[report], selections)
        write_seqs(
            output_records, staged.get(args.outfile, args.outfile), args.outseqformat
        )


def aggregate_main(args):
    expressions = list(getattr(args, "expression", []) or [])
    hit_table = getattr(args, "hit_table", "")
    group_table = getattr(args, "group_table", "")
    report = getattr(args, "selection_report", "")
    transcript_fasta = getattr(args, "orf_transcripts", "")
    if report and (not hit_table or report == "-"):
        raise ValueError("--selection_report requires a file path and --hit_table")
    if transcript_fasta and (not hit_table or transcript_fasta == "-"):
        raise ValueError("--orf_transcripts requires a FASTA file path and --hit_table")
    if transcript_fasta and not Path(transcript_fasta).is_file():
        raise ValueError(f"Source transcript FASTA does not exist: {transcript_fasta}")
    if hit_table or group_table:
        if hit_table and (not (expressions or group_table) or hit_table == "-"):
            raise ValueError(
                "--hit_table requires a file path and an aggregation expression or group table"
            )
        validate_distinct_paths(
            inputs=[args.seqfile, hit_table, group_table, transcript_fasta],
            outputs=[args.outfile, report],
        )
    if len(expressions) == 0:
        sys.stderr.write("Regular expressions for aggregating sequences: (none)\n")
    else:
        sys.stderr.write(
            "Regular expressions for aggregating sequences: "
            + " ".join(expressions)
            + "\n"
        )
    sys.stderr.write(
        "Criterion for aggregated sequences to retain: " + args.mode + "\n"
    )
    expressions = validate_aggregate_expressions(expressions)
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_seqtype(
        records=records,
        seqtype=getattr(args, "seqtype", "auto"),
        label="--seq_file",
    )
    if not (expressions or group_table):
        write_seqs(
            records=records, outfile=args.outfile, outseqformat=args.outseqformat
        )
        return
    threads = resolve_threads(getattr(args, "threads", 1))
    worker = partial(aggregate_name, expressions=expressions)
    aggregated_names = parallel_map_ordered(
        items=[record.id for record in records],
        worker=worker,
        threads=threads,
    )
    if group_table:
        groups = read_group_table(group_table)
        try:
            aggregated_names = [groups[name] for name in aggregated_names]
        except KeyError as error:
            raise ValueError(f"No group assignment for {error.args[0]!r}") from None
    if hit_table:
        aggregate_with_hits(args, records, aggregated_names, hit_table, report)
        return
    uniq: dict[str, SeqRecord] = {}
    for record, newname in zip(records, aggregated_names, strict=True):
        if newname in uniq:
            uniq[newname] = select_aggregate_record(
                existing_record=uniq[newname],
                candidate_record=record,
                mode=args.mode,
            )
        else:
            uniq[newname] = record
    out_records = list(uniq.values())
    write_seqs(
        records=out_records, outfile=args.outfile, outseqformat=args.outseqformat
    )
