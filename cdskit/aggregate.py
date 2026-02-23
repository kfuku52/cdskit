#!/usr/bin/env python

import re
import sys
from functools import partial

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, stop_if_not_seqtype, write_seqs


def aggregate_name(name, expressions):
    aggregated = name
    for expr in expressions:
        aggregated = re.sub(expr, '', aggregated)
    return aggregated


def select_aggregate_record(existing_record, candidate_record, mode):
    if mode == 'longest':
        if len(existing_record.seq) < len(candidate_record.seq):
            return candidate_record
        return existing_record
    sys.stderr.write('different modes to be supported in future.')
    return existing_record


def validate_aggregate_expressions(expressions):
    for expr in expressions:
        try:
            re.compile(expr)
        except re.error as e:
            txt = 'Invalid regex in --expression: {} ({})'
            raise Exception(txt.format(expr, e))


def aggregate_main(args):
    expressions = list(getattr(args, 'expression', []) or [])
    if len(expressions) == 0:
        sys.stderr.write('Regular expressions for aggregating sequences: (none)\n')
    else:
        sys.stderr.write('Regular expressions for aggregating sequences: ' + ' '.join(expressions) + '\n')
    sys.stderr.write('Criterion for aggregated sequences to retain: '+args.mode+'\n')
    validate_aggregate_expressions(expressions)
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_seqtype(
        records=records,
        seqtype=getattr(args, 'seqtype', 'auto'),
        label='--seqfile',
    )
    if len(expressions) == 0:
        write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(aggregate_name, expressions=expressions)
    aggregated_names = parallel_map_ordered(
        items=[record.id for record in records],
        worker=worker,
        threads=threads,
    )
    uniq = {}
    for record, newname in zip(records, aggregated_names):
        if newname in uniq.keys():
            uniq[newname] = select_aggregate_record(
                existing_record=uniq[newname],
                candidate_record=record,
                mode=args.mode,
            )
        else:
            uniq[newname] = record
    out_records = list(uniq.values())
    write_seqs(records=out_records, outfile=args.outfile, outseqformat=args.outseqformat)
