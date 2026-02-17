#!/usr/bin/env python

import re
import sys

from cdskit.util import read_seqs, write_seqs


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


def aggregate_main(args):
    sys.stderr.write('Regular expressions for aggregating sequences: '+' '.join(args.expression)+'\n')
    sys.stderr.write('Criterion for aggregated sequences to retain: '+args.mode+'\n')
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    uniq = {}
    for record in records:
        newname = aggregate_name(record.name, args.expression)
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
