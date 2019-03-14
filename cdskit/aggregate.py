#!/usr/bin/env python

import re
from cdskit.util import *


def aggregate_main(args):
    if not args.quiet:
        sys.stderr.write('cdskit aggregate: start\n')
    if (args.expression[0]=='-')&(len(args.expression)==1):
        args.expression = [':.*', '\..*', '_i.*']
    if not args.quiet:
        sys.stderr.write('regular expressions to aggregate sequences: '+' '.join(args.expression)+'\n')
        sys.stderr.write('criterion for aggregated sequences to retain: '+args.mode+'\n')
    records = read_seqs(args)
    uniq = {}
    for record in records:
        newname = record.name
        for expr in args.expression:
            newname = re.sub(expr, '', newname)
        if newname in uniq.keys():
            if args.mode=='longest':
                if len(uniq[newname].seq) < len(record.seq):
                    uniq[newname] = record
            else:
                sys.stderr.write('different modes to be supported in future.')
        else:
            uniq[newname] = record
    out_records = list(uniq.values())
    write_seqs(out_records, args)
    if not args.quiet:
        sys.stderr.write('cdskit aggregate: end\n')
