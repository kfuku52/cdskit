#!/usr/bin/env python

import sys
import Bio.Seq
import Bio.SeqIO

def read_seqs(args):
    if args.seqfile=='-':
        parsed = sys.stdin
    else:
        parsed = args.seqfile
    records = list(Bio.SeqIO.parse(parsed, args.inseqformat))
    if not args.quiet:
        sys.stderr.write('number of input sequences: {}\n'.format(len(records)))
    return records

def write_seqs(records, args):
    if not args.quiet:
        sys.stderr.write('number of output sequences: {}\n'.format(len(records)))
    if args.outfile=='-':
        Bio.SeqIO.write(records, sys.stdout, args.outseqformat)
    else:
        Bio.SeqIO.write(records, args.outfile, args.inseqformat)


