import os
import re
import sys

from cdskit.util import *

def parsegb_main(args):
    if not args.quiet:
        sys.stderr.write('cdskit parsegb: start\n')
    if (args.verbose)&(not args.quiet):
        sys.stderr.write(str(args)+'\n')
    records = read_seqs(seqfile=args.seqfile, seqformat='genbank', quiet=args.quiet)
    for i in range(len(records)):
        records[i].name = ''
        records[i].description = ''
        records[i].id = get_seqname(records[i], seqnamefmt=args.seqnamefmt)
        if args.extract_cds:
            records[i] = replace_seq2cds(records[i])
    records = [ record for record in records if record is not None ]
    write_seqs(records, args)
    if not args.quiet:
        sys.stderr.write('cdskit parsegb: end\n')