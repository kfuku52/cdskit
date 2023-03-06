import os
import re
import sys

from cdskit.util import *

def parsegb_main(args):
    sys.stderr.write('cdskit parsegb: start\n')
    records = read_seqs(seqfile=args.seqfile, seqformat='genbank')
    for i in range(len(records)):
        if (args.list_seqname_keys):
            sys.stderr.write(str(records[i].annotations)+'\n')
        records[i].name = ''
        records[i].description = ''
        records[i].id = get_seqname(records[i], seqnamefmt=args.seqnamefmt)
        if args.extract_cds:
            records[i] = replace_seq2cds(records[i])
    records = [ record for record in records if record is not None ]
    write_seqs(records, args)
    sys.stderr.write('cdskit parsegb: end\n')