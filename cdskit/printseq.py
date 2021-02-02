import sys
import re
from cdskit.util import *

def printseq_main(args):
    if not args.quiet:
        sys.stderr.write('cdskit printseq: start\n')
    if (args.verbose)&(not args.quiet):
        sys.stderr.write(str(args)+'\n')
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat, quiet=args.quiet)
    for record in records:
        if re.fullmatch(args.seqname, record.name):
            seqtxt = str(record.seq)
            if (args.show_seqname):
                print('>'+record.name)
            print(seqtxt)
    if not args.quiet:
        sys.stderr.write('cdskit printseq: end\n')