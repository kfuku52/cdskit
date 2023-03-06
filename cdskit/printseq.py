import sys
import re
from cdskit.util import *

def printseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    for record in records:
        if re.fullmatch(args.seqname, record.name):
            seqtxt = str(record.seq)
            if (args.show_seqname):
                print('>'+record.name)
            print(seqtxt)
