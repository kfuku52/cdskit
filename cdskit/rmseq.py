import sys
import re
from cdskit.util import *

def rmseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    new_records = list()
    for record in records:
        flag = True
        if re.fullmatch(args.seqname, record.name):
            flag = False
        if args.problematic_percent>0:
            num_problematic_char = 0
            for pc in args.problematic_char:
                num_problematic_char += record.seq.count(pc)
            rate_problematic = num_problematic_char / len(record.seq)
            if rate_problematic >= (args.problematic_percent/100):
                flag = False
        if flag:
            new_records.append(record)
    write_seqs(new_records, args)
