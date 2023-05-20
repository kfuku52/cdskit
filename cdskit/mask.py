#!/usr/bin/env python

import re
import Bio.Seq
from cdskit.util import *

def mask_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(records)
    for record in records:
        nucseq = str(record.seq)
        nucseq_len = len(nucseq)
        flag1 = False
        for i in range(int(nucseq_len/3)):
            start = i * 3
            end = i * 3 + 3
            num_gap = nucseq[start:end].count('-')
            if num_gap!=0:
                if num_gap!=3:
                    flag1 = True
                    nucseq = nucseq[0:start] + args.maskchar*3 + nucseq[end:nucseq_len]
        if flag1:
            record.seq = Bio.Seq.Seq(nucseq)
        aaseq = record.seq.translate(table=args.codontable, to_stop=False, gap="-")
        flag2 = False
        if args.ambiguouscodon=='yes':
            for match in re.finditer("X+", str(aaseq)):
                flag2 = True
                num_X = match.end() - match.start()
                nucseq = nucseq[:match.start() * 3]+args.maskchar * num_X * 3+nucseq[match.end() * 3:]
        if args.stopcodon=='yes':
            for match in re.finditer("\*+", str(aaseq)):
                flag2 = True
                num_stop = match.end() - match.start()
                nucseq = nucseq[:match.start() * 3]+args.maskchar * num_stop * 3+nucseq[match.end() * 3:]
        if flag2:
            record.seq = Bio.Seq.Seq(nucseq)
    write_seqs(records, args)
