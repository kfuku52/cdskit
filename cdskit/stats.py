from cdskit.util import *

import sys

def num_masked_bp(seq):
    return sum(1 for bp in seq if bp.islower())

def stats_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    num_seq = len(records)
    bp_masked = 0
    bp_all = 0
    bp_A = 0
    bp_T = 0
    bp_G = 0
    bp_C = 0
    bp_N = 0
    bp_gap = 0
    for record in records:
        bp_masked += num_masked_bp(record.seq)
        bp_all += len(record.seq)
        bp_A += record.seq.count('A')
        bp_T += record.seq.count('T')
        bp_G += record.seq.count('G')
        bp_C += record.seq.count('C')
        bp_N += record.seq.count('N')
        bp_gap += record.seq.count('-')
    print('Number of sequences: {:,}'.format(num_seq))
    print('Total length: {:,}'.format(bp_all))
    print('Total softmasked length: {:,}'.format(bp_masked))
    print('Total N length: {:,}'.format(bp_N))
    print('Total gap (-) length: {:,}'.format(bp_gap))
    gc_content = ((bp_G + bp_C) / (bp_A + bp_T)) * 100
    print('GC content: {:,.1f}%'.format(gc_content))
