from cdskit.util import *

import sys

def num_masked_bp(seq):
    return sum(1 for bp in seq if bp.islower())

def stats_main(args):
    sys.stderr.write('cdskit stats: start\n')
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    num_seq = len(records)
    bp_masked = 0
    bp_all = 0
    for record in records:
        bp_masked += num_masked_bp(record.seq)
        bp_all += len(record.seq)

    print('Number of sequences: {:,}'.format(num_seq))
    print('Total length: {:,}'.format(bp_all))
    print('Total softmasked length: {:,}'.format(bp_masked))

    sys.stderr.write('cdskit stats: end\n')