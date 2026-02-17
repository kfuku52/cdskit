from cdskit.util import read_seqs

LOWERCASE_DELETE_TABLE = str.maketrans('', '', 'abcdefghijklmnopqrstuvwxyz')

def num_masked_bp(seq):
    seq_str = str(seq)
    return len(seq_str) - len(seq_str.translate(LOWERCASE_DELETE_TABLE))

def stats_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    num_seq = len(records)
    bp_masked = 0
    bp_all = 0
    bp_G = 0
    bp_C = 0
    bp_N = 0
    bp_gap = 0
    for record in records:
        seq_str = str(record.seq)
        bp_masked += num_masked_bp(seq_str)
        bp_all += len(seq_str)
        bp_G += seq_str.count('G')
        bp_C += seq_str.count('C')
        bp_N += seq_str.count('N')
        bp_gap += seq_str.count('-')
    print('Number of sequences: {:,}'.format(num_seq))
    print('Total length: {:,}'.format(bp_all))
    print('Total softmasked length: {:,}'.format(bp_masked))
    print('Total N length: {:,}'.format(bp_N))
    print('Total gap (-) length: {:,}'.format(bp_gap))
    gc_content = ((bp_G + bp_C) / bp_all) * 100
    print('GC content: {:,.1f}%'.format(gc_content))
