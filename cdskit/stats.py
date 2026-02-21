from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, stop_if_not_dna

LOWERCASE_DELETE_TABLE = str.maketrans('', '', 'abcdefghijklmnopqrstuvwxyz')

def num_masked_bp(seq):
    seq_str = str(seq)
    return len(seq_str) - len(seq_str.translate(LOWERCASE_DELETE_TABLE))


def record_stats(record):
    seq_str = str(record.seq)
    seq_upper = seq_str.upper()
    return {
        'bp_masked': num_masked_bp(seq_str),
        'bp_all': len(seq_str),
        'bp_G': seq_upper.count('G'),
        'bp_C': seq_upper.count('C'),
        'bp_N': seq_upper.count('N'),
        'bp_gap': seq_str.count('-'),
    }


def stats_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seqfile')
    threads = resolve_threads(getattr(args, 'threads', 1))
    num_seq = len(records)
    record_count_stats = parallel_map_ordered(items=records, worker=record_stats, threads=threads)
    bp_masked = sum(x['bp_masked'] for x in record_count_stats)
    bp_all = sum(x['bp_all'] for x in record_count_stats)
    bp_G = sum(x['bp_G'] for x in record_count_stats)
    bp_C = sum(x['bp_C'] for x in record_count_stats)
    bp_N = sum(x['bp_N'] for x in record_count_stats)
    bp_gap = sum(x['bp_gap'] for x in record_count_stats)
    print('Number of sequences: {:,}'.format(num_seq))
    print('Total length: {:,}'.format(bp_all))
    print('Total softmasked length: {:,}'.format(bp_masked))
    print('Total N length: {:,}'.format(bp_N))
    print('Total gap (-) length: {:,}'.format(bp_gap))
    gc_content = 0.0
    if bp_all > 0:
        gc_content = ((bp_G + bp_C) / bp_all) * 100
    print('GC content: {:,.1f}%'.format(gc_content))
