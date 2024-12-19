import numpy

import copy

from cdskit.util import *

def hammer_main(args):
    original_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(original_records)
    stop_if_not_aligned(original_records)
    if args.nail=='all':
        nail_value = len(original_records)
        txt = '--nail all was specified. Set to {:,}\n'
        sys.stderr.write(txt.format(nail_value))
    elif (int(args.nail)>len(original_records)):
        nail_value = len(original_records)
        txt = '--nail ({:,}) is greater than the number of input sequences ({:,}). Decreased to {:,}\n'
        sys.stderr.write(txt.format(int(args.nail), len(original_records), nail_value))
    else:
        nail_value = int(args.nail)
    for current_nail in numpy.flip(numpy.arange(nail_value)+1):
        records = copy.deepcopy(original_records)
        max_len = max([len(r.seq) for r in records]) // 3
        missing_site = numpy.zeros(shape=[max_len, ], dtype=int)
        for record in records:
            aaseq = str(record.seq.translate(table=args.codontable, to_stop=False, gap="-"))
            for i in numpy.arange(len(aaseq)):
                if aaseq[i] in ['-','?','X','*']:
                    missing_site[i] += 1
            if len(aaseq)<missing_site.shape[0]:
                missing_site[len(aaseq):] += 1
        non_missing_site = len(records) - missing_site
        non_missing_idx = numpy.argwhere(non_missing_site>=current_nail)
        non_missing_idx = numpy.reshape(non_missing_idx, newshape=[non_missing_idx.shape[0],])
        num_removed_site = max_len - non_missing_idx.shape[0]
        sys.stderr.write('{:,} out of {:,} codon sites will be removed.\n'.format(num_removed_site, max_len))
        for record in records:
            seq = str(record.seq)
            new_seq = ''.join([ seq[nmi*3:nmi*3+3] for nmi in non_missing_idx ])
            record.seq = Bio.Seq.Seq(new_seq)
        if args.prevent_gap_only:
            flag_gap_only = False
            for record in records:
                seq_len = len(record.seq)
                num_hyphen = record.seq.count('-')
                num_N = record.seq.count('N')
                num_X = record.seq.count('X')
                num_gap = num_hyphen + num_N + num_X
                if seq_len==num_gap:
                    flag_gap_only = True
                    txt = 'A gap-only sequence was generated with --nail {}. Will try --nail {}: {}\n'
                    sys.stderr.write(txt.format(current_nail, current_nail-1, record.name))
            if flag_gap_only:
                continue
        break
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
