import numpy
import sys
from collections import Counter
from cdskit.util import *

def label_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if args.replace_chars != '':
        from_chars = list(args.replace_chars.split('--')[0])
        to_char = args.replace_chars.split('--')[1]
        to_chars = [to_char] * len(from_chars)
        replace_count = 0
        for i in range(len(records)):
            if any(c in records[i].id for c in from_chars):
                replace_count += 1
                records[i].id = records[i].id.translate(str.maketrans(''.join(from_chars), to_chars))
        sys.stderr.write('Number of character-replaced sequence labels: {:,}\n'.format(replace_count))
    if args.clip_len != 0:
        clip_count = 0
        for i in range(len(records)):
            if len(records[i].id) > args.clip_len:
                clip_count += 1
                records[i].id = records[i].id[:args.clip_len]
        sys.stderr.write('Number of clipped sequence labels: {:,}\n'.format(clip_count))
    if args.unique:
        nonunique_count = 0
        name_counts = Counter([record.id for record in records])
        suffix_counts = dict()
        for i in range(len(records)):
            if name_counts[records[i].id] > 1:
                nonunique_count += 1
                if not records[i].id in suffix_counts:
                    suffix_counts[records[i].id] = 1
                else:
                    suffix_counts[records[i].id] += 1
                records[i].id += '_{}'.format(suffix_counts[records[i].id])
                records[i].description = ''
        sys.stderr.write('Number of resolved non-unique sequence labels: {:,}\n'.format(nonunique_count))
        sys.stderr.write('Non-unique sequence labels:\n{}\n'.format('\n'.join([name for name in name_counts if name_counts[name] > 1])))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)