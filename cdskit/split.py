import copy
import sys
from cdskit.util import *

def resolve_output_prefix(args):
    if args.prefix != 'INFILE':
        return args.prefix

    outfile_prefix = getattr(args, 'outfile', '-')
    if outfile_prefix not in ('-', '', None):
        return outfile_prefix

    return args.seqfile


def split_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(records)
    first_records = []
    second_records = []
    third_records = []
    for record in records:
        seq = record.seq
        first_record = copy.copy(record)
        second_record = copy.copy(record)
        third_record = copy.copy(record)
        first_record.seq = seq[0::3]
        second_record.seq = seq[1::3]
        third_record.seq = seq[2::3]
        first_records.append(first_record)
        second_records.append(second_record)
        third_records.append(third_record)
    prefix_str = resolve_output_prefix(args)
    sys.stderr.write(f'Writing first codon positions.\n')
    write_seqs(records=first_records, outfile=f'{prefix_str}_1st_codon_positions.{args.outseqformat}', outseqformat=args.outseqformat)
    sys.stderr.write(f'Writing second codon positions.\n')
    write_seqs(records=second_records, outfile=f'{prefix_str}_2nd_codon_positions.{args.outseqformat}', outseqformat=args.outseqformat)
    sys.stderr.write(f'Writing third codon positions.\n')
    write_seqs(records=third_records, outfile=f'{prefix_str}_3rd_codon_positions.{args.outseqformat}', outseqformat=args.outseqformat)
