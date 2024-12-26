import copy
import sys
from cdskit.util import *

def split_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(records)
    first_records = copy.deepcopy(records)
    second_records = copy.deepcopy(records)
    third_records = copy.deepcopy(records)
    for i in range(len(records)):
        first_records[i].seq = records[i].seq[0::3]
        second_records[i].seq = records[i].seq[1::3]
        third_records[i].seq = records[i].seq[2::3]
    sys.stderr.write(f'Writing first codon positions.\n')
    write_seqs(records=first_records, outfile=f'1st_codon_positions.{args.outseqformat}', outseqformat=args.outseqformat)
    sys.stderr.write(f'Writing second codon positions.\n')
    write_seqs(records=second_records, outfile=f'2nd_codon_positions.{args.outseqformat}', outseqformat=args.outseqformat)
    sys.stderr.write(f'Writing third codon positions.\n')
    write_seqs(records=third_records, outfile=f'3rd_codon_positions.{args.outseqformat}', outseqformat=args.outseqformat)
