import numpy

from cdskit.util import *

def intersection_main(args):
    original_records1 = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    original_records1_names = [ rec.name for rec in original_records1 ]
    if args.seqfile2 != '-':
        original_records2 = read_seqs(seqfile=args.seqfile2, seqformat=args.inseqformat2)
        original_records2_names = [ rec.name for rec in original_records2 ]
        intersection_names = list(set(original_records1_names) & set(original_records2_names))
        intersection_records1 = [ rec for rec in original_records1 if rec.name in intersection_names ]
        intersection_records2 = [ rec for rec in original_records2 if rec.name in intersection_names ]
        write_seqs(records=intersection_records1, outfile=args.outfile, outseqformat=args.outseqformat)
        write_seqs(records=intersection_records2, outfile=args.outfile2, outseqformat=args.outseqformat2)
    elif args.ingff != '-':
        original_gff = read_gff(gff_file=args.ingff)
        original_gff_names = [ row[0] for row in original_gff['data'] ]
        intersection_names = list(set(original_records1_names) & set(original_gff_names))
        intersection_records1 = [ rec for rec in original_records1 if rec.name in intersection_names ]
        intersection_gff = dict()
        intersection_gff['header'] = original_gff['header']
        intersection_gff['data'] = [ row for row in original_gff['data'] if row[0] in intersection_names ]
        write_seqs(records=intersection_records1, outfile=args.outfile, outseqformat=args.outseqformat)
        write_gff(gff=intersection_gff, outfile=args.outgff)
    else:
        raise Exception('Either seqfile2 or ingff should be provided.')