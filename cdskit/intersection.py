import numpy
import sys

from cdskit.util import read_gff, read_seqs, write_gff, write_seqs


def filter_records_by_names(records, names):
    return [rec for rec in records if rec.name in names]


def fix_out_of_range_gff_records(filtered_data, seqid_to_seq_len):
    seq_lengths = numpy.array([seqid_to_seq_len[s] for s in filtered_data['seqid']], dtype=int)
    is_gff_entry_start_in_range = (filtered_data['start'] <= seq_lengths)
    if numpy.any(~is_gff_entry_start_in_range):
        sys.stderr.write('Number of fixed out-of-range GFF record start coordinates: {:,}\n'.format(numpy.sum(~is_gff_entry_start_in_range)))
        starts = filtered_data['start']
        starts[~is_gff_entry_start_in_range] = seq_lengths[~is_gff_entry_start_in_range]
        filtered_data['start'] = starts

    is_gff_entry_end_in_range = (filtered_data['end'] <= seq_lengths)
    if numpy.any(~is_gff_entry_end_in_range):
        sys.stderr.write('Number of fixed out-of-range GFF record end coordinates: {:,}\n'.format(numpy.sum(~is_gff_entry_end_in_range)))
        ends = filtered_data['end']
        ends[~is_gff_entry_end_in_range] = seq_lengths[~is_gff_entry_end_in_range]
        filtered_data['end'] = ends

    is_gff_entry_start_greater_than_zero = (filtered_data['start'] > 0)
    if numpy.any(~is_gff_entry_start_greater_than_zero):
        sys.stderr.write('Number of fixed GFF record start coordinates less than 1: {:,}\n'.format(numpy.sum(~is_gff_entry_start_greater_than_zero)))
        starts = filtered_data['start']
        starts[~is_gff_entry_start_greater_than_zero] = 1
        filtered_data['start'] = starts

    is_gff_entry_end_greater_than_zero = (filtered_data['end'] > 0)
    if numpy.any(~is_gff_entry_end_greater_than_zero):
        sys.stderr.write('Number of fixed GFF record end coordinates less than 1: {:,}\n'.format(numpy.sum(~is_gff_entry_end_greater_than_zero)))
        ends = filtered_data['end']
        ends[~is_gff_entry_end_greater_than_zero] = 1
        filtered_data['end'] = ends

    is_gff_start_end_same_coordinate = (filtered_data['start'] == filtered_data['end'])
    if numpy.any(is_gff_start_end_same_coordinate):
        sys.stderr.write('Number of removed GFF records that had the same start & end coordinates: {:,}\n'.format(numpy.sum(is_gff_start_end_same_coordinate)))
        filtered_data = filtered_data[~is_gff_start_end_same_coordinate]
    return filtered_data


def intersect_two_fasta_inputs(original_records1, args):
    original_records2 = read_seqs(seqfile=args.seqfile2, seqformat=args.inseqformat2)
    original_records1_names = [rec.name for rec in original_records1]
    original_records2_names = [rec.name for rec in original_records2]
    intersection_names = set(original_records1_names) & set(original_records2_names)
    intersection_records1 = filter_records_by_names(original_records1, intersection_names)
    intersection_records2 = filter_records_by_names(original_records2, intersection_names)
    write_seqs(records=intersection_records1, outfile=args.outfile, outseqformat=args.outseqformat)
    write_seqs(records=intersection_records2, outfile=args.outfile2, outseqformat=args.outseqformat2)


def intersect_fasta_with_gff(original_records1, args):
    original_records1_names = [rec.name for rec in original_records1]
    original_gff = read_gff(gff_file=args.ingff)
    original_gff_names = numpy.unique(original_gff['data']['seqid'])
    intersection_names = set(original_records1_names) & set(original_gff_names)
    intersection_records1 = filter_records_by_names(original_records1, intersection_names)
    mask = numpy.isin(original_gff['data']['seqid'], list(intersection_names))
    filtered_data = original_gff['data'][mask]

    if args.fix_outrange_gff_records:
        seqid_to_seq_len = {rec.name: len(rec.seq) for rec in intersection_records1}
        filtered_data = fix_out_of_range_gff_records(filtered_data, seqid_to_seq_len)

    intersection_gff = {'header': original_gff['header'], 'data': filtered_data}
    write_seqs(records=intersection_records1, outfile=args.outfile, outseqformat=args.outseqformat)
    write_gff(gff=intersection_gff, outfile=args.outgff)


def intersection_main(args):
    original_records1 = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if args.seqfile2 is not None:
        intersect_two_fasta_inputs(original_records1, args)
    elif args.ingff is not None:
        intersect_fasta_with_gff(original_records1, args)
    else:
        raise Exception('Either seqfile2 or ingff should be provided.')
