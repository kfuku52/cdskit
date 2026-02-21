import numpy
import sys
from functools import partial

from cdskit.util import (
    parallel_map_ordered,
    read_gff,
    read_seqs,
    resolve_threads,
    stop_if_not_dna,
    write_gff,
    write_seqs,
)


_INTERSECTION_PARALLEL_MIN_RECORDS = 5000


def filter_record_chunk_by_names(record_chunk, names):
    return [record for record in record_chunk if record.id in names]


def filter_records_by_names(records, names, threads=1):
    if (threads <= 1) or (len(records) < _INTERSECTION_PARALLEL_MIN_RECORDS):
        return [record for record in records if record.id in names]

    max_workers = min(threads, len(records))
    chunk_size = max(1, (len(records) + max_workers - 1) // max_workers)
    record_chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
    worker = partial(filter_record_chunk_by_names, names=names)
    chunk_results = parallel_map_ordered(items=record_chunks, worker=worker, threads=max_workers)
    filtered_records = list()
    for chunk in chunk_results:
        filtered_records.extend(chunk)
    return filtered_records


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

    is_gff_entry_invalid_range = (filtered_data['start'] > filtered_data['end'])
    if numpy.any(is_gff_entry_invalid_range):
        sys.stderr.write(
            'Number of removed GFF records that had start > end coordinates: {:,}\n'.format(
                numpy.sum(is_gff_entry_invalid_range)
            )
        )
        filtered_data = filtered_data[~is_gff_entry_invalid_range]
    return filtered_data


def intersect_two_fasta_inputs(original_records1, args, threads=1):
    original_records2 = read_seqs(seqfile=args.seqfile2, seqformat=args.inseqformat2)
    stop_if_not_dna(records=original_records2, label='--seqfile2')
    original_records1_names = [rec.id for rec in original_records1]
    original_records2_names = [rec.id for rec in original_records2]
    intersection_names = set(original_records1_names) & set(original_records2_names)
    intersection_records1 = filter_records_by_names(original_records1, intersection_names, threads=threads)
    intersection_records2 = filter_records_by_names(original_records2, intersection_names, threads=threads)
    write_seqs(records=intersection_records1, outfile=args.outfile, outseqformat=args.outseqformat)
    write_seqs(records=intersection_records2, outfile=args.outfile2, outseqformat=args.outseqformat2)


def intersect_fasta_with_gff(original_records1, args, threads=1):
    original_records1_names = [rec.id for rec in original_records1]
    original_gff = read_gff(gff_file=args.ingff)
    original_gff_names = numpy.unique(original_gff['data']['seqid'])
    intersection_names = set(original_records1_names) & set(original_gff_names)
    intersection_records1 = filter_records_by_names(original_records1, intersection_names, threads=threads)
    mask = numpy.isin(original_gff['data']['seqid'], list(intersection_names))
    filtered_data = original_gff['data'][mask]

    if args.fix_outrange_gff_records:
        seqid_to_seq_len = {rec.id: len(rec.seq) for rec in intersection_records1}
        filtered_data = fix_out_of_range_gff_records(filtered_data, seqid_to_seq_len)

    intersection_gff = {'header': original_gff['header'], 'data': filtered_data}
    write_seqs(records=intersection_records1, outfile=args.outfile, outseqformat=args.outseqformat)
    write_gff(gff=intersection_gff, outfile=args.outgff)


def intersection_main(args):
    original_records1 = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=original_records1, label='--seqfile')
    threads = resolve_threads(getattr(args, 'threads', 1))
    if (args.seqfile2 is not None) and (args.ingff is not None):
        raise Exception('Specify either --seqfile2 or --ingff, but not both.')
    if args.seqfile2 is not None:
        intersect_two_fasta_inputs(original_records1, args, threads=threads)
    elif args.ingff is not None:
        intersect_fasta_with_gff(original_records1, args, threads=threads)
    else:
        raise Exception('Either seqfile2 or ingff should be provided.')
