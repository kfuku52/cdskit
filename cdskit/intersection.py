import numpy as np
import sys
from collections import Counter

from cdskit.atomicio import atomic_output_paths
from cdskit.util import (
    read_gff,
    read_seqs,
    resolve_threads,
    stop_if_not_seqtype,
    write_gff,
    write_seqs,
)


def stop_if_duplicate_sequence_ids(records, label="--seq_file"):
    counts = Counter(record.id for record in records)
    duplicated = sorted([seq_id for seq_id, count in counts.items() if count > 1])
    if len(duplicated) == 0:
        return
    shown = ",".join(duplicated[:10])
    if len(duplicated) > 10:
        shown += ",..."
    txt = (
        "Duplicate sequence IDs are not supported when intersecting with GFF "
        "because seqid mapping becomes ambiguous in {}. Duplicate IDs: {}. Exiting.\n"
    )
    raise ValueError(txt.format(label, shown))


def filter_records_by_names(records, names, threads=1):
    del threads
    return [record for record in records if record.id in names]


def fix_out_of_range_gff_records(filtered_data, seqid_to_seq_len):
    seq_lengths = np.array(
        [seqid_to_seq_len[s] for s in filtered_data["seqid"]], dtype=int
    )
    is_gff_entry_start_in_range = filtered_data["start"] <= seq_lengths
    if np.any(~is_gff_entry_start_in_range):
        sys.stderr.write(
            "Number of fixed out-of-range GFF record start coordinates: {:,}\n".format(
                np.sum(~is_gff_entry_start_in_range)
            )
        )
        starts = filtered_data["start"]
        starts[~is_gff_entry_start_in_range] = seq_lengths[~is_gff_entry_start_in_range]
        filtered_data["start"] = starts

    is_gff_entry_end_in_range = filtered_data["end"] <= seq_lengths
    if np.any(~is_gff_entry_end_in_range):
        sys.stderr.write(
            "Number of fixed out-of-range GFF record end coordinates: {:,}\n".format(
                np.sum(~is_gff_entry_end_in_range)
            )
        )
        ends = filtered_data["end"]
        ends[~is_gff_entry_end_in_range] = seq_lengths[~is_gff_entry_end_in_range]
        filtered_data["end"] = ends

    is_gff_entry_start_greater_than_zero = filtered_data["start"] > 0
    if np.any(~is_gff_entry_start_greater_than_zero):
        sys.stderr.write(
            "Number of fixed GFF record start coordinates less than 1: {:,}\n".format(
                np.sum(~is_gff_entry_start_greater_than_zero)
            )
        )
        starts = filtered_data["start"]
        starts[~is_gff_entry_start_greater_than_zero] = 1
        filtered_data["start"] = starts

    is_gff_entry_end_greater_than_zero = filtered_data["end"] > 0
    if np.any(~is_gff_entry_end_greater_than_zero):
        sys.stderr.write(
            "Number of fixed GFF record end coordinates less than 1: {:,}\n".format(
                np.sum(~is_gff_entry_end_greater_than_zero)
            )
        )
        ends = filtered_data["end"]
        ends[~is_gff_entry_end_greater_than_zero] = 1
        filtered_data["end"] = ends

    is_gff_entry_invalid_range = filtered_data["start"] > filtered_data["end"]
    if np.any(is_gff_entry_invalid_range):
        sys.stderr.write(
            "Number of removed GFF records that had start > end coordinates: {:,}\n".format(
                np.sum(is_gff_entry_invalid_range)
            )
        )
        filtered_data = filtered_data[~is_gff_entry_invalid_range]
    return filtered_data


def intersect_two_fasta_inputs(original_records1, args, threads=1):
    original_records2 = read_seqs(seqfile=args.seqfile2, seqformat=args.inseqformat2)
    stop_if_not_seqtype(
        records=original_records2,
        seqtype=getattr(args, "seqtype", "auto"),
        label="--seq_file_2",
    )
    original_records1_names = [rec.id for rec in original_records1]
    original_records2_names = [rec.id for rec in original_records2]
    intersection_names = set(original_records1_names) & set(original_records2_names)
    intersection_records1 = filter_records_by_names(
        original_records1, intersection_names, threads=threads
    )
    intersection_records2 = filter_records_by_names(
        original_records2, intersection_names, threads=threads
    )
    output_paths = [path for path in (args.outfile, args.outfile2) if path != "-"]
    with atomic_output_paths(output_paths) as staged_paths:
        staged = dict(zip(output_paths, staged_paths, strict=False))
        write_seqs(
            records=intersection_records1,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        write_seqs(
            records=intersection_records2,
            outfile=staged.get(args.outfile2, args.outfile2),
            outseqformat=args.outseqformat2,
        )


def intersect_fasta_with_gff(original_records1, args, threads=1):
    stop_if_duplicate_sequence_ids(records=original_records1, label="--seq_file")
    original_records1_names = [rec.id for rec in original_records1]
    original_gff = read_gff(gff_file=args.ingff)
    original_gff_names = np.unique(original_gff["data"]["seqid"])
    intersection_names = set(original_records1_names) & set(original_gff_names)
    intersection_records1 = filter_records_by_names(
        original_records1, intersection_names, threads=threads
    )
    mask = np.isin(original_gff["data"]["seqid"], list(intersection_names))
    filtered_data = original_gff["data"][mask]

    if args.fix_outrange_gff_records:
        seqid_to_seq_len = {rec.id: len(rec.seq) for rec in intersection_records1}
        filtered_data = fix_out_of_range_gff_records(filtered_data, seqid_to_seq_len)

    intersection_gff = {"header": original_gff["header"], "data": filtered_data}
    output_paths = [path for path in (args.outfile, args.outgff) if path != "-"]
    with atomic_output_paths(output_paths) as staged_paths:
        staged = dict(zip(output_paths, staged_paths, strict=False))
        write_seqs(
            records=intersection_records1,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        write_gff(
            gff=intersection_gff,
            outfile=staged.get(args.outgff, args.outgff),
        )


def intersection_main(args):
    original_records1 = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_seqtype(
        records=original_records1,
        seqtype=getattr(args, "seqtype", "auto"),
        label="--seq_file",
    )
    threads = resolve_threads(getattr(args, "threads", 1))
    if (args.seqfile2 is not None) and (args.ingff is not None):
        raise ValueError("Specify either --seq_file_2 or --in_gff, but not both.")
    if args.seqfile2 is not None:
        intersect_two_fasta_inputs(original_records1, args, threads=threads)
    elif args.ingff is not None:
        intersect_fasta_with_gff(original_records1, args, threads=threads)
    else:
        raise ValueError("Either --seq_file_2 or --in_gff should be provided.")
