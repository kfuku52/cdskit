import copy
import sys

from cdskit.atomicio import atomic_output_paths, validate_distinct_paths
from cdskit.util import (
    read_seqs,
    replace_record_sequence,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)


def resolve_output_prefix(args):
    if args.prefix != "INFILE":
        return args.prefix

    outfile_prefix = getattr(args, "outfile", "-")
    if outfile_prefix not in ("-", "", None):
        return outfile_prefix

    if args.seqfile == "-":
        return "stdin"
    return args.seqfile


def split_record_by_codon_position(record):
    seq = record.seq
    first_record = copy.copy(record)
    second_record = copy.copy(record)
    third_record = copy.copy(record)
    replace_record_sequence(first_record, seq[0::3])
    replace_record_sequence(second_record, seq[1::3])
    replace_record_sequence(third_record, seq[2::3])
    return first_record, second_record, third_record


def build_split_output_paths(prefix, outseqformat):
    return (
        f"{prefix}_1st_codon_positions.{outseqformat}",
        f"{prefix}_2nd_codon_positions.{outseqformat}",
        f"{prefix}_3rd_codon_positions.{outseqformat}",
    )


def split_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_not_multiple_of_three(records)
    first_records = list()
    second_records = list()
    third_records = list()
    for record in records:
        first_record, second_record, third_record = split_record_by_codon_position(
            record
        )
        first_records.append(first_record)
        second_records.append(second_record)
        third_records.append(third_record)
    prefix_str = resolve_output_prefix(args)
    first_outfile, second_outfile, third_outfile = build_split_output_paths(
        prefix_str, args.outseqformat
    )
    outputs = (first_outfile, second_outfile, third_outfile)
    validate_distinct_paths(inputs=[args.seqfile], outputs=outputs)
    with atomic_output_paths(outputs) as staged:
        sys.stderr.write("Writing first codon positions.\n")
        write_seqs(
            records=first_records, outfile=staged[0], outseqformat=args.outseqformat
        )
        sys.stderr.write("Writing second codon positions.\n")
        write_seqs(
            records=second_records, outfile=staged[1], outseqformat=args.outseqformat
        )
        sys.stderr.write("Writing third codon positions.\n")
        write_seqs(
            records=third_records, outfile=staged[2], outseqformat=args.outseqformat
        )
