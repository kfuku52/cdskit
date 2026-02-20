import re
from functools import partial

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, write_seqs


def problematic_rate(seq, problematic_chars):
    num_problematic_char = 0
    for problematic_char in problematic_chars:
        num_problematic_char += seq.count(problematic_char)
    return num_problematic_char / len(seq)


def should_remove_record(record, seqname_pattern, problematic_percent, problematic_chars):
    if re.fullmatch(seqname_pattern, record.name):
        return True

    if problematic_percent > 0:
        rate_problematic = problematic_rate(record.seq, problematic_chars)
        if rate_problematic >= (problematic_percent / 100):
            return True

    return False


def should_keep_record(record, seqname_pattern, problematic_percent, problematic_chars):
    return not should_remove_record(
        record=record,
        seqname_pattern=seqname_pattern,
        problematic_percent=problematic_percent,
        problematic_chars=problematic_chars,
    )


def rmseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        should_keep_record,
        seqname_pattern=args.seqname,
        problematic_percent=args.problematic_percent,
        problematic_chars=args.problematic_char,
    )
    keep_flags = parallel_map_ordered(items=records, worker=worker, threads=threads)
    new_records = [record for record, keep in zip(records, keep_flags) if keep]
    write_seqs(records=new_records, outfile=args.outfile, outseqformat=args.outseqformat)
