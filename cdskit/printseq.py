import re
from functools import partial

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads


def record_matches_seqname(record, seqname_pattern):
    return re.fullmatch(seqname_pattern, record.name) is not None


def format_printseq_lines(record, show_seqname):
    lines = []
    if show_seqname:
        lines.append('>' + record.name)
    lines.append(str(record.seq))
    return lines


def printseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    threads = resolve_threads(getattr(args, 'threads', 1))
    match_flags = parallel_map_ordered(
        items=records,
        worker=partial(record_matches_seqname, seqname_pattern=args.seqname),
        threads=threads,
    )
    for record, matched in zip(records, match_flags):
        if matched:
            for line in format_printseq_lines(record, args.show_seqname):
                print(line)
