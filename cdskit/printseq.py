import re

from cdskit.util import read_seqs


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
    for record in records:
        if record_matches_seqname(record, args.seqname):
            for line in format_printseq_lines(record, args.show_seqname):
                print(line)
