import re

from cdskit.util import read_seqs, write_seqs


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


def rmseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    new_records = []
    for record in records:
        if not should_remove_record(
            record=record,
            seqname_pattern=args.seqname,
            problematic_percent=args.problematic_percent,
            problematic_chars=args.problematic_char,
        ):
            new_records.append(record)
    write_seqs(records=new_records, outfile=args.outfile, outseqformat=args.outseqformat)
