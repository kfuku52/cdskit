from Bio.SeqRecord import SeqRecord

from cdskit.util import read_seqs, stop_if_not_multiple_of_three, write_seqs


def translate_record(record, codontable, to_stop):
    translated = record.seq.translate(table=codontable, to_stop=to_stop, gap='-')
    return SeqRecord(
        seq=translated,
        id=record.id,
        name=record.name,
        description=record.description,
    )


def translate_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if len(records) == 0:
        write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    stop_if_not_multiple_of_three(records=records)
    translated_records = [
        translate_record(record=record, codontable=args.codontable, to_stop=args.to_stop)
        for record in records
    ]
    write_seqs(records=translated_records, outfile=args.outfile, outseqformat=args.outseqformat)
