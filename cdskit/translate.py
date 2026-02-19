from functools import partial

from Bio.SeqRecord import SeqRecord

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_multiple_of_three,
    write_seqs,
)


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
    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        translate_record,
        codontable=args.codontable,
        to_stop=args.to_stop,
    )
    translated_records = parallel_map_ordered(items=records, worker=worker, threads=threads)
    write_seqs(records=translated_records, outfile=args.outfile, outseqformat=args.outseqformat)
