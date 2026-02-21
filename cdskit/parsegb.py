import sys
from functools import partial

from cdskit.util import (
    get_seqname,
    parallel_map_ordered,
    read_seqs,
    replace_seq2cds,
    resolve_threads,
    stop_if_not_dna,
    write_seqs,
)


def parsegb_record(record, seqnamefmt, extract_cds=False, list_seqname_keys=False):
    if list_seqname_keys:
        sys.stderr.write(str(record.annotations) + '\n')

    record.name = ''
    record.description = ''
    record.id = get_seqname(record, seqnamefmt=seqnamefmt)
    if extract_cds:
        return replace_seq2cds(record)
    return record


def parsegb_main(args):
    if str(args.inseqformat).lower() not in ('genbank', 'gb'):
        txt = 'parsegb requires --inseqformat genbank (or gb), but got "{}". Exiting.\n'
        raise Exception(txt.format(args.inseqformat))
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seqfile')
    threads = resolve_threads(getattr(args, 'threads', 1))
    if args.list_seqname_keys:
        # Keep deterministic key listing order in stderr output.
        threads = 1
    worker = partial(
        parsegb_record,
        seqnamefmt=args.seqnamefmt,
        extract_cds=args.extract_cds,
        list_seqname_keys=args.list_seqname_keys,
    )
    records = parallel_map_ordered(items=records, worker=worker, threads=threads)
    records = [record for record in records if record is not None]
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
