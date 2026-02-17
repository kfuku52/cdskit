import sys

from cdskit.util import get_seqname, read_seqs, replace_seq2cds, write_seqs


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
    records = read_seqs(seqfile=args.seqfile, seqformat='genbank')
    records = [
        parsegb_record(
            record=record,
            seqnamefmt=args.seqnamefmt,
            extract_cds=args.extract_cds,
            list_seqname_keys=args.list_seqname_keys,
        )
        for record in records
    ]
    records = [record for record in records if record is not None]
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
