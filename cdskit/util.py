import sys
import Bio.Seq
import Bio.SeqIO
import numpy

def read_seqs(seqfile, seqformat, quiet):
    if seqfile=='-':
        parsed = sys.stdin
    else:
        parsed = seqfile
    records = list(Bio.SeqIO.parse(parsed, seqformat))
    if not quiet:
        sys.stderr.write('number of input sequences: {}\n'.format(len(records)))
    return records

def write_seqs(records, args):
    if not args.quiet:
        sys.stderr.write('number of output sequences: {}\n'.format(len(records)))
    if args.outfile=='-':
        Bio.SeqIO.write(records, sys.stdout, args.outseqformat)
    else:
        Bio.SeqIO.write(records, args.outfile, args.outseqformat)

def check_aligned(records):
    seqlens = [ len(seq.seq) for seq in records ]
    is_all_same_len = all([ seqlen==seqlens[0] for seqlen in seqlens ])
    assert is_all_same_len, 'Non-identical sequence lengths were detected. Check if the input sequence is aligned.'

def translate_records(records, codontable):
    pep_records = list()
    for record in records:
        aaseq = record.seq.translate(table=codontable, to_stop=False, gap="-")
        new_record = Bio.SeqRecord.SeqRecord(seq=aaseq, id=record.id)
        pep_records.append(new_record)
    return pep_records

def records2array(records):
    seqlen = max([ len(record.seq) for record in records ])
    seqnum = len(records)
    seqchars = list()
    for record in records:
        seqchars.append(list(record.seq))
    arr = numpy.array(seqchars)
    return arr
