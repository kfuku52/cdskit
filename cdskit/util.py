import Bio.Seq
import Bio.SeqIO
import numpy

import io
import re
import sys

def read_seqs(seqfile, seqformat):
    if seqfile=='-':
        parsed = sys.stdin
    else:
        parsed = seqfile
    records = list(Bio.SeqIO.parse(parsed, seqformat))
    sys.stderr.write('Number of input sequences: {:,}\n'.format(len(records)))
    return records

def write_seqs(records, outfile, outseqformat):
    sys.stderr.write('Number of output sequences: {:,}\n'.format(len(records)))
    if outfile=='-':
        Bio.SeqIO.write(records, sys.stdout, outseqformat)
    else:
        Bio.SeqIO.write(records, outfile, outseqformat)

def stop_if_not_multiple_of_three(records):
    flag_stop = False
    for record in records:
        is_multiple_of_three = (len(record.seq)%3==0)
        if not is_multiple_of_three:
            txt = 'Sequence length is not multiple of three: {}\n'.format(record.id)
            sys.stderr.write(txt)
            flag_stop = True
    if flag_stop:
        txt = 'Input sequence length should be multiple of three. ' \
              'Consider applying `cdskit pad` if the input is truncated coding sequences. Exiting.\n'
        raise Exception(txt)

def stop_if_not_aligned(records):
    seqlens = [ len(seq.seq) for seq in records ]
    is_all_same_len = all([ seqlen==seqlens[0] for seqlen in seqlens ])
    if not is_all_same_len:
        txt = 'Sequence lengths were not identical. Please make sure input sequences are correctly aligned. Exiting.\n'
        raise Exception(txt)

def translate_records(records, codontable):
    pep_records = list()
    for record in records:
        aaseq = record.seq.translate(table=codontable, to_stop=False, gap="-")
        new_record = Bio.SeqRecord.SeqRecord(seq=aaseq, id=record.id)
        pep_records.append(new_record)
    return pep_records

def records2array(records):
    seqchars = list()
    for record in records:
        seqchars.append(list(record.seq))
    arr = numpy.array(seqchars)
    return arr

def read_item_per_line_file(file):
    with open(file, 'r') as f:
        out = f.read().split('\n')
    out = [ o for o in out if o!='' ]
    return out

def get_seqname(record, seqnamefmt):
    name_items = seqnamefmt.split('_')
    seqname = ''
    for name_item in name_items:
        try:
            new_name = record.annotations[name_item]
            if type(new_name) is list:
                new_name = new_name[0]
            seqname += '_'+new_name
        except:
            available_items = ', '.join(list(record.annotations.keys()))
            txt = 'Invalid --seqnamefmt element ({}) in {}. Available elements: {}'
            raise Exception(txt.format(name_item, record.id, available_items))
    seqname = re.sub('^_', '', seqname)
    seqname = re.sub(' ', '_', seqname)
    return seqname

def replace_seq2cds(record):
    flag_no_cds = True
    for feature in record.features:
        if feature.type=="CDS":
            seq = feature.location.extract(record).seq
            record.seq = seq
            flag_no_cds = False
            break
    if flag_no_cds:
        txt = 'Removed from output. No CDS found in: {}\n'
        sys.stderr.write(txt.format(record.id))
        return None
    return record

def read_gff(gff_file):
    header_lines = []
    data_lines = []
    with open(gff_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == '#':
                header_lines.append(line)
            else:
                data_lines.append(line)
    dtype = [
        ('seqid', 'U100'),
        ('source', 'U100'),
        ('type', 'U100'),
        ('start', 'i4'),
        ('end', 'i4'),
        ('score', 'U100'),
        ('strand', 'U10'),
        ('phase', 'U10'),
        ('attributes', 'U500')
    ]
    data = numpy.genfromtxt(io.StringIO('\n'.join(data_lines)), dtype=dtype, delimiter='\t', autostrip=True)
    sys.stderr.write('Number of input GFF header lines: {:,}\n'.format(len(header_lines)))
    sys.stderr.write('Number of input GFF records: {:,}\n'.format(len(data)))
    sys.stderr.write('Number of input GFF unique seqids: {:,}\n'.format(len(numpy.unique(data['seqid']))))
    return {'header': header_lines, 'data': data}

def write_gff(gff, outfile):
    sys.stderr.write('Number of output GFF header lines: {:,}\n'.format(len(gff['header'])))
    sys.stderr.write('Number of output GFF records: {:,}\n'.format(len(gff['data'])))
    sys.stderr.write('Number of output GFF unique seqids: {:,}\n'.format(len(numpy.unique(gff['data']['seqid']))))
    with open(outfile, 'w') as f:
        if gff['header']:
            f.write('\n'.join(gff['header']) + '\n')
        for row in gff['data']:
            f.write('\t'.join(map(str, row)) + '\n')
