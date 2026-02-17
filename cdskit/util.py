import Bio.Seq
import Bio.SeqIO
import numpy

import io
import re
import sys

GFF_DTYPE = [
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


def read_seqs(seqfile, seqformat):
    parsed = sys.stdin if seqfile == '-' else seqfile
    records = list(Bio.SeqIO.parse(parsed, seqformat))
    sys.stderr.write('Number of input sequences: {:,}\n'.format(len(records)))
    return records


def write_seqs(records, outfile, outseqformat):
    sys.stderr.write('Number of output sequences: {:,}\n'.format(len(records)))
    if outfile == '-':
        Bio.SeqIO.write(records, sys.stdout, outseqformat)
    else:
        Bio.SeqIO.write(records, outfile, outseqformat)


def stop_if_not_multiple_of_three(records):
    has_non_triplet_sequence = False
    for record in records:
        if len(record.seq) % 3 != 0:
            txt = 'Sequence length is not multiple of three: {}\n'.format(record.id)
            sys.stderr.write(txt)
            has_non_triplet_sequence = True
    if has_non_triplet_sequence:
        txt = 'Input sequence length should be multiple of three. ' \
              'Consider applying `cdskit pad` if the input is truncated coding sequences. Exiting.\n'
        raise Exception(txt)


def stop_if_not_aligned(records):
    if len(records) <= 1:
        return
    first_len = len(records[0].seq)
    for record in records[1:]:
        if len(record.seq) != first_len:
            txt = 'Sequence lengths were not identical. Please make sure input sequences are correctly aligned. Exiting.\n'
            raise Exception(txt)


def translate_records(records, codontable):
    return [
        Bio.SeqRecord.SeqRecord(
            seq=record.seq.translate(table=codontable, to_stop=False, gap="-"),
            id=record.id,
        )
        for record in records
    ]


def records2array(records):
    return numpy.array([list(record.seq) for record in records])


def read_item_per_line_file(file):
    with open(file, 'r') as f:
        return [line for line in f.read().split('\n') if line != '']


def get_seqname(record, seqnamefmt):
    name_items = seqnamefmt.split('_')
    seqname = ''
    for name_item in name_items:
        if name_item not in record.annotations:
            available_items = ', '.join(list(record.annotations.keys()))
            txt = 'Invalid --seqnamefmt element ({}) in {}. Available elements: {}'
            raise Exception(txt.format(name_item, record.id, available_items))

        try:
            new_name = record.annotations[name_item]
            if isinstance(new_name, list):
                new_name = new_name[0]
            seqname += '_' + new_name
        except Exception:
            available_items = ', '.join(list(record.annotations.keys()))
            txt = 'Invalid --seqnamefmt element ({}) in {}. Available elements: {}'
            raise Exception(txt.format(name_item, record.id, available_items))
    seqname = re.sub('^_', '', seqname)
    seqname = re.sub(' ', '_', seqname)
    return seqname


def replace_seq2cds(record):
    for feature in record.features:
        if feature.type == "CDS":
            record.seq = feature.location.extract(record).seq
            return record
    txt = 'Removed from output. No CDS found in: {}\n'
    sys.stderr.write(txt.format(record.id))
    return None


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
    data = numpy.genfromtxt(io.StringIO('\n'.join(data_lines)), dtype=GFF_DTYPE, delimiter='\t', autostrip=True)
    # Handle single record case: numpy.genfromtxt returns 0-d array for single line
    if data.ndim == 0:
        data = numpy.array([data])
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


def coordinates2ranges(gff_coordinates):
    ranges = []
    if len(gff_coordinates) == 0:
        return ranges
    start = gff_coordinates[0]
    end = gff_coordinates[0]
    for i in range(1, len(gff_coordinates)):
        if gff_coordinates[i] == end + 1:
            end = gff_coordinates[i]
        else:
            ranges.append((start, end))
            start = gff_coordinates[i]
            end = gff_coordinates[i]
    ranges.append((start, end))
    return ranges
