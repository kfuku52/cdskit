import Bio.Seq
import Bio.SeqRecord
import sys

from cdskit.util import read_seqs
from cdskit.util import stop_if_not_aligned
from cdskit.util import write_seqs


AA_GAP_CHARS = {'-', '.'}
AA_WILDCARD_CHARS = {'X', '?'}
CDS_GAP_CHARS = {'-', '.'}


def remove_gap_chars(seq, gap_chars):
    out = str(seq)
    for gap_char in gap_chars:
        out = out.replace(gap_char, '')
    return out


def stop_if_not_multiple_of_three_after_gap_removal(records):
    flag_stop = False
    for record in records:
        seq = remove_gap_chars(record.seq, CDS_GAP_CHARS)
        is_multiple_of_three = (len(seq) % 3 == 0)
        if not is_multiple_of_three:
            txt = 'Sequence length is not multiple of three after removing gaps: {}\n'
            sys.stderr.write(txt.format(record.id))
            flag_stop = True
    if flag_stop:
        txt = 'Input CDS length should be multiple of three after removing gaps. Exiting.\n'
        raise Exception(txt)


def get_record_map(records, label):
    record_map = dict()
    for record in records:
        if record.id in record_map:
            txt = 'Sequence IDs must be unique in {}. Duplicated ID: {}'
            raise Exception(txt.format(label, record.id))
        record_map[record.id] = record
    return record_map


def stop_if_sequence_ids_do_not_match(cdn_records, pep_records):
    cdn_ids = set([record.id for record in cdn_records])
    pep_ids = set([record.id for record in pep_records])
    if cdn_ids == pep_ids:
        return
    missing_in_cds = sorted(list(pep_ids - cdn_ids))
    missing_in_aa = sorted(list(cdn_ids - pep_ids))
    txt = 'Sequence IDs did not match between CDS (--seqfile) and amino acid alignment (--aa_aln).'
    if len(missing_in_cds) > 0:
        txt += ' Missing in CDS: {}.'.format(','.join(missing_in_cds))
    if len(missing_in_aa) > 0:
        txt += ' Missing in amino acid alignment: {}.'.format(','.join(missing_in_aa))
    raise Exception(txt)


def split_codons(seq):
    codons = list()
    for i in range(0, len(seq), 3):
        codons.append(seq[i:i + 3])
    return codons


def translate_codons(codons, codontable):
    translated = list()
    for codon in codons:
        aa = str(Bio.Seq.Seq(codon).translate(table=codontable, to_stop=False, gap='-'))
        translated.append(aa)
    return translated


def translate_cds_seq(cdn_seq, codontable):
    return str(Bio.Seq.Seq(cdn_seq).translate(table=codontable, to_stop=False, gap='-'))


def amino_acid_matches(aa_aln_char, translated_char):
    aa = aa_aln_char.upper()
    tr = translated_char.upper()
    if aa in AA_WILDCARD_CHARS:
        return True
    return aa == tr


def backalign_record(cdn_record, pep_record, codontable):
    cdn_seq = remove_gap_chars(cdn_record.seq, CDS_GAP_CHARS)
    num_codons = len(cdn_seq) // 3
    translated = translate_cds_seq(cdn_seq=cdn_seq, codontable=codontable)
    pep_seq = str(pep_record.seq)
    pep_seq_upper = pep_seq.upper()
    aligned_codons = [''] * len(pep_seq)
    codon_index = 0

    for i, pep_char_upper in enumerate(pep_seq_upper):
        pep_char = pep_seq[i]
        if pep_char in AA_GAP_CHARS:
            aligned_codons[i] = '---'
            continue

        if codon_index >= num_codons:
            txt = 'Protein alignment had too many non-gap sites for {} at amino acid position {}.'
            raise Exception(txt.format(pep_record.id, i + 1))

        translated_char = translated[codon_index]
        codon_start = codon_index * 3
        codon = cdn_seq[codon_start:codon_start+3]
        if (pep_char_upper not in AA_WILDCARD_CHARS) and (pep_char_upper != translated_char):
            txt = 'Amino acid mismatch for {} at aligned position {}: aa_aln={}, translated={}, codon={}'
            raise Exception(txt.format(pep_record.id, i + 1, pep_char, translated_char, codon))

        aligned_codons[i] = codon
        codon_index += 1

    remaining_codons = num_codons - codon_index
    if remaining_codons == 1:
        is_terminal_stop = (len(translated) > 0) and (translated[-1] == '*')
        if is_terminal_stop:
            txt = 'Ignored terminal stop codon not present in amino acid alignment: {}\n'
            sys.stderr.write(txt.format(pep_record.id))
        else:
            txt = 'Unmatched codon remained for {}. The amino acid alignment may be truncated.'
            raise Exception(txt.format(pep_record.id))
    elif remaining_codons != 0:
        txt = '{} codons remained unmatched for {}. The amino acid alignment may be truncated.'
        raise Exception(txt.format(remaining_codons, pep_record.id))

    aligned_seq = ''.join(aligned_codons)
    out_record = Bio.SeqRecord.SeqRecord(
        seq=Bio.Seq.Seq(aligned_seq),
        id=pep_record.id,
        name='',
        description='',
    )
    return out_record


def backalign_main(args):
    cdn_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    pep_records = read_seqs(seqfile=args.aa_aln, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three_after_gap_removal(cdn_records)
    stop_if_not_aligned(records=pep_records)
    stop_if_sequence_ids_do_not_match(cdn_records=cdn_records, pep_records=pep_records)

    _ = get_record_map(cdn_records, '--seqfile')
    pep_record_map = get_record_map(pep_records, '--aa_aln')

    backaligned_records = list()
    for cdn_record in cdn_records:
        pep_record = pep_record_map[cdn_record.id]
        backaligned_record = backalign_record(
            cdn_record=cdn_record,
            pep_record=pep_record,
            codontable=args.codontable,
        )
        backaligned_records.append(backaligned_record)

    stop_if_not_aligned(records=backaligned_records)
    txt = 'Number of aligned nucleotide sites in output codon alignment: {}\n'
    sys.stderr.write(txt.format(len(backaligned_records[0].seq)))
    write_seqs(records=backaligned_records, outfile=args.outfile, outseqformat=args.outseqformat)
