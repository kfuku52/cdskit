from collections import defaultdict
from collections import deque

from cdskit.util import *

def check_same_seq_num(cdn_records, pep_records):
    err_txt = 'The numbers of seqs did not match: seqfile={} and trimmed_aa_aln={}'.format(len(cdn_records), len(pep_records))
    assert len(cdn_records)==len(pep_records), err_txt


def build_column_index(seq_strings):
    col_index = defaultdict(deque)
    if len(seq_strings) == 0:
        return col_index
    for ci, col_chars in enumerate(zip(*seq_strings)):
        key = ''.join(col_chars)
        col_index[key].append(ci)
    return col_index


def backtrim_main(args):
    cdn_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    pep_records = read_seqs(seqfile=args.trimmed_aa_aln, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(cdn_records)
    check_same_seq_num(cdn_records, pep_records)
    stop_if_not_aligned(records=cdn_records)
    stop_if_not_aligned(records=pep_records)
    # check_same_order()
    tcdn_strings = [str(record.seq.translate(table=args.codontable, to_stop=False, gap="-")) for record in cdn_records]
    pep_strings = [str(record.seq) for record in pep_records]
    cdn_strings = [str(record.seq) for record in cdn_records]
    kept_aa_sites = list()
    multiple_matches = set()
    remaining_site_count = len(tcdn_strings[0])
    tcdn_col_index = build_column_index(tcdn_strings)
    for pi, pep_col_chars in enumerate(zip(*pep_strings)):
        if remaining_site_count == 0:
            break
        key = ''.join(pep_col_chars)
        same_sites = tcdn_col_index.get(key)
        if (same_sites is None) or (len(same_sites)==0):
            txt = 'The codon site {} could not be matched to trimmed protein sites. '
            txt += 'The site may contain only missing, ambiguous, and/or stop codons. '
            txt += 'The site will be excluded from the output.\n'
            sys.stderr.write(txt.format(pi))
            continue
        if (len(same_sites)==1):
            kept_aa_sites.append(same_sites.popleft())
            remaining_site_count -= 1
            del tcdn_col_index[key]
        elif (len(same_sites)>1):
            all_same_sites = tuple(same_sites)
            multiple_matches.update(all_same_sites)
            txt = 'The trimmed protein site {} has multiple matches to codon sites({}). Reporting the first match. '
            txt = txt.format(pi, ','.join([ str(ss) for ss in all_same_sites ]))
            txt += 'Site pattern: {}\n'.format(key)
            sys.stderr.write(txt)
            kept_aa_sites.append(same_sites.popleft())
            remaining_site_count -= 1
    num_trimmed_multiple_hit_sites = len(multiple_matches-set(kept_aa_sites))
    txt = '{} codon sites matched to {} protein sites. '
    txt = txt+'Trimmed {} codon sites that matched to multiple protein sites.\n'
    txt = txt.format(len(kept_aa_sites), len(pep_strings[0]), num_trimmed_multiple_hit_sites)
    sys.stderr.write(txt)
    trimmed_cdn_records = list()
    for i in range(len(cdn_records)):
        trimmed_seq = ''.join([cdn_strings[i][codon_site*3:codon_site*3+3] for codon_site in kept_aa_sites])
        trimmed_record = Bio.SeqRecord.SeqRecord(seq=Bio.Seq.Seq(trimmed_seq),
                                                 id=cdn_records[i].id, name='', description='')
        trimmed_cdn_records.append(trimmed_record)
    txt = 'Number of aligned nucleotide sites in untrimmed codon sequences: {}\n'
    sys.stderr.write(txt.format(len(cdn_records[0].seq)))
    txt = 'Number of aligned nucleotide sites in trimmed codon sequences: {}\n'
    sys.stderr.write(txt.format(len(trimmed_cdn_records[0].seq)))
    write_seqs(records=trimmed_cdn_records, outfile=args.outfile, outseqformat=args.outseqformat)
