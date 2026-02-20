from collections import defaultdict
from collections import deque
import sys
from functools import partial

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_aligned,
    stop_if_not_multiple_of_three,
    write_seqs,
)

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


def find_kept_aa_sites(tcdn_strings, pep_strings):
    kept_aa_sites = []
    multiple_matches = set()
    remaining_site_count = len(tcdn_strings[0])
    tcdn_col_index = build_column_index(tcdn_strings)

    for pi, pep_col_chars in enumerate(zip(*pep_strings)):
        if remaining_site_count == 0:
            break
        key = ''.join(pep_col_chars)
        same_sites = tcdn_col_index.get(key)
        if (same_sites is None) or (len(same_sites) == 0):
            txt = 'The codon site {} could not be matched to trimmed protein sites. '
            txt += 'The site may contain only missing, ambiguous, and/or stop codons. '
            txt += 'The site will be excluded from the output.\n'
            sys.stderr.write(txt.format(pi))
            continue
        if len(same_sites) == 1:
            kept_aa_sites.append(same_sites.popleft())
            remaining_site_count -= 1
            del tcdn_col_index[key]
            continue

        all_same_sites = tuple(same_sites)
        multiple_matches.update(all_same_sites)
        txt = 'The trimmed protein site {} has multiple matches to codon sites({}). Reporting the first match. '
        txt = txt.format(pi, ','.join([str(ss) for ss in all_same_sites]))
        txt += 'Site pattern: {}\n'.format(key)
        sys.stderr.write(txt)
        kept_aa_sites.append(same_sites.popleft())
        remaining_site_count -= 1

    num_trimmed_multiple_hit_sites = len(multiple_matches - set(kept_aa_sites))
    return kept_aa_sites, num_trimmed_multiple_hit_sites


def trim_codon_records(cdn_records, kept_aa_sites):
    cdn_strings = [str(record.seq) for record in cdn_records]
    trimmed_cdn_records = []
    for i, record in enumerate(cdn_records):
        trimmed_seq = ''.join([cdn_strings[i][codon_site * 3:codon_site * 3 + 3] for codon_site in kept_aa_sites])
        trimmed_record = SeqRecord(
            seq=Seq(trimmed_seq),
            id=record.id,
            name='',
            description='',
        )
        trimmed_cdn_records.append(trimmed_record)
    return trimmed_cdn_records


def translate_record_to_aa_string(record, codontable):
    return str(record.seq.translate(table=codontable, to_stop=False, gap="-"))


def trim_codon_record(record, kept_aa_sites):
    seq_str = str(record.seq)
    trimmed_seq = ''.join([seq_str[codon_site * 3:codon_site * 3 + 3] for codon_site in kept_aa_sites])
    return SeqRecord(
        seq=Seq(trimmed_seq),
        id=record.id,
        name='',
        description='',
    )


def codon_sites_to_nucleotide_ranges(codon_sites):
    if len(codon_sites) == 0:
        return list()
    ranges = list()
    run_start = codon_sites[0]
    run_end = codon_sites[0]
    for site in codon_sites[1:]:
        if site == run_end + 1:
            run_end = site
            continue
        ranges.append((run_start * 3, (run_end + 1) * 3))
        run_start = site
        run_end = site
    ranges.append((run_start * 3, (run_end + 1) * 3))
    return ranges


def trim_codon_record_with_ranges(record, nucleotide_ranges):
    seq_str = str(record.seq)
    trimmed_seq = ''.join([seq_str[start:end] for start, end in nucleotide_ranges])
    return SeqRecord(
        seq=Seq(trimmed_seq),
        id=record.id,
        name='',
        description='',
    )


def backtrim_main(args):
    cdn_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    pep_records = read_seqs(seqfile=args.trimmed_aa_aln, seqformat=args.inseqformat)
    threads = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_multiple_of_three(cdn_records)
    check_same_seq_num(cdn_records, pep_records)
    stop_if_not_aligned(records=cdn_records)
    stop_if_not_aligned(records=pep_records)
    # check_same_order()
    translate_worker = partial(translate_record_to_aa_string, codontable=args.codontable)
    tcdn_strings = parallel_map_ordered(items=cdn_records, worker=translate_worker, threads=threads)
    pep_strings = [str(record.seq) for record in pep_records]
    kept_aa_sites, num_trimmed_multiple_hit_sites = find_kept_aa_sites(tcdn_strings, pep_strings)
    txt = '{} codon sites matched to {} protein sites. '
    txt += 'Trimmed {} codon sites that matched to multiple protein sites.\n'
    txt = txt.format(len(kept_aa_sites), len(pep_strings[0]), num_trimmed_multiple_hit_sites)
    sys.stderr.write(txt)
    nucleotide_ranges = codon_sites_to_nucleotide_ranges(codon_sites=kept_aa_sites)
    trim_worker = partial(trim_codon_record_with_ranges, nucleotide_ranges=nucleotide_ranges)
    trimmed_cdn_records = parallel_map_ordered(items=cdn_records, worker=trim_worker, threads=threads)
    txt = 'Number of aligned nucleotide sites in untrimmed codon sequences: {}\n'
    sys.stderr.write(txt.format(len(cdn_records[0].seq)))
    txt = 'Number of aligned nucleotide sites in trimmed codon sequences: {}\n'
    sys.stderr.write(txt.format(len(trimmed_cdn_records[0].seq)))
    write_seqs(records=trimmed_cdn_records, outfile=args.outfile, outseqformat=args.outseqformat)
