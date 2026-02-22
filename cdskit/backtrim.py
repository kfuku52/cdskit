from collections import defaultdict
from collections import deque
import sys
from functools import partial

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from cdskit.translate import translate_sequence_string

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_aligned,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)

def check_same_seq_num(cdn_records, pep_records):
    err_txt = 'The numbers of seqs did not match: seqfile={} and trimmed_aa_aln={}'.format(len(cdn_records), len(pep_records))
    if len(cdn_records) != len(pep_records):
        raise Exception(err_txt)


def get_record_map(records, label):
    record_map = dict()
    for record in records:
        if record.id in record_map:
            txt = 'Sequence IDs must be unique in {}. Duplicated ID: {}'
            raise Exception(txt.format(label, record.id))
        record_map[record.id] = record
    return record_map


def reorder_aa_records_by_cds_ids(cdn_records, pep_records):
    cdn_record_map = get_record_map(cdn_records, '--seqfile')
    pep_record_map = get_record_map(pep_records, '--trimmed_aa_aln')
    cdn_ids = set(cdn_record_map.keys())
    pep_ids = set(pep_record_map.keys())
    if cdn_ids != pep_ids:
        missing_in_cds = sorted(list(pep_ids - cdn_ids))
        missing_in_aa = sorted(list(cdn_ids - pep_ids))
        txt = 'Sequence IDs did not match between CDS (--seqfile) and trimmed amino acid alignment (--trimmed_aa_aln).'
        if len(missing_in_cds) > 0:
            txt += ' Missing in CDS: {}.'.format(','.join(missing_in_cds))
        if len(missing_in_aa) > 0:
            txt += ' Missing in trimmed amino acid alignment: {}.'.format(','.join(missing_in_aa))
        raise Exception(txt)
    return [pep_record_map[record.id] for record in cdn_records]


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
    last_kept_site = -1
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
            kept_site = same_sites.popleft()
            if kept_site <= last_kept_site:
                txt = 'The codon site {} would violate codon order at trimmed protein site {}. '
                txt += 'The site will be excluded from the output.\n'
                sys.stderr.write(txt.format(kept_site, pi))
                del tcdn_col_index[key]
                continue
            kept_aa_sites.append(kept_site)
            last_kept_site = kept_site
            remaining_site_count -= 1
            del tcdn_col_index[key]
            continue

        all_same_sites = tuple(same_sites)
        multiple_matches.update(all_same_sites)
        # Prefer the first candidate after the previously selected codon site
        # so codon order follows the trimmed amino acid alignment.
        kept_site = None
        for candidate_site in same_sites:
            if candidate_site > last_kept_site:
                kept_site = candidate_site
                break
        if kept_site is None:
            txt = 'The trimmed protein site {} has multiple matches to codon sites({}), '
            txt += 'but none preserve codon order. The site will be excluded from the output. '
            txt = txt.format(pi, ','.join([str(ss) for ss in all_same_sites]))
            txt += 'Site pattern: {}\n'.format(key)
            sys.stderr.write(txt)
            continue

        txt = 'The trimmed protein site {} has multiple matches to codon sites({}). Reporting codon site {}. '
        txt = txt.format(pi, ','.join([str(ss) for ss in all_same_sites]), kept_site)
        txt += 'Site pattern: {}\n'.format(key)
        sys.stderr.write(txt)
        same_sites.remove(kept_site)
        kept_aa_sites.append(kept_site)
        last_kept_site = kept_site
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
    return translate_sequence_string(
        seq_str=str(record.seq),
        codontable=codontable,
        to_stop=False,
    )


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
    stop_if_not_dna(records=cdn_records, label='--seqfile')
    stop_if_invalid_codontable(args.codontable)
    pep_records = read_seqs(seqfile=args.trimmed_aa_aln, seqformat=args.inseqformat)
    if len(cdn_records) == 0:
        if len(pep_records) != 0:
            txt = 'The numbers of seqs did not match: seqfile={} and trimmed_aa_aln={}'
            raise Exception(txt.format(len(cdn_records), len(pep_records)))
        write_seqs(records=list(), outfile=args.outfile, outseqformat=args.outseqformat)
        return
    threads = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_multiple_of_three(cdn_records)
    check_same_seq_num(cdn_records, pep_records)
    stop_if_not_aligned(records=cdn_records)
    stop_if_not_aligned(records=pep_records)
    pep_records = reorder_aa_records_by_cds_ids(cdn_records=cdn_records, pep_records=pep_records)
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
