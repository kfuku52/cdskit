import copy
import json
import sys

from Bio.Seq import Seq

from cdskit.codonutil import codon_has_missing, codon_is_ambiguous, codon_is_clean, codon_is_stop
from cdskit.util import (
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_aligned,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)


def validate_fraction(name, value):
    value = float(value)
    if (value < 0.0) or (value > 1.0):
        txt = '{} should be between 0 and 1 inclusive. Exiting.\n'
        raise Exception(txt.format(name))
    return value


def summarize_codon_site(seq_strings, codon_site, codontable):
    clean = 0
    missing = 0
    ambiguous = 0
    stop = 0
    start = codon_site * 3
    end = start + 3
    for seq_str in seq_strings:
        codon = seq_str[start:end]
        if codon_is_clean(codon=codon, codontable=codontable):
            clean += 1
            continue
        if codon_has_missing(codon):
            missing += 1
            continue
        if codon_is_ambiguous(codon):
            ambiguous += 1
            continue
        if codon_is_stop(codon=codon, codontable=codontable):
            stop += 1
            continue
    return {
        'codon_site_1based': codon_site + 1,
        'clean_codons': clean,
        'missing_codons': missing,
        'ambiguous_codons': ambiguous,
        'stop_codons': stop,
    }


def choose_kept_codon_sites(site_summaries, num_sequences, min_clean_fraction):
    kept_sites = list()
    for summary in site_summaries:
        clean_fraction = 0.0
        if num_sequences > 0:
            clean_fraction = summary['clean_codons'] / num_sequences
        summary['clean_fraction'] = clean_fraction
        summary['unclean_codons'] = num_sequences - summary['clean_codons']
        keep = clean_fraction >= min_clean_fraction
        summary['keep'] = keep
        if keep:
            kept_sites.append(summary['codon_site_1based'] - 1)
    return kept_sites


def trim_record_to_codon_sites(record, kept_sites):
    trimmed = copy.copy(record)
    seq_str = str(record.seq)
    trimmed.seq = Seq(''.join(seq_str[site * 3:site * 3 + 3] for site in kept_sites))
    return trimmed


def build_trimcodon_summary(site_summaries, kept_sites, num_sequences, args):
    removed_sites = [summary['codon_site_1based'] for summary in site_summaries if not summary['keep']]
    return {
        'num_sequences': num_sequences,
        'num_input_codon_sites': len(site_summaries),
        'num_output_codon_sites': len(kept_sites),
        'num_removed_codon_sites': len(removed_sites),
        'min_clean_fraction': args.min_clean_fraction,
        'kept_codon_sites_1based': [site + 1 for site in kept_sites],
        'removed_codon_sites_1based': removed_sites,
        'site_summaries': site_summaries,
    }


def write_trimcodon_report(report_path, summary):
    if report_path == '':
        return
    if report_path.lower().endswith('.json'):
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('metric\tvalue\n')
        for key in [
            'num_sequences',
            'num_input_codon_sites',
            'num_output_codon_sites',
            'num_removed_codon_sites',
            'min_clean_fraction',
        ]:
            f.write(f'{key}\t{summary[key]}\n')
        f.write('\n')
        f.write(
            'codon_site_1based\tclean_fraction\tclean_codons\tunclean_codons\t'
            'missing_codons\tambiguous_codons\tstop_codons\tkeep\n'
        )
        for site_summary in summary['site_summaries']:
            f.write(
                '{}\t{:.6f}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                    site_summary['codon_site_1based'],
                    site_summary['clean_fraction'],
                    site_summary['clean_codons'],
                    site_summary['unclean_codons'],
                    site_summary['missing_codons'],
                    site_summary['ambiguous_codons'],
                    site_summary['stop_codons'],
                    site_summary['keep'],
                )
            )


def trimcodon_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    _ = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_not_aligned(records=records)
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    args.min_clean_fraction = validate_fraction(name='--min_clean_fraction', value=args.min_clean_fraction)
    if len(records) == 0:
        summary = build_trimcodon_summary(site_summaries=list(), kept_sites=list(), num_sequences=0, args=args)
        write_trimcodon_report(report_path=args.report, summary=summary)
        write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    seq_strings = [str(record.seq) for record in records]
    num_sites = len(seq_strings[0]) // 3
    site_summaries = [
        summarize_codon_site(
            seq_strings=seq_strings,
            codon_site=codon_site,
            codontable=args.codontable,
        )
        for codon_site in range(num_sites)
    ]
    kept_sites = choose_kept_codon_sites(
        site_summaries=site_summaries,
        num_sequences=len(records),
        min_clean_fraction=args.min_clean_fraction,
    )
    summary = build_trimcodon_summary(
        site_summaries=site_summaries,
        kept_sites=kept_sites,
        num_sequences=len(records),
        args=args,
    )
    write_trimcodon_report(report_path=args.report, summary=summary)
    sys.stderr.write('Removed codon sites: {:,}\n'.format(summary['num_removed_codon_sites']))
    out_records = [trim_record_to_codon_sites(record=record, kept_sites=kept_sites) for record in records]
    write_seqs(records=out_records, outfile=args.outfile, outseqformat=args.outseqformat)
