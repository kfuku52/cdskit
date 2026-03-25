import copy
import json
import sys

from Bio.Seq import Seq

from cdskit.codonutil import codon_has_missing, codon_is_ambiguous, codon_is_stop
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
    occupied = 0
    ambiguous = 0
    stop = 0
    start = codon_site * 3
    end = start + 3
    for seq_str in seq_strings:
        codon = seq_str[start:end]
        if codon_has_missing(codon):
            continue
        occupied += 1
        if codon_is_ambiguous(codon):
            ambiguous += 1
            continue
        if codon_is_stop(codon=codon, codontable=codontable):
            stop += 1
    return {
        'codon_site_1based': codon_site + 1,
        'occupied_codons': occupied,
        'ambiguous_codons': ambiguous,
        'stop_codons': stop,
    }


def choose_kept_codon_sites(site_summaries, num_sequences, min_occupancy, max_ambiguous_fraction, drop_stop_codon):
    kept_sites = list()
    for summary in site_summaries:
        occupancy = 0.0
        if num_sequences > 0:
            occupancy = summary['occupied_codons'] / num_sequences
        ambiguous_fraction = 0.0
        if summary['occupied_codons'] > 0:
            ambiguous_fraction = summary['ambiguous_codons'] / summary['occupied_codons']
        summary['occupancy'] = occupancy
        summary['ambiguous_fraction'] = ambiguous_fraction
        keep = (
            (occupancy >= min_occupancy)
            and (ambiguous_fraction <= max_ambiguous_fraction)
            and ((not drop_stop_codon) or (summary['stop_codons'] == 0))
        )
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
        'min_occupancy': args.min_occupancy,
        'max_ambiguous_fraction': args.max_ambiguous_fraction,
        'drop_stop_codon': bool(args.drop_stop_codon),
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
            'min_occupancy',
            'max_ambiguous_fraction',
            'drop_stop_codon',
        ]:
            f.write(f'{key}\t{summary[key]}\n')
        f.write('\n')
        f.write('codon_site_1based\toccupancy\toccupied_codons\tambiguous_codons\tambiguous_fraction\tstop_codons\tkeep\n')
        for site_summary in summary['site_summaries']:
            f.write(
                '{}\t{:.6f}\t{}\t{}\t{:.6f}\t{}\t{}\n'.format(
                    site_summary['codon_site_1based'],
                    site_summary['occupancy'],
                    site_summary['occupied_codons'],
                    site_summary['ambiguous_codons'],
                    site_summary['ambiguous_fraction'],
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
    args.min_occupancy = validate_fraction(name='--min_occupancy', value=args.min_occupancy)
    args.max_ambiguous_fraction = validate_fraction(
        name='--max_ambiguous_fraction',
        value=args.max_ambiguous_fraction,
    )
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
        min_occupancy=args.min_occupancy,
        max_ambiguous_fraction=args.max_ambiguous_fraction,
        drop_stop_codon=args.drop_stop_codon,
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
