import json
import sys
from functools import partial

from cdskit.codonutil import (
    codon_has_missing,
    codon_is_ambiguous,
    codon_is_clean,
    codon_is_stop,
    has_internal_stop,
)
from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    write_seqs,
)
from cdskit.tsvio import json_cell, write_sectioned_tsv


def validate_fraction(name, value):
    value = float(value)
    if (value < 0.0) or (value > 1.0):
        txt = '{} should be between 0 and 1 inclusive. Exiting.\n'
        raise Exception(txt.format(name))
    return value


def analyze_record(record, codontable, inspect_internal_stop):
    seq_str = str(record.seq)
    length_nt = len(seq_str)
    clean_codons = 0
    missing_codons = 0
    ambiguous_codons = 0
    stop_codons = 0
    total_codons = length_nt // 3
    for idx in range(total_codons):
        codon = seq_str[idx * 3:idx * 3 + 3]
        if codon_is_clean(codon=codon, codontable=codontable):
            clean_codons += 1
            continue
        if codon_has_missing(codon):
            missing_codons += 1
            continue
        if codon_is_ambiguous(codon):
            ambiguous_codons += 1
            continue
        if codon_is_stop(codon=codon, codontable=codontable):
            stop_codons += 1
            continue
    clean_codon_fraction = 0.0
    if total_codons > 0:
        clean_codon_fraction = clean_codons / total_codons
    return {
        'id': record.id,
        'length_nt': length_nt,
        'tail_nt': length_nt % 3,
        'non_triplet': (length_nt % 3) != 0,
        'internal_stop': inspect_internal_stop and has_internal_stop(seq=seq_str, codontable=codontable),
        'total_codons': total_codons,
        'clean_codons': clean_codons,
        'unclean_codons': total_codons - clean_codons,
        'missing_codons': missing_codons,
        'ambiguous_codons': ambiguous_codons,
        'stop_codons': stop_codons,
        'clean_codon_fraction': clean_codon_fraction,
    }


def choose_duplicate_winners(records, candidate_indices, dedup):
    if dedup == 'no':
        return set(candidate_indices), list()
    if dedup == 'keep-first':
        seen = set()
        kept = set()
        dropped = list()
        for idx in candidate_indices:
            seq_id = records[idx].id
            if seq_id in seen:
                dropped.append(idx)
                continue
            seen.add(seq_id)
            kept.add(idx)
        return kept, dropped
    if dedup == 'keep-longest':
        winners = dict()
        for idx in candidate_indices:
            seq_id = records[idx].id
            winner_idx = winners.get(seq_id)
            if winner_idx is None:
                winners[seq_id] = idx
                continue
            if len(records[idx].seq) > len(records[winner_idx].seq):
                winners[seq_id] = idx
        kept = set(winners.values())
        dropped = [idx for idx in candidate_indices if idx not in kept]
        return kept, dropped
    raise Exception('Invalid --dedup: {}. Exiting.\n'.format(dedup))


def summarize_filter(records, analyses, args):
    reasons = {
        'non_triplet': list(),
        'internal_stop': list(),
        'clean_codon_fraction': list(),
        'duplicate': list(),
    }
    index_reasons = {idx: list() for idx in range(len(records))}
    surviving_indices = list()
    for idx, analysis in enumerate(analyses):
        should_drop = False
        if args.drop_non_triplet and analysis['non_triplet']:
            reasons['non_triplet'].append(records[idx].id)
            index_reasons[idx].append('non_triplet')
            should_drop = True
        if args.drop_internal_stop and analysis['internal_stop']:
            reasons['internal_stop'].append(records[idx].id)
            index_reasons[idx].append('internal_stop')
            should_drop = True
        if analysis['clean_codon_fraction'] < args.min_clean_codon_fraction:
            reasons['clean_codon_fraction'].append(records[idx].id)
            index_reasons[idx].append('clean_codon_fraction')
            should_drop = True
        if not should_drop:
            surviving_indices.append(idx)

    kept_index_set, duplicate_dropped_indices = choose_duplicate_winners(
        records=records,
        candidate_indices=surviving_indices,
        dedup=args.dedup,
    )
    for idx in duplicate_dropped_indices:
        reasons['duplicate'].append(records[idx].id)
        index_reasons[idx].append('duplicate')

    kept_indices = [idx for idx in surviving_indices if idx in kept_index_set]
    dropped_indices = [idx for idx in range(len(records)) if idx not in kept_index_set]
    sequence_reports = list()
    for idx, analysis in enumerate(analyses):
        sequence_reports.append({
            'input_order': idx + 1,
            'id': records[idx].id,
            'kept': idx in kept_index_set,
            'drop_reasons': index_reasons[idx],
            'length_nt': analysis['length_nt'],
            'tail_nt': analysis['tail_nt'],
            'non_triplet': analysis['non_triplet'],
            'internal_stop': analysis['internal_stop'],
            'total_codons': analysis['total_codons'],
            'clean_codons': analysis['clean_codons'],
            'unclean_codons': analysis['unclean_codons'],
            'missing_codons': analysis['missing_codons'],
            'ambiguous_codons': analysis['ambiguous_codons'],
            'stop_codons': analysis['stop_codons'],
            'clean_codon_fraction': analysis['clean_codon_fraction'],
        })

    return {
        'num_input_sequences': len(records),
        'num_output_sequences': len(kept_indices),
        'num_dropped_sequences': len(dropped_indices),
        'drop_non_triplet': bool(args.drop_non_triplet),
        'drop_internal_stop': bool(args.drop_internal_stop),
        'min_clean_codon_fraction': args.min_clean_codon_fraction,
        'dedup': args.dedup,
        'drop_counts_by_reason': {key: len(ids) for key, ids in reasons.items()},
        'dropped_ids_by_reason': reasons,
        'kept_ids': [records[idx].id for idx in kept_indices],
        'dropped_ids': [records[idx].id for idx in dropped_indices],
        'sequence_reports': sequence_reports,
    }, kept_indices


def write_filter_report(report_path, summary):
    if report_path == '':
        return
    if report_path.lower().endswith('.json'):
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return
    rows = []
    for key in [
        'num_input_sequences',
        'num_output_sequences',
        'num_dropped_sequences',
        'drop_non_triplet',
        'drop_internal_stop',
        'min_clean_codon_fraction',
        'dedup',
    ]:
        rows.append({'section': 'summary', 'metric': key, 'value': json_cell(summary[key])})
    for key in ['non_triplet', 'internal_stop', 'clean_codon_fraction', 'duplicate']:
        rows.append({
            'section': 'id_set',
            'metric': 'dropped_{}_ids'.format(key),
            'ids': json_cell(summary['dropped_ids_by_reason'][key]),
        })
    rows.extend([
        {'section': 'id_set', 'metric': 'kept_ids', 'ids': json_cell(summary['kept_ids'])},
        {'section': 'id_set', 'metric': 'dropped_ids', 'ids': json_cell(summary['dropped_ids'])},
    ])
    for key in ['non_triplet', 'internal_stop', 'clean_codon_fraction', 'duplicate']:
        rows.append({
            'section': 'drop_reason',
            'drop_reason': key,
            'count': summary['drop_counts_by_reason'][key],
        })
    for source_row in summary['sequence_reports']:
        row = dict(source_row)
        row['section'] = 'sequence'
        row['kept'] = json_cell(row['kept'])
        row['non_triplet'] = json_cell(row['non_triplet'])
        row['internal_stop'] = json_cell(row['internal_stop'])
        row['drop_reasons'] = json_cell(row['drop_reasons'])
        row['clean_codon_fraction'] = '{:.6g}'.format(row['clean_codon_fraction'])
        rows.append(row)
    write_sectioned_tsv(
        path=report_path,
        fieldnames=[
            'section', 'metric', 'value', 'ids', 'drop_reason', 'count',
            'input_order', 'id', 'kept', 'drop_reasons', 'length_nt', 'tail_nt',
            'non_triplet', 'internal_stop', 'total_codons', 'clean_codons',
            'unclean_codons', 'missing_codons', 'ambiguous_codons', 'stop_codons',
            'clean_codon_fraction',
        ],
        rows=rows,
    )


def filter_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seq_file')
    stop_if_invalid_codontable(args.codontable)
    args.min_clean_codon_fraction = validate_fraction(
        name='--min_clean_codon_fraction',
        value=getattr(args, 'min_clean_codon_fraction', 0.5),
    )
    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        analyze_record,
        codontable=args.codontable,
        inspect_internal_stop=args.drop_internal_stop,
    )
    analyses = parallel_map_ordered(items=records, worker=worker, threads=threads)
    summary, kept_indices = summarize_filter(records=records, analyses=analyses, args=args)
    write_filter_report(report_path=args.report, summary=summary)
    sys.stderr.write('Dropped sequences: {:,}\n'.format(summary['num_dropped_sequences']))
    out_records = [records[idx] for idx in kept_indices]
    write_seqs(records=out_records, outfile=args.outfile, outseqformat=args.outseqformat)
