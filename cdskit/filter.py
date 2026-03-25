import json
import sys
from functools import partial

from cdskit.codonutil import ambiguous_codon_counts, has_internal_stop, is_gap_only_sequence
from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    write_seqs,
)


def validate_fraction(name, value):
    value = float(value)
    if (value < 0.0) or (value > 1.0):
        txt = '{} should be between 0 and 1 inclusive. Exiting.\n'
        raise Exception(txt.format(name))
    return value


def analyze_record(record, codontable, inspect_internal_stop):
    seq_str = str(record.seq)
    ambiguous_codons, evaluable_codons = ambiguous_codon_counts(seq=seq_str)
    ambiguous_codon_rate = 0.0
    if evaluable_codons > 0:
        ambiguous_codon_rate = ambiguous_codons / evaluable_codons
    return {
        'id': record.id,
        'non_triplet': (len(seq_str) % 3) != 0,
        'gap_only': is_gap_only_sequence(seq=seq_str),
        'internal_stop': inspect_internal_stop and has_internal_stop(seq=seq_str, codontable=codontable),
        'ambiguous_codons': ambiguous_codons,
        'evaluable_codons': evaluable_codons,
        'ambiguous_codon_rate': ambiguous_codon_rate,
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
        'gap_only': list(),
        'internal_stop': list(),
        'ambiguous_rate': list(),
        'duplicate': list(),
    }
    surviving_indices = list()
    for idx, analysis in enumerate(analyses):
        should_drop = False
        if args.drop_non_triplet and analysis['non_triplet']:
            reasons['non_triplet'].append(records[idx].id)
            should_drop = True
        if args.drop_gap_only and analysis['gap_only']:
            reasons['gap_only'].append(records[idx].id)
            should_drop = True
        if args.drop_internal_stop and analysis['internal_stop']:
            reasons['internal_stop'].append(records[idx].id)
            should_drop = True
        if analysis['ambiguous_codon_rate'] > args.max_ambiguous_codon_rate:
            reasons['ambiguous_rate'].append(records[idx].id)
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

    kept_indices = [idx for idx in surviving_indices if idx in kept_index_set]
    dropped_indices = [idx for idx in range(len(records)) if idx not in kept_index_set]

    return {
        'num_input_sequences': len(records),
        'num_output_sequences': len(kept_indices),
        'num_dropped_sequences': len(dropped_indices),
        'drop_non_triplet': bool(args.drop_non_triplet),
        'drop_gap_only': bool(args.drop_gap_only),
        'drop_internal_stop': bool(args.drop_internal_stop),
        'max_ambiguous_codon_rate': args.max_ambiguous_codon_rate,
        'dedup': args.dedup,
        'dropped_ids_by_reason': reasons,
        'kept_ids': [records[idx].id for idx in kept_indices],
        'dropped_ids': [records[idx].id for idx in dropped_indices],
    }, kept_indices


def write_filter_report(report_path, summary):
    if report_path == '':
        return
    if report_path.lower().endswith('.json'):
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('metric\tvalue\n')
        for key in [
            'num_input_sequences',
            'num_output_sequences',
            'num_dropped_sequences',
            'drop_non_triplet',
            'drop_gap_only',
            'drop_internal_stop',
            'max_ambiguous_codon_rate',
            'dedup',
        ]:
            f.write(f'{key}\t{summary[key]}\n')
        f.write('\n')
        f.write('section\tids\n')
        for key in [
            'non_triplet',
            'gap_only',
            'internal_stop',
            'ambiguous_rate',
            'duplicate',
        ]:
            ids = ','.join(summary['dropped_ids_by_reason'][key])
            f.write(f'dropped_{key}_ids\t{ids}\n')
        f.write(f"kept_ids\t{','.join(summary['kept_ids'])}\n")
        f.write(f"dropped_ids\t{','.join(summary['dropped_ids'])}\n")


def filter_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seqfile')
    if args.drop_internal_stop:
        stop_if_invalid_codontable(args.codontable)
    args.max_ambiguous_codon_rate = validate_fraction(
        name='--max_ambiguous_codon_rate',
        value=args.max_ambiguous_codon_rate,
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
