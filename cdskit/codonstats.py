import sys
from collections import Counter

from cdskit.codonutil import (
    UNAMBIGUOUS_NT,
    ambiguous_codon_counts,
    codon_has_missing,
    codon_is_stop,
    get_forward_table,
)
from cdskit.util import (
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
)


def gc_percent(gc_count, total_count):
    if total_count == 0:
        return 0.0
    return (gc_count / total_count) * 100


def summarize_record(record, codontable):
    seq_str = str(record.seq)
    codons_total = len(seq_str) // 3
    codons_missing = 0
    codons_stop = 0
    usage = Counter()
    gc_counts = [0, 0, 0]
    gc_denoms = [0, 0, 0]
    for i in range(0, len(seq_str), 3):
        codon = seq_str[i:i + 3]
        codon_upper = codon.upper()
        for pos, ch in enumerate(codon_upper):
            if ch in UNAMBIGUOUS_NT:
                gc_denoms[pos] += 1
                if ch in ('G', 'C'):
                    gc_counts[pos] += 1
        if codon_has_missing(codon_upper):
            codons_missing += 1
            continue
        if not any(ch not in UNAMBIGUOUS_NT for ch in codon_upper):
            usage[codon_upper] += 1
            if codon_is_stop(codon=codon_upper, codontable=codontable):
                codons_stop += 1
    codons_ambiguous, evaluable_codons = ambiguous_codon_counts(seq=seq_str)
    codons_complete = evaluable_codons
    gc_total = sum(gc_counts)
    gc_denom_total = sum(gc_denoms)
    return {
        'seq_id': record.id,
        'nt_length': len(seq_str),
        'codons_total': codons_total,
        'codons_complete': codons_complete,
        'codons_missing': codons_missing,
        'codons_ambiguous': codons_ambiguous,
        'codons_stop': codons_stop,
        'gc_all': gc_percent(gc_total, gc_denom_total),
        'gc1': gc_percent(gc_counts[0], gc_denoms[0]),
        'gc2': gc_percent(gc_counts[1], gc_denoms[1]),
        'gc3': gc_percent(gc_counts[2], gc_denoms[2]),
        'usage': usage,
    }


def print_summary_table(summaries):
    sys.stdout.write(
        'seq_id\tnt_length\tcodons_total\tcodons_complete\tcodons_missing\tcodons_ambiguous\tcodons_stop\tgc_all\tgc1\tgc2\tgc3\n'
    )
    for summary in summaries:
        sys.stdout.write(
            '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(
                summary['seq_id'],
                summary['nt_length'],
                summary['codons_total'],
                summary['codons_complete'],
                summary['codons_missing'],
                summary['codons_ambiguous'],
                summary['codons_stop'],
                summary['gc_all'],
                summary['gc1'],
                summary['gc2'],
                summary['gc3'],
            )
        )


def print_usage_table(summaries, codontable):
    usage = Counter()
    for summary in summaries:
        usage.update(summary['usage'])
    total = sum(usage.values())
    forward_table = get_forward_table(codontable=codontable)
    sys.stdout.write('codon\taa\tcount\tfraction\n')
    for codon in sorted(usage):
        aa = forward_table.get(codon, '*')
        fraction = 0.0
        if total > 0:
            fraction = usage[codon] / total
        sys.stdout.write(f'{codon}\t{aa}\t{usage[codon]}\t{fraction:.6f}\n')


def codonstats_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    _ = resolve_threads(getattr(args, 'threads', 1))
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_not_multiple_of_three(records=records)
    stop_if_invalid_codontable(args.codontable)
    summaries = [summarize_record(record=record, codontable=args.codontable) for record in records]
    if args.mode in ('summary', 'both'):
        print_summary_table(summaries=summaries)
    if args.mode == 'both':
        sys.stdout.write('\n')
    if args.mode in ('usage', 'both'):
        print_usage_table(summaries=summaries, codontable=args.codontable)
