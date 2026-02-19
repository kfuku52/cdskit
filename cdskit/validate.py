import json
import sys
from collections import Counter
from functools import partial

import Bio.Seq

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads


MISSING_CHARS = frozenset('-?.')
GAP_ONLY_CHARS = frozenset('-?.NXnx')
UNAMBIGUOUS_NT = frozenset('ACGTacgt')


def chunk_codons(seq):
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


def is_gap_only_sequence(seq):
    return len(seq) > 0 and all(ch in GAP_ONLY_CHARS for ch in seq)


def has_internal_stop(seq, codontable):
    clean_seq = ''.join(ch for ch in seq if ch not in MISSING_CHARS)
    if (len(clean_seq) < 3) or (len(clean_seq) % 3 != 0):
        return False
    aa = str(Bio.Seq.Seq(clean_seq).translate(table=codontable, to_stop=False))
    if len(aa) <= 1:
        return False
    return '*' in aa[:-1]


def is_ambiguous_codon(codon):
    if any(ch in MISSING_CHARS for ch in codon):
        return False
    return any(ch not in UNAMBIGUOUS_NT for ch in codon)


def sequence_ambiguous_codon_counts(seq):
    ambiguous = 0
    evaluable = 0
    for codon in chunk_codons(seq):
        if len(codon) != 3:
            continue
        if any(ch in MISSING_CHARS for ch in codon):
            continue
        evaluable += 1
        if is_ambiguous_codon(codon):
            ambiguous += 1
    return ambiguous, evaluable


def get_duplicate_ids(records):
    counts = Counter(record.id for record in records)
    return sorted([seq_id for seq_id, count in counts.items() if count > 1])


def summarize_single_record(record, codontable):
    seq = str(record.seq)
    ambiguous, evaluable = sequence_ambiguous_codon_counts(seq)
    return {
        'id': record.id,
        'is_non_triplet': (len(record.seq) % 3 != 0),
        'is_gap_only': is_gap_only_sequence(seq),
        'has_internal_stop': has_internal_stop(seq, codontable=codontable),
        'ambiguous_codons': ambiguous,
        'evaluable_codons': evaluable,
    }


def summarize_records(records, codontable, threads=1):
    lengths = [len(record.seq) for record in records]
    aligned = len(set(lengths)) <= 1
    duplicate_ids = get_duplicate_ids(records)
    worker = partial(summarize_single_record, codontable=codontable)
    worker_threads = resolve_threads(threads=threads)
    per_record = parallel_map_ordered(items=records, worker=worker, threads=worker_threads)

    non_triplet_ids = [entry['id'] for entry in per_record if entry['is_non_triplet']]
    gap_only_ids = [entry['id'] for entry in per_record if entry['is_gap_only']]
    internal_stop_ids = [entry['id'] for entry in per_record if entry['has_internal_stop']]
    ambiguous_by_seq = dict()
    total_ambiguous = 0
    total_evaluable = 0
    for entry in per_record:
        ambiguous_by_seq[entry['id']] = entry['ambiguous_codons']
        total_ambiguous += entry['ambiguous_codons']
        total_evaluable += entry['evaluable_codons']
    ambiguous_rate = 0.0
    if total_evaluable > 0:
        ambiguous_rate = total_ambiguous / total_evaluable

    issue_ids = set(non_triplet_ids) | set(gap_only_ids) | set(internal_stop_ids)
    for seq_id, count in ambiguous_by_seq.items():
        if count > 0:
            issue_ids.add(seq_id)
    issue_ids |= set(duplicate_ids)

    return {
        'num_sequences': len(records),
        'aligned': aligned,
        'non_triplet_ids': non_triplet_ids,
        'duplicate_ids': duplicate_ids,
        'gap_only_ids': gap_only_ids,
        'internal_stop_ids': internal_stop_ids,
        'ambiguous_codons': total_ambiguous,
        'evaluable_codons': total_evaluable,
        'ambiguous_codon_rate': ambiguous_rate,
        'ambiguous_codons_by_sequence': ambiguous_by_seq,
        'num_sequences_with_issues': len(issue_ids),
        'sequence_ids_with_issues': sorted(issue_ids),
    }


def write_validate_report(report_path, summary):
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
            'aligned',
            'num_non_triplet_sequences',
            'num_duplicate_ids',
            'num_gap_only_sequences',
            'num_internal_stop_sequences',
            'ambiguous_codons',
            'evaluable_codons',
            'ambiguous_codon_rate',
            'num_sequences_with_issues',
        ]:
            if key == 'num_non_triplet_sequences':
                value = len(summary['non_triplet_ids'])
            elif key == 'num_duplicate_ids':
                value = len(summary['duplicate_ids'])
            elif key == 'num_gap_only_sequences':
                value = len(summary['gap_only_ids'])
            elif key == 'num_internal_stop_sequences':
                value = len(summary['internal_stop_ids'])
            else:
                value = summary[key]
            f.write(f'{key}\t{value}\n')
        f.write('\n')
        f.write('section\tids\n')
        f.write(f"non_triplet_ids\t{','.join(summary['non_triplet_ids'])}\n")
        f.write(f"duplicate_ids\t{','.join(summary['duplicate_ids'])}\n")
        f.write(f"gap_only_ids\t{','.join(summary['gap_only_ids'])}\n")
        f.write(f"internal_stop_ids\t{','.join(summary['internal_stop_ids'])}\n")
        f.write(f"sequence_ids_with_issues\t{','.join(summary['sequence_ids_with_issues'])}\n")


def print_validate_summary(summary):
    sys.stdout.write('Validation summary\n')
    sys.stdout.write(f"num_sequences\t{summary['num_sequences']}\n")
    sys.stdout.write(f"aligned\t{summary['aligned']}\n")
    sys.stdout.write(f"num_non_triplet_sequences\t{len(summary['non_triplet_ids'])}\n")
    sys.stdout.write(f"num_duplicate_ids\t{len(summary['duplicate_ids'])}\n")
    sys.stdout.write(f"num_gap_only_sequences\t{len(summary['gap_only_ids'])}\n")
    sys.stdout.write(f"num_internal_stop_sequences\t{len(summary['internal_stop_ids'])}\n")
    sys.stdout.write(f"ambiguous_codons\t{summary['ambiguous_codons']}\n")
    sys.stdout.write(f"evaluable_codons\t{summary['evaluable_codons']}\n")
    sys.stdout.write(f"ambiguous_codon_rate\t{summary['ambiguous_codon_rate']:.6f}\n")
    sys.stdout.write(f"num_sequences_with_issues\t{summary['num_sequences_with_issues']}\n")

    if len(summary['non_triplet_ids']) > 0:
        sys.stdout.write(f"non_triplet_ids\t{','.join(summary['non_triplet_ids'])}\n")
    if len(summary['duplicate_ids']) > 0:
        sys.stdout.write(f"duplicate_ids\t{','.join(summary['duplicate_ids'])}\n")
    if len(summary['gap_only_ids']) > 0:
        sys.stdout.write(f"gap_only_ids\t{','.join(summary['gap_only_ids'])}\n")
    if len(summary['internal_stop_ids']) > 0:
        sys.stdout.write(f"internal_stop_ids\t{','.join(summary['internal_stop_ids'])}\n")


def validate_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    summary = summarize_records(
        records=records,
        codontable=args.codontable,
        threads=getattr(args, 'threads', 1),
    )
    print_validate_summary(summary=summary)
    report_path = getattr(args, 'report', '')
    write_validate_report(report_path=report_path, summary=summary)
