import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import Bio.Data.CodonTable

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads


MISSING_CHARS = frozenset('-?.')
GAP_ONLY_CHARS = frozenset('-?.NXnx')
UNAMBIGUOUS_NT = frozenset('ACGTacgt')
_DROP_MISSING_CHARS_TABLE = str.maketrans('', '', ''.join(sorted(MISSING_CHARS)))
_STOP_CODON_CACHE = dict()
_AMBIGUOUS_CODON_CLASS_CACHE = dict()
_PROCESS_PARALLEL_MIN_RECORDS = 2000


def chunk_codons(seq):
    return [seq[i:i + 3] for i in range(0, len(seq), 3)]


def get_stop_codons(codontable):
    stop_codons = _STOP_CODON_CACHE.get(codontable)
    if stop_codons is None:
        table = Bio.Data.CodonTable.unambiguous_dna_by_id[codontable]
        stop_codons = frozenset([codon.upper() for codon in table.stop_codons])
        _STOP_CODON_CACHE[codontable] = stop_codons
    return stop_codons


def is_gap_only_sequence(seq):
    return len(seq) > 0 and all(ch in GAP_ONLY_CHARS for ch in seq)


def has_internal_stop_with_stop_codons(seq, stop_codons):
    clean_seq = seq.translate(_DROP_MISSING_CHARS_TABLE).upper()
    clean_len = len(clean_seq)
    if (clean_len < 3) or (clean_len % 3 != 0):
        return False
    # Ignore terminal codon: terminal stop is not an internal stop.
    internal_stop_limit = clean_len - 3
    seq_find = clean_seq.find
    for stop_codon in stop_codons:
        pos = seq_find(stop_codon)
        while pos != -1:
            if (pos % 3 == 0) and (pos < internal_stop_limit):
                return True
            pos = seq_find(stop_codon, pos + 1)
    return False


def has_internal_stop(seq, codontable):
    stop_codons = get_stop_codons(codontable=codontable)
    return has_internal_stop_with_stop_codons(seq=seq, stop_codons=stop_codons)


def is_ambiguous_codon(codon):
    if any(ch in MISSING_CHARS for ch in codon):
        return False
    return any(ch not in UNAMBIGUOUS_NT for ch in codon)


def sequence_ambiguous_codon_counts(seq):
    ambiguous = 0
    evaluable = 0
    seq_len = len(seq)
    codon_class_cache = _AMBIGUOUS_CODON_CLASS_CACHE
    for i in range(0, seq_len - 2, 3):
        codon = seq[i:i + 3]
        codon_class = codon_class_cache.get(codon)
        if codon_class is None:
            ch0 = codon[0]
            ch1 = codon[1]
            ch2 = codon[2]
            if (ch0 in MISSING_CHARS) or (ch1 in MISSING_CHARS) or (ch2 in MISSING_CHARS):
                codon_class = (0, 0)
            else:
                codon_class = (
                    1,
                    int((ch0 not in UNAMBIGUOUS_NT) or (ch1 not in UNAMBIGUOUS_NT) or (ch2 not in UNAMBIGUOUS_NT)),
                )
            codon_class_cache[codon] = codon_class
        evaluable += codon_class[0]
        ambiguous += codon_class[1]
    return ambiguous, evaluable


def get_duplicate_ids(records):
    counts = Counter(record.id for record in records)
    return sorted([seq_id for seq_id, count in counts.items() if count > 1])


def summarize_single_sequence(seq_id, seq, stop_codons):
    ambiguous, evaluable = sequence_ambiguous_codon_counts(seq)
    return (
        seq_id,
        (len(seq) % 3 != 0),
        is_gap_only_sequence(seq),
        has_internal_stop_with_stop_codons(seq=seq, stop_codons=stop_codons),
        ambiguous,
        evaluable,
    )


def summarize_single_record(record, stop_codons):
    return summarize_single_sequence(
        seq_id=record.id,
        seq=str(record.seq),
        stop_codons=stop_codons,
    )


def summarize_single_payload(payload, stop_codons):
    seq_id, seq = payload
    return summarize_single_sequence(seq_id=seq_id, seq=seq, stop_codons=stop_codons)


def summarize_records_process_parallel(payloads, stop_codons, threads):
    worker = partial(summarize_single_payload, stop_codons=stop_codons)
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def summarize_records(records, codontable, threads=1):
    if len(records) <= 1:
        aligned = True
    else:
        first_len = len(records[0].seq)
        aligned = all(len(record.seq) == first_len for record in records[1:])
    duplicate_ids = get_duplicate_ids(records)
    worker_threads = resolve_threads(threads=threads)
    stop_codons = get_stop_codons(codontable=codontable)
    per_record = None
    if (worker_threads > 1) and (len(records) >= _PROCESS_PARALLEL_MIN_RECORDS):
        try:
            payloads = [(record.id, str(record.seq)) for record in records]
            per_record = summarize_records_process_parallel(
                payloads=payloads,
                stop_codons=stop_codons,
                threads=worker_threads,
            )
        except (OSError, PermissionError):
            sys.stderr.write('Process-based parallelism unavailable; falling back to threads.\n')
    if per_record is None:
        worker = partial(summarize_single_record, stop_codons=stop_codons)
        per_record = parallel_map_ordered(items=records, worker=worker, threads=worker_threads)

    non_triplet_ids = [entry[0] for entry in per_record if entry[1]]
    gap_only_ids = [entry[0] for entry in per_record if entry[2]]
    internal_stop_ids = [entry[0] for entry in per_record if entry[3]]
    ambiguous_by_seq = dict()
    total_ambiguous = 0
    total_evaluable = 0
    for entry in per_record:
        ambiguous_by_seq[entry[0]] = entry[4]
        total_ambiguous += entry[4]
        total_evaluable += entry[5]
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
