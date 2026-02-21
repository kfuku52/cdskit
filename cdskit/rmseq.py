import re
import math
from functools import partial

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, stop_if_not_dna, write_seqs


def normalize_problematic_chars(problematic_chars):
    if problematic_chars is None:
        return frozenset()

    normalized = set()
    if isinstance(problematic_chars, str):
        source_iter = [problematic_chars]
    else:
        source_iter = problematic_chars

    for token in source_iter:
        if token is None:
            continue
        for ch in str(token):
            normalized.add(ch.upper())
    return frozenset(normalized)


def problematic_rate(seq, problematic_chars):
    seq_len = len(seq)
    if seq_len == 0:
        return 0.0
    normalized_chars = normalize_problematic_chars(problematic_chars)
    if len(normalized_chars) == 0:
        return 0.0
    num_problematic_char = sum(1 for ch in str(seq).upper() if ch in normalized_chars)
    return num_problematic_char / seq_len


def should_remove_record(record, seqname_pattern, problematic_percent, problematic_chars):
    if hasattr(seqname_pattern, 'fullmatch'):
        matched = seqname_pattern.fullmatch(record.id) is not None
    else:
        matched = re.fullmatch(seqname_pattern, record.id) is not None
    if matched:
        return True

    if problematic_percent > 0:
        rate_problematic = problematic_rate(record.seq, problematic_chars)
        if rate_problematic >= (problematic_percent / 100):
            return True

    return False


def should_keep_record(record, seqname_pattern, problematic_percent, problematic_chars):
    return not should_remove_record(
        record=record,
        seqname_pattern=seqname_pattern,
        problematic_percent=problematic_percent,
        problematic_chars=problematic_chars,
    )


def compile_seqname_regex(seqname_pattern):
    try:
        return re.compile(seqname_pattern)
    except re.error as e:
        txt = 'Invalid regex in --seqname: {} ({})'
        raise Exception(txt.format(seqname_pattern, e))


def validate_problematic_percent(problematic_percent):
    if not math.isfinite(problematic_percent):
        txt = '--problematic_percent should be finite within [0, 100], but got {}. Exiting.\n'
        raise Exception(txt.format(problematic_percent))
    if (problematic_percent < 0) or (problematic_percent > 100):
        txt = '--problematic_percent should be within [0, 100], but got {}. Exiting.\n'
        raise Exception(txt.format(problematic_percent))


def validate_problematic_chars(problematic_chars, problematic_percent):
    normalized_chars = normalize_problematic_chars(problematic_chars)
    if (problematic_percent > 0) and (len(normalized_chars) == 0):
        txt = '--problematic_char must contain at least one character when --problematic_percent > 0. Exiting.\n'
        raise Exception(txt)
    return normalized_chars


def rmseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seqfile')
    validate_problematic_percent(args.problematic_percent)
    normalized_problematic_chars = validate_problematic_chars(
        problematic_chars=args.problematic_char,
        problematic_percent=args.problematic_percent,
    )
    compiled_pattern = compile_seqname_regex(args.seqname)
    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        should_keep_record,
        seqname_pattern=compiled_pattern,
        problematic_percent=args.problematic_percent,
        problematic_chars=normalized_problematic_chars,
    )
    keep_flags = parallel_map_ordered(items=records, worker=worker, threads=threads)
    new_records = [record for record, keep in zip(records, keep_flags) if keep]
    write_seqs(records=new_records, outfile=args.outfile, outseqformat=args.outseqformat)
