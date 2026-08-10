from functools import partial

from cdskit.util import (
    compile_safe_regex,
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_seqtype,
)


def record_matches_seqname(record, seqname_pattern):
    if hasattr(seqname_pattern, "fullmatch"):
        return seqname_pattern.fullmatch(record.id) is not None
    compiled = compile_safe_regex(
        seqname_pattern,
        label="regex in --seq_name_regex",
    )
    return compiled.fullmatch(record.id) is not None


def compile_seqname_regex(seqname_pattern):
    return compile_safe_regex(
        seqname_pattern,
        label="regex in --seq_name_regex",
    )


def format_printseq_lines(record, show_seqname):
    lines = []
    if show_seqname:
        lines.append(">" + record.id)
    lines.append(str(record.seq))
    return lines


def printseq_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_seqtype(
        records=records,
        seqtype=getattr(args, "seqtype", "auto"),
        label="--seq_file",
    )
    compiled_pattern = compile_seqname_regex(args.seqname)
    threads = resolve_threads(getattr(args, "threads", 1))
    match_flags = parallel_map_ordered(
        items=records,
        worker=partial(record_matches_seqname, seqname_pattern=compiled_pattern),
        threads=threads,
    )
    for record, matched in zip(records, match_flags, strict=False):
        if matched:
            for line in format_printseq_lines(record, args.show_seqname):
                print(line)
