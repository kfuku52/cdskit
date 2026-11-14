import numpy as np
import sys
from functools import partial
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.translate import translate_sequence_codes
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

GAP_ONLY_CHARS = frozenset("-NXnxn?.")
GAP_ONLY_CHARS_BYTES = np.array(
    [ch.encode("ascii") for ch in sorted(GAP_ONLY_CHARS)], dtype="S1"
)
AA_MISSING_CODES = np.array([ord("-"), ord("?"), ord("X"), ord("*")], dtype=np.uint8)
_GAP_TRANSLATION = bytes(int(chr(code) in GAP_ONLY_CHARS) for code in range(256))
_AA_NON_MISSING = np.ones(256, dtype=bool)
_AA_NON_MISSING[AA_MISSING_CODES] = False
_AA_NON_MISSING.flags.writeable = False


def codon_is_gap_like(codon):
    for ch in codon:
        if ch not in GAP_ONLY_CHARS:
            return False
    return True


def resolve_nail_value(nail, num_records):
    if nail == "all":
        nail_value = num_records
        txt = "--nail all was specified. Set to {:,}\n"
        sys.stderr.write(txt.format(nail_value))
        return nail_value

    nail_value = int(nail)
    if nail_value <= 0:
        txt = '--nail should be a positive integer or "all", but got {}. Exiting.\n'
        raise ValueError(txt.format(nail))
    if nail_value > num_records:
        txt = "--nail ({:,}) is greater than the number of input sequences ({:,}). Decreased to {:,}\n"
        sys.stderr.write(txt.format(nail_value, num_records, num_records))
        return num_records
    return nail_value


def build_codon_gap_like_matrix(records):
    num_records = len(records)
    if num_records == 0:
        return np.zeros((0, 0), dtype=bool)
    seq_len = len(records[0].seq)
    if seq_len == 0:
        return np.zeros((num_records, 0), dtype=bool)

    seq_bytes = "".join([str(record.seq) for record in records]).encode("ascii")
    nt_matrix = np.frombuffer(
        seq_bytes.translate(_GAP_TRANSLATION), dtype=bool
    ).reshape(num_records, seq_len)
    codon_matrix = nt_matrix.reshape(num_records, seq_len // 3, 3)
    return codon_matrix.all(axis=2)


def build_non_missing_site(records, codontable, threads):
    del threads
    num_records = len(records)
    if num_records == 0:
        return np.zeros(0, dtype=int)
    seq_len = len(records[0].seq)
    if seq_len == 0:
        return np.zeros(0, dtype=int)
    amino_acids = translate_sequence_codes(
        seq_str="".join(str(record.seq) for record in records),
        codontable=codontable,
    ).reshape(num_records, seq_len // 3)
    return _AA_NON_MISSING[amino_acids].sum(axis=0)


def select_codon_site_indices(
    non_missing_site,
    nail_value,
    max_len,
    prevent_gap_only,
    codon_gap_like_matrix,
    original_records,
):
    if not prevent_gap_only:
        selected = np.flatnonzero(non_missing_site >= nail_value)
        _report_removed_sites(max_len - selected.size, max_len)
        return selected

    # Each row first becomes usable at the highest occupancy of a column
    # containing a non-gap-like codon in that row. Reuse these thresholds
    # and the column counts while reproducing the relaxation diagnostics.
    row_thresholds = np.array(
        [non_missing_site.max(where=~row, initial=0) for row in codon_gap_like_matrix]
    )
    below_nail = np.bincount(non_missing_site, minlength=nail_value).cumsum()
    final_nail = max(1, min(nail_value, int(row_thresholds.min(initial=nail_value))))
    for current_nail in range(nail_value, final_nail - 1, -1):
        removed = max_len - non_missing_site.size + below_nail[current_nail - 1]
        _report_removed_sites(removed, max_len)
        for i in np.flatnonzero(row_thresholds < current_nail):
            txt = "A gap-only sequence was generated with --nail {}. Will try --nail {}: {}\n"
            sys.stderr.write(
                txt.format(current_nail, current_nail - 1, original_records[i].name)
            )
    return np.flatnonzero(non_missing_site >= final_nail)


def _report_removed_sites(removed, total):
    sys.stderr.write(f"{removed:,} out of {total:,} codon sites will be removed.\n")


def codon_sites_to_nucleotide_ranges(codon_sites):
    if len(codon_sites) == 0:
        return list()
    ranges = list()
    run_start = codon_sites[0]
    run_end = codon_sites[0]
    for codon_site in codon_sites[1:]:
        if codon_site == run_end + 1:
            run_end = codon_site
            continue
        ranges.append((run_start * 3, (run_end + 1) * 3))
        run_start = codon_site
        run_end = codon_site
    ranges.append((run_start * 3, (run_end + 1) * 3))
    return ranges


def build_hammer_output_record(record, selected_nucleotide_ranges):
    seq_str = str(record.seq)
    new_seq = "".join([seq_str[start:end] for start, end in selected_nucleotide_ranges])
    return SeqRecord(
        seq=Seq(new_seq),
        id=record.id,
        name=record.name,
        description=record.description,
    )


def hammer_main(args):
    original_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=original_records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)
    if len(original_records) == 0:
        write_seqs(
            records=original_records,
            outfile=args.outfile,
            outseqformat=args.outseqformat,
        )
        return
    threads = resolve_threads(getattr(args, "threads", 1))
    stop_if_not_multiple_of_three(original_records)
    stop_if_not_aligned(original_records)
    nail_value = resolve_nail_value(args.nail, len(original_records))
    non_missing_site = build_non_missing_site(
        records=original_records,
        codontable=args.codontable,
        threads=threads,
    )
    max_len = non_missing_site.shape[0]
    codon_gap_like_matrix = None
    if args.prevent_gap_only:
        codon_gap_like_matrix = build_codon_gap_like_matrix(original_records)
    selected_non_missing_idx = select_codon_site_indices(
        non_missing_site=non_missing_site,
        nail_value=nail_value,
        max_len=max_len,
        prevent_gap_only=args.prevent_gap_only,
        codon_gap_like_matrix=codon_gap_like_matrix,
        original_records=original_records,
    )
    selected_nucleotide_ranges = codon_sites_to_nucleotide_ranges(
        selected_non_missing_idx.tolist()
    )
    worker = partial(
        build_hammer_output_record,
        selected_nucleotide_ranges=selected_nucleotide_ranges,
    )
    records = parallel_map_ordered(
        items=original_records,
        worker=worker,
        threads=threads,
    )
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
