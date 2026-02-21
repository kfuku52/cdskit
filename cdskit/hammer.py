import numpy
import sys
from functools import partial
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from cdskit.translate import translate_sequence_string

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

GAP_ONLY_CHARS = frozenset('-NXnx')
GAP_ONLY_CHARS_BYTES = numpy.array([ch.encode('ascii') for ch in sorted(GAP_ONLY_CHARS)], dtype='S1')
AA_MISSING_CHARS = numpy.array([b'-', b'?', b'X', b'*'], dtype='S1')


def codon_is_gap_like(codon):
    for ch in codon:
        if ch not in GAP_ONLY_CHARS:
            return False
    return True


def resolve_nail_value(nail, num_records):
    if nail == 'all':
        nail_value = num_records
        txt = '--nail all was specified. Set to {:,}\n'
        sys.stderr.write(txt.format(nail_value))
        return nail_value

    nail_value = int(nail)
    if nail_value <= 0:
        txt = '--nail should be a positive integer or "all", but got {}. Exiting.\n'
        raise Exception(txt.format(nail))
    if nail_value > num_records:
        txt = '--nail ({:,}) is greater than the number of input sequences ({:,}). Decreased to {:,}\n'
        sys.stderr.write(txt.format(nail_value, num_records, num_records))
        return num_records
    return nail_value


def build_codon_gap_like_matrix(records):
    num_records = len(records)
    if num_records == 0:
        return numpy.zeros((0, 0), dtype=bool)
    seq_len = len(records[0].seq)
    if seq_len == 0:
        return numpy.zeros((num_records, 0), dtype=bool)

    seq_bytes = ''.join([str(record.seq) for record in records]).encode('ascii')
    nt_matrix = numpy.frombuffer(seq_bytes, dtype='S1').reshape(num_records, seq_len)
    codon_matrix = nt_matrix.reshape(num_records, seq_len // 3, 3)
    return numpy.isin(codon_matrix, GAP_ONLY_CHARS_BYTES).all(axis=2)


def translate_record_to_aa_string(record, codontable):
    return translate_sequence_string(
        seq_str=str(record.seq),
        codontable=codontable,
        to_stop=False,
    )


def build_non_missing_site(records, codontable, threads):
    worker = partial(translate_record_to_aa_string, codontable=codontable)
    aa_strings = parallel_map_ordered(items=records, worker=worker, threads=threads)
    aa_len = len(aa_strings[0])
    aa_bytes = b''.join([aa_seq.encode('ascii') for aa_seq in aa_strings])
    aa_matrix = numpy.frombuffer(aa_bytes, dtype='S1').reshape(len(aa_strings), aa_len)
    missing_site = numpy.isin(aa_matrix, AA_MISSING_CHARS).sum(axis=0)
    return len(records) - missing_site


def select_codon_site_indices(non_missing_site, nail_value, max_len, prevent_gap_only, codon_gap_like_matrix, original_records):
    selected_non_missing_idx = None
    last_non_missing_idx = numpy.array([], dtype=int)

    for current_nail in range(nail_value, 0, -1):
        non_missing_idx = numpy.flatnonzero(non_missing_site >= current_nail)
        last_non_missing_idx = non_missing_idx
        num_removed_site = max_len - non_missing_idx.shape[0]
        sys.stderr.write('{:,} out of {:,} codon sites will be removed.\n'.format(num_removed_site, max_len))
        if prevent_gap_only:
            gap_only_mask = numpy.all(codon_gap_like_matrix[:, non_missing_idx], axis=1)
            if numpy.any(gap_only_mask):
                for i in numpy.flatnonzero(gap_only_mask):
                    txt = 'A gap-only sequence was generated with --nail {}. Will try --nail {}: {}\n'
                    sys.stderr.write(txt.format(current_nail, current_nail - 1, original_records[i].name))
                continue
        selected_non_missing_idx = non_missing_idx
        break

    if selected_non_missing_idx is None:
        return last_non_missing_idx
    return selected_non_missing_idx


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
    new_seq = ''.join([seq_str[start:end] for start, end in selected_nucleotide_ranges])
    return SeqRecord(
        seq=Seq(new_seq),
        id=record.id,
        name=record.name,
        description=record.description,
    )


def hammer_main(args):
    original_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=original_records, label='--seqfile')
    stop_if_invalid_codontable(args.codontable)
    if len(original_records) == 0:
        write_seqs(records=original_records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    threads = resolve_threads(getattr(args, 'threads', 1))
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
    selected_nucleotide_ranges = codon_sites_to_nucleotide_ranges(selected_non_missing_idx.tolist())
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
