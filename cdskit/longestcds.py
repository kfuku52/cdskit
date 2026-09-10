import sys
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from bisect import bisect_right
from functools import partial

import Bio.Data.CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.atomicio import atomic_output_paths
from cdskit.codonreport import validate_codon_output_paths, write_codon_report
from cdskit.codonutil import analyze_codon, codon_matches_stop_set

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    should_use_process_pool,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    write_seqs,
)

_CODON_TABLE_CACHE: dict = {}
_CODON_SCAN_CACHE: dict = {}
_REVCOMP_TABLE = str.maketrans(
    "ACGTRYMKWSBDHVN",
    "TGCAYRKMWSVHDBN",
)


@dataclass
class CdsCandidate:
    strand: str
    frame: int
    start_1based: int
    end_1based: int
    has_start: bool
    has_stop: bool
    category: str
    nt_len: int
    aa_len: int
    start_idx: int
    end_idx: int
    sort_key: tuple
    output_seq: str


def get_start_stop_codons(codontable):
    cached = _CODON_TABLE_CACHE.get(codontable)
    if cached is not None:
        return cached
    try:
        table = Bio.Data.CodonTable.unambiguous_dna_by_id[int(codontable)]
    except (KeyError, TypeError, ValueError):
        table = Bio.Data.CodonTable.unambiguous_dna_by_name[str(codontable)]
    codons = (
        set(table.start_codons),
        set(table.stop_codons) - set(table.forward_table),
    )
    _CODON_TABLE_CACHE[codontable] = codons
    return codons


def get_scan_codons(codontable):
    cached = _CODON_SCAN_CACHE.get(codontable)
    if cached is not None:
        return cached
    start_codons, stop_codons = get_start_stop_codons(codontable=codontable)
    scan_codons = (tuple(sorted(start_codons)), tuple(sorted(stop_codons)))
    _CODON_SCAN_CACHE[codontable] = scan_codons
    return scan_codons


def frame_end_index(seq_len, frame_offset):
    return seq_len - ((seq_len - frame_offset) % 3)


def original_coordinates(strand, seq_len, start_idx, end_idx):
    if strand == "+":
        return start_idx + 1, end_idx
    # Coordinates on original (plus) strand while preserving coding orientation via strand symbol.
    return seq_len - end_idx + 1, seq_len - start_idx


def build_candidate_sort_key(
    category, nt_len, strand, has_stop, has_start, frame, start_1based
):
    rank = 2 if category == "complete" else 1 if category == "partial" else 0
    strand_order = 0 if strand == "+" else 1
    return (
        rank,
        nt_len,
        -strand_order,
        -int(has_stop),
        -int(has_start),
        -frame,
        -start_1based,
    )


def update_best_candidate(
    best_sort_key,
    best_fields,
    strand,
    seq_len,
    frame_offset,
    start_idx,
    end_idx,
    has_start,
    has_stop,
    category,
):
    nt_len = end_idx - start_idx
    start_1based, end_1based = original_coordinates(
        strand=strand,
        seq_len=seq_len,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    frame = frame_offset + 1
    sort_key = build_candidate_sort_key(
        category=category,
        nt_len=nt_len,
        strand=strand,
        has_stop=has_stop,
        has_start=has_start,
        frame=frame,
        start_1based=start_1based,
    )
    if (best_sort_key is None) or (sort_key > best_sort_key):
        return sort_key, (
            strand,
            frame,
            start_1based,
            end_1based,
            has_start,
            has_stop,
            category,
            nt_len,
            start_idx,
            end_idx,
        )
    return best_sort_key, best_fields


def build_candidate_from_fields(candidate_fields, sort_key):
    return CdsCandidate(
        strand=candidate_fields[0],
        frame=candidate_fields[1],
        start_1based=candidate_fields[2],
        end_1based=candidate_fields[3],
        has_start=candidate_fields[4],
        has_stop=candidate_fields[5],
        category=candidate_fields[6],
        nt_len=candidate_fields[7],
        aa_len=candidate_fields[7] // 3,
        start_idx=candidate_fields[8],
        end_idx=candidate_fields[9],
        sort_key=sort_key,
        output_seq="",
    )


def collect_start_stop_positions_by_frame(strand_seq, start_codons, stop_codons):
    start_positions: list[list[int]] = [[], [], []]
    stop_positions: list[list[int]] = [[], [], []]
    if any(ch not in "ACGT" for ch in strand_seq):
        stops = frozenset(stop_codons)
        for pos in range(len(strand_seq) - 2):
            codon = strand_seq[pos : pos + 3]
            if codon in start_codons:
                start_positions[pos % 3].append(pos)
            if codon_matches_stop_set(codon, stops):
                stop_positions[pos % 3].append(pos)
        return start_positions, stop_positions
    seq_find = strand_seq.find

    for codon in start_codons:
        pos = seq_find(codon)
        while pos != -1:
            start_positions[pos % 3].append(pos)
            pos = seq_find(codon, pos + 1)

    for codon in stop_codons:
        pos = seq_find(codon)
        while pos != -1:
            stop_positions[pos % 3].append(pos)
            pos = seq_find(codon, pos + 1)

    for frame_offset in range(3):
        if len(start_positions[frame_offset]) > 1:
            start_positions[frame_offset].sort()
        if len(stop_positions[frame_offset]) > 1:
            stop_positions[frame_offset].sort()
    return start_positions, stop_positions


def scan_start_based_best_candidate(
    strand,
    seq_len,
    frame_ends,
    start_positions_by_frame,
    stop_positions_by_frame,
    best_sort_key,
    best_fields,
):
    best_rank = -1 if best_sort_key is None else best_sort_key[0]
    best_len = -1 if best_sort_key is None else best_sort_key[1]
    for frame_offset in range(3):
        max_end = frame_ends[frame_offset]
        if max_end - frame_offset < 3:
            continue
        start_positions = start_positions_by_frame[frame_offset]
        if len(start_positions) == 0:
            continue
        stop_positions = stop_positions_by_frame[frame_offset]
        stop_idx = len(stop_positions) - 1
        nearest_stop_pos = -1
        for start_idx in reversed(start_positions):
            if (best_rank == 2) and ((max_end - start_idx) < best_len):
                continue
            while (stop_idx >= 0) and (stop_positions[stop_idx] > start_idx):
                nearest_stop_pos = stop_positions[stop_idx]
                stop_idx -= 1
            if nearest_stop_pos != -1:
                end_idx = nearest_stop_pos + 3
                nt_len = end_idx - start_idx
                if (best_rank == 2) and (nt_len < best_len):
                    continue
                best_sort_key, best_fields = update_best_candidate(
                    best_sort_key=best_sort_key,
                    best_fields=best_fields,
                    strand=strand,
                    seq_len=seq_len,
                    frame_offset=frame_offset,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    has_start=True,
                    has_stop=True,
                    category="complete",
                )
                best_rank = best_sort_key[0]
                best_len = best_sort_key[1]
            elif max_end - start_idx >= 3:
                if best_rank > 1:
                    continue
                nt_len = max_end - start_idx
                if (best_rank == 1) and (nt_len < best_len):
                    continue
                best_sort_key, best_fields = update_best_candidate(
                    best_sort_key=best_sort_key,
                    best_fields=best_fields,
                    strand=strand,
                    seq_len=seq_len,
                    frame_offset=frame_offset,
                    start_idx=start_idx,
                    end_idx=max_end,
                    has_start=True,
                    has_stop=False,
                    category="partial",
                )
                best_rank = best_sort_key[0]
                best_len = best_sort_key[1]
    return best_sort_key, best_fields


def scan_stop_free_best_candidate(
    strand,
    seq_len,
    frame_ends,
    stop_positions_by_frame,
    best_sort_key,
    best_fields,
):
    best_rank = -1 if best_sort_key is None else best_sort_key[0]
    best_len = -1 if best_sort_key is None else best_sort_key[1]
    for frame_offset in range(3):
        max_end = frame_ends[frame_offset]
        if max_end - frame_offset < 3:
            continue
        segment_start = frame_offset
        for stop_pos in stop_positions_by_frame[frame_offset]:
            if stop_pos - segment_start >= 3:
                if best_rank > 0:
                    segment_start = stop_pos + 3
                    continue
                nt_len = stop_pos - segment_start
                if (best_rank == 0) and (nt_len < best_len):
                    segment_start = stop_pos + 3
                    continue
                best_sort_key, best_fields = update_best_candidate(
                    best_sort_key=best_sort_key,
                    best_fields=best_fields,
                    strand=strand,
                    seq_len=seq_len,
                    frame_offset=frame_offset,
                    start_idx=segment_start,
                    end_idx=stop_pos,
                    has_start=False,
                    has_stop=False,
                    category="no_start",
                )
                best_rank = best_sort_key[0]
                best_len = best_sort_key[1]
            segment_start = stop_pos + 3
        if max_end - segment_start >= 3:
            if best_rank > 0:
                continue
            nt_len = max_end - segment_start
            if (best_rank == 0) and (nt_len < best_len):
                continue
            best_sort_key, best_fields = update_best_candidate(
                best_sort_key=best_sort_key,
                best_fields=best_fields,
                strand=strand,
                seq_len=seq_len,
                frame_offset=frame_offset,
                start_idx=segment_start,
                end_idx=max_end,
                has_start=False,
                has_stop=False,
                category="no_start",
            )
            best_rank = best_sort_key[0]
            best_len = best_sort_key[1]
    return best_sort_key, best_fields


def rank_value(candidate):
    if candidate.category == "complete":
        return 2
    if candidate.category == "partial":
        return 1
    return 0


def candidate_sort_key(candidate):
    return candidate.sort_key


def choose_best_candidate(seq_str, codontable, selection="complete-first"):
    if selection == "longest":
        candidate = max(
            iter_candidates(seq_str, codontable, include_sequence=False),
            key=lambda c: selection_key(c, selection),
            default=None,
        )
        if candidate is not None:
            sequence = seq_str.upper()
            if candidate.strand == "-":
                sequence = sequence.translate(_REVCOMP_TABLE)[::-1]
            candidate.output_seq = sequence[candidate.start_idx : candidate.end_idx]
        return candidate
    if selection != "complete-first":
        raise ValueError(f"Unknown ORF selection: {selection}")
    seq_upper = seq_str.upper()
    seq_len = len(seq_upper)
    if seq_len < 3:
        return None

    start_codons, stop_codons = get_scan_codons(codontable=codontable)
    best_sort_key = None
    best_fields = None
    strand_contexts = list()
    plus_frame_ends = (
        frame_end_index(seq_len=seq_len, frame_offset=0),
        frame_end_index(seq_len=seq_len, frame_offset=1),
        frame_end_index(seq_len=seq_len, frame_offset=2),
    )
    plus_start_positions, plus_stop_positions = collect_start_stop_positions_by_frame(
        strand_seq=seq_upper,
        start_codons=start_codons,
        stop_codons=stop_codons,
    )
    strand_contexts.append(
        ("+", plus_frame_ends, plus_start_positions, plus_stop_positions)
    )
    best_sort_key, best_fields = scan_start_based_best_candidate(
        strand="+",
        seq_len=seq_len,
        frame_ends=plus_frame_ends,
        start_positions_by_frame=plus_start_positions,
        stop_positions_by_frame=plus_stop_positions,
        best_sort_key=best_sort_key,
        best_fields=best_fields,
    )
    if (
        (best_sort_key is not None)
        and (best_sort_key[0] == 2)
        and (best_sort_key[1] == seq_len)
    ):
        candidate = build_candidate_from_fields(
            candidate_fields=best_fields, sort_key=best_sort_key
        )
        candidate.output_seq = seq_upper[candidate.start_idx : candidate.end_idx]
        return candidate

    reverse_complement = seq_upper.translate(_REVCOMP_TABLE)[::-1]
    minus_frame_ends = (
        frame_end_index(seq_len=seq_len, frame_offset=0),
        frame_end_index(seq_len=seq_len, frame_offset=1),
        frame_end_index(seq_len=seq_len, frame_offset=2),
    )
    minus_start_positions, minus_stop_positions = collect_start_stop_positions_by_frame(
        strand_seq=reverse_complement,
        start_codons=start_codons,
        stop_codons=stop_codons,
    )
    strand_contexts.append(
        ("-", minus_frame_ends, minus_start_positions, minus_stop_positions)
    )
    best_sort_key, best_fields = scan_start_based_best_candidate(
        strand="-",
        seq_len=seq_len,
        frame_ends=minus_frame_ends,
        start_positions_by_frame=minus_start_positions,
        stop_positions_by_frame=minus_stop_positions,
        best_sort_key=best_sort_key,
        best_fields=best_fields,
    )

    if best_fields is None:
        for strand, frame_ends, _, stop_positions in strand_contexts:
            best_sort_key, best_fields = scan_stop_free_best_candidate(
                strand=strand,
                seq_len=seq_len,
                frame_ends=frame_ends,
                stop_positions_by_frame=stop_positions,
                best_sort_key=best_sort_key,
                best_fields=best_fields,
            )

    if best_fields is None:
        return None
    candidate = build_candidate_from_fields(
        candidate_fields=best_fields, sort_key=best_sort_key
    )
    if candidate.strand == "+":
        candidate.output_seq = seq_upper[candidate.start_idx : candidate.end_idx]
    else:
        candidate.output_seq = reverse_complement[
            candidate.start_idx : candidate.end_idx
        ]
    return candidate


def selection_key(candidate, selection):
    if selection == "complete-first":
        return candidate.sort_key
    if selection == "longest":
        rank, length, *ties = candidate.sort_key
        return (length, rank, *ties)
    raise ValueError(f"Unknown ORF selection: {selection}")


def iter_candidates(seq_str, codontable, include_sequence=True):
    """Yield start-to-first-stop candidates and maximal stop-free segments.

    Dual-coding codons never establish complete boundaries without annotation.
    All six frames are enumerated, including no-start candidates in length mode.
    """
    sequence = seq_str.upper()
    starts, stops = get_scan_codons(codontable)
    for strand, oriented in (
        ("+", sequence),
        ("-", sequence.translate(_REVCOMP_TABLE)[::-1]),
    ):
        start_positions, stop_positions = collect_start_stop_positions_by_frame(
            oriented, starts, stops
        )
        for frame in range(3):
            end = frame_end_index(len(sequence), frame)
            boundaries = stop_positions[frame]
            for start in start_positions[frame]:
                next_index = bisect_right(boundaries, start)
                has_stop = next_index < len(boundaries)
                stop = boundaries[next_index] + 3 if has_stop else end
                yield make_candidate(
                    oriented,
                    strand,
                    frame,
                    start,
                    stop,
                    True,
                    has_stop,
                    include_sequence,
                )
            segment_start = frame
            for stop in [*boundaries, end]:
                if stop - segment_start >= 3:
                    yield make_candidate(
                        oriented,
                        strand,
                        frame,
                        segment_start,
                        stop,
                        False,
                        False,
                        include_sequence,
                    )
                segment_start = stop + 3


def make_candidate(
    sequence, strand, frame, start, end, has_start, has_stop, include_sequence=True
):
    category = (
        "complete" if has_start and has_stop else "partial" if has_start else "no_start"
    )
    key, fields = update_best_candidate(
        None,
        None,
        strand,
        len(sequence),
        frame,
        start,
        end,
        has_start,
        has_stop,
        category,
    )
    candidate = build_candidate_from_fields(fields, key)
    if include_sequence:
        candidate.output_seq = sequence[start:end]
    return candidate


def candidate_attribute_prefixes(sequence, codontable):
    """Count each frame once instead of rescanning nested candidate sequences."""
    prefixes = {}
    for strand, oriented in (
        ("+", sequence),
        ("-", sequence.translate(_REVCOMP_TABLE)[::-1]),
    ):
        for frame in range(3):
            counts = [array("Q", [0]) for _ in range(4)]
            for start in range(frame, len(oriented) - 2, 3):
                meaning = analyze_codon(oriented[start : start + 3], codontable)
                flags = (
                    meaning.possible_stop,
                    meaning.context_dependent,
                    meaning.missing,
                    meaning.ambiguous,
                )
                for prefix, flag in zip(counts, flags, strict=True):
                    prefix.append(prefix[-1] + int(flag))
            prefixes[strand, frame + 1] = counts
    return prefixes


def candidate_report(record, codontable, selection, input_order):
    sequence = str(record.seq).upper()
    candidates = sorted(
        iter_candidates(sequence, codontable, include_sequence=False),
        key=lambda c: selection_key(c, selection),
        reverse=True,
    )
    prefixes = candidate_attribute_prefixes(sequence, codontable) if candidates else {}
    rows = []
    best_primary = selection_key(candidates[0], selection)[:2] if candidates else None
    for index, candidate in enumerate(candidates):
        left = candidate.start_idx // 3
        right = candidate.end_idx // 3
        possible, context, missing, ambiguous = (
            prefix[right] - prefix[left]
            for prefix in prefixes[candidate.strand, candidate.frame]
        )
        fields = asdict(candidate)
        fields.pop("output_seq")
        if index == 0:
            selected_sequence = (
                sequence
                if candidate.strand == "+"
                else sequence.translate(_REVCOMP_TABLE)[::-1]
            )
            fields["output_seq"] = selected_sequence[
                candidate.start_idx : candidate.end_idx
            ]
        rows.append(
            {
                **fields,
                "sort_key": selection_key(candidate, selection),
                "rank": index + 1,
                "selected": index == 0,
                "tied_before_deterministic_order": selection_key(candidate, selection)[
                    :2
                ]
                == best_primary,
                "stop_evidence": "definite" if candidate.has_stop else "unconfirmed",
                "possible_stop_codons": possible,
                "context_dependent_codons": context,
                "missing_codons": missing,
                "ambiguous_codons": ambiguous,
            }
        )
    return {
        "seq_id": record.id,
        "input_order": input_order,
        "selection": selection,
        "candidates": rows,
        "selected_candidate_0based": 0 if rows else None,
    }


def choose_best_candidate_from_record(record, codontable, selection="complete-first"):
    return choose_best_candidate(
        seq_str=str(record.seq), codontable=codontable, selection=selection
    )


def choose_candidates_process_parallel(
    seq_strings, codontable, threads, selection="complete-first"
):
    worker = partial(choose_best_candidate, codontable=codontable, selection=selection)
    max_workers = min(threads, len(seq_strings))
    chunk_size = max(1, len(seq_strings) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, seq_strings, chunksize=chunk_size))


def build_output_record(record, candidate, annotate_seqname):
    if candidate is None:
        txt = "No CDS candidate found for {}. Output sequence is empty.\n"
        sys.stderr.write(txt.format(record.id))
        meta = "strand=NA frame=NA start=NA end=NA nt_len=0 aa_len=0 category=none"
        if annotate_seqname:
            description = f"{record.id} {meta}"
        else:
            description = record.description
        return SeqRecord(
            seq=Seq(""),
            id=record.id,
            name=record.name,
            description=description,
        )

    meta = (
        f"strand={candidate.strand} frame={candidate.frame} "
        f"start={candidate.start_1based} end={candidate.end_1based} "
        f"nt_len={candidate.nt_len} aa_len={candidate.aa_len} category={candidate.category}"
    )
    if candidate.output_seq != "":
        output_seq = candidate.output_seq
    else:
        seq_upper = str(record.seq).upper()
        if candidate.strand == "+":
            output_seq = seq_upper[candidate.start_idx : candidate.end_idx]
        else:
            reverse_complement = seq_upper.translate(_REVCOMP_TABLE)[::-1]
            output_seq = reverse_complement[candidate.start_idx : candidate.end_idx]
    if annotate_seqname:
        description = f"{record.id} {meta}"
    else:
        description = record.description
    return SeqRecord(
        seq=Seq(output_seq),
        id=record.id,
        name=record.name,
        description=description,
    )


def longestcds_main(args):
    validate_codon_output_paths(args.seqfile, args.outfile, getattr(args, "report", ""))
    annotate_seqname = bool(getattr(args, "annotate_seqname", False))
    threads = resolve_threads(getattr(args, "threads", 1))
    selection = getattr(args, "selection", "complete-first")
    report = getattr(args, "report", "")
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)
    candidates = None
    if should_use_process_pool(records=records, threads=threads):
        try:
            seq_strings = [str(record.seq) for record in records]
            candidates = choose_candidates_process_parallel(
                seq_strings=seq_strings,
                codontable=args.codontable,
                threads=threads,
                selection=selection,
            )
        except (OSError, PermissionError):
            sys.stderr.write(
                "Process-based parallelism unavailable; falling back to threads.\n"
            )
    if candidates is None:
        worker = partial(
            choose_best_candidate_from_record,
            codontable=args.codontable,
            selection=selection,
        )
        candidates = parallel_map_ordered(items=records, worker=worker, threads=threads)

    output_records = list()
    for record, candidate in zip(records, candidates, strict=False):
        output_records.append(
            build_output_record(
                record=record,
                candidate=candidate,
                annotate_seqname=annotate_seqname,
            )
        )

    report_records = (
        [
            candidate_report(record, args.codontable, selection, index + 1)
            for index, record in enumerate(records)
        ]
        if report
        else []
    )
    outputs = [path for path in (args.outfile, report) if path not in ("", "-")]
    with atomic_output_paths(outputs) as temporary_paths:
        staged = dict(zip(outputs, temporary_paths, strict=True))
        write_seqs(
            records=output_records,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        write_codon_report(
            staged.get(report, report), "longestorf", args.codontable, report_records
        )
