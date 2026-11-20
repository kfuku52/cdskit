import sys
import re
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from bisect import bisect_left, bisect_right
from functools import lru_cache, partial
from itertools import islice

import Bio.Data.CodonTable
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np

from cdskit.atomicio import atomic_output_paths, validate_distinct_paths
from cdskit.codonarray import codon_features
from cdskit.codonreport import validate_codon_output_paths, write_codon_report
from cdskit.codonutil import (
    analyze_codon,
    definite_stop_patterns,
    get_codon_table_components,
)

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
# Below these bounds, repeated C str.find calls avoid NumPy's setup cost.
_POSITION_SCAN_MIN_LENGTH = 16384
_POSITION_SCAN_MIN_PATTERNS = 8
_POSITION_SCAN_BLOCK_SIZE = 65536
_BEST_BLOCK_SCAN_MIN_LENGTH = 262_144
_CANDIDATE_PREFIX_BLOCK_NT = 3 * 16384
_PARTIAL_SELECTIONS = {"partial-5prime", "partial-3prime", "partial-both"}
_REVCOMP_TABLE = str.maketrans(
    "ACGTRYMKWSBDHVN",
    "TGCAYRKMWSVHDBN",
)
_CONCRETE_DNA = re.compile(r"[ACGT]*")
ALT_ORF_SUFFIX = r"\.cdskit-orf-[0-9]+$"


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


def _orf_scan_codons(codontable, start_policy):
    starts, stops = get_scan_codons(codontable)
    if start_policy == "table":
        return starts, stops
    if start_policy == "atg-only":
        return (("ATG",) if "ATG" in starts else ()), stops
    raise ValueError(f"Unknown ORF start policy: {start_policy}")


def _orf_directions(strand):
    if strand not in ("both", "forward", "reverse"):
        raise ValueError(f"Unknown ORF strand: {strand}")
    if strand != "reverse":
        yield "+"
    if strand != "forward":
        yield "-"


def _oriented_slice(sequence, direction, start, end):
    if direction == "+":
        return sequence[start:end]
    length = len(sequence)
    return sequence[length - end : length - start].translate(_REVCOMP_TABLE)[::-1]


def _oriented_sequences(sequence, strand):
    for direction in _orf_directions(strand):
        yield (
            direction,
            sequence
            if direction == "+"
            else _oriented_slice(sequence, direction, 0, len(sequence)),
        )


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


@lru_cache(maxsize=128)
def _position_scan_lookup(start_codons, stop_codons):
    alphabet = "ACGTRYSWKMBDHVNX"
    values = {base: index + 1 for index, base in enumerate(alphabet)}
    encode = np.zeros(256, dtype=np.uint16)
    for base, value in values.items():
        encode[ord(base)] = value
    lookup = np.zeros(1 << 15, dtype=np.uint8)
    for patterns, flag in (
        (start_codons, 1),
        (definite_stop_patterns(frozenset(stop_codons)), 2),
    ):
        for codon in patterns:
            key = (values[codon[0]] << 10) | (values[codon[1]] << 5) | values[codon[2]]
            lookup[key] |= flag
    return encode, lookup


def _position_flag_blocks(sequence, direction, start_codons, stop_codons):
    encode, lookup = _position_scan_lookup(start_codons, stop_codons)
    for left in range(0, len(sequence) - 2, _POSITION_SCAN_BLOCK_SIZE):
        # Include the two following bases, but classify each triplet start only
        # once. Replacing each non-ASCII character preserves Python offsets.
        right = min(len(sequence), left + _POSITION_SCAN_BLOCK_SIZE + 2)
        data = _oriented_slice(sequence, direction, left, right).encode(
            "ascii", "replace"
        )
        bases = encode[np.frombuffer(data, dtype=np.uint8)]
        keys = (bases[:-2] << 10) | (bases[1:-1] << 5) | bases[2:]
        yield left, lookup[keys]


def _collect_start_stop_positions_blocked(strand_seq, start_codons, stop_codons):
    start_positions: list[list[int]] = [[], [], []]
    stop_positions: list[list[int]] = [[], [], []]
    for left, flags in _position_flag_blocks(
        strand_seq, "+", start_codons, stop_codons
    ):
        for frame in range(3):
            offset = (frame - left) % 3
            frame_flags = flags[offset::3]
            start_positions[frame].extend(
                (np.flatnonzero(frame_flags & 1) * 3 + left + offset).tolist()
            )
            stop_positions[frame].extend(
                (np.flatnonzero(frame_flags & 2) * 3 + left + offset).tolist()
            )
    return start_positions, stop_positions


def _has_block_scan_patterns(start_codons, stop_codons):
    for starts, stops in tuple(_CODON_SCAN_CACHE.values()):
        if start_codons is starts and stop_codons is stops:
            return (
                len(starts) + len(definite_stop_patterns(frozenset(stops)))
                >= _POSITION_SCAN_MIN_PATTERNS
            )
    return False


def collect_start_stop_positions_by_frame(strand_seq, start_codons, stop_codons):
    if (
        type(strand_seq) is str
        and len(strand_seq) >= _POSITION_SCAN_MIN_LENGTH
        and _has_block_scan_patterns(start_codons, stop_codons)
    ):
        # Only canonical table patterns use the LUT. External patterns retain
        # literal find semantics, including duplicates and non-triplet starts.
        return _collect_start_stop_positions_blocked(
            strand_seq, start_codons, stop_codons
        )
    start_positions: list[list[int]] = [[], [], []]
    stop_positions: list[list[int]] = [[], [], []]
    seq_find = strand_seq.find

    for codon in start_codons:
        pos = seq_find(codon)
        while pos != -1:
            start_positions[pos % 3].append(pos)
            pos = seq_find(codon, pos + 1)

    for codon in definite_stop_patterns(frozenset(stop_codons)):
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


def _choose_without_stops(sequence, start_codons, strand):
    length = len(sequence)
    full_frame_length = frame_end_index(length, 0)
    best = None
    for direction in _orf_directions(strand):
        if direction == "+":
            motifs = start_codons
            hits = (sequence.find(codon) for codon in motifs)
            first = min(
                (position for position in hits if position >= 0), default=length
            )
        else:
            motifs = tuple(
                codon.translate(_REVCOMP_TABLE)[::-1] for codon in start_codons
            )
            hits = (sequence.rfind(codon) for codon in motifs)
            first = min(
                (length - position - 3 for position in hits if position >= 0),
                default=length,
            )
        # The first start gives the longest possible length on this strand.
        # At most two following positions can tie after rounding to codons.
        for start in range(first, min(first + 3, length - 2)):
            left = start if direction == "+" else length - start - 3
            if sequence[left : left + 3] not in motifs:
                continue
            frame = start % 3
            candidate = make_candidate(
                sequence,
                direction,
                frame,
                start,
                frame_end_index(length, frame),
                True,
                False,
                include_sequence=False,
            )
            if best is None or candidate.sort_key > best.sort_key:
                best = candidate
        if best is not None and best.nt_len == full_frame_length:
            break
    if best is None:
        direction = "-" if strand == "reverse" else "+"
        best = make_candidate(
            sequence,
            direction,
            0,
            0,
            full_frame_length,
            False,
            False,
            include_sequence=False,
        )
    best.output_seq = _oriented_slice(
        sequence, best.strand, best.start_idx, best.end_idx
    )
    return best


def _has_dense_orf_events(sequence, starts, stops, strand):
    # Small samples avoid the vector setup for long runs of missing bases or
    # low-complexity sequence that contain very few start/stop matches.
    width = 2048
    middle = len(sequence) // 2
    samples = (sequence[:width], sequence[middle : middle + width], sequence[-width:])
    for direction in _orf_directions(strand):
        motifs = starts + stops
        if direction == "-":
            motifs = tuple(codon.translate(_REVCOMP_TABLE)[::-1] for codon in motifs)
        if sum(sample.count(codon) for sample in samples for codon in motifs) >= 48:
            return True
    return False


def _best_block_interval(best, direction, length, frame, starts, ends, *, complete):
    widths = ends - starts
    if direction == "+":
        index = int(widths.argmax())
    else:
        # The last equally long hit has the smaller original reverse start.
        index = len(widths) - 1 - int(widths[::-1].argmax())
    if widths[index] < (6 if complete else 3):
        return best
    return update_best_candidate(
        best[0],
        best[1],
        direction,
        length,
        frame,
        int(starts[index]),
        int(ends[index]),
        complete,
        complete,
        "complete" if complete else "no_start",
    )


def _blockwise_strand_candidates(sequence, direction, starts, stops, best, no_start):
    length = len(sequence)
    previous_stop = [-3, -2, -1]
    pending_start: list[int | None] = [None, None, None]
    for left, flags in _position_flag_blocks(sequence, direction, starts, stops):
        for frame in range(3):
            offset = (frame - left) % 3
            hits = flags[offset::3]
            start_positions = np.flatnonzero(hits & 1) * 3 + left + offset
            stop_positions = np.flatnonzero(hits & 2) * 3 + left + offset
            if not len(stop_positions):
                if pending_start[frame] is None and len(start_positions):
                    pending_start[frame] = int(start_positions[0])
                continue

            preceding = np.empty_like(stop_positions)
            preceding[0] = previous_stop[frame]
            preceding[1:] = stop_positions[:-1]
            # The earliest start after the preceding stop is the longest
            # candidate for this stop. An incoming start can span blocks.
            nearest = np.append(start_positions, length)[
                start_positions.searchsorted(preceding, side="left")
            ]
            incoming = pending_start[frame]
            if incoming is not None:
                nearest[0] = incoming
            best = _best_block_interval(
                best,
                direction,
                length,
                frame,
                nearest,
                stop_positions + 3,
                complete=True,
            )
            if best[1] is None:
                no_start = _best_block_interval(
                    no_start,
                    direction,
                    length,
                    frame,
                    preceding + 3,
                    stop_positions,
                    complete=False,
                )
            last = int(stop_positions[-1])
            previous_stop[frame] = last
            next_start = start_positions.searchsorted(last, side="left")
            pending_start[frame] = (
                int(start_positions[next_start])
                if next_start < len(start_positions)
                else None
            )

    for frame, start in enumerate(pending_start):
        end = frame_end_index(length, frame)
        if start is not None:
            best = update_best_candidate(
                best[0],
                best[1],
                direction,
                length,
                frame,
                start,
                end,
                True,
                False,
                "partial",
            )
        elif best[1] is None and end - previous_stop[frame] >= 6:
            no_start = update_best_candidate(
                no_start[0],
                no_start[1],
                direction,
                length,
                frame,
                previous_stop[frame] + 3,
                end,
                False,
                False,
                "no_start",
            )
    return best, no_start


def _choose_blockwise(sequence, starts, stops, strand):
    best, no_start = (None, None), (None, None)
    for direction in _orf_directions(strand):
        best, no_start = _blockwise_strand_candidates(
            sequence, direction, starts, stops, best, no_start
        )
        if best[0] is not None and best[0][:2] == (2, len(sequence)):
            break
    key, fields = best if best[1] is not None else no_start
    if fields is None:
        return None
    candidate = build_candidate_from_fields(fields, key)
    candidate.output_seq = _oriented_slice(
        sequence, candidate.strand, candidate.start_idx, candidate.end_idx
    )
    return candidate


def choose_best_candidate(
    seq_str,
    codontable,
    selection="complete-first",
    *,
    strand="both",
    start_policy="table",
):
    if selection == "longest" or selection in _PARTIAL_SELECTIONS:
        eligible = (
            longest_frame_candidates(seq_str, codontable, strand, start_policy)
            if selection == "longest"
            else _iter_partial_candidates(
                seq_str, codontable, selection, strand, start_policy
            )
        )
        candidate = max(
            eligible,
            key=lambda c: selection_key(c, selection),
            default=None,
        )
        if candidate is not None:
            sequence = seq_str.upper()
            candidate.output_seq = _oriented_slice(
                sequence, candidate.strand, candidate.start_idx, candidate.end_idx
            )
        return candidate
    if selection != "complete-first":
        raise ValueError(f"Unknown ORF selection: {selection}")
    seq_upper = seq_str.upper()
    seq_len = len(seq_upper)
    if seq_len < 3:
        return None

    start_codons, stop_codons = _orf_scan_codons(codontable, start_policy)
    if not stop_codons:
        return _choose_without_stops(seq_upper, start_codons, strand)
    if seq_len >= _BEST_BLOCK_SCAN_MIN_LENGTH and _has_dense_orf_events(
        seq_upper, start_codons, stop_codons, strand
    ):
        return _choose_blockwise(seq_upper, start_codons, stop_codons, strand)
    best_sort_key = None
    best_fields = None
    frame_ends = tuple(frame_end_index(seq_len, offset) for offset in range(3))
    oriented_sequences = {}
    strand_stops = {}
    for direction, oriented in _oriented_sequences(seq_upper, strand):
        start_positions, stop_positions = collect_start_stop_positions_by_frame(
            strand_seq=oriented,
            start_codons=start_codons,
            stop_codons=stop_codons,
        )
        oriented_sequences[direction] = oriented
        strand_stops[direction] = stop_positions
        best_sort_key, best_fields = scan_start_based_best_candidate(
            strand=direction,
            seq_len=seq_len,
            frame_ends=frame_ends,
            start_positions_by_frame=start_positions,
            stop_positions_by_frame=stop_positions,
            best_sort_key=best_sort_key,
            best_fields=best_fields,
        )
        # A complete sequence-length hit cannot be improved by a later strand.
        if best_sort_key is not None and best_sort_key[:2] == (2, seq_len):
            break

    if best_fields is None:
        for direction, stop_positions in strand_stops.items():
            best_sort_key, best_fields = scan_stop_free_best_candidate(
                strand=direction,
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
    candidate.output_seq = oriented_sequences[candidate.strand][
        candidate.start_idx : candidate.end_idx
    ]
    return candidate


def selection_key(candidate, selection):
    if selection == "complete-first" or selection in _PARTIAL_SELECTIONS:
        return candidate.sort_key
    if selection == "longest":
        rank, length, *ties = candidate.sort_key
        return (length, rank, *ties)
    raise ValueError(f"Unknown ORF selection: {selection}")


def longest_frame_candidates(sequence, codontable, strand, start_policy):
    """Yield at most one started and one start-free candidate per frame."""
    sequence = sequence.upper()
    starts, stops = _orf_scan_codons(codontable, start_policy)
    for direction, oriented in _oriented_sequences(sequence, strand):
        frame_starts, frame_stops = collect_start_stop_positions_by_frame(
            oriented, starts, stops
        )
        # The last tie-breaker is the lower coordinate on the original strand.
        forward = direction == "+"
        for frame, (starts_at, stops_at) in enumerate(
            zip(frame_starts, frame_stops, strict=True)
        ):
            end = frame_end_index(len(oriented), frame)
            best = (0, False, 0)
            index = 0
            stop_index = 0
            while index < len(starts_at):
                start = starts_at[index]
                stop_index = bisect_right(stops_at, start, lo=stop_index)
                has_stop = stop_index < len(stops_at)
                stop = stops_at[stop_index] + 3 if has_stop else end
                key = (stop - start, has_stop, -start if forward else stop)
                if key > best:
                    best = key
                    best_start, best_end = start, stop
                if not has_stop:
                    break
                # Later starts sharing this stop are strictly shorter.
                index = bisect_left(starts_at, stop - 3, lo=index + 1)
            if best[0] >= 3:
                yield make_candidate(
                    oriented,
                    direction,
                    frame,
                    best_start,
                    best_end,
                    True,
                    best[1],
                    False,
                )

            best = (0, 0)
            start = frame
            for stop in (*stops_at, end):
                key = (stop - start, -start if forward else stop)
                if key > best:
                    best = key
                    best_start, best_end = start, stop
                start = stop + 3
            if best[0] >= 3:
                yield make_candidate(
                    oriented,
                    direction,
                    frame,
                    best_start,
                    best_end,
                    False,
                    False,
                    False,
                )


def _selection_candidates(sequence, codontable, selection, *, strand, start_policy):
    if selection in _PARTIAL_SELECTIONS:
        return _iter_partial_candidates(
            sequence, codontable, selection, strand, start_policy
        )
    return iter_candidates(
        sequence,
        codontable,
        include_sequence=False,
        strand=strand,
        start_policy=start_policy,
    )


def _iter_partial_candidates(sequence, codontable, selection, strand, start_policy):
    missing_start = selection != "partial-3prime"
    missing_stop = selection != "partial-5prime"
    starts, stops = _orf_scan_codons(codontable, start_policy)
    for direction, oriented in _oriented_sequences(sequence.upper(), strand):
        start_positions, stop_positions = collect_start_stop_positions_by_frame(
            oriented, () if missing_start else starts, stops
        )
        for frame, boundaries in enumerate(stop_positions):
            end = frame_end_index(len(oriented), frame)
            if missing_start:
                if missing_stop and boundaries:
                    continue
                if not missing_stop:
                    if not boundaries or boundaries[0] == frame:
                        continue
                    end = boundaries[0] + 3
                eligible = [frame]
            else:
                # For a clipped 3' end, an observed start must follow the
                # last definite stop in this frame.
                observed = start_positions[frame]
                offset = bisect_right(observed, boundaries[-1]) if boundaries else 0
                eligible = observed[offset:]
            for start in eligible:
                if end - start >= 3:
                    yield make_candidate(
                        oriented,
                        direction,
                        frame,
                        start,
                        end,
                        not missing_start,
                        not missing_stop,
                        False,
                        category="partial",
                    )


def iter_candidates(
    seq_str, codontable, include_sequence=True, *, strand="both", start_policy="table"
):
    """Yield start-to-first-stop candidates and maximal stop-free segments.

    Dual-coding codons never establish complete boundaries without annotation.
    Enumerate all frames on the selected strands, including no-start segments.
    """
    sequence = seq_str.upper()
    starts, stops = _orf_scan_codons(codontable, start_policy)
    for direction, oriented in _oriented_sequences(sequence, strand):
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
                    direction,
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
                        direction,
                        frame,
                        segment_start,
                        stop,
                        False,
                        False,
                        include_sequence,
                    )
                segment_start = stop + 3


def make_candidate(
    sequence,
    strand,
    frame,
    start,
    end,
    has_start,
    has_stop,
    include_sequence=True,
    *,
    category=None,
):
    if category is None:
        category = (
            "complete"
            if has_start and has_stop
            else "partial"
            if has_start
            else "no_start"
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


def candidate_attribute_prefixes(sequence, codontable, *, strand="both"):
    """Count each frame once instead of rescanning nested candidate sequences."""
    prefixes = {}
    for direction, oriented in _oriented_sequences(sequence, strand):
        for frame in range(3):
            prefixes[direction, frame + 1] = _frame_attribute_prefixes(
                oriented, frame, codontable
            )
    return prefixes


def _frame_attribute_prefixes(sequence, frame, codontable):
    counts = [array("Q", [0]) for _ in range(4)]
    end = frame_end_index(len(sequence), frame)
    for start in range(frame, end, _CANDIDATE_PREFIX_BLOCK_NT):
        block = sequence[start : min(end, start + _CANDIDATE_PREFIX_BLOCK_NT)]
        values = codon_features(block, codontable) if len(block) >= 192 else None
        if values is not None:
            features, _ = values
            # Outside-alphabet triplets take the scalar path below. For
            # supported DNA the ambiguity/invalid flag is exactly ambiguity.
            for prefix, index in zip(counts, (4, 5, 1, 2), strict=True):
                cumulative = np.cumsum(features[:, index], dtype=np.uint64)
                cumulative += prefix[-1]
                prefix.frombytes(cumulative.tobytes())
        else:
            for offset in range(0, len(block), 3):
                meaning = analyze_codon(block[offset : offset + 3], codontable)
                flags = (
                    meaning.possible_stop,
                    meaning.context_dependent,
                    meaning.missing,
                    meaning.ambiguous,
                )
                for prefix, flag in zip(counts, flags, strict=True):
                    prefix.append(prefix[-1] + int(flag))
    return counts


def candidate_report(
    record, codontable, selection, input_order, *, strand="both", start_policy="table"
):
    sequence = str(record.seq).upper()
    candidates = sorted(
        _selection_candidates(
            sequence,
            codontable,
            selection,
            strand=strand,
            start_policy=start_policy,
        ),
        key=lambda c: selection_key(c, selection),
        reverse=True,
    )
    prefixes = {}
    if candidates:
        table = get_codon_table_components(codontable)
        context_dependent = not table["stop_codons"].isdisjoint(table["forward_table"])
        if context_dependent or _CONCRETE_DNA.fullmatch(sequence) is None:
            prefixes = candidate_attribute_prefixes(sequence, codontable, strand=strand)
    rows = []
    best_primary = selection_key(candidates[0], selection)[:2] if candidates else None
    for index, candidate in enumerate(candidates):
        possible = context = missing = ambiguous = 0
        if prefixes:
            left = candidate.start_idx // 3
            right = candidate.end_idx // 3
            possible, context, missing, ambiguous = (
                prefix[right] - prefix[left]
                for prefix in prefixes[candidate.strand, candidate.frame]
            )
        # Candidate fields are immutable scalars or tuples of scalars.
        fields = vars(candidate).copy()
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
        sort_key = selection_key(candidate, selection)
        rows.append(
            {
                **fields,
                "sort_key": sort_key,
                "rank": index + 1,
                "selected": index == 0,
                "tied_before_deterministic_order": sort_key[:2] == best_primary,
                "stop_evidence": "definite" if candidate.has_stop else "unconfirmed",
                "possible_stop_codons": possible,
                "context_dependent_codons": context,
                "missing_codons": missing,
                "ambiguous_codons": ambiguous,
            }
        )
    report = {
        "seq_id": record.id,
        "input_order": input_order,
        "selection": selection,
        "candidates": rows,
        "selected_candidate_0based": 0 if rows else None,
    }
    if strand != "both":
        report["search_strand"] = strand
    if start_policy != "table":
        report["start_policy"] = start_policy
    return report


def choose_best_candidate_from_record(
    record,
    codontable,
    selection="complete-first",
    *,
    strand="both",
    start_policy="table",
):
    return choose_best_candidate(
        seq_str=str(record.seq),
        codontable=codontable,
        selection=selection,
        strand=strand,
        start_policy=start_policy,
    )


def choose_candidates_process_parallel(
    seq_strings,
    codontable,
    threads,
    selection="complete-first",
    *,
    strand="both",
    start_policy="table",
):
    worker = partial(
        choose_best_candidate,
        codontable=codontable,
        selection=selection,
        strand=strand,
        start_policy=start_policy,
    )
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
        description = f"{record.id} {candidate_metadata(candidate)}"
    else:
        description = record.description
    return SeqRecord(
        seq=Seq(output_seq),
        id=record.id,
        name=record.name,
        description=description,
    )


def candidate_metadata(candidate):
    return (
        f"strand={candidate.strand} frame={candidate.frame} "
        f"start={candidate.start_1based} end={candidate.end_1based} "
        f"nt_len={candidate.nt_len} aa_len={candidate.aa_len} category={candidate.category}"
    )


def alternate_records(record, baseline, codontable, selection, search_options):
    best = {}
    for candidate in _selection_candidates(
        str(record.seq), codontable, selection, **search_options
    ):
        key = candidate.strand, candidate.frame, candidate.has_start
        previous = best.get(key)
        if previous is None or selection_key(candidate, selection) > selection_key(
            previous, selection
        ):
            best[key] = candidate
    yield from distinct_alternates(
        record.id, str(record.seq), str(baseline.seq), best.values(), selection
    )


def distinct_alternates(identifier, sequence, baseline, candidates, selection):
    seen = {baseline}
    sequence = sequence.upper()
    alternatives = sorted(
        candidates,
        key=lambda candidate: selection_key(candidate, selection),
        reverse=True,
    )
    for index, candidate in enumerate(alternatives, 1):
        coding = _oriented_slice(
            sequence, candidate.strand, candidate.start_idx, candidate.end_idx
        )
        if coding in seen:
            continue
        seen.add(coding)
        name = f"{identifier}.cdskit-orf-{index}"
        yield SeqRecord(
            Seq(coding), id=name, description=f"{name} {candidate_metadata(candidate)}"
        )


def longest_with_alternates(item, codontable, strand, start_policy):
    identifier, sequence = item
    candidates = list(
        longest_frame_candidates(sequence, codontable, strand, start_policy)
    )
    best = max(
        candidates, key=partial(selection_key, selection="longest"), default=None
    )
    if best is None:
        return None, ""
    best.output_seq = _oriented_slice(
        sequence.upper(), best.strand, best.start_idx, best.end_idx
    )
    models = distinct_alternates(
        identifier, sequence, best.output_seq, candidates, "longest"
    )
    return best, "".join(model.format("fasta") for model in models)


def write_longest_with_alternates(records, worker, threads, stream):
    def inputs():
        return ((record.id, str(record.seq)) for record in records)

    candidates = []
    if should_use_process_pool(records=records, threads=threads):
        try:
            with ProcessPoolExecutor(
                max_workers=min(threads, len(records))
            ) as executor:
                pending = inputs()
                while batch := list(islice(pending, threads * 256)):
                    for candidate, fasta in executor.map(worker, batch, chunksize=32):
                        candidates.append(candidate)
                        stream.write(fasta)
            return candidates
        except (OSError, PermissionError):
            sys.stderr.write(
                "Process-based parallelism unavailable; falling back to serial processing.\n"
            )
            candidates.clear()
            stream.seek(0)
            stream.truncate()
    for candidate, fasta in map(worker, inputs()):
        candidates.append(candidate)
        stream.write(fasta)
    return candidates


def choose_primary_candidates(records, codontable, threads, selection, search_options):
    if should_use_process_pool(records=records, threads=threads):
        try:
            seq_strings = [str(record.seq) for record in records]
            return choose_candidates_process_parallel(
                seq_strings=seq_strings,
                codontable=codontable,
                threads=threads,
                selection=selection,
                **search_options,
            )
        except (OSError, PermissionError):
            sys.stderr.write(
                "Process-based parallelism unavailable; falling back to threads.\n"
            )
    worker = partial(
        choose_best_candidate_from_record,
        codontable=codontable,
        selection=selection,
        **search_options,
    )
    return parallel_map_ordered(items=records, worker=worker, threads=threads)


def longestcds_main(args):
    validate_codon_output_paths(args.seqfile, args.outfile, getattr(args, "report", ""))
    annotate_seqname = bool(getattr(args, "annotate_seqname", False))
    threads = resolve_threads(getattr(args, "threads", 1))
    selection = getattr(args, "selection", "complete-first")
    search_options = dict(
        strand=getattr(args, "strand", "both"),
        start_policy=getattr(args, "start_policy", "table"),
    )
    report = getattr(args, "report", "")
    alternate = getattr(args, "alternate_orfs", "")
    if alternate:
        if alternate == "-":
            raise ValueError("--alternate_orfs requires a FASTA file path")
        validate_distinct_paths(
            inputs=[args.seqfile], outputs=[args.outfile, report, alternate]
        )
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if alternate:
        ids = {record.id for record in records}
        if len(ids) != len(records) or any(
            not name or re.search(ALT_ORF_SUFFIX, name) for name in ids
        ):
            raise ValueError(
                "--alternate_orfs requires unique IDs without a reserved .cdskit-orf-N suffix"
            )
    stop_if_not_dna(records=records, label="--seq_file")
    stop_if_invalid_codontable(args.codontable)

    stream_report = bool(report) and args.outfile not in ("", "-") and report != "-"
    candidate_reports = (
        (
            candidate_report(
                record, args.codontable, selection, index + 1, **search_options
            )
            for index, record in enumerate(records)
        )
        if report
        else ()
    )
    # Complete candidate analysis before any nontransactional stdout write.
    report_records = candidate_reports if stream_report else list(candidate_reports)
    outputs = [
        path for path in (args.outfile, report, alternate) if path not in ("", "-")
    ]
    with atomic_output_paths(outputs) as temporary_paths:
        staged = dict(zip(outputs, temporary_paths, strict=True))
        combined = alternate and selection == "longest"
        if combined:
            worker = partial(
                longest_with_alternates, codontable=args.codontable, **search_options
            )
            with open(staged[alternate], "w", encoding="utf-8") as stream:
                candidates = write_longest_with_alternates(
                    records, worker, threads, stream
                )
        else:
            candidates = choose_primary_candidates(
                records, args.codontable, threads, selection, search_options
            )
        output_records = [
            build_output_record(record, candidate, annotate_seqname)
            for record, candidate in zip(records, candidates, strict=True)
        ]

        if alternate and not combined:
            with open(staged[alternate], "w", encoding="utf-8") as stream:
                SeqIO.write(
                    (
                        model
                        for record, baseline in zip(
                            records, output_records, strict=True
                        )
                        for model in alternate_records(
                            record, baseline, args.codontable, selection, search_options
                        )
                    ),
                    stream,
                    "fasta",
                )
        write_seqs(
            records=output_records,
            outfile=staged.get(args.outfile, args.outfile),
            outseqformat=args.outseqformat,
        )
        write_codon_report(
            staged.get(report, report),
            "longestorf",
            args.codontable,
            report_records,
            stream=stream_report,
        )
