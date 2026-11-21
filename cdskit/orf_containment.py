"""Recognize shorter representations of the same coding sequence."""

import re
from pathlib import Path

from Bio.Seq import reverse_complement

from cdskit.longestcds import ALT_ORF_SUFFIX, get_start_stop_codons
from cdskit.sequence_selection import fasta_records

_DNA = re.compile(r"[ACGT]+")
_ALTERNATE_SUFFIX = re.compile(ALT_ORF_SUFFIX)


def _coding_body(sequence, stops):
    seq = str(sequence).upper().replace("U", "T")
    if len(seq) % 3 or not _DNA.fullmatch(seq):
        return ""
    if seq[-3:] in stops:
        seq = seq[:-3]
    return seq


def _containment_reason(long, short, starts):
    if not short or len(long) <= len(short):
        return ""
    offset = long.find(short)
    while offset >= 0:
        if offset % 3 == 0:
            return "retain_inframe_containing_baseline"
        offset = long.find(short, offset + 1)
    # Compare an integral number of codons covering at least 98% of the short CDS.
    codons = len(short) // 3
    span = 3 * ((98 * codons + 99) // 100)
    if short[:3] not in starts and long[-span:] == short[-span:]:
        return "retain_shared_stop_baseline"
    return ""


def _placement(transcript, body):
    if transcript is None:
        return None
    placements = []
    for strand, strand_seq in (
        ("+", transcript),
        ("-", reverse_complement(transcript)),
    ):
        start = strand_seq.find(body)
        if start < 0:
            continue
        if strand_seq.find(body, start + 1) >= 0:
            return None
        end = start + len(body)
        placements.append((strand, start, end, strand_seq[end : end + 3]))
    return placements[0] if len(placements) == 1 else None


def _same_coding_path(long_place, short_place, same_transcript, reason, stops):
    if long_place is None or short_place is None:
        return False
    strand, start, end, stop = long_place
    other_strand, other_start, other_end, other_stop = short_place
    if strand != other_strand:
        return False
    if reason == "retain_inframe_containing_baseline":
        return not same_transcript or (
            start <= other_start and other_end <= end and (other_start - start) % 3 == 0
        )
    return (
        stop == other_stop
        and stop in stops
        and (not same_transcript or end == other_end)
    )


def retain_nested_orfs(selections, transcript_fasta, codon_table):
    """Retain longer in-frame ORFs after checking their source transcripts."""
    starts, stops = get_start_stop_codons(codon_table)
    comparisons = {}
    wanted: set[str] = set()
    for index, (_, baseline, selected, _, _, _) in enumerate(selections):
        if len(baseline.seq) <= len(selected.seq):
            continue
        long, short = (
            _coding_body(record.seq, stops) for record in (baseline, selected)
        )
        reason = _containment_reason(long, short, starts)
        if reason:
            # Only candidate pairs that could be retained need a codon-by-codon scan.
            if any(
                body[pos : pos + 3] in stops
                for body in (long, short)
                for pos in range(0, len(body), 3)
            ):
                continue
            ids = tuple(
                _ALTERNATE_SUFFIX.sub("", record.id) for record in (baseline, selected)
            )
            comparisons[index] = (ids, long, short, reason)
            wanted.update(ids)
    if not wanted:
        return selections

    transcripts = {}
    for identifier, _, sequence in fasta_records(
        Path(transcript_fasta), identifiers=wanted
    ):
        transcripts[identifier] = (
            None if identifier in transcripts else sequence.upper().replace("U", "T")
        )
    for index, ((long_id, short_id), long, short, reason) in comparisons.items():
        long_place = _placement(transcripts.get(long_id), long)
        short_place = _placement(transcripts.get(short_id), short)
        if _same_coding_path(
            long_place, short_place, long_id == short_id, reason, stops
        ):
            group, baseline, _, _, hit, _ = selections[index]
            selections[index] = (group, baseline, baseline, reason, hit, hit)
    return selections
