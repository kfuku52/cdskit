import sys
from functools import partial

import Bio.Data.CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, write_seqs

_CODON_TABLE_CACHE = dict()


class CdsCandidate:
    def __init__(self, nt_seq, strand, frame, start_1based, end_1based, has_start, has_stop, category):
        self.nt_seq = nt_seq
        self.strand = strand
        self.frame = frame
        self.start_1based = start_1based
        self.end_1based = end_1based
        self.has_start = has_start
        self.has_stop = has_stop
        self.category = category
        self.nt_len = len(nt_seq)
        self.aa_len = self.nt_len // 3


def get_start_stop_codons(codontable):
    cached = _CODON_TABLE_CACHE.get(codontable)
    if cached is not None:
        return cached
    table = Bio.Data.CodonTable.unambiguous_dna_by_id[codontable]
    codons = (set(table.start_codons), set(table.stop_codons))
    _CODON_TABLE_CACHE[codontable] = codons
    return codons


def frame_end_index(seq_len, frame_offset):
    return seq_len - ((seq_len - frame_offset) % 3)


def original_coordinates(strand, seq_len, start_idx, end_idx):
    if strand == '+':
        return start_idx + 1, end_idx
    # Coordinates on original (plus) strand while preserving coding orientation via strand symbol.
    return seq_len - end_idx + 1, seq_len - start_idx


def build_candidate(strand_seq, strand, seq_len, frame_offset, start_idx, end_idx, has_start, has_stop, category):
    nt_seq = strand_seq[start_idx:end_idx]
    start_1based, end_1based = original_coordinates(
        strand=strand,
        seq_len=seq_len,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    return CdsCandidate(
        nt_seq=nt_seq,
        strand=strand,
        frame=frame_offset + 1,
        start_1based=start_1based,
        end_1based=end_1based,
        has_start=has_start,
        has_stop=has_stop,
        category=category,
    )


def scan_start_based_candidates(strand_seq, strand, seq_len, start_codons, stop_codons):
    candidates = list()
    seq_len_local = len(strand_seq)
    for frame_offset in range(3):
        max_end = frame_end_index(seq_len=seq_len_local, frame_offset=frame_offset)
        i = frame_offset
        while i + 3 <= seq_len_local:
            codon = strand_seq[i:i + 3]
            if codon in start_codons:
                j = i + 3
                found_stop = False
                while j + 3 <= seq_len_local:
                    stop_codon = strand_seq[j:j + 3]
                    if stop_codon in stop_codons:
                        candidates.append(build_candidate(
                            strand_seq=strand_seq,
                            strand=strand,
                            seq_len=seq_len,
                            frame_offset=frame_offset,
                            start_idx=i,
                            end_idx=j + 3,
                            has_start=True,
                            has_stop=True,
                            category='complete',
                        ))
                        found_stop = True
                        break
                    j += 3
                if (not found_stop) and (max_end - i >= 3):
                    candidates.append(build_candidate(
                        strand_seq=strand_seq,
                        strand=strand,
                        seq_len=seq_len,
                        frame_offset=frame_offset,
                        start_idx=i,
                        end_idx=max_end,
                        has_start=True,
                        has_stop=False,
                        category='partial',
                    ))
            i += 3
    return candidates


def scan_stop_free_candidates(strand_seq, strand, seq_len, stop_codons):
    candidates = list()
    seq_len_local = len(strand_seq)
    for frame_offset in range(3):
        segment_start = frame_offset
        i = frame_offset
        while i + 3 <= seq_len_local:
            codon = strand_seq[i:i + 3]
            if codon in stop_codons:
                if i - segment_start >= 3:
                    candidates.append(build_candidate(
                        strand_seq=strand_seq,
                        strand=strand,
                        seq_len=seq_len,
                        frame_offset=frame_offset,
                        start_idx=segment_start,
                        end_idx=i,
                        has_start=False,
                        has_stop=False,
                        category='no_start',
                    ))
                segment_start = i + 3
            i += 3
        max_end = frame_end_index(seq_len=seq_len_local, frame_offset=frame_offset)
        if max_end - segment_start >= 3:
            candidates.append(build_candidate(
                strand_seq=strand_seq,
                strand=strand,
                seq_len=seq_len,
                frame_offset=frame_offset,
                start_idx=segment_start,
                end_idx=max_end,
                has_start=False,
                has_stop=False,
                category='no_start',
            ))
    return candidates


def rank_value(candidate):
    if candidate.category == 'complete':
        return 2
    if candidate.category == 'partial':
        return 1
    return 0


def candidate_sort_key(candidate):
    strand_order = 0 if candidate.strand == '+' else 1
    return (
        rank_value(candidate),
        candidate.nt_len,
        -strand_order,
        -candidate.has_stop,
        -candidate.has_start,
        -candidate.frame,
        -candidate.start_1based,
    )


def choose_best_candidate(seq_str, codontable):
    seq_upper = seq_str.upper()
    seq_len = len(seq_upper)
    if seq_len < 3:
        return None

    start_codons, stop_codons = get_start_stop_codons(codontable=codontable)
    reverse_complement = str(Seq(seq_upper).reverse_complement())
    all_candidates = list()

    for strand, strand_seq in [('+', seq_upper), ('-', reverse_complement)]:
        all_candidates.extend(scan_start_based_candidates(
            strand_seq=strand_seq,
            strand=strand,
            seq_len=seq_len,
            start_codons=start_codons,
            stop_codons=stop_codons,
        ))

    if len(all_candidates) == 0:
        for strand, strand_seq in [('+', seq_upper), ('-', reverse_complement)]:
            all_candidates.extend(scan_stop_free_candidates(
                strand_seq=strand_seq,
                strand=strand,
                seq_len=seq_len,
                stop_codons=stop_codons,
            ))

    if len(all_candidates) == 0:
        return None

    return max(all_candidates, key=candidate_sort_key)


def choose_best_candidate_from_record(record, codontable):
    return choose_best_candidate(seq_str=str(record.seq), codontable=codontable)


def build_output_record(record, candidate, annotate_seqname):
    if candidate is None:
        txt = 'No CDS candidate found for {}. Output sequence is empty.\n'
        sys.stderr.write(txt.format(record.id))
        meta = 'strand=NA frame=NA start=NA end=NA nt_len=0 aa_len=0 category=none'
        if annotate_seqname:
            description = f'{record.id} {meta}'
        else:
            description = record.description
        return SeqRecord(
            seq=Seq(''),
            id=record.id,
            name=record.name,
            description=description,
        )

    meta = (
        f"strand={candidate.strand} frame={candidate.frame} "
        f"start={candidate.start_1based} end={candidate.end_1based} "
        f"nt_len={candidate.nt_len} aa_len={candidate.aa_len} category={candidate.category}"
    )
    if annotate_seqname:
        description = f'{record.id} {meta}'
    else:
        description = record.description
    return SeqRecord(
        seq=Seq(candidate.nt_seq),
        id=record.id,
        name=record.name,
        description=description,
    )


def longestcds_main(args):
    annotate_seqname = bool(getattr(args, 'annotate_seqname', False))
    threads = resolve_threads(getattr(args, 'threads', 1))
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if len(records) == 0:
        write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
        return

    worker = partial(choose_best_candidate_from_record, codontable=args.codontable)
    candidates = parallel_map_ordered(items=records, worker=worker, threads=threads)

    output_records = list()
    for record, candidate in zip(records, candidates):
        output_records.append(build_output_record(
            record=record,
            candidate=candidate,
            annotate_seqname=annotate_seqname,
        ))

    write_seqs(records=output_records, outfile=args.outfile, outseqformat=args.outseqformat)
