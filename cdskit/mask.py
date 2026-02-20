#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import Bio.Data.CodonTable
import Bio.Seq

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_multiple_of_three,
    write_seqs,
)

_CODON_CLASS_CACHE = dict()
_MASK_DECISION_CACHE = dict()
_PROCESS_PARALLEL_MIN_RECORDS = 2000


def codon_chunks(nucseq):
    return [nucseq[i:i + 3] for i in range(0, len(nucseq), 3)]


def mask_partial_gap_codons(codons, mask_triplet):
    changed = False
    for i, codon in enumerate(codons):
        if ('-' in codon) and (codon != '---'):
            codons[i] = mask_triplet
            changed = True
    return changed


def should_mask_amino_acid(aa, mask_ambiguous, mask_stop):
    if mask_ambiguous and mask_stop:
        return aa in ['X', '*']
    if mask_ambiguous:
        return aa == 'X'
    if mask_stop:
        return aa == '*'
    return False


def mask_translated_codons(codons, aaseq, mask_triplet, mask_ambiguous, mask_stop):
    changed = False
    for i, aa in enumerate(aaseq):
        if should_mask_amino_acid(aa, mask_ambiguous, mask_stop):
            codons[i] = mask_triplet
            changed = True
    return changed


def get_codon_class_sets(codontable):
    cached = _CODON_CLASS_CACHE.get(codontable)
    if cached is not None:
        return cached

    if isinstance(codontable, int):
        table = Bio.Data.CodonTable.unambiguous_dna_by_id[codontable]
    else:
        table = Bio.Data.CodonTable.unambiguous_dna_by_name[str(codontable)]
    stop_codons = frozenset([codon.upper() for codon in table.stop_codons])
    forward_codons = frozenset([codon.upper() for codon in table.forward_table.keys()])
    _CODON_CLASS_CACHE[codontable] = (stop_codons, forward_codons)
    return _CODON_CLASS_CACHE[codontable]


def get_mask_decision_cache(codontable, mask_ambiguous, mask_stop):
    key = (codontable, bool(mask_ambiguous), bool(mask_stop))
    cache = _MASK_DECISION_CACHE.get(key)
    if cache is None:
        cache = dict()
        _MASK_DECISION_CACHE[key] = cache
    return cache


def mask_sequence_string(nucseq, codontable, mask_triplet, mask_ambiguous, mask_stop):
    if not mask_ambiguous and not mask_stop:
        out = list()
        changed = False
        for i in range(0, len(nucseq), 3):
            codon = nucseq[i:i + 3]
            if ('-' in codon) and (codon != '---'):
                out.append(mask_triplet)
                changed = True
            else:
                out.append(codon)
        if changed:
            return ''.join(out)
        return nucseq

    stop_codons, forward_codons = get_codon_class_sets(codontable=codontable)
    mask_decision_cache = get_mask_decision_cache(
        codontable=codontable,
        mask_ambiguous=mask_ambiguous,
        mask_stop=mask_stop,
    )
    out = list()
    changed = False
    for i in range(0, len(nucseq), 3):
        codon = nucseq[i:i + 3]
        codon_upper = codon.upper()
        should_mask = mask_decision_cache.get(codon_upper)
        if should_mask is None:
            if ('-' in codon_upper) and (codon_upper != '---'):
                should_mask = True
            elif codon_upper == '---':
                should_mask = False
            elif mask_stop and (codon_upper in stop_codons):
                should_mask = True
            elif mask_ambiguous and (codon_upper not in forward_codons) and (codon_upper not in stop_codons):
                should_mask = True
            else:
                should_mask = False
            mask_decision_cache[codon_upper] = should_mask
        if should_mask:
            out.append(mask_triplet)
            changed = True
            continue

        out.append(codon)

    if changed:
        return ''.join(out)
    return nucseq


def mask_record(record, codontable, mask_triplet, mask_ambiguous, mask_stop):
    nucseq = str(record.seq)
    return mask_sequence_string(
        nucseq=nucseq,
        codontable=codontable,
        mask_triplet=mask_triplet,
        mask_ambiguous=mask_ambiguous,
        mask_stop=mask_stop,
    )


def mask_payload(payload, codontable, mask_triplet, mask_ambiguous, mask_stop):
    seq_id, nucseq = payload
    masked = mask_sequence_string(
        nucseq=nucseq,
        codontable=codontable,
        mask_triplet=mask_triplet,
        mask_ambiguous=mask_ambiguous,
        mask_stop=mask_stop,
    )
    return seq_id, masked


def mask_payloads_process_parallel(payloads, codontable, mask_triplet, mask_ambiguous, mask_stop, threads):
    worker = partial(
        mask_payload,
        codontable=codontable,
        mask_triplet=mask_triplet,
        mask_ambiguous=mask_ambiguous,
        mask_stop=mask_stop,
    )
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def mask_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three(records)
    mask_triplet = args.maskchar * 3
    mask_ambiguous = (args.ambiguouscodon == 'yes')
    mask_stop = (args.stopcodon == 'yes')
    threads = resolve_threads(getattr(args, 'threads', 1))
    masked_seqs = None
    if (threads > 1) and (len(records) >= _PROCESS_PARALLEL_MIN_RECORDS):
        try:
            payloads = [(record.id, str(record.seq)) for record in records]
            masked_payloads = mask_payloads_process_parallel(
                payloads=payloads,
                codontable=args.codontable,
                mask_triplet=mask_triplet,
                mask_ambiguous=mask_ambiguous,
                mask_stop=mask_stop,
                threads=threads,
            )
            masked_seqs = [masked_seq for _, masked_seq in masked_payloads]
        except (OSError, PermissionError):
            pass
    if masked_seqs is None:
        worker = partial(
            mask_record,
            codontable=args.codontable,
            mask_triplet=mask_triplet,
            mask_ambiguous=mask_ambiguous,
            mask_stop=mask_stop,
        )
        masked_seqs = parallel_map_ordered(items=records, worker=worker, threads=threads)
    for record, masked_seq in zip(records, masked_seqs):
        if str(record.seq) != masked_seq:
            record.seq = Bio.Seq.Seq(masked_seq)
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
