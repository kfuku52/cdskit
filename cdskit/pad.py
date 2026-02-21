#!/usr/bin/env python

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import Bio.Data.CodonTable
import Bio.Seq
import Bio.SeqIO
import sys

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    write_seqs,
)

_STOP_CODON_CACHE = {}
_STOP_CODON_SCAN_CACHE = {}
_PROCESS_PARALLEL_MIN_RECORDS = 2000


def get_stop_codons(codon_table):
    if codon_table in _STOP_CODON_CACHE:
        return _STOP_CODON_CACHE[codon_table]
    if isinstance(codon_table, int):
        table = Bio.Data.CodonTable.unambiguous_dna_by_id[codon_table]
    else:
        table = Bio.Data.CodonTable.unambiguous_dna_by_name[str(codon_table)]
    stop_codons = set(table.stop_codons)
    _STOP_CODON_CACHE[codon_table] = stop_codons
    return stop_codons


def get_stop_codon_scan_list(codon_table):
    cached = _STOP_CODON_SCAN_CACHE.get(codon_table)
    if cached is not None:
        return cached
    scan_codons = tuple(sorted(get_stop_codons(codon_table)))
    _STOP_CODON_SCAN_CACHE[codon_table] = scan_codons
    return scan_codons


def count_internal_stop_codons(seq, codon_table):
    seq_str = seq if isinstance(seq, str) else str(seq)
    seq_upper = seq_str.upper()
    internal_stop_limit = len(seq_upper) - 3
    if internal_stop_limit <= 0:
        return 0
    num_stop = 0
    seq_find = seq_upper.find
    for codon in get_stop_codon_scan_list(codon_table):
        pos = seq_find(codon)
        while pos != -1:
            if (pos % 3 == 0) and (pos < internal_stop_limit):
                num_stop += 1
            pos = seq_find(codon, pos + 1)
    return num_stop


class padseqs:
    def __init__(self, original_seq, codon_table='Standard', padchar='N'):
        self.new_seqs = list()
        self.num_stops = list()
        self.headn = list()
        self.tailn = list()
        self.original_seq = str(original_seq)
        self.codon_table = codon_table
        self.padchar = padchar
    def add(self, headn=0, tailn=0):
        new_seq = Bio.Seq.Seq((self.padchar*headn)+self.original_seq+(self.padchar*tailn))
        self.new_seqs.append(new_seq)
        self.num_stops.append(count_internal_stop_codons(new_seq, self.codon_table))
        self.headn.append(headn)
        self.tailn.append(tailn)
    def get_minimum_num_stop(self):
        min_index = min(range(len(self.num_stops)), key=lambda i: self.num_stops[i])
        out = {
            'new_seq':self.new_seqs[min_index],
            'num_stop':self.num_stops[min_index],
            'headn':self.headn[min_index],
            'tailn':self.tailn[min_index],
        }
        return out


def get_adjusted_length_and_tailpadded_sequence(clean_seq, padchar):
    seqlen = len(clean_seq)
    if seqlen % 3 == 0:
        return seqlen, clean_seq
    adjlen = ((seqlen // 3) + 1) * 3
    return adjlen, clean_seq.ljust(adjlen, padchar)


def get_padding_candidates(num_stop_input, num_missing, seqlen):
    candidates = []
    if num_stop_input:
        if (num_missing == 0) or (num_missing == 3):
            candidates.extend([(0, 0), (1, 2), (2, 1)])
        elif num_missing == 1:
            candidates.extend([(0, 1), (1, 0), (2, 2)])
        elif num_missing == 2:
            candidates.extend([(0, 2), (2, 0), (1, 1)])
    if (not num_stop_input) and (seqlen % 3):
        candidates.append((0, num_missing))
    return candidates


def choose_best_padding(clean_seq, codon_table, padchar, num_stop_input, num_missing, seqlen, tailpad_seq):
    best = None
    for headn, tailn in get_padding_candidates(num_stop_input, num_missing, seqlen):
        if (headn == 0) and (tailn == num_missing):
            # Reuse already evaluated tail-padded sequence.
            new_seq = tailpad_seq
            num_stop = num_stop_input
        else:
            new_seq = (padchar * headn) + clean_seq + (padchar * tailn)
            num_stop = count_internal_stop_codons(new_seq, codon_table)
        if (best is None) or (num_stop < best['num_stop']):
            best = {
                'new_seq': new_seq,
                'num_stop': num_stop,
                'headn': headn,
                'tailn': tailn,
            }
    return best


def process_record_padding(record_name, record_seq, codon_table, padchar):
    clean_seq = record_seq.replace('X', 'N')
    seqlen = len(clean_seq)
    adjlen, tailpad_seq = get_adjusted_length_and_tailpadded_sequence(clean_seq, padchar)
    num_stop_input = count_internal_stop_codons(tailpad_seq, codon_table)

    if not (num_stop_input or (seqlen % 3)):
        return {
            'new_seq': record_seq,
            'is_no_stop': True,
            'was_padded': False,
            'log': '',
        }

    num_missing = adjlen - seqlen
    best_padseq = choose_best_padding(
        clean_seq=clean_seq,
        codon_table=codon_table,
        padchar=padchar,
        num_stop_input=num_stop_input,
        num_missing=num_missing,
        seqlen=seqlen,
        tailpad_seq=tailpad_seq,
    )
    is_no_stop = (best_padseq['num_stop'] == 0)
    txt = f'{record_name}, original_seqlen={seqlen}, head_padding={best_padseq["headn"]}, tail_padding={best_padseq["tailn"]}, '
    txt += f'original_num_stop={num_stop_input}, new_num_stop={best_padseq["num_stop"]}\n'
    was_padded = not ((best_padseq['headn'] == 0) and (best_padseq['tailn'] == 0))
    return {
        'new_seq': str(best_padseq['new_seq']),
        'is_no_stop': is_no_stop,
        'was_padded': was_padded,
        'log': txt,
    }


def process_record_padding_entry(record, codon_table, padchar):
    return process_record_padding(
        record_name=record.name,
        record_seq=str(record.seq),
        codon_table=codon_table,
        padchar=padchar,
    )


def process_record_padding_payload(payload, codon_table, padchar):
    record_name, record_seq = payload
    return process_record_padding(
        record_name=record_name,
        record_seq=record_seq,
        codon_table=codon_table,
        padchar=padchar,
    )


def process_padding_payloads_process_parallel(payloads, codon_table, padchar, threads):
    worker = partial(
        process_record_padding_payload,
        codon_table=codon_table,
        padchar=padchar,
    )
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def pad_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label='--seqfile')
    stop_if_invalid_codontable(args.codontable)
    threads = resolve_threads(getattr(args, 'threads', 1))
    results = None
    if (threads > 1) and (len(records) >= _PROCESS_PARALLEL_MIN_RECORDS):
        try:
            payloads = [(record.name, str(record.seq)) for record in records]
            results = process_padding_payloads_process_parallel(
                payloads=payloads,
                codon_table=args.codontable,
                padchar=args.padchar,
                threads=threads,
            )
        except (OSError, PermissionError):
            pass
    if results is None:
        worker = partial(
            process_record_padding_entry,
            codon_table=args.codontable,
            padchar=args.padchar,
        )
        results = parallel_map_ordered(items=records, worker=worker, threads=threads)
    is_no_stop = []
    seqnum_padded = 0
    log_lines = list()
    for i, result in enumerate(results):
        records[i].seq = Bio.Seq.Seq(result['new_seq'])
        if result['log'] != '':
            log_lines.append(result['log'])
        is_no_stop.append(result['is_no_stop'])
        if result['was_padded']:
            seqnum_padded += 1
    if len(log_lines) > 0:
        sys.stderr.write(''.join(log_lines))
    if args.nopseudo:
        records = [records[i] for i in range(len(records)) if is_no_stop[i]]
    sys.stderr.write('Number of padded sequences: {:,} / {:,}\n'.format(seqnum_padded, len(records)))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
