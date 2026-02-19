#!/usr/bin/env python

from functools import partial

import Bio.Data.CodonTable
import Bio.Seq
import Bio.SeqIO
import numpy
import sys

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, write_seqs

_STOP_CODON_CACHE = {}


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


def count_internal_stop_codons(seq, codon_table):
    seq_str = str(seq)
    stop_codons = get_stop_codons(codon_table)
    end = len(seq_str) - 3
    if end <= 0:
        return 0
    num_stop = 0
    for i in range(0, end, 3):
        if seq_str[i:i+3] in stop_codons:
            num_stop += 1
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
        min_index = numpy.argmin(self.num_stops)
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


def choose_best_padding(clean_seq, codon_table, padchar, num_stop_input, num_missing, seqlen):
    seqs = padseqs(original_seq=clean_seq, codon_table=codon_table, padchar=padchar)
    for headn, tailn in get_padding_candidates(num_stop_input, num_missing, seqlen):
        seqs.add(headn=headn, tailn=tailn)
    return seqs.get_minimum_num_stop()


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


def pad_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    threads = resolve_threads(getattr(args, 'threads', 1))
    worker = partial(
        process_record_padding_entry,
        codon_table=args.codontable,
        padchar=args.padchar,
    )
    results = parallel_map_ordered(items=records, worker=worker, threads=threads)
    is_no_stop = []
    seqnum_padded = 0
    for i, result in enumerate(results):
        records[i].seq = Bio.Seq.Seq(result['new_seq'])
        if result['log'] != '':
            sys.stderr.write(result['log'])
        is_no_stop.append(result['is_no_stop'])
        if result['was_padded']:
            seqnum_padded += 1
    if args.nopseudo:
        records = [records[i] for i in range(len(records)) if is_no_stop[i]]
    sys.stderr.write('Number of padded sequences: {:,} / {:,}\n'.format(seqnum_padded, len(records)))
    write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
