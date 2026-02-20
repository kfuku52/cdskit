from concurrent.futures import ProcessPoolExecutor
from functools import partial

import Bio.Data.CodonTable
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.util import (
    parallel_map_ordered,
    read_seqs,
    resolve_threads,
    stop_if_not_multiple_of_three,
    write_seqs,
)

_PROCESS_PARALLEL_MIN_RECORDS = 2000
_CODON_TRANSLATOR_CACHE = dict()


def get_codon_translator(codontable):
    cached = _CODON_TRANSLATOR_CACHE.get(codontable)
    if cached is not None:
        return cached
    translator = {
        'codon_to_aa': dict(),
        'codon_error': dict(),
    }
    _CODON_TRANSLATOR_CACHE[codontable] = translator
    return translator


def translate_sequence_string(seq_str, codontable, to_stop):
    translator = get_codon_translator(codontable=codontable)
    codon_to_aa = translator['codon_to_aa']
    codon_error = translator['codon_error']
    seq_upper = seq_str.upper()
    aa_chars = list()
    for i in range(0, len(seq_upper), 3):
        codon = seq_upper[i:i + 3]
        aa = codon_to_aa.get(codon)
        if aa is None:
            message = codon_error.get(codon)
            if message is not None:
                raise Bio.Data.CodonTable.TranslationError(message)
            try:
                aa = str(Seq(codon).translate(table=codontable, to_stop=False, gap='-'))
            except Exception as exc:
                message = str(exc)
                codon_error[codon] = message
                raise Bio.Data.CodonTable.TranslationError(message)
            codon_to_aa[codon] = aa
        if to_stop and (aa == '*'):
            break
        aa_chars.append(aa)
    return ''.join(aa_chars)


def translate_record(record, codontable, to_stop):
    translated = translate_sequence_string(
        seq_str=str(record.seq),
        codontable=codontable,
        to_stop=to_stop,
    )
    return SeqRecord(
        seq=Seq(translated),
        id=record.id,
        name=record.name,
        description=record.description,
    )


def translate_payload(payload, codontable, to_stop):
    seq_id, seq_name, seq_description, seq_str = payload
    translated = translate_sequence_string(
        seq_str=seq_str,
        codontable=codontable,
        to_stop=to_stop,
    )
    return seq_id, seq_name, seq_description, translated


def translate_payloads_process_parallel(payloads, codontable, to_stop, threads):
    worker = partial(translate_payload, codontable=codontable, to_stop=to_stop)
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def translate_main(args):
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    if len(records) == 0:
        write_seqs(records=records, outfile=args.outfile, outseqformat=args.outseqformat)
        return
    stop_if_not_multiple_of_three(records=records)
    threads = resolve_threads(getattr(args, 'threads', 1))
    translated_records = None
    if (threads > 1) and (len(records) >= _PROCESS_PARALLEL_MIN_RECORDS):
        try:
            payloads = [(record.id, record.name, record.description, str(record.seq)) for record in records]
            translated_payloads = translate_payloads_process_parallel(
                payloads=payloads,
                codontable=args.codontable,
                to_stop=args.to_stop,
                threads=threads,
            )
            translated_records = [
                SeqRecord(
                    seq=Seq(translated_seq),
                    id=seq_id,
                    name=seq_name,
                    description=seq_description,
                )
                for seq_id, seq_name, seq_description, translated_seq in translated_payloads
            ]
        except (OSError, PermissionError):
            pass
    if translated_records is None:
        worker = partial(
            translate_record,
            codontable=args.codontable,
            to_stop=args.to_stop,
        )
        translated_records = parallel_map_ordered(items=records, worker=worker, threads=threads)
    write_seqs(records=translated_records, outfile=args.outfile, outseqformat=args.outseqformat)
