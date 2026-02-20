import Bio.Seq
import Bio.SeqRecord
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import Bio.Data.CodonTable

from cdskit.util import parallel_map_ordered, read_seqs, resolve_threads, stop_if_not_aligned, write_seqs


AA_GAP_CHARS = {'-', '.'}
AA_WILDCARD_CHARS = {'X', '?'}
CDS_GAP_CHARS = {'-', '.'}
_GAP_DROP_TABLE_CACHE = dict()
_CODON_TRANSLATOR_CACHE = dict()
_PROCESS_PARALLEL_MIN_RECORDS = 2000


def remove_gap_chars(seq, gap_chars):
    gap_chars = tuple(sorted(gap_chars))
    table = _GAP_DROP_TABLE_CACHE.get(gap_chars)
    if table is None:
        table = str.maketrans('', '', ''.join(gap_chars))
        _GAP_DROP_TABLE_CACHE[gap_chars] = table
    return str(seq).translate(table)


def stop_if_not_multiple_of_three_after_gap_removal(records):
    flag_stop = False
    for record in records:
        seq = remove_gap_chars(record.seq, CDS_GAP_CHARS)
        is_multiple_of_three = (len(seq) % 3 == 0)
        if not is_multiple_of_three:
            txt = 'Sequence length is not multiple of three after removing gaps: {}\n'
            sys.stderr.write(txt.format(record.id))
            flag_stop = True
    if flag_stop:
        txt = 'Input CDS length should be multiple of three after removing gaps. Exiting.\n'
        raise Exception(txt)


def get_record_map(records, label):
    record_map = dict()
    for record in records:
        if record.id in record_map:
            txt = 'Sequence IDs must be unique in {}. Duplicated ID: {}'
            raise Exception(txt.format(label, record.id))
        record_map[record.id] = record
    return record_map


def stop_if_sequence_ids_do_not_match(cdn_records, pep_records):
    cdn_ids = set([record.id for record in cdn_records])
    pep_ids = set([record.id for record in pep_records])
    if cdn_ids == pep_ids:
        return
    missing_in_cds = sorted(list(pep_ids - cdn_ids))
    missing_in_aa = sorted(list(cdn_ids - pep_ids))
    txt = 'Sequence IDs did not match between CDS (--seqfile) and amino acid alignment (--aa_aln).'
    if len(missing_in_cds) > 0:
        txt += ' Missing in CDS: {}.'.format(','.join(missing_in_cds))
    if len(missing_in_aa) > 0:
        txt += ' Missing in amino acid alignment: {}.'.format(','.join(missing_in_aa))
    raise Exception(txt)


def split_codons(seq):
    codons = list()
    for i in range(0, len(seq), 3):
        codons.append(seq[i:i + 3])
    return codons


def translate_codons(codons, codontable):
    translated = list()
    for codon in codons:
        aa = str(Bio.Seq.Seq(codon).translate(table=codontable, to_stop=False, gap='-'))
        translated.append(aa)
    return translated


def get_codon_translator(codontable):
    cached = _CODON_TRANSLATOR_CACHE.get(codontable)
    if cached is not None:
        return cached
    if isinstance(codontable, int):
        table = Bio.Data.CodonTable.ambiguous_dna_by_id[codontable]
    else:
        table = Bio.Data.CodonTable.ambiguous_dna_by_name[str(codontable)]
    translator = {
        'forward_table': table.forward_table,
        'stop_codons': frozenset([codon.upper() for codon in table.stop_codons]),
        'cache': dict(),
    }
    _CODON_TRANSLATOR_CACHE[codontable] = translator
    return translator


def translate_single_codon(codon, translator):
    codon_upper = codon.upper()
    cached = translator['cache'].get(codon_upper)
    if cached is not None:
        return cached
    if codon_upper in translator['stop_codons']:
        aa = '*'
    else:
        try:
            aa = translator['forward_table'][codon_upper]
        except Exception:
            aa = 'X'
    translator['cache'][codon_upper] = aa
    return aa


def translate_cds_seq(cdn_seq, codontable):
    return str(Bio.Seq.Seq(cdn_seq).translate(table=codontable, to_stop=False, gap='-'))


def amino_acid_matches(aa_aln_char, translated_char):
    aa = aa_aln_char.upper()
    tr = translated_char.upper()
    if aa in AA_WILDCARD_CHARS:
        return True
    return aa == tr


def backalign_record(cdn_record, pep_record, codontable):
    aligned_seq = backalign_sequence_strings(
        cdn_seq_raw=str(cdn_record.seq),
        pep_seq_raw=str(pep_record.seq),
        codontable=codontable,
        seq_id=pep_record.id,
        emit_terminal_stop_warning=True,
    )
    out_record = Bio.SeqRecord.SeqRecord(
        seq=Bio.Seq.Seq(aligned_seq),
        id=pep_record.id,
        name='',
        description='',
    )
    return out_record


def backalign_sequence_strings(cdn_seq_raw, pep_seq_raw, codontable, seq_id, emit_terminal_stop_warning):
    cdn_seq = remove_gap_chars(cdn_seq_raw, CDS_GAP_CHARS)
    cdn_seq_upper = cdn_seq.upper()
    num_codons = len(cdn_seq_upper) // 3
    translator = get_codon_translator(codontable=codontable)
    cache = translator['cache']
    stop_codons = translator['stop_codons']
    forward_table = translator['forward_table']
    pep_seq = pep_seq_raw
    pep_seq_upper = pep_seq.upper()
    aligned_codons = [''] * len(pep_seq)
    codon_index = 0

    for i, pep_char in enumerate(pep_seq_upper):
        if pep_char in AA_GAP_CHARS:
            aligned_codons[i] = '---'
            continue

        if codon_index >= num_codons:
            txt = 'Protein alignment had too many non-gap sites for {} at amino acid position {}.'
            raise Exception(txt.format(seq_id, i + 1))

        codon_start = codon_index * 3
        codon = cdn_seq[codon_start:codon_start + 3]
        codon_key = cdn_seq_upper[codon_start:codon_start + 3]
        translated_char = cache.get(codon_key)
        if translated_char is None:
            if codon_key in stop_codons:
                translated_char = '*'
            else:
                try:
                    translated_char = forward_table[codon_key]
                except Exception:
                    translated_char = 'X'
            cache[codon_key] = translated_char
        if (pep_char not in AA_WILDCARD_CHARS) and (pep_char != translated_char):
            txt = 'Amino acid mismatch for {} at aligned position {}: aa_aln={}, translated={}, codon={}'
            raise Exception(txt.format(seq_id, i + 1, pep_seq[i], translated_char, codon))

        aligned_codons[i] = codon
        codon_index += 1

    remaining_codons = num_codons - codon_index
    if remaining_codons == 1:
        terminal_codon_start = codon_index * 3
        terminal_codon = cdn_seq_upper[terminal_codon_start:terminal_codon_start + 3]
        terminal_aa = cache.get(terminal_codon)
        if terminal_aa is None:
            if terminal_codon in stop_codons:
                terminal_aa = '*'
            else:
                try:
                    terminal_aa = forward_table[terminal_codon]
                except Exception:
                    terminal_aa = 'X'
            cache[terminal_codon] = terminal_aa
        is_terminal_stop = terminal_aa == '*'
        if is_terminal_stop:
            if emit_terminal_stop_warning:
                txt = 'Ignored terminal stop codon not present in amino acid alignment: {}\n'
                sys.stderr.write(txt.format(seq_id))
        else:
            txt = 'Unmatched codon remained for {}. The amino acid alignment may be truncated.'
            raise Exception(txt.format(seq_id))
    elif remaining_codons != 0:
        txt = '{} codons remained unmatched for {}. The amino acid alignment may be truncated.'
        raise Exception(txt.format(remaining_codons, seq_id))

    return ''.join(aligned_codons)


def backalign_record_from_cds_record(cdn_record, pep_record_map, codontable):
    pep_record = pep_record_map[cdn_record.id]
    return backalign_record(
        cdn_record=cdn_record,
        pep_record=pep_record,
        codontable=codontable,
    )


def backalign_payload(payload, codontable):
    seq_id, cdn_seq, pep_seq = payload
    aligned_seq = backalign_sequence_strings(
        cdn_seq_raw=cdn_seq,
        pep_seq_raw=pep_seq,
        codontable=codontable,
        seq_id=seq_id,
        emit_terminal_stop_warning=False,
    )
    return seq_id, aligned_seq


def backalign_payloads_process_parallel(payloads, codontable, threads):
    worker = partial(backalign_payload, codontable=codontable)
    max_workers = min(threads, len(payloads))
    chunk_size = max(1, len(payloads) // (max_workers * 16))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, payloads, chunksize=chunk_size))


def backalign_main(args):
    cdn_records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    pep_records = read_seqs(seqfile=args.aa_aln, seqformat=args.inseqformat)
    stop_if_not_multiple_of_three_after_gap_removal(cdn_records)
    stop_if_not_aligned(records=pep_records)
    stop_if_sequence_ids_do_not_match(cdn_records=cdn_records, pep_records=pep_records)

    _ = get_record_map(cdn_records, '--seqfile')
    pep_record_map = get_record_map(pep_records, '--aa_aln')

    threads = resolve_threads(getattr(args, 'threads', 1))
    backaligned_records = None
    if (threads > 1) and (len(cdn_records) >= _PROCESS_PARALLEL_MIN_RECORDS):
        try:
            payloads = [(record.id, str(record.seq), str(pep_record_map[record.id].seq)) for record in cdn_records]
            aligned = backalign_payloads_process_parallel(
                payloads=payloads,
                codontable=args.codontable,
                threads=threads,
            )
            backaligned_records = [
                Bio.SeqRecord.SeqRecord(seq=Bio.Seq.Seq(aligned_seq), id=seq_id, name='', description='')
                for seq_id, aligned_seq in aligned
            ]
        except (OSError, PermissionError):
            sys.stderr.write('Process-based parallelism unavailable; falling back to threads.\n')
    if backaligned_records is None:
        worker = partial(
            backalign_record_from_cds_record,
            pep_record_map=pep_record_map,
            codontable=args.codontable,
        )
        backaligned_records = parallel_map_ordered(items=cdn_records, worker=worker, threads=threads)

    stop_if_not_aligned(records=backaligned_records)
    txt = 'Number of aligned nucleotide sites in output codon alignment: {}\n'
    sys.stderr.write(txt.format(len(backaligned_records[0].seq)))
    write_seqs(records=backaligned_records, outfile=args.outfile, outseqformat=args.outseqformat)
