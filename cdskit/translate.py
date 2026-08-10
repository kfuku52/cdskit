import Bio.Data.CodonTable
import Bio.SeqIO
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from cdskit.atomicio import atomic_output_path
from cdskit.util import (
    iter_seq_chunks,
    read_seqs,
    resolve_threads,
    stop_if_invalid_codontable,
    stop_if_not_dna,
    stop_if_not_multiple_of_three,
    write_seqs,
)

_CODON_TRANSLATOR_CACHE: dict = {}
_TRANSLATION_ALPHABET = "ACGTRYSWKMBDHVNX-?."


def get_codon_translator(codontable):
    cached = _CODON_TRANSLATOR_CACHE.get(codontable)
    if cached is not None:
        return cached
    alphabet_size = len(_TRANSLATION_ALPHABET)
    encode_table = bytearray([255] * 256)
    for code, character in enumerate(_TRANSLATION_ALPHABET):
        encode_table[ord(character)] = code
        encode_table[ord(character.lower())] = code
    aa_lut = np.empty(alphabet_size**3, dtype=np.uint8)
    error_lut = np.zeros(alphabet_size**3, dtype=bool)
    errors_by_index = {}
    for first_code, first in enumerate(_TRANSLATION_ALPHABET):
        for second_code, second in enumerate(_TRANSLATION_ALPHABET):
            for third_code, third in enumerate(_TRANSLATION_ALPHABET):
                codon = first + second + third
                index = (
                    first_code * alphabet_size * alphabet_size
                    + second_code * alphabet_size
                    + third_code
                )
                if all(ch in "-." for ch in codon):
                    aa = "-"
                elif any(ch in "-?." for ch in codon):
                    aa = "X"
                else:
                    try:
                        aa = str(
                            Seq(codon).translate(
                                table=codontable, to_stop=False, gap="-"
                            )
                        )
                    except Exception as exc:
                        errors_by_index[index] = str(exc)
                        error_lut[index] = True
                        aa = "X"
                aa_lut[index] = ord(aa)
    translator = {
        "alphabet_size": alphabet_size,
        "encode_table": bytes(encode_table),
        "aa_lut": aa_lut,
        "error_lut": error_lut,
        "errors_by_index": errors_by_index,
    }
    _CODON_TRANSLATOR_CACHE[codontable] = translator
    return translator


def translate_sequence_codes(seq_str, codontable, to_stop=False):
    """Translate complete codons to an ASCII uint8 array without string assembly."""
    translator = get_codon_translator(codontable=codontable)
    if len(seq_str) == 0:
        return np.empty(0, dtype=np.uint8)
    if len(seq_str) % 3 != 0:
        raise Bio.Data.CodonTable.TranslationError(
            "Sequence length should be a multiple of three."
        )
    try:
        encoded = seq_str.encode("ascii").translate(translator["encode_table"])
    except UnicodeEncodeError:
        encoded = b"\xff"
    codes = np.frombuffer(encoded, dtype=np.uint8)
    if np.any(codes == 255):
        try:
            translated = str(
                Seq(seq_str).translate(
                    table=codontable,
                    to_stop=to_stop,
                    gap="-",
                )
            )
            return np.frombuffer(translated.encode("ascii"), dtype=np.uint8)
        except Exception as exc:
            raise Bio.Data.CodonTable.TranslationError(str(exc)) from exc
    codons = codes.reshape(-1, 3).astype(np.int32, copy=False)
    alphabet_size = translator["alphabet_size"]
    indices = (
        codons[:, 0] * alphabet_size * alphabet_size
        + codons[:, 1] * alphabet_size
        + codons[:, 2]
    )
    amino_acids = translator["aa_lut"][indices]
    limit = len(indices)
    if to_stop:
        stop_indices = np.flatnonzero(amino_acids == ord("*"))
        if len(stop_indices) > 0:
            limit = int(stop_indices[0])
    error_positions = np.flatnonzero(translator["error_lut"][indices[:limit]])
    if len(error_positions) > 0:
        error_index = int(indices[int(error_positions[0])])
        raise Bio.Data.CodonTable.TranslationError(
            translator["errors_by_index"][error_index]
        )
    return amino_acids[:limit]


def translate_sequence_string(seq_str, codontable, to_stop):
    if len(seq_str) % 3 != 0:
        # CLI callers reject partial codons, but retain the historical helper
        # behavior for direct callers that use missing or gap tail fragments.
        amino_acids = []
        for start in range(0, len(seq_str), 3):
            codon = seq_str[start : start + 3].upper()
            if all(ch in "-." for ch in codon):
                amino_acid = "-"
            elif any(ch in "-?." for ch in codon):
                amino_acid = "X"
            else:
                try:
                    amino_acid = str(
                        Seq(codon).translate(
                            table=codontable,
                            to_stop=False,
                            gap="-",
                        )
                    )
                except Exception as exc:
                    raise Bio.Data.CodonTable.TranslationError(str(exc)) from exc
            if to_stop and amino_acid == "*":
                break
            amino_acids.append(amino_acid)
        return "".join(amino_acids)
    amino_acids = translate_sequence_codes(
        seq_str=seq_str,
        codontable=codontable,
        to_stop=to_stop,
    )
    return amino_acids.tobytes().decode("ascii")


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


def translate_records(records, codontable, to_stop, threads):
    del threads
    return [
        translate_record(record, codontable=codontable, to_stop=to_stop)
        for record in records
    ]


def translate_main(args):
    stop_if_invalid_codontable(args.codontable)
    threads = resolve_threads(getattr(args, "threads", 1))
    if args.outfile != "-" and str(args.outseqformat).lower() in {
        "fasta",
        "fasta-2line",
        "tab",
    }:
        output_count = 0
        with atomic_output_path(args.outfile) as temporary:
            with open(temporary, "w", encoding="utf-8") as output_handle:
                for records in iter_seq_chunks(
                    seqfile=args.seqfile,
                    seqformat=args.inseqformat,
                    max_chunk_residues=8_000_000,
                ):
                    stop_if_not_dna(records=records, label="--seq_file")
                    stop_if_not_multiple_of_three(records=records)
                    translated_records = translate_records(
                        records=records,
                        codontable=args.codontable,
                        to_stop=args.to_stop,
                        threads=threads,
                    )
                    output_count += Bio.SeqIO.write(
                        translated_records,
                        output_handle,
                        args.outseqformat,
                    )
        import sys

        sys.stderr.write("Number of output sequences: {:,}\n".format(output_count))
        return
    records = read_seqs(seqfile=args.seqfile, seqformat=args.inseqformat)
    stop_if_not_dna(records=records, label="--seq_file")
    if len(records) == 0:
        write_seqs(
            records=records, outfile=args.outfile, outseqformat=args.outseqformat
        )
        return
    stop_if_not_multiple_of_three(records=records)
    translated_records = translate_records(
        records=records,
        codontable=args.codontable,
        to_stop=args.to_stop,
        threads=threads,
    )
    write_seqs(
        records=translated_records, outfile=args.outfile, outseqformat=args.outseqformat
    )
