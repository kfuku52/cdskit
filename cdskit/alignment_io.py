"""FASTA identity and sequence preparation for MAFFT and domain search."""

from __future__ import annotations

import csv
import gzip
import hashlib
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

from Bio.SeqIO.FastaIO import SimpleFastaParser

from cdskit.atomicio import atomic_output_paths, validate_distinct_paths
from cdskit.sequence_io import open_text
from cdskit.translate import translate_sequence_string

GAPS = str.maketrans("", "", "-.")
DNA_SYMBOLS = set("ACGTRYSWKMBDHVN")
AA_SYMBOLS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*?")


def read_fasta(path, *, aligned=False):
    with nullcontext(sys.stdin) if str(path) == "-" else open_text(path) as handle:
        sequences = {}
        for title, row in SimpleFastaParser(handle):
            fields = title.split(None, 1)
            gene = fields[0] if fields else ""
            row = row.upper()
            if not aligned:
                row = row.translate(GAPS)
            if not gene or gene in sequences or not row:
                raise ValueError(
                    f"Expected unique IDs and nonempty FASTA sequences in {path}: {gene!r}"
                )
            sequences[gene] = row
    if not sequences:
        raise ValueError(f"No FASTA records in {path}.")
    return sequences


def validate_alphabet(rows, seq_type):
    alphabet = AA_SYMBOLS if seq_type == "aa" else DNA_SYMBOLS
    for gene, sequence in rows.items():
        if not alphabet.issuperset(sequence):
            raise ValueError(f"Unexpected {seq_type} character in {gene}.")
        if seq_type == "codon" and len(sequence) % 3:
            raise ValueError(f"CDS length must be a multiple of three: {gene}.")


def alignment_sequences(raw, seq_type, codon_table, aa_file=None):
    validate_alphabet(raw, seq_type)
    if seq_type != "codon":
        if aa_file:
            raise ValueError("--aa_file requires --seq_type codon.")
        return raw
    translated = {
        g: translate_sequence_string(s, codon_table, False) for g, s in raw.items()
    }
    if not aa_file:
        return translated
    aa = read_fasta(aa_file)
    validate_alphabet(aa, "aa")
    if aa.keys() != translated.keys():
        raise ValueError("Precomputed AA IDs must match the CDS IDs exactly.")
    for g, canonical in translated.items():
        if len(canonical) != len(aa[g]) or any(
            found not in (expected, "X")
            for expected, found in zip(canonical, aa[g], strict=True)
        ):
            raise ValueError(f"Precomputed AA does not match the translation for {g}.")
    return {g: aa[g] for g in raw}


def write_fasta(path: Path, sequences: dict[str, str], ids: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for gene in ids:
            handle.write(f">{gene}\n{sequences[gene]}\n")


def sequence_digest(sequences):
    hasher = hashlib.sha256()
    for gene in sorted(sequences):
        hasher.update(f">{gene}\n{sequences[gene]}\n".encode())
    return hasher.hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_table(path, header, rows, *, quoting=csv.QUOTE_MINIMAL):
    opener = gzip.open if Path(path).suffix.lower() == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(
            handle,
            delimiter="\t",
            lineterminator="\n",
            quoting=quoting,
            quotechar=None if quoting == csv.QUOTE_NONE else '"',
        )
        writer.writerow(header)
        writer.writerows(rows)


@contextmanager
def staged_outputs(inputs, outputs) -> Iterator[dict]:
    """Use the shared transaction for files and keep one stream on stdout."""
    targets = {key: value for key, value in outputs.items() if value is not None}
    if sum(value == "-" for value in targets.values()) > 1:
        raise ValueError("Only one output may use standard output.")
    paths = {key: value for key, value in targets.items() if value != "-"}
    validate_distinct_paths(inputs=inputs, outputs=paths.values())
    with atomic_output_paths(paths.values()) as staged:
        yield dict(targets, **dict(zip(paths, staged, strict=True)))
