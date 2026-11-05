"""Strict, streaming FASTA artifact profiles, independent of quality reporting."""

from __future__ import annotations
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

DNA_ALPHABET = frozenset("ACGTURYSWKMBDHVN-?.")
PROTEIN_ALPHABET = frozenset("ABCDEFGHIKLMNPQRSTVWXYZJUO*-?.")


class SequenceContractError(RuntimeError):
    """Raised when a FASTA artifact violates its declared sequence type."""


@dataclass(frozen=True)
class FastaSummary:
    records: int
    residues: int
    aligned_length: int


def _open_text(path: Path) -> TextIO:
    with path.open("rb") as handle:
        magic = handle.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="strict")
    return path.open("r", encoding="utf-8", errors="strict")


def validate_fasta(path: Path, expected: str) -> FastaSummary:
    """Validate one plain or gzip-compressed FASTA file.

    ``codon`` requires an aligned IUPAC nucleotide FASTA whose aligned and
    ungapped sequence lengths are multiples of three. ``dna`` validates only
    the alphabet, while ``protein`` accepts the IUPAC amino-acid alphabet.
    """

    if expected not in {"dna", "codon", "protein"}:
        raise SequenceContractError(f"Unsupported FASTA sequence type: {expected}")
    if not path.is_file():
        raise FileNotFoundError(path)

    with _open_text(path) as handle:
        return validate_fasta_stream(handle, expected, path=path)


def validate_fasta_stream(
    handle: TextIO, expected: str, *, path: Path = Path("<stream>")
) -> FastaSummary:
    """Apply the same sequence checks to raw or archive-backed text streams."""
    if expected not in {"dna", "codon", "protein"}:
        raise SequenceContractError(f"Unsupported FASTA sequence type: {expected}")
    allowed = PROTEIN_ALPHABET if expected == "protein" else DNA_ALPHABET
    record_count = 0
    aligned_length: int | None = None
    has_empty = unequal_lengths = non_triplet_ungapped = False
    current_length: int | None = None
    current_ungapped = 0
    invalid: set[str] = set()
    residues = 0

    def finish_record():
        nonlocal \
            record_count, \
            aligned_length, \
            has_empty, \
            unequal_lengths, \
            non_triplet_ungapped
        if current_length is None:
            return
        record_count += 1
        if aligned_length is None:
            aligned_length = current_length
        unequal_lengths |= current_length != aligned_length
        has_empty |= current_length == 0
        non_triplet_ungapped |= current_ungapped % 3 != 0

    for line_number, raw_line in enumerate(handle, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if len(line) == 1:
                raise SequenceContractError(
                    f"Empty FASTA header at line {line_number}: {path}"
                )
            if current_length is not None:
                finish_record()
            current_length = 0
            current_ungapped = 0
            continue
        if current_length is None:
            raise SequenceContractError(
                f"FASTA sequence data appears before the first header at line {line_number}: {path}"
            )
        sequence = "".join(line.split()).upper()
        invalid.update(set(sequence) - allowed)
        current_length += len(sequence)
        current_ungapped += (
            len(sequence)
            - sequence.count("-")
            - sequence.count("?")
            - sequence.count(".")
        )
        residues += len(sequence)

    if current_length is not None:
        finish_record()
    if not record_count:
        raise SequenceContractError(f"FASTA contains no records: {path}")
    if not residues or has_empty:
        raise SequenceContractError(f"FASTA contains an empty sequence: {path}")
    if invalid:
        rendered = "".join(sorted(invalid))
        raise SequenceContractError(
            f"FASTA contains characters outside the {expected} alphabet: {rendered} ({path})"
        )

    assert aligned_length is not None
    if expected == "codon":
        if unequal_lengths:
            raise SequenceContractError(f"Codon FASTA is not an alignment: {path}")
        if aligned_length % 3:
            raise SequenceContractError(
                f"Codon FASTA aligned length is not a multiple of three: {aligned_length} ({path})"
            )
        if non_triplet_ungapped:
            raise SequenceContractError(
                f"Codon FASTA has ungapped sequence lengths that are not multiples of three: {path}"
            )

    return FastaSummary(
        records=record_count,
        residues=residues,
        aligned_length=aligned_length,
    )
