"""Policy-explicit alignment lengths and codon-position nucleotide frequencies."""

from pathlib import Path
from collections import Counter
import sys


def alignment_length_stats(file) -> dict[str, int]:
    """Return num_site, num_seq, len_max, and len_min for UTF-8 FASTA.

    Gapped lengths must match. Only '-' is excluded from ungapped lengths;
    the sequence alphabet is not validated. An empty file returns all zeros.
    """
    file = Path(file)
    count = 0
    aligned_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    unequal = False
    seq_w_gap_len = 0
    seq_len = 0
    has_sequence = False

    def finish_record():
        nonlocal count, aligned_length, minimum, maximum, unequal
        count += 1
        if aligned_length is None:
            aligned_length = seq_w_gap_len
        unequal |= seq_w_gap_len != aligned_length
        if minimum is None or seq_len < minimum:
            minimum = seq_len
        if maximum is None or seq_len > maximum:
            maximum = seq_len

    try:
        with file.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith(">"):
                    if has_sequence:
                        finish_record()
                    seq_w_gap_len = 0
                    seq_len = 0
                    has_sequence = True
                    continue
                seq_line = line.strip()
                if not seq_line:
                    continue
                if not has_sequence:
                    raise ValueError(
                        "alignment file must be FASTA-formatted with header lines starting with '>'"
                    )
                seq_w_gap_len += len(seq_line)
                seq_len += len(seq_line) - seq_line.count("-")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Failed to read file: {file}") from exc
    if has_sequence:
        finish_record()

    if not count:
        return dict(num_site=0, num_seq=0, len_max=0, len_min=0)

    if unequal:
        raise ValueError("all FASTA sequences must have the same aligned length")

    assert aligned_length is not None and minimum is not None and maximum is not None
    return dict(
        num_site=aligned_length, num_seq=count, len_max=maximum, len_min=minimum
    )


VALID_NUCLEOTIDES = ("A", "C", "G", "T")
_BASE_CODES = bytes(
    VALID_NUCLEOTIDES.index(chr(code)) if chr(code) in VALID_NUCLEOTIDES else 4
    for code in range(256)
)
_FREQUENCY_BLOCK_NT = 3 * 65_536
_FREQUENCY_ARRAY_MIN_NT = 16_384
_FREQUENCY_COLD_MIN_NT = 3_000_000


def normalize_nuc_counts(counts):
    total = sum(counts.values())
    if not total:
        return {key: 0.25 for key in VALID_NUCLEOTIDES}
    return {key: value / total for key, value in counts.items()}


def _concrete_codon_histogram(sequence, usable_len):
    """Count uppercase ASCII A/C/G/T triplets in a complete-codon prefix."""
    import numpy as np

    histogram = np.zeros(64, dtype=np.int64)
    for start in range(0, usable_len, _FREQUENCY_BLOCK_NT):
        end = min(start + _FREQUENCY_BLOCK_NT, usable_len)
        encoded = sequence[start:end].encode("ascii").translate(_BASE_CODES)
        first, second, third = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 3).T
        # The fifth code sets bit 2 in any unusable position.
        valid = (first | second | third) < 4
        keys = first * 16 + second * 4 + third
        histogram += np.bincount(keys if valid.all() else keys[valid], minlength=64)
    return histogram


def _positional_counts_ascii(sequence, usable_len):
    codons = _concrete_codon_histogram(sequence, usable_len).reshape(4, 4, 4)
    return (codons.sum(axis=(1, 2)), codons.sum(axis=(0, 2)), codons.sum(axis=(0, 1)))


def positional_nucleotide_frequencies(sequences):
    """F3X4: U→T, omit an entire ambiguous codon, include definite stop codons."""
    counts = [{nuc: 0.0 for nuc in VALID_NUCLEOTIDES} for _ in range(3)]
    long_ascii_nt = 0
    for sequence in sequences:
        seq = str(sequence).upper().replace("U", "T")
        usable_len = len(seq) - (len(seq) % 3)
        # Short sequences seldom repeat enough codons to offset aggregation.
        long_ascii = usable_len >= _FREQUENCY_ARRAY_MIN_NT and seq.isascii()
        if long_ascii:
            long_ascii_nt += usable_len
        # Avoid the first NumPy import for small standalone library workloads.
        if long_ascii and (
            long_ascii_nt >= _FREQUENCY_COLD_MIN_NT or "numpy" in sys.modules
        ):
            for position, values in zip(
                counts, _positional_counts_ascii(seq, usable_len), strict=True
            ):
                for base, value in zip(VALID_NUCLEOTIDES, values, strict=True):
                    position[base] += int(value)
        elif usable_len < 384:
            for i in range(0, usable_len, 3):
                codon = seq[i : (i + 3)]
                if any(base not in VALID_NUCLEOTIDES for base in codon):
                    continue
                for pos, base in enumerate(codon):
                    counts[pos][base] += 1.0
        else:
            codon_counts = Counter(seq[i : (i + 3)] for i in range(0, usable_len, 3))
            for codon, count in codon_counts.items():
                if any(base not in VALID_NUCLEOTIDES for base in codon):
                    continue
                for pos, base in enumerate(codon):
                    counts[pos][base] += count
    return [normalize_nuc_counts(c) for c in counts]
