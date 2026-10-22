"""Row-wise codon arrays using the authoritative scalar scientific semantics."""

from functools import lru_cache
from collections import Counter
from typing import NamedTuple

import numpy as np

from cdskit.codonutil import analyze_codon, classify_codon

_ALPHABET = "ACGTRYSWKMBDHVNX-?."
_SIZE = len(_ALPHABET)
_encoding = bytearray([255] * 256)
for _index, _character in enumerate(_ALPHABET):
    _encoding[ord(_character)] = _encoding[ord(_character.lower())] = _index
_ENCODING = bytes(_encoding)
_HISTOGRAM_BLOCK_CODONS = 65536


class CodonHistogram(NamedTuple):
    """Aggregate semantic flags, position-wise GC counts and ordered usage.

    Counts follow codon_features: clean, missing, ambiguous/invalid, definite
    stop, possible stop, context dependent. GC includes concrete bases even
    within missing or ambiguous codons.
    """

    counts: np.ndarray
    gc_counts: np.ndarray
    gc_denoms: np.ndarray
    usage: Counter[str]


@lru_cache(maxsize=16)
def _tables(codontable):
    # Fill only encountered triplets: small inputs do not pay for all IUPAC
    # expansions, and repeated rows share the same bounded lookup table.
    return np.full((_SIZE**3, 6), -1, dtype=np.int8), np.full(
        _SIZE**3, -1, dtype=np.int8
    )


def _triplet_indices(sequence):
    if len(sequence) % 3:
        return None
    try:
        encoded = sequence.encode("ascii").translate(_ENCODING)
    except UnicodeEncodeError:
        return None
    if b"\xff" in encoded:
        return None
    bases = np.frombuffer(encoded, dtype=np.uint8).reshape(-1, 3)
    indices = bases[:, 0].astype(np.uint16) * _SIZE**2
    indices += bases[:, 1].astype(np.uint16) * _SIZE
    indices += bases[:, 2]
    return indices


def _decode_triplet(index):
    first, rest = divmod(int(index), _SIZE**2)
    second, third = divmod(rest, _SIZE)
    return _ALPHABET[first] + _ALPHABET[second] + _ALPHABET[third]


def _resolved_tables(indices, codontable):
    features, states = _tables(codontable)
    for index in np.unique(indices[states[indices] < 0]):
        codon = _decode_triplet(index)
        meaning = analyze_codon(codon, codontable)
        state = classify_codon(codon, codontable)
        features[index] = (
            state == 0,
            state == 1,
            meaning.ambiguous or meaning.invalid,
            state == 3,
            meaning.possible_stop,
            meaning.context_dependent,
        )
        states[index] = state
    return features, states


def codon_features(sequence, codontable):
    """Return six count flags and class codes, or None outside aligned DNA."""
    indices = _triplet_indices(sequence)
    if indices is None:
        return None
    features, states = _resolved_tables(indices, codontable)
    return features[indices], states[indices]


@lru_cache(maxsize=1)
def _base_weights():
    indices = np.arange(_SIZE**3, dtype=np.uint16)
    bases = np.column_stack(
        (indices // _SIZE**2, (indices // _SIZE) % _SIZE, indices % _SIZE)
    )
    return (
        (bases < 4).astype(np.int8),
        ((bases == 1) | (bases == 2)).astype(np.int8),
        np.all(bases < 4, axis=1),
    )


def codon_histogram(sequence, codontable):
    """Aggregate DNA in bounded blocks, or return None outside aligned DNA.

    Counts use the same scalar-backed flags as codon_features. Temporary index
    and first-occurrence arrays are bounded by 65536 codons and the alphabet,
    rather than the record length. Usage retains first-encounter insertion order.
    """
    if len(sequence) % 3:
        return None
    counts = np.zeros(_SIZE**3, dtype=np.int64)
    gc_denoms, gc_counts, concrete = _base_weights()
    usage_order = []
    for start in range(0, len(sequence), _HISTOGRAM_BLOCK_CODONS * 3):
        indices = _triplet_indices(
            sequence[start : start + _HISTOGRAM_BLOCK_CODONS * 3]
        )
        if indices is None:
            return None
        block_counts = np.bincount(indices, minlength=_SIZE**3)
        unseen = np.flatnonzero((block_counts > 0) & (counts == 0) & concrete)
        if unseen.size:
            first = np.full(_SIZE**3, len(indices), dtype=np.intp)
            np.minimum.at(first, indices, np.arange(len(indices), dtype=np.intp))
            usage_order.extend(unseen[np.argsort(first[unseen])].tolist())
        counts += block_counts
    observed = np.flatnonzero(counts)
    features, _ = _resolved_tables(observed, codontable)
    multiplicities = counts[observed]
    return CodonHistogram(
        multiplicities @ features[observed],
        multiplicities @ gc_counts[observed],
        multiplicities @ gc_denoms[observed],
        Counter({_decode_triplet(index): int(counts[index]) for index in usage_order}),
    )
