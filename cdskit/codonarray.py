"""Row-wise codon arrays using the authoritative scalar scientific semantics."""

from functools import lru_cache

import numpy as np

from cdskit.codonutil import analyze_codon, classify_codon

_ALPHABET = "ACGTRYSWKMBDHVNX-?."
_SIZE = len(_ALPHABET)
_encoding = bytearray([255] * 256)
for _index, _character in enumerate(_ALPHABET):
    _encoding[ord(_character)] = _encoding[ord(_character.lower())] = _index
_ENCODING = bytes(_encoding)


@lru_cache(maxsize=16)
def _tables(codontable):
    # Fill only encountered triplets: small inputs do not pay for all IUPAC
    # expansions, and repeated rows share the same bounded lookup table.
    return np.full((_SIZE**3, 6), -1, dtype=np.int8), np.full(
        _SIZE**3, -1, dtype=np.int8
    )


def codon_features(sequence, codontable):
    """Return six count flags and class codes, or None outside aligned DNA."""
    if len(sequence) % 3:
        return None
    try:
        encoded = sequence.encode("ascii").translate(_ENCODING)
    except UnicodeEncodeError:
        return None
    if b"\xff" in encoded:
        return None
    bases = np.frombuffer(encoded, dtype=np.uint8).astype(np.int32).reshape(-1, 3)
    indices = bases[:, 0] * _SIZE**2 + bases[:, 1] * _SIZE + bases[:, 2]
    features, states = _tables(codontable)
    for index in np.unique(indices[states[indices] < 0]):
        first, rest = divmod(int(index), _SIZE**2)
        second, third = divmod(rest, _SIZE)
        codon = _ALPHABET[first] + _ALPHABET[second] + _ALPHABET[third]
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
    return features[indices], states[indices]
