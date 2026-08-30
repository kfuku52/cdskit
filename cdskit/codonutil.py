from __future__ import annotations

from typing import Any, TypeAlias

import Bio.Data.CodonTable


MISSING_CHARS = frozenset("-?.")
GAP_ONLY_CHARS = frozenset("-?.NXnx")
UNAMBIGUOUS_NT = frozenset("ACGT")
DNA_BASES = ("A", "C", "G", "T")

CodonTableKey: TypeAlias = int | str
CodonComponents: TypeAlias = dict[str, Any]

_CODON_TABLE_CACHE: dict[CodonTableKey, CodonComponents] = {}
_CODON_TRANSLATOR_CACHE: dict[CodonTableKey, CodonComponents] = {}
_CODON_CLASSIFICATION_CACHE: dict[CodonTableKey, dict[str, int]] = {}
_DEGENERACY_CACHE: dict[
    CodonTableKey,
    dict[str, tuple[int, int, int] | None],
] = {}

CODON_CLEAN = 0
CODON_MISSING = 1
CODON_AMBIGUOUS = 2
CODON_STOP = 3


def get_codon_translator(codontable: CodonTableKey) -> CodonComponents:
    """Cache an ambiguous-DNA translator with forward-table precedence.

    Context-dependent stop codons (tables 27, 28 and 31) encode amino acids in
    ordinary translation. Terminal-stop acceptance is a separate CDS decision.
    """
    if codontable not in _CODON_TRANSLATOR_CACHE:
        try:
            table = Bio.Data.CodonTable.ambiguous_dna_by_id[int(codontable)]
        except (KeyError, TypeError, ValueError):
            table = Bio.Data.CodonTable.ambiguous_dna_by_name[str(codontable)]
        _CODON_TRANSLATOR_CACHE[codontable] = {
            "forward_table": table.forward_table,
            "stop_codons": frozenset(table.stop_codons),
            "cache": {},
        }
    return _CODON_TRANSLATOR_CACHE[codontable]


def translate_single_codon(codon: str, translator: CodonComponents) -> str:
    """Translate a codon; raise KeyError for invalid DNA, use X for ambiguity."""
    codon = codon.upper()
    cache: dict[str, str] = translator["cache"]
    if codon not in cache:
        try:
            aa = translator["forward_table"][codon]
        except Bio.Data.CodonTable.TranslationError:
            aa = "X"
        except KeyError:
            if codon not in translator["stop_codons"]:
                raise
            aa = "*"
        cache[codon] = aa
    return cache[codon]


def get_codon_table_components(codontable: CodonTableKey) -> CodonComponents:
    cached = _CODON_TABLE_CACHE.get(codontable)
    if cached is not None:
        return cached
    try:
        table = Bio.Data.CodonTable.unambiguous_dna_by_id[int(codontable)]
    except (KeyError, TypeError, ValueError):
        table = Bio.Data.CodonTable.unambiguous_dna_by_name[str(codontable)]
    cached = {
        "forward_table": {
            codon.upper(): aa for codon, aa in table.forward_table.items()
        },
        "stop_codons": frozenset(codon.upper() for codon in table.stop_codons),
        "start_codons": frozenset(codon.upper() for codon in table.start_codons),
    }
    _CODON_TABLE_CACHE[codontable] = cached
    return cached


def get_forward_table(codontable: CodonTableKey) -> dict[str, str]:
    return get_codon_table_components(codontable=codontable)["forward_table"]


def get_stop_codons(codontable: CodonTableKey) -> frozenset[str]:
    return get_codon_table_components(codontable=codontable)["stop_codons"]


def classify_codon(codon: str, codontable: CodonTableKey) -> int:
    """Classify a codon once and cache the result for the selected code."""
    codon_upper = codon.upper()
    cache = _CODON_CLASSIFICATION_CACHE.get(codontable)
    if cache is None:
        cache = {}
        _CODON_CLASSIFICATION_CACHE[codontable] = cache
    cached = cache.get(codon_upper)
    if cached is not None:
        return cached
    if any(ch in MISSING_CHARS for ch in codon_upper):
        state = CODON_MISSING
    elif any(ch not in UNAMBIGUOUS_NT for ch in codon_upper):
        state = CODON_AMBIGUOUS
    elif codon_upper in get_stop_codons(codontable=codontable):
        state = CODON_STOP
    else:
        state = CODON_CLEAN
    cache[codon_upper] = state
    return state


def codon_has_missing(codon: str) -> bool:
    codon_upper = codon.upper()
    return any(ch in MISSING_CHARS for ch in codon_upper)


def codon_is_gap_only(codon: str) -> bool:
    return len(codon) > 0 and all(ch in GAP_ONLY_CHARS for ch in codon)


def codon_is_ambiguous(codon: str) -> bool:
    codon_upper = codon.upper()
    if codon_has_missing(codon_upper):
        return False
    return any(ch not in UNAMBIGUOUS_NT for ch in codon_upper)


def codon_is_stop(codon: str, codontable: CodonTableKey) -> bool:
    return classify_codon(codon=codon, codontable=codontable) == CODON_STOP


def codon_is_clean(codon: str, codontable: CodonTableKey) -> bool:
    return classify_codon(codon=codon, codontable=codontable) == CODON_CLEAN


def ambiguous_codon_counts(seq: str) -> tuple[int, int]:
    seq_upper = seq.upper()
    ambiguous = 0
    evaluable = 0
    for i in range(0, len(seq_upper) - 2, 3):
        codon = seq_upper[i : i + 3]
        if codon_has_missing(codon):
            continue
        evaluable += 1
        if any(ch not in UNAMBIGUOUS_NT for ch in codon):
            ambiguous += 1
    return ambiguous, evaluable


def summarize_codons(seq: str, codontable: CodonTableKey) -> dict[str, int | bool]:
    """Return codon-state counts and internal-stop status in one pass."""
    counts = [0, 0, 0, 0]
    last_evaluable_index = None
    stop_indices = []
    total_codons = len(seq) // 3
    for codon_index in range(total_codons):
        start = codon_index * 3
        state = classify_codon(seq[start : start + 3], codontable=codontable)
        counts[state] += 1
        if state != CODON_MISSING:
            last_evaluable_index = codon_index
        if state == CODON_STOP:
            stop_indices.append(codon_index)
    internal_stop = last_evaluable_index is not None and any(
        index != last_evaluable_index for index in stop_indices
    )
    return {
        "total": total_codons,
        "clean": counts[CODON_CLEAN],
        "missing": counts[CODON_MISSING],
        "ambiguous": counts[CODON_AMBIGUOUS],
        "stop": counts[CODON_STOP],
        "evaluable": total_codons - counts[CODON_MISSING],
        "internal_stop": internal_stop,
    }


def is_gap_only_sequence(seq: str) -> bool:
    return len(seq) > 0 and all(ch in GAP_ONLY_CHARS for ch in seq)


def has_internal_stop(seq: str, codontable: CodonTableKey) -> bool:
    return bool(summarize_codons(seq=seq, codontable=codontable)["internal_stop"])


def degeneracy_fold_by_position(
    codon: str,
    codontable: CodonTableKey,
) -> tuple[int, int, int] | None:
    codon_upper = codon.upper()
    cache = _DEGENERACY_CACHE.get(codontable)
    if cache is None:
        cache = {}
        _DEGENERACY_CACHE[codontable] = cache
    if codon_upper in cache:
        return cache[codon_upper]
    if codon_has_missing(codon_upper):
        cache[codon_upper] = None
        return None
    if any(ch not in UNAMBIGUOUS_NT for ch in codon_upper):
        cache[codon_upper] = None
        return None
    forward_table = get_forward_table(codontable=codontable)
    aa = forward_table.get(codon_upper)
    if aa is None:
        cache[codon_upper] = None
        return None
    folds: list[int] = []
    for pos in range(3):
        synonymous = 0
        for base in DNA_BASES:
            alt = codon_upper[:pos] + base + codon_upper[pos + 1 :]
            if forward_table.get(alt) == aa:
                synonymous += 1
        folds.append(0 if synonymous == 1 else synonymous)
    result = (folds[0], folds[1], folds[2])
    cache[codon_upper] = result
    return result
