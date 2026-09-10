from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any, Literal, TypeAlias

import Bio.Data.CodonTable
from Bio.Data.IUPACData import ambiguous_dna_values


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
CODON_SEMANTICS_VERSION = "2"
CodonContext: TypeAlias = Literal["ordinary", "unknown", "complete_terminal"]


@lru_cache(maxsize=8192)
def expand_dna_codon(codon: str) -> tuple[str, ...]:
    """Return all valid IUPAC DNA expansions, or none for missing/invalid input."""
    codon = codon.upper()
    if len(codon) != 3 or any(ch not in ambiguous_dna_values for ch in codon):
        return ()
    return tuple(
        "".join(bases) for bases in product(*(ambiguous_dna_values[ch] for ch in codon))
    )


def codon_matches_stop_set(
    codon: str, unconditional_stops: set[str] | frozenset[str]
) -> bool:
    """Compatibility for explicit unconditional-stop sets (not raw dual tables)."""
    expansions = expand_dna_codon(codon)
    return bool(expansions) and all(item in unconditional_stops for item in expansions)


@dataclass(frozen=True)
class CodonMeaning:
    """Independent sequence and coding attributes (counts may overlap).

    ``possible_stop`` means uncertain termination, excluding definite stops.
    ``context_dependent`` identifies a sense/stop overlap in the code table.
    A complete_terminal context is an explicit caller assertion, never inferred.
    """

    ambiguous: bool
    missing: bool
    invalid: bool
    partial: bool
    amino_acids: tuple[str, ...]
    definite_stop: bool
    possible_stop: bool
    context_dependent: bool
    terminal_stop_compatible: bool

    @property
    def clean(self) -> bool:
        return not (
            self.ambiguous
            or self.missing
            or self.invalid
            or self.partial
            or self.definite_stop
        )


def analyze_codon(
    codon: str, codontable: CodonTableKey, context: CodonContext = "unknown"
) -> CodonMeaning:
    """Resolve IUPAC expansions without confusing uncertainty with termination."""
    if context not in ("ordinary", "unknown", "complete_terminal"):
        raise ValueError(f"Unknown codon context: {context}")
    return _analyze_codon(codon.upper(), codontable, context)


@lru_cache(maxsize=131072)
def _analyze_codon(
    codon: str, codontable: CodonTableKey, context: CodonContext
) -> CodonMeaning:
    missing = any(ch in MISSING_CHARS for ch in codon)
    invalid = any(
        ch not in ambiguous_dna_values and ch not in MISSING_CHARS for ch in codon
    )
    partial = len(codon) != 3
    ambiguous = not missing and any(
        ch in ambiguous_dna_values and ch not in UNAMBIGUOUS_NT for ch in codon
    )
    if missing or invalid or partial:
        return CodonMeaning(
            ambiguous, missing, invalid, partial, (), False, False, False, False
        )
    table = get_codon_table_components(codontable)
    forward = table["forward_table"]
    stops = table["stop_codons"]
    expansions = expand_dna_codon(codon)
    terminal = all(item in stops for item in expansions)
    definite = all(
        item in stops and (context == "complete_terminal" or item not in forward)
        for item in expansions
    )
    return CodonMeaning(
        ambiguous,
        False,
        False,
        False,
        tuple(sorted({forward[item] for item in expansions if item in forward})),
        definite,
        any(item in stops for item in expansions) and not definite,
        any(item in stops and item in forward for item in expansions),
        terminal,
    )


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
    meaning = analyze_codon(codon_upper, codontable)
    if meaning.missing:
        state = CODON_MISSING
    elif meaning.definite_stop:
        state = CODON_STOP
    elif meaning.ambiguous or meaning.invalid or meaning.partial:
        state = CODON_AMBIGUOUS
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


def summarize_codons(
    seq: str,
    codontable: CodonTableKey,
    terminal_policy: Literal["last_evaluable", "physical"] = "last_evaluable",
) -> dict[str, int | bool]:
    """Count independent attributes; terminal exceptions do not certify a CDS.

    Aligned QC ignores trailing missing codons. Padding uses the physical end,
    so artificial gap padding cannot erase a stop preceding it. A partial tail
    does not establish a terminal complete codon under either policy.
    """
    if terminal_policy not in ("last_evaluable", "physical"):
        raise ValueError(f"Unknown terminal policy: {terminal_policy}")
    counts = [0, 0, 0, 0]
    last_evaluable_index = None
    stop_indices = []
    possible_indices = []
    context_indices = []
    ambiguous = 0
    total_codons = len(seq) // 3
    for codon_index in range(total_codons):
        start = codon_index * 3
        codon = seq[start : start + 3]
        meaning = analyze_codon(codon, codontable)
        state = classify_codon(codon, codontable=codontable)
        ambiguous += int(meaning.ambiguous or meaning.invalid)
        counts[state] += 1
        if state != CODON_MISSING:
            last_evaluable_index = codon_index
        if state == CODON_STOP:
            stop_indices.append(codon_index)
        if meaning.possible_stop:
            possible_indices.append(codon_index)
        if meaning.context_dependent:
            context_indices.append(codon_index)
    terminal_index = (
        last_evaluable_index
        if terminal_policy == "last_evaluable"
        else total_codons - 1
    )
    if len(seq) % 3:
        terminal_index = None
    internal_count = sum(index != terminal_index for index in stop_indices)
    return {
        "total": total_codons,
        "clean": counts[CODON_CLEAN],
        "missing": counts[CODON_MISSING],
        "ambiguous": ambiguous,
        "stop": counts[CODON_STOP],
        "evaluable": total_codons - counts[CODON_MISSING],
        "internal_stop": internal_count > 0,
        "internal_stop_count": internal_count,
        "possible_stop": len(possible_indices),
        "context_dependent": len(context_indices),
        "internal_possible_stop_count": sum(
            index != terminal_index for index in possible_indices
        ),
        "internal_context_dependent_count": sum(
            index != terminal_index for index in context_indices
        ),
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
