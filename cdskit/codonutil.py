import Bio.Data.CodonTable


MISSING_CHARS = frozenset('-?.')
GAP_ONLY_CHARS = frozenset('-?.NXnx')
UNAMBIGUOUS_NT = frozenset('ACGT')
DNA_BASES = ('A', 'C', 'G', 'T')

_CODON_TABLE_CACHE = dict()
_DEGENERACY_CACHE = dict()


def get_codon_table_components(codontable):
    cached = _CODON_TABLE_CACHE.get(codontable)
    if cached is not None:
        return cached
    try:
        table = Bio.Data.CodonTable.unambiguous_dna_by_id[int(codontable)]
    except (KeyError, TypeError, ValueError):
        table = Bio.Data.CodonTable.unambiguous_dna_by_name[str(codontable)]
    cached = {
        'forward_table': {codon.upper(): aa for codon, aa in table.forward_table.items()},
        'stop_codons': frozenset(codon.upper() for codon in table.stop_codons),
        'start_codons': frozenset(codon.upper() for codon in table.start_codons),
    }
    _CODON_TABLE_CACHE[codontable] = cached
    return cached


def get_forward_table(codontable):
    return get_codon_table_components(codontable=codontable)['forward_table']


def get_stop_codons(codontable):
    return get_codon_table_components(codontable=codontable)['stop_codons']


def codon_has_missing(codon):
    codon_upper = codon.upper()
    return any(ch in MISSING_CHARS for ch in codon_upper)


def codon_is_gap_only(codon):
    return len(codon) > 0 and all(ch in GAP_ONLY_CHARS for ch in codon)


def codon_is_ambiguous(codon):
    codon_upper = codon.upper()
    if codon_has_missing(codon_upper):
        return False
    return any(ch not in UNAMBIGUOUS_NT for ch in codon_upper)


def codon_is_stop(codon, codontable):
    codon_upper = codon.upper()
    if codon_has_missing(codon_upper):
        return False
    if any(ch not in UNAMBIGUOUS_NT for ch in codon_upper):
        return False
    return codon_upper in get_stop_codons(codontable=codontable)


def ambiguous_codon_counts(seq):
    seq_upper = seq.upper()
    ambiguous = 0
    evaluable = 0
    for i in range(0, len(seq_upper) - 2, 3):
        codon = seq_upper[i:i + 3]
        if codon_has_missing(codon):
            continue
        evaluable += 1
        if any(ch not in UNAMBIGUOUS_NT for ch in codon):
            ambiguous += 1
    return ambiguous, evaluable


def is_gap_only_sequence(seq):
    return len(seq) > 0 and all(ch in GAP_ONLY_CHARS for ch in seq)


def has_internal_stop(seq, codontable):
    stop_codons = get_stop_codons(codontable=codontable)
    codons = [seq[i:i + 3].upper() for i in range(0, len(seq) - 2, 3)]
    evaluable_indices = [i for i, codon in enumerate(codons) if not codon_has_missing(codon)]
    if len(evaluable_indices) <= 1:
        return False
    terminal_index = evaluable_indices[-1]
    for i in evaluable_indices:
        if i == terminal_index:
            continue
        codon = codons[i]
        if any(ch not in UNAMBIGUOUS_NT for ch in codon):
            continue
        if codon in stop_codons:
            return True
    return False


def degeneracy_fold_by_position(codon, codontable):
    codon_upper = codon.upper()
    cache = _DEGENERACY_CACHE.get(codontable)
    if cache is None:
        cache = dict()
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
    folds = list()
    for pos in range(3):
        synonymous = 0
        for base in DNA_BASES:
            alt = codon_upper[:pos] + base + codon_upper[pos + 1:]
            if forward_table.get(alt) == aa:
                synonymous += 1
        folds.append(0 if synonymous == 1 else synonymous)
    result = tuple(folds)
    cache[codon_upper] = result
    return result
