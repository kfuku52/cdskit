"""Batched Yang--Nielsen (2000) estimation with equal-weighted sense paths.

The estimator uses F3x4 frequencies and nondegenerate/fourfold sites to infer
kappa. Its weighting=0 numerical contract is checked against PAML/YN00; it is
not a substitute for codeml maximum likelihood or weighted-path YN00.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from itertools import permutations, product
from typing import Any

import numpy as np

from cdskit.atomicio import validate_distinct_paths, validate_output_paths
from cdskit.codonutil import (
    CODON_SEMANTICS_VERSION,
    definite_stop_patterns,
    get_forward_table,
)
from cdskit.tsvio import TSV_REPORT_SCHEMA_VERSION, read_tsv, write_tsv
from cdskit.util import resolve_threads

BASES = "TCAG"
BATCH_SIZE = 256
METHOD = "YN00_weighting0_F3x4"
DEFAULT_KAPPA = 4.6  # YN00 initializes each independent data set with this value.
REPORT_COLUMNS = (
    "schema_version",
    "pair_id",
    "method",
    "codon_table",
    "codon_semantics_version",
    "dS",
    "dN",
    "dS_diagnostic",
    "dN_diagnostic",
    "kappa",
    "S",
    "N",
    "aligned_codons",
    "retained_codons",
    "excluded_codons",
    "dS_model",
    "dN_model",
    "kappa_defaulted",
    "status",
)


@lru_cache(maxsize=1)
def _codon_lookup() -> Any:
    """Map three validated ASCII bytes to a concrete/missing/invalid codon."""
    lookup = np.full(1 << 21, -2, dtype=np.int8)
    alphabet = "ACGTRYSWKMBDHVNX-?."
    chars = np.array([ord(base) for base in alphabet], dtype=np.int32)
    bases = np.array([BASES.find(base) for base in alphabet], dtype=np.int16)
    keys = (
        (chars[:, None, None] << 14)
        | (chars[None, :, None] << 7)
        | chars[None, None, :]
    )
    concrete = (
        (bases[:, None, None] >= 0)
        & (bases[None, :, None] >= 0)
        & (bases[None, None, :] >= 0)
    )
    values = 16 * bases[:, None, None] + 4 * bases[None, :, None] + bases[None, None, :]
    lookup[keys] = np.where(concrete, values, -1)
    lookup.flags.writeable = False
    return lookup


@lru_cache(maxsize=64)
def _geometry(code: int) -> dict[str, Any]:
    forward = get_forward_table(code)
    codons = ["".join(parts) for parts in product(BASES, repeat=3)]
    amino = [forward.get(codon) for codon in codons]
    sense = np.array([aa is not None for aa in amino])
    bases = np.array([[BASES.index(nt) for nt in codon] for codon in codons])
    lookup = {codon: i for i, codon in enumerate(codons)}
    neighbors = np.empty((64, 9), dtype=int)
    synonymous = np.zeros((64, 9), dtype=bool)
    transition = np.zeros((64, 9), dtype=bool)
    source_bases = np.zeros((64, 9, 4))
    nondegenerate = np.ones((64, 3), dtype=bool)
    fourfold = np.ones(64, dtype=bool)
    position_counts = np.zeros((64, 12))
    for i, codon in enumerate(codons):
        neighbor_index = 0
        for pos in range(3):
            position_counts[i, pos * 4 + bases[i, pos]] = 1
            for base in BASES:
                if base == codon[pos]:
                    continue
                neighbor = codon[:pos] + base + codon[pos + 1 :]
                dest = lookup[neighbor]
                neighbors[i, neighbor_index] = dest
                same = amino[i] is not None and amino[i] == amino[dest]
                synonymous[i, neighbor_index] = same
                transition[i, neighbor_index] = {codon[pos], base} in (
                    {"T", "C"},
                    {"A", "G"},
                )
                source_bases[i, neighbor_index, bases[i, pos]] = 1
                if same:
                    nondegenerate[i, pos] = False
                if pos == 2 and not same:
                    fourfold[i] = False
                neighbor_index += 1
    differences = np.zeros((4096, 4))
    classes = np.zeros((4096, 32))
    for i, j in product(range(64), repeat=2):
        if not sense[i] or not sense[j]:
            continue
        index = i * 64 + j
        for pos in range(3):
            if nondegenerate[i, pos] and nondegenerate[j, pos]:
                a, b = bases[i, pos], bases[j, pos]
                classes[index, a * 4 + b] += 0.5
                classes[index, b * 4 + a] += 0.5
        if fourfold[i] and fourfold[j] and amino[i] == amino[j]:
            a, b = bases[i, 2], bases[j, 2]
            classes[index, 16 + a * 4 + b] += 0.5
            classes[index, 16 + b * 4 + a] += 0.5
        changed = [pos for pos in range(3) if codons[i][pos] != codons[j][pos]]
        paths = []
        for steps in permutations(changed):
            current = codons[i]
            counts = np.zeros(4)
            for pos in steps:
                following = current[:pos] + codons[j][pos] + current[pos + 1 :]
                if following not in forward:
                    break
                same = forward[current] == forward[following]
                ts = {current[pos], following[pos]} in ({"T", "C"}, {"A", "G"})
                counts[(0 if same else 2) + (0 if ts else 1)] += 1
                current = following
            else:
                paths.append(counts)
        if paths:
            differences[index] = np.mean(paths, axis=0)
        elif changed:
            # The documented YN00 all-stop-path convention, not an NG86 switch.
            differences[index, 2:] = (0.5, len(changed) - 0.5)
    alphabet = np.full(256, -2, dtype=np.int16)
    for base in "RYSWKMBDHVNX-?.":
        alphabet[ord(base)] = -1
    for i, base in enumerate(BASES):
        alphabet[ord(base)] = i
    triangle = np.triu_indices(4)
    ordinary_stops = frozenset(
        codon for codon, aa in zip(codons, amino, strict=True) if aa is None
    )
    ambiguous_stops = frozenset(
        pattern
        for pattern in definite_stop_patterns(ordinary_stops)
        if any(base not in BASES for base in pattern)
    )
    return dict(
        sense=sense,
        bases=bases,
        neighbors=neighbors,
        synonymous=synonymous,
        transition=transition,
        source_bases=source_bases,
        position_counts=position_counts,
        differences=differences,
        classes=classes,
        class_triangle=triangle,
        class_projection=classes.reshape(4096, 2, 4, 4)[:, :, triangle[0], triangle[1]]
        .reshape(4096, 20)
        .copy(),
        alphabet=alphabet,
        codon_lookup=_codon_lookup(),
        ambiguous_stops=ambiguous_stops,
        ambiguous_stop_keys=np.array(
            [
                (ord(pattern[0]) << 14) | (ord(pattern[1]) << 7) | ord(pattern[2])
                for pattern in sorted(ambiguous_stops)
            ],
            dtype=np.int32,
        ),
        ambiguous_stop_symbols=sorted(
            {
                base
                for pattern in ambiguous_stops
                for base in pattern
                if base not in BASES
            }
        ),
    )


def _encode_pair(
    first: str, second: str, geometry: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    if not first.isascii() or not second.isascii():
        raise ValueError("Invalid DNA alphabet in dnds input")
    seqs = [seq.upper().replace("U", "T") for seq in (first, second)]
    if len(seqs[0]) != len(seqs[1]) or len(seqs[0]) % 3:
        raise ValueError("dnds requires equally aligned CDS lengths divisible by three")
    if any(set(seq) - set("ACGTRYSWKMBDHVNX-?.") for seq in seqs):
        raise ValueError("Invalid DNA alphabet in dnds input")
    arrays = [
        geometry["alphabet"][
            np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        ].reshape(-1, 3)
        for seq in seqs
    ]
    concrete = [np.all(array >= 0, axis=1) for array in arrays]
    encoded = [(np.maximum(array, 0) * (16, 4, 1)).sum(axis=1) for array in arrays]
    metadata = {"aligned_codons": len(arrays[0]), "status": "ok"}
    stops = [concrete[i] & ~geometry["sense"][encoded[i]] for i in (0, 1)]
    for side, sequence in enumerate(seqs):
        if any(symbol in sequence for symbol in geometry["ambiguous_stop_symbols"]):
            # The reference path resolves triplets independently of the bulk
            # byte lookup. TAR/TRA can be definite stops despite ambiguity.
            stops[side] |= np.fromiter(
                (
                    sequence[start : start + 3] in geometry["ambiguous_stops"]
                    for start in range(0, len(sequence), 3)
                ),
                dtype=bool,
            )
    for i in (0, 1):
        if not np.any(stops[i]):
            continue
        # Use last_evaluable, as in validate: ambiguity is evaluable, gaps are
        # not. Trailing N codons cannot disguise an earlier internal stop.
        chars = np.frombuffer(seqs[i].encode("ascii"), dtype=np.uint8).reshape(-1, 3)
        positions = np.flatnonzero(
            ~np.isin(chars, [ord(ch) for ch in "-?."]).any(axis=1)
        )
        if len(positions) == 0 or np.any(np.flatnonzero(stops[i]) != positions[-1]):
            metadata.update(
                status="internal_stop",
                retained_codons=0,
                excluded_codons=len(arrays[0]),
            )
            return None, metadata
    valid = concrete[0] & concrete[1] & ~stops[0] & ~stops[1]
    retained = int(valid.sum())
    metadata.update(retained_codons=retained, excluded_codons=len(arrays[0]) - retained)
    if not retained:
        metadata["status"] = "no_aligned_sense_codons"
        return None, metadata
    return np.bincount(
        encoded[0][valid] * 64 + encoded[1][valid], minlength=4096
    ), metadata


def _distance(
    sites: Any,
    transitions: Any,
    transversions: Any,
    frequencies: Any,
    proportions: bool = False,
) -> tuple[Any, Any, Any]:
    """Vectorized F84, with explicitly identified K80/JC69 corrections."""
    count = len(sites)
    distance = np.full(count, np.nan)
    kappa = np.full(count, -1.0)
    model = np.full(count, "no_sites", dtype=object)
    with np.errstate(divide="ignore", invalid="ignore"):
        p, q = (
            (transitions, transversions)
            if proportions
            else (transitions / sites, transversions / sites)
        )
        y, r = frequencies[:, :2].sum(axis=1), frequencies[:, 2:].sum(axis=1)
        tc, ag = (
            frequencies[:, 0] * frequencies[:, 1],
            frequencies[:, 2] * frequencies[:, 3],
        )
        a, b, c = tc / y + ag / r, tc + ag, y * r
        u = (2 * b + 2 * (tc * r / y + ag * y / r) * (1 - q / (2 * c)) - p) / (2 * a)
        v = 1 - q / (2 * c)
        valid = (sites > 0) & np.all(np.isfinite(frequencies), axis=1)
        f84 = (
            valid
            & (q >= np.minimum(1e-10, 0.1 / sites))
            & (u > 0)
            & (v > 0)
            & (v < 1)
            & (p + q <= 1)
        )
        intermediate = np.log(u) / np.log(v) - 1
        f84_kappa = (b + a * intermediate) / b
        f84_distance = (
            -2
            * np.log(v)
            * (tc * (1 + intermediate / y) + ag * (1 + intermediate / r) + c)
        )
        distance[f84], kappa[f84], model[f84] = f84_distance[f84], f84_kappa[f84], "F84"
        k80 = (
            valid
            & ~f84
            & (q >= np.minimum(1e-10, 0.1 / sites))
            & (1 - 2 * p - q > 0)
            & (1 - 2 * q > 0)
            & (q > 0)
        )
        log_a, log_b = -np.log(1 - 2 * p - q), -np.log(1 - 2 * q)
        distance[k80] = (0.5 * log_a + 0.25 * log_b)[k80]
        kappa[k80] = (2 * log_a / log_b - 1)[k80]
        model[k80] = "K80"
        jc = valid & ~f84 & ~k80 & (p + q < 0.75)
        distance[jc] = (-0.75 * np.log1p(-4 * (p + q) / 3))[jc]
        model[jc] = "JC69"
        saturated = valid & ~f84 & ~k80 & ~jc
        # Retain PAML's finite-sample diagnostic correction for comparison,
        # but callers must not present this as an estimable saturated dS.
        distance[saturated] = np.minimum(99, (-0.75 * np.log(1 / sites))[saturated])
        excessive = valid & (p + q > 1)
        distance[excessive], kappa[excessive] = 99, 1
        model[saturated] = "saturated"
    return distance, np.minimum(kappa, 999), model


def _encode_batch(
    pairs: list[tuple[str, str]], geometry: dict[str, Any]
) -> list[tuple[Any, dict[str, Any]]]:
    """Encode variable-length pairs together, preserving individual stop rules."""
    if not pairs:
        return []
    if any(not a.isascii() or not b.isascii() for a, b in pairs):
        raise ValueError("Invalid DNA alphabet in dnds input")
    sequences = [
        (a.upper().replace("U", "T"), b.upper().replace("U", "T")) for a, b in pairs
    ]
    for a, b in sequences:
        if len(a) != len(b) or len(a) % 3:
            raise ValueError(
                "dnds requires equally aligned CDS lengths divisible by three"
            )
    lengths = np.array([len(a) // 3 for a, _ in sequences])
    pair_indices = np.repeat(np.arange(len(pairs)), lengths)
    encoded = []
    stops = []
    for side in (0, 1):
        sequence = "".join(pair[side] for pair in sequences)
        chars = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8).reshape(-1, 3)
        keys = chars[:, 0].astype(np.int32) << 14
        keys |= chars[:, 1].astype(np.int32) << 7
        keys |= chars[:, 2]
        codons = geometry["codon_lookup"][keys].astype(np.int16)
        encoded.append(codons)
        stop = (codons >= 0) & ~geometry["sense"][codons]
        # Shared codon semantics includes ambiguous definite stops. A cheap
        # symbol scan keeps the usual A/C/G/T/N/gap path free of extra lookups.
        if any(symbol in sequence for symbol in geometry["ambiguous_stop_symbols"]):
            stop |= np.isin(keys, geometry["ambiguous_stop_keys"])
        stops.append(stop)
    if any(np.any(array < -1) for array in encoded):
        raise ValueError("Invalid DNA alphabet in dnds input")
    concrete = [array >= 0 for array in encoded]
    valid = concrete[0] & concrete[1] & ~stops[0] & ~stops[1]
    stop_pairs = np.unique(pair_indices[stops[0] | stops[1]])
    if len(stop_pairs):
        # Rare stop-containing pairs use the same last-evaluable-codon rule;
        # the bulk path does not infer terminal stops from concatenated CDS.
        valid[np.isin(pair_indices, stop_pairs)] = False
    counts = np.bincount(
        pair_indices[valid] * 4096 + encoded[0][valid] * 64 + encoded[1][valid],
        minlength=len(pairs) * 4096,
    ).reshape(len(pairs), 4096)
    retained = counts.sum(axis=1)
    result = [
        (
            counts[i] if retained[i] else None,
            dict(
                aligned_codons=int(length),
                retained_codons=int(retained[i]),
                excluded_codons=int(length - retained[i]),
                status="ok" if retained[i] else "no_aligned_sense_codons",
            ),
        )
        for i, length in enumerate(lengths)
    ]
    for i in stop_pairs:
        first, second = sequences[i]
        result[i] = _encode_pair(first, second, geometry)
    return result


def _estimate_batch(pairs: list[tuple[str, str]], code: int) -> list[dict[str, Any]]:
    geometry = _geometry(code)
    encoded = _encode_batch(pairs, geometry)
    rows = [
        dict(
            dS=None,
            dN=None,
            dS_diagnostic=None,
            dN_diagnostic=None,
            kappa=None,
            S=None,
            N=None,
            dS_model="not_estimated",
            dN_model="not_estimated",
            kappa_defaulted=False,
            **metadata,
        )
        for _, metadata in encoded
    ]
    usable = [i for i, (counts, _) in enumerate(encoded) if counts is not None]
    if usable:
        counts = np.asarray([encoded[i][0] for i in usable], dtype=float)
        pair_counts = counts.reshape(-1, 64, 64)
        hist = np.stack((pair_counts.sum(axis=2), pair_counts.sum(axis=1)), axis=1)
        lengths = hist.sum(axis=2)[:, 0]
        # Each class matrix is symmetric. Project only its ten distinct
        # entries, then restore the full matrix before the original numerical
        # normalization (important near kappa correction boundaries).
        projected = (counts @ geometry["class_projection"]).reshape(-1, 2, 10)
        class_counts = np.empty((len(usable), 2, 4, 4))
        a, b = geometry["class_triangle"]
        class_counts[:, :, a, b] = projected
        class_counts[:, :, b, a] = projected
        class_sites = class_counts.sum(axis=(2, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = class_counts / class_sites[:, :, None, None]
            freqs = normalized.sum(axis=3)
            ts = 2 * (normalized[:, :, 0, 1] + normalized[:, :, 2, 3])
            tv = 1 - np.trace(normalized, axis1=2, axis2=3) - ts
        _, kap, _ = _distance(
            class_sites.ravel(),
            ts.ravel(),
            np.maximum(tv, 0).ravel(),
            freqs.reshape(-1, 4),
            proportions=True,
        )
        kap = kap.reshape(-1, 2)
        weights = np.where(kap > 0, class_sites, 0)
        total = weights.sum(axis=1)
        kappa = np.divide(
            (kap * weights).sum(axis=1),
            total,
            out=np.full(len(usable), DEFAULT_KAPPA),
            where=total > 0,
        )
        f3x4 = (hist.mean(axis=1) @ geometry["position_counts"]).reshape(
            -1, 3, 4
        ) / lengths[:, None, None]
        bases = geometry["bases"]
        pi = (
            f3x4[:, 0, bases[:, 0]]
            * f3x4[:, 1, bases[:, 1]]
            * f3x4[:, 2, bases[:, 2]]
            * geometry["sense"]
        )
        pi /= pi.sum(axis=1, keepdims=True)
        rates = pi[:, geometry["neighbors"]] * np.where(
            geometry["transition"], kappa[:, None, None], 1
        )
        syn_rates = rates * geometry["synonymous"]
        non_rates = rates * ~geometry["synonymous"]
        raw_sites = [
            np.einsum("bsc,bc->bs", hist, rate.sum(axis=2))
            for rate in (syn_rates, non_rates)
        ]
        with np.errstate(divide="ignore", invalid="ignore"):
            scaling = 3 * lengths[:, None] / (raw_sites[0] + raw_sites[1])
            sites = [(raw * scaling).mean(axis=1) for raw in raw_sites]
            site_freqs = []
            for rate in (syn_rates, non_rates):
                per_codon = np.einsum("bcj,cjk->bck", rate, geometry["source_bases"])
                frequency = np.einsum("bsc,bck->bsk", hist, per_codon)
                site_freqs.append(
                    (frequency / frequency.sum(axis=2, keepdims=True)).mean(axis=1)
                )
        differences = counts @ geometry["differences"]
        ds, _, ds_model = _distance(
            sites[0], differences[:, 0], differences[:, 1], site_freqs[0]
        )
        dn, _, dn_model = _distance(
            sites[1], differences[:, 2], differences[:, 3], site_freqs[1]
        )
        # Convert entire columns once; scalar NumPy checks and conversions in
        # each row otherwise dominate both Python time and thread contention.
        columns = dict(
            dS=_finite_column(ds, ds_model != "saturated"),
            dN=_finite_column(dn, dn_model != "saturated"),
            dS_diagnostic=_finite_column(ds),
            dN_diagnostic=_finite_column(dn),
            kappa=kappa.tolist(),
            S=_finite_column(sites[0]),
            N=_finite_column(sites[1]),
            dS_model=ds_model.tolist(),
            dN_model=dn_model.tolist(),
            kappa_defaulted=(total == 0).tolist(),
            status=np.where(
                ds_model == "saturated",
                "saturated",
                np.where(np.isfinite(ds), "ok", "no_synonymous_sites"),
            ).tolist(),
        )
        for index, values in zip(
            usable, zip(*columns.values(), strict=True), strict=True
        ):
            rows[index].update(zip(columns, values, strict=True))
    return rows


def _finite_column(values: np.ndarray[Any, Any], usable: Any = True) -> list[Any]:
    column = values.astype(object)
    column[~(np.isfinite(values) & usable)] = None
    return column.tolist()


def _batch_size(pair_count: int, workers: int) -> int:
    # Larger parallel batches amortize GIL/scheduling overhead. Keep at least
    # four batches per worker when possible to balance unequal CDS lengths,
    # and cap dense per-worker arrays at four times the serial batch size.
    return min(
        BATCH_SIZE * min(workers, 4), max(BATCH_SIZE, pair_count // (workers * 4))
    )


def estimate_pairs(
    pairs: list[tuple[str, str]], codon_table: int = 1, threads: int = 1
) -> list[dict[str, Any]]:
    """Estimate independent aligned pairs; retain input order and NA statuses."""
    workers = resolve_threads(threads)
    _geometry(codon_table)
    size = _batch_size(len(pairs), workers)
    chunks = [pairs[i : i + size] for i in range(0, len(pairs), size)]
    if workers == 1 or len(chunks) < 2:
        return [row for chunk in chunks for row in _estimate_batch(chunk, codon_table)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [
            row
            for chunk in pool.map(
                lambda chunk: _estimate_batch(chunk, codon_table), chunks
            )
            for row in chunk
        ]


def dnds_main(args: Any) -> None:
    # This handler is also called directly by workflow consumers, without the
    # CLI's path preflight. Protect their inputs and existing outputs as well.
    validate_distinct_paths(inputs=[args.pairs_file], outputs=[args.outfile])
    if args.outfile != "-":
        validate_output_paths([args.outfile])
    rows = read_tsv(
        args.pairs_file, required_columns=("pair_id", "sequence_1", "sequence_2")
    )
    ids = [row["pair_id"] for row in rows]
    if any(not identifier.strip() for identifier in ids) or len(ids) != len(set(ids)):
        raise ValueError("dnds pair_id values must be nonempty and unique")
    values = estimate_pairs(
        [(row["sequence_1"], row["sequence_2"]) for row in rows],
        args.codontable,
        args.threads,
    )
    report = (
        dict(
            schema_version=TSV_REPORT_SCHEMA_VERSION,
            pair_id=identifier,
            method=METHOD,
            codon_table=args.codontable,
            codon_semantics_version=CODON_SEMANTICS_VERSION,
            **{key: "" if value is None else value for key, value in result.items()},
        )
        for identifier, result in zip(ids, values, strict=True)
    )
    validate_distinct_paths(inputs=[args.pairs_file], outputs=[args.outfile])
    write_tsv(args.outfile, report, REPORT_COLUMNS)
