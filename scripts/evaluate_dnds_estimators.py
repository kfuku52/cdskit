"""Compare native dN/dS estimators against known stationary codon distances."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from Bio.Data.CodonTable import unambiguous_dna_by_id
from scipy.linalg import expm

from cdskit.atomicio import atomic_write_json
from cdskit.dnds import estimate_pairs

FREQUENCIES = (
    np.array(
        [[0.12, 0.39, 0.11, 0.38], [0.10, 0.40, 0.10, 0.40], [0.06, 0.44, 0.05, 0.45]]
    ),
    np.array(
        [[0.60, 0.04, 0.31, 0.05], [0.03, 0.31, 0.04, 0.62], [0.43, 0.06, 0.43, 0.08]]
    ),
)
METRICS = ("dS", "dN", "dN_over_dS")
METHODS = ("yn00", "gy94")


def codon_model(code, frequency, kappa, omega, nonproduct):
    amino_acids = unambiguous_dna_by_id[code].forward_table
    codons = sorted(amino_acids)
    pi = np.array(
        [
            np.prod(
                [
                    FREQUENCIES[frequency][i, "ACGT".index(base)]
                    for i, base in enumerate(codon)
                ]
            )
            for codon in codons
        ]
    )
    if nonproduct:
        rng = np.random.default_rng(43710 + code)
        pi *= np.exp(rng.normal(0, 0.7, len(pi)))
    pi /= pi.sum()
    synonymous = np.zeros((len(pi), len(pi)))
    nonsynonymous = np.zeros_like(synonymous)
    transitions = {frozenset("AG"), frozenset("CT")}
    for i, first in enumerate(codons):
        for j, second in enumerate(codons):
            changes = [(a, b) for a, b in zip(first, second, strict=True) if a != b]
            if len(changes) == 1:
                matrix = (
                    synonymous
                    if amino_acids[first] == amino_acids[second]
                    else nonsynonymous
                )
                matrix[i, j] = pi[j] * (
                    kappa if frozenset(changes[0]) in transitions else 1
                )
    syn_rate = float(pi @ synonymous.sum(axis=1))
    non_rate = float(pi @ nonsynonymous.sum(axis=1))
    generator = synonymous + omega * nonsynonymous
    np.fill_diagonal(generator, -generator.sum(axis=1))
    scale = syn_rate + omega * non_rate
    return (
        np.asarray(codons, dtype="S3"),
        pi,
        generator / scale,
        (syn_rate + non_rate) / (3 * scale),
    )


def simulate_pairs(codons, pi, generator, distance, length, count, seed):
    # Sample both endpoints directly from the stationary joint distribution.
    joint = np.maximum(expm(generator * distance), 0) * pi[:, None]
    joint /= joint.sum()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    draws = rng.choice(joint.size, size=(count, length), p=joint.ravel())
    first, second = codons[draws // len(codons)], codons[draws % len(codons)]
    return [
        (a.tobytes().decode("ascii"), b.tobytes().decode("ascii"))
        for a, b in zip(first, second, strict=True)
    ]


def distances(rows):
    values = np.array(
        [
            [
                row[metric] if row[metric] is not None else np.nan
                for metric in ("dS", "dN")
            ]
            for row in rows
        ]
    )
    ratio = np.divide(
        values[:, 1],
        values[:, 0],
        out=np.full(len(values), np.nan),
        where=values[:, 0] > 0,
    )
    return np.column_stack((values, ratio))


def error_summary(expected, values, common):
    result = {}
    for index, metric in enumerate(METRICS):
        errors = values[common[:, index], index] - expected[index]
        result[metric] = {
            "common_pairs": len(errors),
            "finite_pairs": int(np.isfinite(values[:, index]).sum()),
            "bias": float(errors.mean()) if len(errors) else None,
            "mae": float(np.abs(errors).mean()) if len(errors) else None,
            "rmse": float(np.sqrt((errors**2).mean())) if len(errors) else None,
        }
    return result


def evaluate(setting):
    code, frequency, kappa, omega, distance, length, nonproduct, count = setting
    codons, pi, generator, ds_scale = codon_model(
        code, frequency, kappa, omega, nonproduct
    )
    seed = [
        20261111,
        code,
        frequency,
        int(kappa * 100),
        int(omega * 100),
        int(distance * 100),
        length,
    ]
    pairs = simulate_pairs(codons, pi, generator, distance, length, count, seed)
    expected = np.array([distance * ds_scale, distance * ds_scale * omega, omega])
    estimates = {
        method: estimate_pairs(pairs, code, method=method) for method in METHODS
    }
    predictions = {method: distances(estimates[method]) for method in METHODS}
    common = np.isfinite(predictions["yn00"]) & np.isfinite(predictions["gy94"])
    return {
        "genetic_code": code,
        "frequency": frequency,
        "kappa": kappa,
        "omega": omega,
        "distance": distance,
        "codons": length,
        "nonproduct": nonproduct,
        "replicates": count,
        "expected": dict(zip(METRICS, expected.tolist(), strict=True)),
        "methods": {
            method: error_summary(expected, values, common)
            for method, values in predictions.items()
        },
        "gy94_statuses": {
            status: sum(row["status"] == status for row in estimates["gy94"])
            for status in sorted({row["status"] for row in estimates["gy94"]})
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=160)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replicates, args.workers) < 1:
        parser.error("replicates and workers must be positive")
    design = product(
        (1, 2, 6),
        range(len(FREQUENCIES)),
        (2.0, 8.0),
        (0.05, 0.35, 1.6),
        (0.04, 0.35, 1.1, 2.4),
        (60, 240, 900),
        (False, True),
        (args.replicates,),
    )
    if args.workers == 1:
        strata = list(map(evaluate, design))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            strata = list(executor.map(evaluate, design))
    atomic_write_json(args.output, {"strata": strata}, indent=2)


if __name__ == "__main__":
    main()
