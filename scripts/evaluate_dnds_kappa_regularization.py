"""Compare GY94 log-kappa regularization against held-out codon simulations."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np

from cdskit.atomicio import atomic_write_json
from cdskit.dnds import estimate_pairs
from evaluate_dnds_estimators import (
    distances,
    error_summary,
    sample_joint,
    simulation_distribution,
)

PROFILES = (
    np.array(
        [[0.14, 0.33, 0.18, 0.35], [0.24, 0.25, 0.25, 0.26], [0.14, 0.42, 0.09, 0.35]]
    ),
    np.array(
        [[0.47, 0.11, 0.34, 0.08], [0.10, 0.34, 0.11, 0.45], [0.41, 0.09, 0.43, 0.07]]
    ),
)
METHODS = ("gy94-regularized", "gy94-regularized-kappa")


def evaluate(setting):
    code, profile, kappa, omega, distance, length, model, count = setting
    codons, pi, conditional, ds = simulation_distribution(
        code, profile, kappa, omega, distance, model, PROFILES, (0.2, 1.8)
    )
    seed = [
        135573061,
        code,
        profile,
        round(1000 * kappa),
        round(1000 * omega),
        round(1000 * distance),
        length,
    ]
    pairs = sample_joint(codons, pi, conditional, length, count, seed)
    expected = np.array([ds, ds * omega, omega])
    rows = {method: estimate_pairs(pairs, code, method=method) for method in METHODS}
    estimates = {method: distances(results) for method, results in rows.items()}
    common = np.logical_and.reduce(
        [np.isfinite(values) for values in estimates.values()]
    )
    results = {
        method: error_summary(expected, values, common)
        for method, values in estimates.items()
    }
    kappas = {
        method: np.array([row["kappa"] for row in values], dtype=float)
        for method, values in rows.items()
    }
    common_kappa = np.logical_and.reduce(
        [np.isfinite(values) for values in kappas.values()]
    )
    for method, values in kappas.items():
        errors = np.log(values[common_kappa] / kappa)
        results[method]["log_kappa"] = {
            "common_pairs": len(errors),
            "finite_pairs": int(np.isfinite(values).sum()),
            "mae": float(np.abs(errors).mean()) if len(errors) else None,
        }
    return {
        "genetic_code": code,
        "profile": profile,
        "kappa": kappa,
        "omega": omega,
        "distance": distance,
        "codons": length,
        "simulation": model,
        "replicates": count,
        "methods": results,
        "statuses": {
            method: dict(Counter(row["status"] for row in values))
            for method, values in rows.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replicates, args.workers) < 1:
        parser.error("replicates and workers must be positive")
    design = product(
        (1, 2, 11, 28),
        range(len(PROFILES)),
        (0.25, 0.8, 3.5, 8, 20),
        (0.008, 0.08, 0.55, 2.7),
        (0.02, 0.25, 1.3, 2.2),
        (40, 140, 650),
        ("product", "mixture", "nonproduct"),
        (args.replicates,),
    )
    if args.workers == 1:
        strata = list(map(evaluate, design))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            strata = list(executor.map(evaluate, design, chunksize=2))
    atomic_write_json(args.output, {"strata": strata}, indent=2)


if __name__ == "__main__":
    main()
