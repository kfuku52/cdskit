"""Evaluate log-ratio regularization under held-out codon simulation settings."""

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
        [[0.18, 0.34, 0.21, 0.27], [0.29, 0.22, 0.33, 0.16], [0.07, 0.51, 0.11, 0.31]]
    ),
    np.array(
        [[0.38, 0.13, 0.14, 0.35], [0.16, 0.27, 0.43, 0.14], [0.47, 0.05, 0.36, 0.12]]
    ),
)
METHODS = ("yn00", "gy94", "gy94-regularized")


def evaluate(setting):
    code, frequency, kappa, omega, distance, length, model, count = setting
    codons, pi, conditional, ds = simulation_distribution(
        code, frequency, kappa, omega, distance, model, PROFILES, (0.15, 1.85)
    )
    seed = [
        754687,
        code,
        frequency,
        int(kappa * 100),
        int(omega * 1000),
        int(distance * 100),
        length,
    ]
    pairs = sample_joint(codons, pi, conditional, length, count, seed)
    expected = np.array([ds, ds * omega, omega])
    rows = {method: estimate_pairs(pairs, code, method=method) for method in METHODS}
    estimates = {method: distances(results) for method, results in rows.items()}
    common = np.logical_and.reduce(
        [np.isfinite(values) for values in estimates.values()]
    )
    return {
        "genetic_code": code,
        "frequency": frequency,
        "kappa": kappa,
        "omega": omega,
        "distance": distance,
        "codons": length,
        "simulation": model,
        "replicates": count,
        "methods": {
            method: error_summary(expected, values, common)
            for method, values in estimates.items()
        },
        "statuses": {
            method: dict(Counter(row["status"] for row in results))
            for method, results in rows.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replicates, args.workers) < 1:
        parser.error("replicates and workers must be positive")
    design = product(
        (1, 2, 6),
        range(len(PROFILES)),
        (1.3, 6.0),
        (0.005, 0.07, 0.4, 0.8, 1.2, 2.0, 5.0),
        (0.01, 0.12, 0.65, 2.2),
        (40, 90, 360, 1200),
        ("product", "nonproduct", "mixture"),
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
