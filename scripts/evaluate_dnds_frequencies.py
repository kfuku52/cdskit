"""Measure the effect of empirical codon frequency shrinkage on GY94 distances."""

import argparse
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
        [[0.09, 0.41, 0.13, 0.37], [0.18, 0.31, 0.21, 0.30], [0.09, 0.49, 0.11, 0.31]]
    ),
    np.array(
        [[0.51, 0.08, 0.32, 0.09], [0.11, 0.40, 0.13, 0.36], [0.42, 0.05, 0.48, 0.05]]
    ),
)
FREQUENCY_MODELS = ("cf3x4", "empirical-shrinkage")


def evaluate(setting):
    code, profile, kappa, omega, distance, length, model, count = setting
    codons, pi, conditional, ds = simulation_distribution(
        code, profile, kappa, omega, distance, model, PROFILES, (0.3, 1.7)
    )
    seed = [
        1066523,
        code,
        profile,
        int(kappa * 100),
        int(omega * 1000),
        int(distance * 100),
        length,
    ]
    pairs = sample_joint(codons, pi, conditional, length, count, seed)
    expected = np.array([ds, ds * omega, omega])
    estimates = {
        frequency: distances(
            estimate_pairs(
                pairs, code, method="gy94-regularized", frequency_model=frequency
            )
        )
        for frequency in FREQUENCY_MODELS
    }
    common = np.isfinite(estimates["cf3x4"]) & np.isfinite(
        estimates["empirical-shrinkage"]
    )
    return {
        "genetic_code": code,
        "profile": profile,
        "kappa": kappa,
        "omega": omega,
        "distance": distance,
        "codons": length,
        "simulation": model,
        "replicates": count,
        "frequencies": {
            frequency: error_summary(expected, values, common)
            for frequency, values in estimates.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=36)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.replicates, args.workers) < 1:
        parser.error("replicates and workers must be positive")
    design = product(
        (1, 2, 11),
        range(len(PROFILES)),
        (1.8, 7.3),
        (0.008, 0.08, 0.45, 1.7, 4.5),
        (0.05, 0.4, 1.4, 2.2),
        (60, 180, 520, 1500),
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
