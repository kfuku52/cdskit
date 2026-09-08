#!/usr/bin/env python3
"""Audit fixed HPA predictions against all training-source sequences."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cdskit.deeploc_benchmark import (
    _read_prepared_tsv,
    DEEPLOC_LOCALIZATION_LABELS,
    compute_multilabel_metrics,
)
from cdskit.localize_evaluation import probability_metrics
from cdskit.perox_benchmark import mmseqs_homology_report
from cdskit.util import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--prepared_dir", default="data/localize_bench/deeploc21")
    args = parser.parse_args()
    directory, root = Path(args.prepared_dir), Path(args.experiment_dir)
    train = _read_prepared_tsv(
        str(directory / "deeploc21_localization_train_validation.tsv")
    )
    test = _read_prepared_tsv(str(directory / "deeploc21_hpa_test.tsv"))
    report = mmseqs_homology_report(
        [dict(row, peroxisome=0) for row in train],
        [dict(row, peroxisome=0) for row in test],
        threads=2,
        include_hit_indices=True,
    )
    if report["status"] != "ok":
        raise ValueError("Homology audit failed: {}".format(report))
    hits = set(report.pop("_hit_eval_indices"))
    report = {
        k: v for k, v in report.items() if "positive" not in k and "negative" not in k
    }
    results = {}
    for path in (root / "final").glob("*_hpa.npz"):
        arrays = np.load(path)
        for name, include in [("hit", True), ("nohit", False)]:
            ids = [i for i in range(len(test)) if (i in hits) == include]
            if not ids:
                continue
            metrics = compute_multilabel_metrics(
                arrays["target"][ids],
                arrays["prediction"][ids],
                DEEPLOC_LOCALIZATION_LABELS,
            )
            metrics.update(
                probability_metrics(
                    arrays["target"][ids],
                    arrays["probability"][ids],
                    DEEPLOC_LOCALIZATION_LABELS,
                )
            )
            results[path.stem + "_" + name] = metrics
    atomic_write_json(
        str(root / "external_homology.json"), dict(report=report, subsets=results)
    )
    print("External homologous queries:", len(hits), "of", len(test), flush=True)


if __name__ == "__main__":
    main()
