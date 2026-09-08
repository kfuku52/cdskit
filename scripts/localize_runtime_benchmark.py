#!/usr/bin/env python3
"""Measure each fixed model in a fresh process on the same length-stratified proteins."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cdskit.deeploc_benchmark import _read_prepared_tsv, _predict_model_on_rows
from cdskit.localize_model import load_localize_model
from cdskit.localize_evaluation import dataset_digest
from cdskit.util import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--input", default="data/localize_bench/deeploc21/deeploc21_hpa_test.tsv"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        for model in args.models:
            path = str(Path(args.output) / (Path(model).stem + ".json"))
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--models",
                    model,
                    "--input",
                    args.input,
                    "--output",
                    path,
                ],
                check=True,
            )
        return
    import torch

    torch.set_num_threads(1)
    rows = sorted(
        _read_prepared_tsv(args.input),
        key=lambda row: (len(row["sequence"]), row["accession"]),
    )
    rows = [
        rows[int(i)]
        for i in np.linspace(0, len(rows) - 1, min(256, len(rows))).astype(int)
    ]
    started = time.perf_counter()
    model = load_localize_model(args.models[0])
    load_seconds = time.perf_counter() - started
    _predict_model_on_rows(model, rows[:32])
    durations, reference = [], None
    for _ in range(3):
        started = time.perf_counter()
        pred = _predict_model_on_rows(model, rows)
        durations.append(time.perf_counter() - started)
        if reference is not None:
            np.testing.assert_allclose(
                pred["prob_matrix"], reference, atol=1e-7, rtol=1e-6
            )
        reference = pred["prob_matrix"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    report = dict(
        model=args.models[0],
        rows=len(rows),
        dataset_sha256=dataset_digest(rows),
        min_length=min(len(row["sequence"]) for row in rows),
        max_length=max(len(row["sequence"]) for row in rows),
        cpu_threads=1,
        device="cpu",
        load_seconds=load_seconds,
        wall_seconds=durations,
        median_seconds=float(np.median(durations)),
        peak_process_rss_bytes=peak_bytes,
        model_bytes=Path(args.models[0]).stat().st_size,
        note="Separate models may produce different outputs; these timings describe an accuracy/runtime tradeoff, not equivalent-output optimization.",
    )
    atomic_write_json(args.output, report)
    np.savez_compressed(
        args.output + ".npz",
        probability=reference,
        prediction=pred["prediction_matrix"],
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
