#!/usr/bin/env python3
"""Replay adopted integrated checkpoints and independently verify their exports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from cdskit.deeploc_benchmark import (
    _read_prepared_tsv,
    _predict_model_on_rows,
    build_label_matrix,
    DEEPLOC_LOCALIZATION_LABELS as LABELS,
    compute_multilabel_metrics,
)
from cdskit.localize_evaluation import (
    assert_disjoint,
    dataset_digest,
    stratified_metrics,
)
from cdskit.localize_model import load_localize_model
from cdskit.localize_specialists import (
    predict_specialists,
    sequence_features,
    calibrate_blend,
)
from cdskit.util import atomic_write_json


def verify_metadata(model, train, val):
    meta = model["metadata"]
    assert meta["training_data_sha256"] == dataset_digest(train)
    assert meta["validation_data_sha256"] == dataset_digest(val)
    assert meta["num_training_rows"] == len(train)
    assert meta["num_validation_rows"] == len(val)
    assert model["localization_model"]["class_order"] == list(LABELS)
    assert_disjoint(train, val)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--prepared_dir", default="data/localize_bench/deeploc21")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    root, prepared = Path(args.experiment_dir), Path(args.prepared_dir)
    out = root / "audit"
    out.mkdir(exist_ok=True)
    rows = _read_prepared_tsv(
        str(prepared / "deeploc21_localization_train_validation.tsv")
    )
    hpa = _read_prepared_tsv(str(prepared / "deeploc21_hpa_test.tsv"))
    folds = np.asarray([r["fold_id"] for r in rows])
    target = build_label_matrix(rows, LABELS, "localization_labels")
    report = {"dataset_sha256": dataset_digest(rows), "oof": [], "checkpoints": {}}
    assert_disjoint(rows, hpa)
    for seed in (1, 2, 3):
        directory = root / f"integration_seed{seed}"
        with np.load(directory / "integrated_oof.npz") as saved:
            np.testing.assert_array_equal(saved["target"], target)
            np.testing.assert_array_equal(saved["folds"], folds)
            for fold in sorted(set(folds)):
                val_fold = sorted(set(folds) - {fold})[-1]
                train = [r for r in rows if r["fold_id"] not in (fold, val_fold)]
                val = [r for r in rows if r["fold_id"] == val_fold]
                ids = np.flatnonzero(folds == fold)
                test = [rows[i] for i in ids]
                assert_disjoint(train + val, test)
                path = directory / f"integrated_fold{fold}.pt"
                model = load_localize_model(str(path))
                verify_metadata(model, train, val)
                assert model["metadata"]["cnn_params"]["seed"] == seed
                pred = _predict_model_on_rows(model, test)
                error = float(
                    np.max(np.abs(pred["prob_matrix"] - saved["probability"][ids]))
                )
                np.testing.assert_allclose(
                    pred["prob_matrix"], saved["probability"][ids], atol=1e-7, rtol=1e-6
                )
                np.testing.assert_array_equal(
                    pred["prediction_matrix"], saved["prediction"][ids]
                )
                report["oof"].append(
                    dict(
                        seed=seed,
                        fold=fold,
                        rows=len(test),
                        max_probability_error=error,
                    )
                )
                report["checkpoints"][str(path)] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                atomic_write_json(str(out / "progress.json"), report)
                print("OOF verified", seed, fold, error, flush=True)
            metrics = {
                a: f1_score(target, saved["prediction"], average=a, zero_division=0)
                for a in ("macro", "micro")
            }
            print("Independent sklearn F1", metrics, flush=True)
            baseline = np.load(root / f"cnn_legacy_seed{seed}" / "oof.npz")
            report[f"strata_seed{seed}"] = {
                "integrated": stratified_metrics(
                    rows,
                    target,
                    saved["prediction"],
                    saved["probability"],
                    LABELS,
                    compute_multilabel_metrics,
                ),
                "baseline": stratified_metrics(
                    rows,
                    target,
                    baseline["prediction"],
                    baseline["probability"],
                    LABELS,
                    compute_multilabel_metrics,
                ),
            }
    path = root / "final" / "integrated.pt"
    final = load_localize_model(str(path))
    train = [r for r in rows if r["fold_id"] != "4"]
    val = [r for r in rows if r["fold_id"] == "4"]
    verify_metadata(final, train, val)
    head = final["localization_model"]["specialist_head"]
    x, vx, hx = [
        sequence_features([r["sequence"] for r in subset])
        for subset in (train, val, hpa)
    ]
    y = build_label_matrix(train, LABELS, "localization_labels")
    expert_val = predict_specialists(vx, head)
    expert_hpa = predict_specialists(hx, head)
    reference_errors = {}
    for i, name in enumerate(LABELS):
        estimator = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=1,
        ).fit(x.astype(np.float32), y[:, i])
        ref = estimator.predict_proba(np.vstack([vx, hx]).astype(np.float32))[:, 1]
        exported = np.r_[expert_val[:, i], expert_hpa[:, i]]
        np.testing.assert_allclose(ref, exported, atol=1e-7, rtol=1e-6)
        reference_errors[name] = float(np.max(np.abs(ref - exported)))
        print("sklearn reference verified", name, reference_errors[name], flush=True)
    report["sklearn_refit_max_error"] = reference_errors
    base = load_localize_model(str(root / "final" / "cnn_legacy.pt"))
    weights, thresholds = calibrate_blend(
        _predict_model_on_rows(base, val)["prob_matrix"],
        expert_val,
        build_label_matrix(val, LABELS, "localization_labels"),
        LABELS,
    )
    assert weights == final["localization_model"]["specialist_weights"]
    assert thresholds == final["localization_model"]["class_thresholds"]
    pred = _predict_model_on_rows(final, hpa)
    with np.load(root / "final" / "integrated_hpa.npz") as saved:
        np.testing.assert_array_equal(
            saved["target"], build_label_matrix(hpa, LABELS, "localization_labels")
        )
        np.testing.assert_array_equal(saved["prediction"], pred["prediction_matrix"])
        np.testing.assert_allclose(
            saved["probability"], pred["prob_matrix"], atol=1e-7, rtol=1e-6
        )
    fasta = out / "hpa.fasta"
    fasta.write_text(
        "".join(">" + r["accession"] + "\n" + r["sequence"] + "\n" for r in hpa)
    )
    cli = out / "hpa_cli.tsv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cdskit.cli",
            "localize",
            "--seq_file",
            str(fasta),
            "--seq_type",
            "protein",
            "--model",
            str(path),
            "--threads",
            "1",
            "--report",
            str(cli),
        ],
        check=True,
    )
    with cli.open() as stream:
        cli_rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [r["seq_id"] for r in cli_rows] == [r["accession"] for r in hpa]
    cp = np.asarray([[float(r["p_" + c]) for c in LABELS] for r in cli_rows])
    np.testing.assert_allclose(cp, pred["prob_matrix"], atol=1e-6, rtol=1e-6)
    assert [r["predicted_labels"] for r in cli_rows] == [
        ";".join(c for c, yes in zip(LABELS, p, strict=True) if yes)
        for p in pred["prediction_matrix"]
    ]
    report["hpa_cli_rows"] = len(cli_rows)
    report["hpa_cli_max_probability_error"] = float(
        np.max(np.abs(cp - pred["prob_matrix"]))
    )
    report["checkpoints"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["status"] = "passed"
    atomic_write_json(str(out / "verification.json"), report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k in ("status", "hpa_cli_rows", "hpa_cli_max_probability_error")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
