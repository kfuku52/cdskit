#!/usr/bin/env python3
"""Full-data localization experiments with resumable, audited outer folds."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cdskit.deeploc_benchmark import (
    DEEPLOC_LOCALIZATION_LABELS as LABELS,
    _read_prepared_tsv,
    _fold_ids_from_rows,
    fit_deeploc_multilabel_model,
    _predict_model_on_rows,
    build_label_matrix,
    compute_multilabel_metrics,
)
from cdskit.localize_evaluation import (
    assert_disjoint,
    assert_model_partitions,
    dataset_digest,
    probability_metrics,
)
from cdskit.localize_model import save_localize_model, load_localize_model
from cdskit.util import atomic_write_json


def run_baselines(rows, output, recipes, seeds, device, epochs):
    import torch

    torch.set_num_threads(1)
    folds = _fold_ids_from_rows(rows)
    labels = list(LABELS)
    y = build_label_matrix(rows, labels, "localization_labels")
    for recipe in recipes:
        for seed in seeds:
            directory = output / "{}_seed{}".format(recipe, seed)
            directory.mkdir(parents=True, exist_ok=True)
            parameters = dict(
                epochs=epochs,
                patience=3,
                batch_size=256,
                seed=seed,
                device=device,
                sequence_layout="legacy"
                if recipe == "cnn_legacy"
                else "separate_termini",
                mask_padding=recipe != "cnn_legacy",
            )
            config = dict(
                dataset_sha256=dataset_digest(rows),
                parameters=parameters,
                recipe=recipe,
            )
            config_path = directory / "config.json"
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError(
                    "Existing experiment configuration differs: " + str(directory)
                )
            atomic_write_json(str(config_path), config)
            probs, predictions, reports = (
                np.zeros(y.shape),
                np.zeros(y.shape, dtype=int),
                [],
            )
            for fold in sorted(set(folds)):
                path = directory / ("fold" + fold + ".pt")
                mask = folds == fold
                train = [row for i, row in enumerate(rows) if not mask[i]]
                test = [row for i, row in enumerate(rows) if mask[i]]
                started = time.perf_counter()
                if path.exists():
                    model = load_localize_model(str(path))
                else:
                    print("Training", recipe, "seed", seed, "outer", fold, flush=True)
                    model = fit_deeploc_multilabel_model(
                        train,
                        labels,
                        "localization_labels",
                        "localization",
                        "cnn",
                        parameters,
                    )
                    save_localize_model(model, str(path))
                validation_fold = sorted({row["fold_id"] for row in train})[-1]
                fit_rows = [row for row in train if row["fold_id"] != validation_fold]
                validation_rows = [
                    row for row in train if row["fold_id"] == validation_fold
                ]
                assert_model_partitions(model, fit_rows, validation_rows)
                assert_disjoint(train, test)
                pred = _predict_model_on_rows(model, test)
                probs[mask], predictions[mask] = (
                    pred["prob_matrix"],
                    pred["prediction_matrix"],
                )
                metrics = compute_multilabel_metrics(y[mask], predictions[mask], labels)
                metrics.update(
                    fold=fold,
                    elapsed_seconds=time.perf_counter() - started,
                    selected_epoch=model["localization_model"]["selected_epoch"],
                )
                reports.append(metrics)
                atomic_write_json(str(directory / ("fold" + fold + ".json")), metrics)
                print(
                    recipe,
                    seed,
                    fold,
                    "macro",
                    metrics["macro_f1"],
                    "micro",
                    metrics["micro_f1"],
                    "seconds",
                    metrics["elapsed_seconds"],
                    flush=True,
                )
            np.savez_compressed(
                directory / "oof.npz",
                probability=probs,
                prediction=predictions,
                target=y,
                folds=folds.astype(str),
            )
            metrics = compute_multilabel_metrics(y, predictions, labels)
            metrics.update(probability_metrics(y, probs, labels))
            metrics["folds"] = reports
            metrics["config"] = config
            atomic_write_json(str(directory / "metrics.json"), metrics)
            print(
                "COMPLETE",
                recipe,
                seed,
                metrics["macro_f1"],
                metrics["micro_f1"],
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_tsv",
        default="data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--recipes", default="cnn_legacy,cnn_termini")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if any(
        recipe not in ("cnn_legacy", "cnn_termini")
        for recipe in args.recipes.split(",")
    ):
        parser.error("Unknown base recipe.")
    rows = _read_prepared_tsv(args.training_tsv)
    run_baselines(
        rows,
        Path(args.output),
        args.recipes.split(","),
        [int(x) for x in args.seeds.split(",")],
        args.device,
        args.epochs,
    )


if __name__ == "__main__":
    main()
