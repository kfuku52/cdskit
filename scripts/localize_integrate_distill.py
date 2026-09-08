#!/usr/bin/env python3
"""Train/validate specialists and distilled CNNs within each audited outer fold."""

import argparse
import copy
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
    build_label_matrix,
    _predict_model_on_rows,
    compute_multilabel_metrics,
)
from cdskit.localize_model import (
    load_localize_model,
    save_localize_model,
    _tune_binary_threshold,
)
from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier
from cdskit.localize_specialists import (
    fit_specialists,
    predict_specialists,
    calibrate_blend,
    sequence_features,
)
from cdskit.localize_evaluation import (
    assert_disjoint,
    assert_model_partitions,
    dataset_digest,
    probability_metrics,
)
from cdskit.util import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_tsv",
        default="data/localize_bench/deeploc21/deeploc21_localization_train_validation.tsv",
    )
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--teacher_weight", type=float, default=0.5)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    rows = _read_prepared_tsv(args.training_tsv)
    folds = _fold_ids_from_rows(rows)
    y = build_label_matrix(rows, LABELS, "localization_labels")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), dataset_sha256=dataset_digest(rows))
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing experiment configuration differs.")
    atomic_write_json(str(config_path), config)
    matrices = {
        name: dict(
            probability=np.zeros(y.shape), prediction=np.zeros(y.shape, dtype=int)
        )
        for name in ["integrated", "student_control", "distilled"]
    }
    all_features = sequence_features([row["sequence"] for row in rows])
    for fold in sorted(set(folds)):
        validation_fold = sorted(set(folds) - {fold})[-1]
        train_ids = np.flatnonzero((folds != fold) & (folds != validation_fold))
        val_ids, test_ids = (
            np.flatnonzero(folds == validation_fold),
            np.flatnonzero(folds == fold),
        )
        train, val, test = [
            [rows[i] for i in ids] for ids in (train_ids, val_ids, test_ids)
        ]
        assert_disjoint(train, val)
        assert_disjoint(train + val, test)
        base_path = Path(args.base_dir) / ("fold" + fold + ".pt")
        teacher_path = output / ("integrated_fold" + fold + ".pt")
        if teacher_path.exists():
            teacher = load_localize_model(str(teacher_path))
        else:
            teacher = load_localize_model(str(base_path))
            assert_model_partitions(teacher, train, val)
            if teacher["metadata"]["cnn_params"]["seed"] != args.seed:
                raise ValueError("Base checkpoint seed differs from the experiment.")
            print("Specialists outer", fold, flush=True)
            head = fit_specialists(
                all_features[train_ids], y[train_ids], seed=args.seed
            )
            base_val = _predict_model_on_rows(teacher, val)["prob_matrix"]
            weights, thresholds = calibrate_blend(
                base_val,
                predict_specialists(all_features[val_ids], head),
                y[val_ids],
                LABELS,
            )
            teacher["localization_model"]["specialist_head"] = head
            teacher["localization_model"]["specialist_weights"] = weights
            teacher["localization_model"]["class_thresholds"] = thresholds
            teacher["metadata"]["specialist_calibration_fold"] = validation_fold
            save_localize_model(teacher, str(teacher_path))
            print("Weights", weights, flush=True)
        assert_model_partitions(teacher, train, val)
        if teacher["metadata"]["cnn_params"]["seed"] != args.seed:
            raise ValueError("Integrated checkpoint seed differs from the experiment.")
        pred = _predict_model_on_rows(teacher, test)
        for key in ["probability", "prediction"]:
            matrices["integrated"][key][test_ids] = pred[
                "prob_matrix" if key == "probability" else "prediction_matrix"
            ]
        teacher_prob = None
        for name in ["student_control", "distilled"]:
            path = output / (name + "_fold" + fold + ".pt")
            start = time.perf_counter()
            if path.exists():
                student = load_localize_model(str(path))
            else:
                if name == "distilled" and teacher_prob is None:
                    teacher_prob = _predict_model_on_rows(teacher, train)["prob_matrix"]
                print(name, "outer", fold, flush=True)
                head = fit_multilabel_cnn_classifier(
                    [row["sequence"] for row in train],
                    y[train_ids],
                    LABELS,
                    feature_matrix=all_features[train_ids],
                    num_filters=32,
                    epochs=args.epochs,
                    batch_size=256,
                    seed=args.seed,
                    device=args.device,
                    patience=3,
                    validation_sequences=[row["sequence"] for row in val],
                    validation_labels=y[val_ids],
                    validation_features=all_features[val_ids],
                    teacher_probabilities=teacher_prob if name == "distilled" else None,
                    distillation_weight=args.teacher_weight,
                )
                head["sequence_only_features"] = True
                student = dict(
                    model_type="multilabel_cnn_v1",
                    localization_model=head,
                    feature_names=teacher["feature_names"],
                    perox_model={"mode": "embedded_multilabel"},
                    metadata=copy.deepcopy(teacher["metadata"]),
                )
                student["metadata"]["experiment_role"] = name
                student["metadata"]["cnn_params"].update(
                    num_filters=32,
                    feature_fusion=True,
                    sequence_layout="separate_termini",
                    mask_padding=True,
                )
                student["metadata"]["teacher_path"] = (
                    str(teacher_path) if name == "distilled" else None
                )
                probability = _predict_model_on_rows(student, val)["prob_matrix"]
                for i, label in enumerate(LABELS):
                    if len(np.unique(y[val_ids, i])) == 2:
                        head["class_thresholds"][label] = _tune_binary_threshold(
                            probability[:, i], y[val_ids, i]
                        )
                save_localize_model(student, str(path))
            assert_model_partitions(student, train, val)
            pred = _predict_model_on_rows(student, test)
            matrices[name]["probability"][test_ids] = pred["prob_matrix"]
            matrices[name]["prediction"][test_ids] = pred["prediction_matrix"]
            metrics = compute_multilabel_metrics(
                y[test_ids], pred["prediction_matrix"], LABELS
            )
            print(
                name,
                fold,
                metrics["macro_f1"],
                metrics["micro_f1"],
                "seconds",
                time.perf_counter() - start,
                flush=True,
            )
        atomic_write_json(
            str(output / ("fold" + fold + ".json")),
            {
                name: compute_multilabel_metrics(
                    y[test_ids], values["prediction"][test_ids], LABELS
                )
                for name, values in matrices.items()
            },
        )
    for name, values in matrices.items():
        np.savez_compressed(
            output / (name + "_oof.npz"), target=y, folds=folds.astype(str), **values
        )
        metrics = compute_multilabel_metrics(y, values["prediction"], LABELS)
        metrics.update(probability_metrics(y, values["probability"], LABELS))
        metrics["config"] = config
        atomic_write_json(str(output / (name + "_metrics.json")), metrics)
        print("COMPLETE", name, metrics["macro_f1"], metrics["micro_f1"], flush=True)


if __name__ == "__main__":
    main()
