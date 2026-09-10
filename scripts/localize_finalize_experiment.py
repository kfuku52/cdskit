#!/usr/bin/env python3
"""Fit fixed development-selected models and evaluate HPA once without tuning it."""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from cdskit.localize_schema import CURRENT_FEATURE_SCHEMA
from cdskit.deeploc_benchmark import (
    DEEPLOC_LOCALIZATION_LABELS as LABELS,
    _read_prepared_tsv,
    fit_deeploc_multilabel_model,
    _predict_model_on_rows,
    build_label_matrix,
    compute_multilabel_metrics,
)
from cdskit.localize_model import (
    save_localize_model,
    load_localize_model,
    _tune_binary_threshold,
)
from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier
from cdskit.localize_specialists import (
    fit_specialists,
    predict_specialists,
    sequence_features,
    calibrate_blend,
)
from cdskit.localize_evaluation import (
    assert_disjoint,
    assert_model_partitions,
    probability_metrics,
    dataset_digest,
)
from cdskit.util import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", required=True)
    parser.add_argument("--prepared_dir", default="data/localize_bench/deeploc21")
    parser.add_argument(
        "--base_recipe", required=True, choices=["cnn_legacy", "cnn_termini"]
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument(
        "--fit_only", action="store_true", help="Fit/export models without reading HPA."
    )
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    output = Path(args.experiment_dir) / "final"
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_prepared_tsv(
        str(Path(args.prepared_dir) / "deeploc21_localization_train_validation.tsv")
    )
    val_fold = sorted({row["fold_id"] for row in rows})[-1]
    train = [row for row in rows if row["fold_id"] != val_fold]
    val = [row for row in rows if row["fold_id"] == val_fold]
    assert_disjoint(train, val)
    config = dict(
        {key: value for key, value in vars(args).items() if key != "fit_only"},
        feature_schema=CURRENT_FEATURE_SCHEMA,
        dataset_sha256=dataset_digest(rows),
        validation_fold=val_fold,
    )
    path = output / "config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Existing final configuration differs.")
    atomic_write_json(str(path), config)
    models = {}
    for recipe in sorted({"cnn_legacy", args.base_recipe}):
        path = output / (recipe + ".pt")
        if path.exists():
            model = load_localize_model(str(path))
        else:
            print("Final base", recipe, flush=True)
            model = fit_deeploc_multilabel_model(
                rows,
                LABELS,
                "localization_labels",
                "localization",
                "cnn",
                dict(
                    epochs=args.epochs,
                    device=args.device,
                    seed=1,
                    batch_size=256,
                    sequence_layout="legacy"
                    if recipe == "cnn_legacy"
                    else "separate_termini",
                    mask_padding=recipe != "cnn_legacy",
                ),
            )
            save_localize_model(model, str(path))
        assert_model_partitions(model, train, val)
        models[recipe] = model
    base = models[args.base_recipe]
    x, vx = (
        sequence_features([row["sequence"] for row in train]),
        sequence_features([row["sequence"] for row in val]),
    )
    y, vy = (
        build_label_matrix(train, LABELS, "localization_labels"),
        build_label_matrix(val, LABELS, "localization_labels"),
    )
    path = output / "integrated.pt"
    if path.exists():
        teacher = load_localize_model(str(path))
    else:
        print("Final specialists", flush=True)
        teacher = load_localize_model(str(output / (args.base_recipe + ".pt")))
        expert = fit_specialists(x, y, seed=1)
        weights, thresholds = calibrate_blend(
            _predict_model_on_rows(base, val)["prob_matrix"],
            predict_specialists(vx, expert),
            vy,
            LABELS,
        )
        teacher["localization_model"].update(
            specialist_head=expert,
            specialist_weights=weights,
            class_thresholds=thresholds,
        )
        teacher["metadata"]["specialist_calibration_fold"] = val_fold
        save_localize_model(teacher, str(path))
    assert_model_partitions(teacher, train, val)
    models["integrated"] = teacher
    teacher_prob = None
    for name in ["student_control", "distilled"]:
        path = output / (name + ".pt")
        if path.exists():
            student = load_localize_model(str(path))
        else:
            print("Final", name, flush=True)
            if name == "distilled":
                teacher_prob = _predict_model_on_rows(teacher, train)["prob_matrix"]
            head = fit_multilabel_cnn_classifier(
                [row["sequence"] for row in train],
                y,
                LABELS,
                feature_matrix=x,
                num_filters=32,
                epochs=args.epochs,
                batch_size=256,
                seed=1,
                device=args.device,
                validation_sequences=[row["sequence"] for row in val],
                validation_labels=vy,
                validation_features=vx,
                teacher_probabilities=teacher_prob if name == "distilled" else None,
                distillation_weight=0.5,
            )
            head["sequence_only_features"] = True
            student = dict(
                model_type="multilabel_cnn_v1",
                localization_model=head,
                feature_names=base["feature_names"],
                perox_model={"mode": "embedded_multilabel"},
                metadata=copy.deepcopy(base["metadata"]),
            )
            student["metadata"]["experiment_role"] = name
            student["metadata"]["cnn_params"].update(
                num_filters=32,
                feature_fusion=True,
                sequence_layout="separate_termini",
                mask_padding=True,
            )
            probability = _predict_model_on_rows(student, val)["prob_matrix"]
            for i, label in enumerate(LABELS):
                if len(np.unique(vy[:, i])) == 2:
                    head["class_thresholds"][label] = _tune_binary_threshold(
                        probability[:, i], vy[:, i]
                    )
            save_localize_model(student, str(path))
        assert_model_partitions(student, train, val)
        models[name] = student
    if args.fit_only:
        return
    # HPA labels are loaded only after every model/threshold has been frozen.
    test = _read_prepared_tsv(str(Path(args.prepared_dir) / "deeploc21_hpa_test.tsv"))
    assert_disjoint(rows, test)
    target = build_label_matrix(test, LABELS, "localization_labels")
    reports = {}
    for name, model in models.items():
        pred = _predict_model_on_rows(model, test)
        metrics = compute_multilabel_metrics(target, pred["prediction_matrix"], LABELS)
        metrics.update(probability_metrics(target, pred["prob_matrix"], LABELS))
        metrics["test_sha256"] = dataset_digest(test)
        metrics["model_bytes"] = (output / (name + ".pt")).stat().st_size
        reports[name] = metrics
        np.savez_compressed(
            output / (name + "_hpa.npz"),
            target=target,
            probability=pred["prob_matrix"],
            prediction=pred["prediction_matrix"],
        )
        print("HPA", name, metrics["macro_f1"], metrics["micro_f1"], flush=True)
    atomic_write_json(str(output / "hpa_metrics.json"), reports)


if __name__ == "__main__":
    main()
