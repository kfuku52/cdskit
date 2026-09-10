"""Learning stages; orchestration and artifact ownership live in localize_pipeline."""

import json

from cdskit.localize_schema import CURRENT_FEATURE_SCHEMA

import numpy as np

from cdskit.deeploc_benchmark import build_label_matrix, compute_multilabel_metrics
from cdskit.localize_evaluation import (
    dataset_digest,
    probability_metrics,
    stratified_metrics,
    paired_cluster_bootstrap,
)
from cdskit.localize_model import (
    _tune_binary_threshold,
    load_localize_model,
    save_localize_model,
)
from cdskit.localize_pipeline_config import digest, file_digest
from cdskit.util import atomic_write_json


def target(rows, labels):
    return build_label_matrix(rows, labels, "localization_labels")


def partition_identity(partitions):
    return {name: dataset_digest(rows) for name, rows in partitions.items()}


def predict(model, rows, device="cpu", batch_size=8, thresholds=True):
    sequences = [row["sequence"] for row in rows]
    head = model["localization_model"]
    if model["model_type"] == "multilabel_plm_v1":
        from cdskit.localize_multilabel_plm import predict_multilabel_plm

        return predict_multilabel_plm(
            sequences,
            head,
            device=device,
            batch_size=batch_size,
            apply_thresholds=thresholds,
        )
    from cdskit.localize_multilabel_cnn import predict_multilabel_cnn_batch

    return predict_multilabel_cnn_batch(
        sequences,
        head,
        device=device,
        batch_size=batch_size,
        apply_thresholds=thresholds,
    )


def wrap_model(head, config, partitions, kind):
    head["ensure_one_label"] = config["ensure_one_label"]
    head["feature_schema"] = CURRENT_FEATURE_SCHEMA
    head["decision_policy"] = config["decision_policy"]
    return {
        "model_type": "multilabel_{}_v1".format(kind),
        "feature_schema": CURRENT_FEATURE_SCHEMA,
        "localization_model": head,
        "feature_names": [],
        "perox_model": {"mode": "embedded_multilabel"},
        "metadata": {
            "task": "localization",
            "label_contract": "observed_binary_v1"
            if config["schema_version"] == 2
            else "legacy_closed_world",
            "evidence_policy": config["data"]["evidence_policy"]
            if config["schema_version"] == 2
            else "dataset",
            "seqtype": "protein",
            "partition_identity": partition_identity(partitions),
            "labels": config["labels"],
            "num_training_rows": len(partitions["train"]),
            "num_validation_rows": len(partitions["validation"]),
            "training_data_sha256": dataset_digest(partitions["train"]),
            "validation_data_sha256": dataset_digest(partitions["validation"]),
            "threshold_source": "validation_partition",
        },
    }


def calibrate(model, config, rows, settings):
    labels = config["labels"]
    result = predict(
        model, rows, settings["device"], settings["batch_size"], thresholds=False
    )
    probability = require_scored_rows(result)
    y = target(rows, labels)
    # Probability validation also catches non-finite model outputs before export.
    probability_metrics(y, probability, labels)
    thresholds = {label: 0.5 for label in labels}
    for i, label in enumerate(labels):
        observed = np.isfinite(y[:, i])
        if len(np.unique(y[observed, i])) == 2:
            thresholds[label] = _tune_binary_threshold(
                probability[observed, i], y[observed, i], objective="f1"
            )
    model["localization_model"]["class_thresholds"] = thresholds


def require_scored_rows(result):
    """Never fit thresholds or a student to an abstention's numeric placeholders."""
    available = np.asarray(result["score_available"], dtype=bool)
    if available.shape != (len(result["prob_matrix"]),) or not available.all():
        raise ValueError(
            "Unscored rows cannot be used for calibration or distillation."
        )
    return result["prob_matrix"]


def fit_teacher(config, partitions, output):
    from cdskit.localize_multilabel_plm import fit_multilabel_plm

    settings = config["teacher"]
    encoder = {
        key: settings[key]
        for key in (
            "model_name",
            "revision",
            "cache_dir",
            "local_files_only",
            "pooling",
            "window",
            "overlap",
        )
    }
    train, validation = partitions["train"], partitions["validation"]
    head = fit_multilabel_plm(
        [row["sequence"] for row in train],
        target(train, config["labels"]),
        config["labels"],
        encoder,
        validation_sequences=[row["sequence"] for row in validation],
        validation_y=target(validation, config["labels"]),
        **{
            key: settings[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "patience",
                "seed",
                "device",
            )
        },
    )
    model = wrap_model(head, config, partitions, "plm")
    calibrate(model, config, validation, settings)
    save_localize_model(model, str(output / "model.pt"))
    atomic_write_json(
        str(output / "training.json"),
        {"selected_epoch": head["selected_epoch"], "history": head["training_history"]},
    )


def prediction_identity(config, partitions, teacher_path):
    return {
        "schema_version": 2,
        "feature_schema": config["feature_schema"],
        "decision_policy": config["decision_policy"],
        "labels": config["labels"],
        "training_data_sha256": dataset_digest(partitions["train"]),
        "teacher_sha256": file_digest(teacher_path),
    }


def make_teacher_predictions(config, partitions, teacher_path, output):
    model = load_localize_model(str(teacher_path))
    metadata = model.get("metadata", {})
    if metadata.get("labels") != config["labels"] or any(
        metadata.get("partition_identity", {}).get(name)
        != dataset_digest(partitions[name])
        for name in ("train", "validation")
    ):
        raise ValueError(
            "Teacher training/validation partitions or label order differ."
        )
    rows = partitions["train"]
    settings = config["teacher"]
    result = predict(model, rows, settings["device"], settings["batch_size"], False)
    probabilities = require_scored_rows(result)
    probability_metrics(target(rows, config["labels"]), probabilities, config["labels"])
    identity = prediction_identity(config, partitions, teacher_path)
    np.savez_compressed(
        output / "probabilities.npz",
        probability=probabilities,
        accessions=np.asarray([row["accession"] for row in rows]),
        sequence_sha256=np.asarray([digest(row["sequence"]) for row in rows]),
        identity=np.asarray(json.dumps(identity, sort_keys=True)),
    )


def read_teacher_predictions(config, partitions, teacher_path, prediction_path):
    expected = prediction_identity(config, partitions, teacher_path)
    rows = partitions["train"]
    with np.load(prediction_path, allow_pickle=False) as saved:
        if json.loads(str(saved["identity"].item())) != expected:
            raise ValueError("Teacher prediction provenance differs.")
        if saved["accessions"].tolist() != [row["accession"] for row in rows] or saved[
            "sequence_sha256"
        ].tolist() != [digest(row["sequence"]) for row in rows]:
            raise ValueError("Teacher prediction row alignment differs.")
        probabilities = saved["probability"].copy()
    probability_metrics(target(rows, config["labels"]), probabilities, config["labels"])
    return probabilities


def fit_students(config, partitions, teacher_path, prediction_path, output):
    from cdskit.localize_multilabel_cnn import fit_multilabel_cnn_classifier

    probabilities = read_teacher_predictions(
        config, partitions, teacher_path, prediction_path
    )
    settings = config["student"]
    train, validation = partitions["train"], partitions["validation"]
    parameters = {
        key: value
        for key, value in settings.items()
        if key not in ("train_control", "distillation_weight")
    }
    for name in ["student", "control"] if settings["train_control"] else ["student"]:
        head = fit_multilabel_cnn_classifier(
            [row["sequence"] for row in train],
            target(train, config["labels"]),
            config["labels"],
            validation_sequences=[row["sequence"] for row in validation],
            validation_labels=target(validation, config["labels"]),
            teacher_probabilities=probabilities if name == "student" else None,
            distillation_weight=settings["distillation_weight"]
            if name == "student"
            else 0,
            **parameters,
        )
        model = wrap_model(head, config, partitions, "cnn")
        model["metadata"]["experiment_role"] = name
        model["metadata"]["teacher_sha256"] = file_digest(teacher_path)
        calibrate(model, config, validation, settings)
        save_localize_model(model, str(output / (name + ".pt")))
        atomic_write_json(
            str(output / (name + "-training.json")),
            {
                "selected_epoch": head["selected_epoch"],
                "history": head["training_history"],
            },
        )


def evaluate_students(config, partitions, student_dir, output):
    rows = partitions["test"]
    if not rows:
        raise ValueError("The evaluate stage requires a nonempty test partition.")
    labels = config["labels"]
    report = {
        "test_data_sha256": dataset_digest(rows),
        "models": {},
        "evaluation_scope": "provided_test_partition; independence from earlier model selection is a protocol responsibility",
    }
    predictions = {}
    for name in (
        ["student", "control"] if config["student"]["train_control"] else ["student"]
    ):
        model_path = student_dir / (name + ".pt")
        model = load_localize_model(str(model_path))
        result = predict(
            model, rows, device="cpu", batch_size=config["student"]["batch_size"]
        )
        y = target(rows, labels)
        metrics = compute_multilabel_metrics(y, result["prediction_matrix"], labels)
        available = np.asarray(result["score_available"], dtype=bool)
        metrics.update(
            probability_metrics(y[available], result["prob_matrix"][available], labels)
        )
        metrics["scored_rows"] = int(available.sum())
        metrics["unscored_rows"] = int((~available).sum())
        metrics["score_coverage"] = float(available.mean())
        metrics["accepted_only"] = (
            compute_multilabel_metrics(
                y[available], result["prediction_matrix"][available], labels
            )
            if available.any()
            else None
        )
        metrics["model_sha256"] = file_digest(model_path)
        metrics["strata"] = stratified_metrics(
            rows,
            y,
            result["prediction_matrix"],
            result["prob_matrix"],
            labels,
            compute_multilabel_metrics,
            score_available=available,
        )
        predictions[name] = result["prediction_matrix"]
        report["models"][name] = metrics
        np.savez_compressed(
            output / (name + ".npz"),
            target=y,
            observation_mask=np.isfinite(y),
            probability=result["prob_matrix"],
            score_available=available,
            decision_status=np.asarray(result["decision_status"]),
            prediction=result["prediction_matrix"],
            labels=np.asarray(labels),
            accessions=np.asarray([row["accession"] for row in rows]),
        )
    if "control" in predictions:
        report["student_minus_control"] = (
            paired_cluster_bootstrap(
                y,
                predictions["control"],
                predictions["student"],
                [row["cluster_id"] for row in rows],
                labels,
                compute_multilabel_metrics,
            )
            if all(row.get("cluster_id") for row in rows)
            else {
                "status": "unavailable",
                "reason": "Homology cluster IDs required for paired intervals.",
            }
        )
    atomic_write_json(str(output / "metrics.json"), report)
