"""Learning stages; orchestration and artifact ownership live in localize_pipeline."""

import json

import numpy as np

from cdskit.deeploc_benchmark import build_label_matrix, compute_multilabel_metrics
from cdskit.localize_evaluation import dataset_digest, probability_metrics
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
    return {
        "model_type": "multilabel_{}_v1".format(kind),
        "localization_model": head,
        "feature_names": [],
        "perox_model": {"mode": "embedded_multilabel"},
        "metadata": {
            "task": "localization",
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
    probability = predict(
        model, rows, settings["device"], settings["batch_size"], thresholds=False
    )["prob_matrix"]
    y = target(rows, labels)
    # Probability validation also catches non-finite model outputs before export.
    probability_metrics(y, probability, labels)
    thresholds = {label: 0.5 for label in labels}
    for i, label in enumerate(labels):
        if len(np.unique(y[:, i])) == 2:
            thresholds[label] = _tune_binary_threshold(
                probability[:, i], y[:, i], objective="f1"
            )
    model["localization_model"]["class_thresholds"] = thresholds


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
        "schema_version": 1,
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
    probabilities = predict(
        model, rows, settings["device"], settings["batch_size"], False
    )["prob_matrix"]
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
    report = {"test_data_sha256": dataset_digest(rows), "models": {}}
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
        metrics.update(probability_metrics(y, result["prob_matrix"], labels))
        metrics["model_sha256"] = file_digest(model_path)
        report["models"][name] = metrics
        np.savez_compressed(
            output / (name + ".npz"),
            target=y,
            probability=result["prob_matrix"],
            prediction=result["prediction_matrix"],
            labels=np.asarray(labels),
            accessions=np.asarray([row["accession"] for row in rows]),
        )
    atomic_write_json(str(output / "metrics.json"), report)
