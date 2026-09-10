"""Freeze an external evaluation protocol and models before scoring any test rows."""

import json
from pathlib import Path

import numpy as np

from cdskit.deeploc_benchmark import (
    _predict_model_on_rows,
    build_label_matrix,
    compute_multilabel_metrics,
)
from cdskit.localize_evaluation import (
    assert_disjoint,
    dataset_digest,
    paired_cluster_bootstrap,
    probability_metrics,
    stratified_metrics,
)
from cdskit.localize_model import load_localize_model
from cdskit.localize_pipeline import code_identity, run_lock
from cdskit.localize_pipeline_config import (
    digest,
    file_digest,
    load_config,
    load_partitions,
)
from cdskit.localize_splits import audit_homology_partitions
from cdskit.tsvio import read_tsv
from cdskit.util import atomic_write_json


def freeze_evaluation(config_path, models, protocol_path, output):
    """Record declared selection history; software cannot establish historical non-use."""
    config_path, protocol_path, output = [
        Path(p).resolve() for p in (config_path, protocol_path, output)
    ]
    files = {str(p): file_digest(p) for p in (config_path, protocol_path)}
    code_sha256 = code_identity()
    config = load_config(config_path)
    files[config["data"]["path"]] = file_digest(config["data"]["path"])
    if (
        config["schema_version"] != 2
        or config["data"]["evidence_policy"] != "experimental"
    ):
        raise ValueError(
            "Frozen scientific evaluation requires observed experimental schema 2."
        )
    partitions = load_partitions(config)
    if not partitions["test"]:
        raise ValueError("A nonempty external test partition is required.")
    protocol = json.loads(protocol_path.read_text())
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object.")
    if any(
        not isinstance(protocol.get(key), str) or not protocol[key].strip()
        for key in (
            "population",
            "annotation_protocol",
            "selection_history",
            "primary_metric",
        )
    ):
        raise ValueError(
            "Protocol requires population, annotation_protocol, selection_history and primary_metric."
        )
    if protocol["primary_metric"] not in ("macro_f1", "micro_f1"):
        raise ValueError("Primary metric must be macro_f1 or micro_f1.")
    if protocol.get("test_used_for_selection") is not False:
        raise ValueError(
            "Previously inspected selection data cannot be declared a new final holdout."
        )
    if len(models) < 2 or any(
        not name or not name.replace("_", "").isalnum() for name in models
    ):
        raise ValueError("Supply at least two models with simple distinct names.")
    comparison = protocol.get("comparison", {})
    if (
        not isinstance(comparison, dict)
        or comparison.get("reference") not in models
        or comparison.get("candidate") not in models
        or comparison["reference"] == comparison["candidate"]
    ):
        raise ValueError(
            "Protocol must specify distinct reference and candidate model names."
        )
    development = partitions["train"] + partitions["validation"]
    extra = protocol.get("development_tsvs", [])
    if not isinstance(extra, list) or any(not isinstance(p, str) for p in extra):
        raise ValueError(
            "development_tsvs must list every additional supervised development source."
        )
    for source in extra:
        path = (protocol_path.parent / source).resolve()
        files[str(path)] = file_digest(path)
        development.extend(read_tsv(str(path), required_columns=["sequence"]))
    target = build_label_matrix(
        partitions["test"], config["labels"], "localization_labels"
    )
    if not ((target == 1).any(axis=0) & (target == 0).any(axis=0)).all():
        raise ValueError(
            "An F1 comparison requires observed positives and negatives for every label; a positive-only panel is insufficient."
        )
    assert_disjoint(development, partitions["test"])
    audit = audit_homology_partitions(
        {"development": development, "test": partitions["test"]}, config["threads"]
    )
    model_files, encoders = {}, {}
    for name, source in models.items():
        path = Path(source).resolve()
        files[str(path)] = file_digest(path)
        model_files[name] = str(path)
        head = load_localize_model(str(path))["localization_model"]
        if head.get("class_order") != config["labels"]:
            raise ValueError(
                "Frozen model label order differs from the evaluation contract."
            )
        if "encoder" in head:
            from cdskit.localize_multilabel_plm import ResidueEncoder

            actual = ResidueEncoder(head["encoder"]).identity
            if actual != head.get("encoder_identity"):
                raise ValueError("PLM encoder differs from trained model identity.")
            encoders[name] = dict(config=head["encoder"], identity=actual)
    identity = dict(
        schema_version=1,
        config_path=str(config_path),
        files=files,
        models=model_files,
        protocol=protocol,
        code_sha256=code_sha256,
        encoders=encoders,
        test_sha256=dataset_digest(partitions["test"]),
        homology_audit=audit,
    )
    manifest = {
        "identity": identity,
        "fingerprint": digest(identity),
        "status": "frozen_unscored",
    }
    _verify(manifest)
    # Exclusive output ownership prevents replacing an already frozen evaluation.
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(
        str(output / "protocol.json"),
        manifest,
    )
    return output / "protocol.json"


def _verify(manifest):
    identity = manifest["identity"]
    if (
        manifest["fingerprint"] != digest(identity)
        or identity["code_sha256"] != code_identity()
    ):
        raise ValueError("Frozen protocol or evaluation code changed.")
    if any(file_digest(path) != value for path, value in identity["files"].items()):
        raise ValueError("Frozen model, data, configuration or protocol changed.")
    for encoder in identity.get("encoders", {}).values():
        from cdskit.localize_multilabel_plm import ResidueEncoder

        if ResidueEncoder(encoder["config"]).identity != encoder["identity"]:
            raise ValueError("Frozen PLM encoder changed.")
    return identity


def evaluate_frozen(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    with run_lock(root):
        manifest = json.loads(manifest_path.read_text())
        identity = _verify(manifest)
        if (root / "metrics.json").exists():
            raise ValueError(
                "This frozen evaluation was already scored; preserve its result."
            )
        config = load_config(identity["config_path"])
        rows = load_partitions(config)["test"]
        if dataset_digest(rows) != identity["test_sha256"]:
            raise ValueError("Frozen test rows changed.")
        outputs = [root / "metrics.json"] + [
            root / (name + ".npz") for name in identity["models"]
        ]
        if any(str(p.resolve()) in identity["files"] or p.exists() for p in outputs):
            raise ValueError("Evaluation output would overwrite an existing artifact.")
        labels = config["labels"]
        target = build_label_matrix(rows, labels, "localization_labels")
        results, arrays = {}, {}
        for name, path in identity["models"].items():
            model = load_localize_model(path)
            if model["localization_model"].get("class_order") != labels:
                raise ValueError(
                    "Frozen model label order differs from the evaluation contract."
                )
            prediction = _predict_model_on_rows(model, rows)
            probability, decisions = (
                prediction["prob_matrix"],
                prediction["prediction_matrix"],
            )
            scores = compute_multilabel_metrics(target, decisions, labels)
            scores.update(probability_metrics(target, probability, labels))
            scores["strata"] = stratified_metrics(
                rows, target, decisions, probability, labels, compute_multilabel_metrics
            )
            results[name] = scores
            arrays[name] = dict(
                target=target,
                observation_mask=np.isfinite(target),
                probability=probability,
                prediction=decisions,
                labels=np.asarray(labels),
                accessions=np.asarray([row["accession"] for row in rows]),
            )
        comparison = identity["protocol"]["comparison"]
        difference = paired_cluster_bootstrap(
            target,
            arrays[comparison["reference"]]["prediction"],
            arrays[comparison["candidate"]]["prediction"],
            [row["cluster_id"] for row in rows],
            labels,
            compute_multilabel_metrics,
        )
        _verify(manifest)
        report = dict(
            protocol_fingerprint=manifest["fingerprint"],
            primary_metric=identity["protocol"]["primary_metric"],
            models=results,
            paired_difference=difference,
            selection_history_is_user_declared=True,
        )
        for name, values in arrays.items():
            np.savez_compressed(root / (name + ".npz"), **values)
        atomic_write_json(str(root / "metrics.json"), report)
    return root / "metrics.json"
