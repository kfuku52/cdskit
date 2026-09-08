"""Partition audits and probability metrics shared by localization experiments."""

import hashlib
import json

import numpy as np


def assert_disjoint(train_rows, test_rows):
    accessions = {row["accession"] for row in train_rows if row.get("accession")}
    sequences = {str(row["sequence"]).upper() for row in train_rows}
    clusters = {str(row["cluster_id"]) for row in train_rows if row.get("cluster_id")}
    overlap = [
        row
        for row in test_rows
        if row.get("accession", "") in accessions
        or str(row["sequence"]).upper() in sequences
        or (row.get("cluster_id") and str(row["cluster_id"]) in clusters)
    ]
    if overlap:
        raise ValueError(
            "Partition overlap: {} evaluation rows share accession, sequence or cluster with training.".format(
                len(overlap)
            )
        )


def dataset_digest(rows):
    # Retain row order: out-of-fold probabilities are aligned to this exact table.
    payload = [{key: row[key] for key in sorted(row)} for row in rows]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def assert_model_partitions(model, train_rows, validation_rows):
    """Check resumed checkpoint provenance, including the calibration partition."""
    metadata = model.get("metadata", {})
    for prefix, rows in [("training", train_rows), ("validation", validation_rows)]:
        if metadata.get(prefix + "_data_sha256") != dataset_digest(
            rows
        ) or metadata.get("num_" + prefix + "_rows") != len(rows):
            raise ValueError(
                "Checkpoint {} partition differs from the experiment.".format(prefix)
            )
    assert_disjoint(train_rows, validation_rows)


def grouped_folds(rows, groups, n_folds=5, seed=1):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    if len(members) < 2:
        raise ValueError("At least two independent groups are required for evaluation.")
    rng = np.random.default_rng(seed)
    keys = sorted(members)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(members[key]), reverse=True)
    sizes = np.zeros(min(n_folds, len(keys)), dtype=int)
    folds = np.empty(len(rows), dtype=object)
    for group in keys:
        fold = int(sizes.argmin())
        folds[members[group]] = str(fold)
        sizes[fold] += len(members[group])
    return folds


def average_precision(target, scores):
    target, scores = np.asarray(target), np.asarray(scores)
    if not target.sum():
        return None  # Undefined for a label with no positives; never silently report 0.
    order = np.argsort(-scores, kind="stable")
    truth, ordered = target[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ordered)), len(ordered) - 1]
    tp = np.cumsum(truth)[ends]
    precision = tp / (ends + 1)
    recall = tp / target.sum()
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def probability_metrics(target, probability, labels):
    target, probability = np.asarray(target), np.asarray(probability)
    if (
        target.shape != probability.shape
        or target.ndim != 2
        or target.shape[1] != len(labels)
    ):
        raise ValueError("Invalid probability metric dimensions.")
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("Probabilities must be finite values in [0, 1].")
    per_label = {
        name: average_precision(target[:, i], probability[:, i])
        for i, name in enumerate(labels)
    }
    supported = [value for value in per_label.values() if value is not None]
    return {
        "average_precision_by_label": per_label,
        "macro_average_precision_observed": float(np.mean(supported))
        if supported
        else None,
        "micro_average_precision": average_precision(
            target.ravel(), probability.ravel()
        ),
        "brier_score": float(np.mean((target - probability) ** 2))
        if target.size
        else None,
    }


def stratified_metrics(rows, target, prediction, probability, labels, metric_fn):
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        organism = str(row.get("organism_group", "unknown")) or "unknown"
        length = len(row["sequence"])
        length_bin = (
            "<=512" if length <= 512 else "513-1022" if length <= 1022 else ">1022"
        )
        for group in ["organism:" + organism, "length:" + length_bin]:
            groups.setdefault(group, []).append(i)
    result = {}
    for name, ids in sorted(groups.items()):
        metrics = metric_fn(target[ids], prediction[ids], labels)
        metrics.update(probability_metrics(target[ids], probability[ids], labels))
        result[name] = metrics
    return result


def cluster_bootstrap(
    target, prediction, groups, labels, metric_fn, iterations=200, seed=1
):
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    if len(members) < 2:
        return {"status": "insufficient_clusters"}
    clusters = list(members.values())
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"macro_f1": [], "micro_f1": []}
    for _ in range(iterations):
        ids = np.concatenate(
            [clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]
        )
        metrics = metric_fn(target[ids], prediction[ids], labels)
        for key in samples:
            samples[key].append(metrics[key])
    return {
        "status": "ok",
        "iterations": iterations,
        "seed": seed,
        "cluster_count": len(clusters),
        "percentile_95": {
            key: np.quantile(values, [0.025, 0.975]).tolist()
            for key, values in samples.items()
        },
    }
