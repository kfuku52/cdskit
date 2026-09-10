"""Partition audits and probability metrics shared by localization experiments."""

import hashlib
import json

import numpy as np

from cdskit.localize_labels import observed_targets


def normalize_partition_id(value):
    """Keep zero-valued identifiers; normalize missing table values to None."""
    if value is None or (
        isinstance(value, (float, np.floating)) and not np.isfinite(value)
    ):
        return None
    key = str(value).strip()
    return None if key.lower() in {"", "none", "nan", "na", "null"} else key


def _group_members(groups, rows):
    if len(groups) != rows:
        raise ValueError("Complete groups must match evaluation rows.")
    members: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        key = normalize_partition_id(group)
        if key is None:
            raise ValueError("Complete groups must match evaluation rows.")
        members.setdefault(key, []).append(i)
    return members


def _bootstrap_inputs(target, predictions, groups, labels, iterations):
    if (
        isinstance(iterations, (bool, np.bool_))
        or not isinstance(iterations, (int, np.integer))
        or iterations < 1
    ):
        raise ValueError("Bootstrap iterations must be a positive integer.")
    target = np.asarray(target)
    predictions = [np.asarray(prediction) for prediction in predictions]
    if (
        target.ndim != 2
        or target.shape[1] != len(labels)
        or any(prediction.shape != target.shape for prediction in predictions)
    ):
        raise ValueError(
            "Bootstrap predictions and labels must match target dimensions."
        )
    # Validate the full population before resampling: a bad row might otherwise
    # escape validation when no draw happens to contain it.
    observed_targets(target)
    if any(not np.isin(prediction, [0, 1]).all() for prediction in predictions):
        raise ValueError("Bootstrap predictions must be binary.")
    return target, predictions, _group_members(groups, len(target))


def assert_disjoint(train_rows, test_rows):
    from cdskit.localize_model import to_canonical_aa_sequence

    accessions = {
        normalize_partition_id(row.get("accession")) for row in train_rows
    } - {None}
    sequences = {to_canonical_aa_sequence(row["sequence"]) for row in train_rows}
    clusters = {normalize_partition_id(row.get("cluster_id")) for row in train_rows} - {
        None
    }
    overlap = [
        row
        for row in test_rows
        if normalize_partition_id(row.get("accession")) in accessions
        or to_canonical_aa_sequence(row["sequence"]) in sequences
        or normalize_partition_id(row.get("cluster_id")) in clusters
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
    if (
        isinstance(n_folds, (bool, np.bool_))
        or not isinstance(n_folds, (int, np.integer))
        or n_folds < 2
    ):
        raise ValueError("n_folds must be an integer of at least 2.")
    members = _group_members(groups, len(rows))
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
    if (
        target.ndim != 1
        or target.shape != scores.shape
        or not np.isin(target, [0, 1]).all()
        or scores.dtype.kind not in "biuf"
        or not np.isfinite(scores).all()
    ):
        raise ValueError(
            "AP requires matching binary targets and finite numeric scores."
        )
    if not target.sum():
        return None  # Undefined for a label with no positives; never silently report 0.
    # Negating unsigned integers wraps at zero; negating booleans fails.
    # Ties are evaluated as a group, so reversing their internal order is safe.
    order = np.argsort(scores, kind="stable")[::-1]
    truth, ordered = target[order], scores[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(ordered) - 1]
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
    target, mask = observed_targets(target)
    per_label = {
        name: average_precision(target[mask[:, i], i], probability[mask[:, i], i])
        for i, name in enumerate(labels)
    }
    supported = [value for value in per_label.values() if value is not None]
    return {
        "average_precision_by_label": per_label,
        "reliability_by_label": {
            name: reliability_bins(target[mask[:, i], i], probability[mask[:, i], i])
            for i, name in enumerate(labels)
        },
        "macro_average_precision_observed": float(np.mean(supported))
        if supported
        else None,
        "micro_average_precision": average_precision(target[mask], probability[mask]),
        "observed_count": int(mask.sum()),
        "observed_count_by_label": {
            name: int(mask[:, i].sum()) for i, name in enumerate(labels)
        },
        "unknown_count": int((~mask).sum()),
        "brier_score": float(np.mean((target[mask] - probability[mask]) ** 2))
        if mask.any()
        else None,
    }


def stratified_metrics(
    rows, target, prediction, probability, labels, metric_fn, score_available=None
):
    available = (
        np.ones(len(rows), dtype=bool)
        if score_available is None
        else np.asarray(score_available, dtype=bool)
    )
    if available.shape != (len(rows),):
        raise ValueError("Score availability must match evaluation rows.")
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        organism = str(row.get("organism_group", "unknown")) or "unknown"
        length = len(row["sequence"])
        length_bin = (
            "<=512" if length <= 512 else "513-1022" if length <= 1022 else ">1022"
        )
        strata = ["organism:" + organism, "length:" + length_bin]
        strata.extend(
            key + ":" + str(row.get(key) or "unknown")
            for key in ("compartment", "fragment", "isoform")
        )
        for group in strata:
            groups.setdefault(group, []).append(i)
    result = {}
    for name, ids in sorted(groups.items()):
        metrics = metric_fn(target[ids], prediction[ids], labels)
        scored = np.asarray(ids)[available[ids]]
        metrics.update(probability_metrics(target[scored], probability[scored], labels))
        if score_available is not None:
            metrics["scored_rows"] = len(scored)
            metrics["unscored_rows"] = len(ids) - len(scored)
            metrics["score_coverage"] = len(scored) / len(ids)
        result[name] = metrics
    return result


def cluster_bootstrap(
    target, prediction, groups, labels, metric_fn, iterations=200, seed=1
):
    target, predictions, members = _bootstrap_inputs(
        target, [prediction], groups, labels, iterations
    )
    prediction = predictions[0]
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
            if metrics[key] is not None:
                samples[key].append(metrics[key])
    return {
        "status": "ok" if all(samples.values()) else "insufficient_observations",
        "iterations": iterations,
        "seed": seed,
        "cluster_count": len(clusters),
        "percentile_95": {
            key: np.quantile(values, [0.025, 0.975]).tolist() if values else None
            for key, values in samples.items()
        },
    }


def reliability_bins(target, probability, bins=10):
    """Descriptive calibration bins over observed labels (not proof of calibration)."""
    target, probability = np.asarray(target), np.asarray(probability)
    result = []
    for i in range(bins):
        selected = (probability >= i / bins) & (
            (probability < (i + 1) / bins) if i + 1 < bins else (probability <= 1)
        )
        result.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": int(selected.sum()),
                "mean_probability": float(probability[selected].mean())
                if selected.any()
                else None,
                "positive_fraction": float(target[selected].mean())
                if selected.any()
                else None,
            }
        )
    return result


def paired_cluster_bootstrap(
    target,
    prediction_a,
    prediction_b,
    groups,
    labels,
    metric_fn,
    iterations=1000,
    seed=1,
):
    """Paired B-minus-A intervals; related proteins are resampled together."""
    target, predictions, members = _bootstrap_inputs(
        target, [prediction_a, prediction_b], groups, labels, iterations
    )
    if len(members) < 2:
        return {"status": "insufficient_clusters", "cluster_count": len(members)}
    clusters = list(members.values())
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"macro_f1": [], "micro_f1": []}
    for _ in range(iterations):
        ids = np.concatenate(
            [clusters[i] for i in rng.integers(len(clusters), size=len(clusters))]
        )
        a, b = [
            metric_fn(target[ids], prediction[ids], labels)
            for prediction in predictions
        ]
        for key in samples:
            if a[key] is not None and b[key] is not None:
                samples[key].append(float(b[key] - a[key]))
    return {
        "status": "ok" if all(samples.values()) else "insufficient_observations",
        "direction": "B minus A",
        "cluster_count": len(clusters),
        "iterations": iterations,
        "seed": seed,
        "valid_iterations": {key: len(value) for key, value in samples.items()},
        "percentile_95": {
            key: np.quantile(value, [0.025, 0.975]).tolist() if value else None
            for key, value in samples.items()
        },
    }
