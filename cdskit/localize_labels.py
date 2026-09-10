"""Observed localization targets: NaN is unobserved, never a negative label.

NaN is an in-memory/NPZ interchange value only. TSV uses explicit positive and
negative label lists; JSON evidence uses the string ``unknown``.
"""

import numpy as np


def observed_targets(target, observation_mask=None):
    """Return finite targets and a boolean observation mask without imputing truth."""
    values = np.asarray(target, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Targets must be a two-dimensional matrix.")
    mask = ~np.isnan(values)
    if observation_mask is not None:
        supplied = np.asarray(observation_mask)
        if supplied.shape != values.shape or supplied.dtype != np.bool_:
            raise ValueError("Observation mask must be boolean and match target shape.")
        mask &= supplied
    if not np.isin(values[mask], [0, 1]).all():
        raise ValueError("Observed targets must be binary; use NaN for unknown.")
    return np.where(mask, values, 0).astype(np.float32), mask


def require_observed(target, name, each_label=False):
    values, mask = observed_targets(target)
    if not mask.any():
        raise ValueError("{} has no observed targets.".format(name))
    if each_label and not mask.any(axis=0).all():
        raise ValueError(
            "{} contains a head without observed targets; remove it from the configured label set.".format(
                name
            )
        )
    return values, mask


def masked_bce(logits, target, pos_weight=None, reduction="mean"):
    """BCE on observed cells; an all-unknown batch has zero loss and gradient."""
    import torch
    from torch.nn import functional as F

    mask = ~torch.isnan(target)
    safe = torch.where(mask, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(
        logits, safe, pos_weight=pos_weight, reduction="none"
    )
    total = torch.where(mask, loss, torch.zeros_like(loss)).sum()
    if reduction == "sum":
        return total
    if reduction != "mean":
        raise ValueError("Unsupported masked BCE reduction.")
    return total / mask.sum().clamp_min(1)


def masked_multilabel_metrics(target, prediction, labels, metric_fn):
    """Metrics conditional on annotation; whole-set match requires complete rows."""
    y, mask = observed_targets(target)
    prediction = np.asarray(prediction)
    if prediction.shape != y.shape or not np.isin(prediction, [0, 1]).all():
        raise ValueError("Predictions must be binary and match targets.")
    if len(labels) != y.shape[1]:
        raise ValueError("Label count does not match targets.")
    # Reuse the established closed-world definitions only on observed entries.
    flat = metric_fn(
        y[mask].reshape(-1, 1), prediction[mask].reshape(-1, 1), ["observed"]
    )
    result = dict(flat)
    by_label = {}
    for i, label in enumerate(labels):
        selected = mask[:, i]
        scores = metric_fn(
            y[selected, i : i + 1], prediction[selected, i : i + 1], [label]
        )["by_label"][label]
        scores["observed_count"] = int(selected.sum())
        scores["unknown_count"] = int((~selected).sum())
        if not selected.any():
            for key in (
                "precision",
                "recall",
                "sensitivity",
                "specificity",
                "f1",
                "mcc",
            ):
                scores[key] = None
        by_label[label] = scores
    supported = [row["f1"] for row in by_label.values() if row["observed_count"]]
    positive = [row["f1"] for row in by_label.values() if row["support"]]
    complete = mask.all(axis=1)
    active = mask.any(axis=1)
    intersections = ((y == 1) & (prediction == 1) & mask).sum(axis=1)[active]
    unions = (((y == 1) | (prediction == 1)) & mask).sum(axis=1)[active]
    sums = (((y == 1) & mask).sum(axis=1) + ((prediction == 1) & mask).sum(axis=1))[
        active
    ]
    jaccard = np.divide(
        intersections, unions, out=np.ones(len(unions)), where=unions > 0
    )
    sample_f1 = np.divide(
        2 * intersections, sums, out=np.ones(len(sums)), where=sums > 0
    )
    result.update(
        n_rows=len(y),
        n_labels=len(labels),
        by_label=by_label,
        label_contract="observed_binary_v1",
        observed_count=int(mask.sum()),
        unknown_count=int((~mask).sum()),
        fully_observed_rows=int(complete.sum()),
        macro_f1=float(np.mean(supported)) if supported else None,
        macro_f1_observed_labels=float(np.mean(positive)) if positive else None,
        subset_accuracy=float(
            np.mean(np.all(y[complete] == prediction[complete], axis=1))
        )
        if complete.any()
        else None,
        jaccard=float(jaccard.mean()) if len(jaccard) else None,
        sample_f1=float(sample_f1.mean()) if len(sample_f1) else None,
        accuracy=float(sample_f1.mean()) if len(sample_f1) else None,
    )
    if not mask.any():
        for key in (
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "hamming_loss",
            "predicted_per_true",
        ):
            result[key] = None
    return result


def validate_label_evidence(text, positive, negative, policy="experimental"):
    """Validate a curated evidence ledger, not the truth of a cited experiment."""
    import json

    try:
        records = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "label_evidence must be a JSON list of evidence records."
        ) from exc
    if not isinstance(records, list):
        raise ValueError("label_evidence must be a JSON list.")
    if policy not in ("experimental", "declared"):
        raise ValueError("Unknown evidence policy.")
    if set(positive) & set(negative):
        raise ValueError("Positive and negative labels conflict.")
    expected = {label: "positive" for label in positive}
    expected.update({label: "negative" for label in negative})
    covered = set()
    for record in records:
        if not isinstance(record, dict) or any(
            not isinstance(record.get(key), str) or not record[key].strip()
            for key in (
                "label",
                "state",
                "evidence_type",
                "source",
                "source_version",
                "reference",
            )
        ):
            raise ValueError(
                "Evidence requires label, state, evidence_type, source, source_version and reference."
            )
        label, state = record["label"], record["state"]
        if state != expected.get(label):
            raise ValueError("Evidence state disagrees with positive/negative labels.")
        kind = record["evidence_type"]
        if kind not in (
            "experimental",
            "dataset",
            "similarity",
            "sequence_analysis",
            "localization_proxy",
        ):
            raise ValueError("Unknown evidence type.")
        if policy == "experimental" and kind != "experimental":
            raise ValueError(
                "Experimental policy excludes proxy, predicted and dataset labels."
            )
        covered.add(label)
    if covered != set(expected):
        raise ValueError(
            "Every observed label requires evidence; absent labels are unknown."
        )
    return records
