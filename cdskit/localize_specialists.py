"""Portable per-localization boosted-tree specialists and validated score blends.

Only numeric tree nodes are exported; inference needs no sklearn pickle loading.
"""

from typing import Any

from cdskit.localize_schema import (
    with_model_feature_schema,
    extraction_schema,
)

import numpy as np

from cdskit.localize_labels import observed_targets, require_observed


def validate_specialists(model, class_order=None):
    """Reject inconsistent or cyclic numeric trees before model inference."""
    if model.get("mode") != "localization_specialists_v1":
        raise ValueError("Unsupported specialist model mode.")
    dim = model.get("feature_dim")
    if not isinstance(dim, int) or dim < 1:
        raise ValueError("Invalid specialist feature dimension.")
    labels = model.get("labels")
    if (
        not isinstance(labels, list)
        or not labels
        or (class_order is not None and len(labels) != len(class_order))
    ):
        raise ValueError("Specialist label count differs from class order.")
    for label in labels:
        if "constant" in label:
            if not np.isfinite(label["constant"]) or not 0 <= label["constant"] <= 1:
                raise ValueError("Invalid specialist constant probability.")
            continue
        if not np.isfinite(label.get("bias", np.nan)) or not label.get("trees"):
            raise ValueError("Invalid specialist bias or trees.")
        for tree in label["trees"]:
            fields = [
                "value",
                "feature_idx",
                "num_threshold",
                "missing_go_to_left",
                "left",
                "right",
                "is_leaf",
            ]
            if any(key not in tree for key in fields):
                raise ValueError("Missing specialist tree fields.")
            n = len(tree["value"])
            if not n or any(np.asarray(tree[key]).shape != (n,) for key in fields):
                raise ValueError("Inconsistent specialist tree dimensions.")
            if (
                not np.isfinite(tree["value"]).all()
                or np.isnan(tree["num_threshold"]).any()
            ):
                raise ValueError("Invalid specialist tree values.")
            for key in [
                "feature_idx",
                "left",
                "right",
                "is_leaf",
                "missing_go_to_left",
            ]:
                a = np.asarray(tree[key])
                if not np.isfinite(a).all() or np.any(a != np.floor(a)):
                    raise ValueError(
                        "Specialist tree indices and flags must be integers."
                    )
            for key in ["is_leaf", "missing_go_to_left"]:
                if not np.isin(tree[key], [0, 1]).all():
                    raise ValueError("Invalid specialist tree flags.")
            pending, visited = [0], set()
            while pending:
                node = int(pending.pop())
                if node < 0 or node >= n or node in visited:
                    raise ValueError("Specialist tree has invalid children or a cycle.")
                visited.add(node)
                if not tree["is_leaf"][node]:
                    if not 0 <= tree["feature_idx"][node] < dim:
                        raise ValueError("Invalid specialist tree feature index.")
                    pending.extend([tree["left"][node], tree["right"][node]])
            if len(visited) != n:
                raise ValueError("Specialist tree contains unreachable nodes.")


def fit_specialists(features, labels, seed=1, max_iter=100, feature_schema=None):
    feature_schema = extraction_schema(feature_schema)
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = np.asarray(features, dtype=np.float32), np.asarray(labels)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or not len(x):
        raise ValueError("Invalid specialist input dimensions.")
    require_observed(y, "Specialist training data", each_label=True)
    models: list[dict[str, Any]] = []
    for column in y.T:
        observed = np.isfinite(column)
        column = column[observed]
        if len(np.unique(column)) == 1:
            models.append({"constant": float(column[0])})
            continue
        estimator = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            min_samples_leaf=20,
            early_stopping=False,
            random_state=seed,
        )
        estimator.fit(x[observed], column)
        trees = []
        for predictors in estimator._predictors:
            nodes = predictors[0].nodes
            if nodes["is_categorical"].any():
                raise ValueError("Categorical specialist trees are not supported.")
            trees.append(
                {
                    name: nodes[name].tolist()
                    for name in [
                        "value",
                        "feature_idx",
                        "num_threshold",
                        "missing_go_to_left",
                        "left",
                        "right",
                        "is_leaf",
                    ]
                }
            )
        exported = {"bias": float(estimator._baseline_prediction[0, 0]), "trees": trees}
        # Verify the public predictor against the exported representation before accepting it.
        np.testing.assert_allclose(
            _predict_one(x[:256], exported),
            estimator.predict_proba(x[:256])[:, 1],
            atol=1e-7,
            rtol=1e-6,
        )
        models.append(exported)
    return {
        "mode": "localization_specialists_v1",
        "feature_schema": feature_schema,
        "feature_dim": x.shape[1],
        "labels": models,
        "seed": seed,
        "max_iter": max_iter,
    }


def _predict_one(features, model):
    if "constant" in model:
        return np.full(len(features), model["constant"])
    score = np.full(len(features), model["bias"], dtype=float)
    for tree in model["trees"]:
        node = np.zeros(len(features), dtype=int)
        leaf = np.asarray(tree["is_leaf"], dtype=bool)
        feature = np.asarray(tree["feature_idx"], dtype=int)
        threshold = np.asarray(tree["num_threshold"])
        left, right = (
            np.asarray(tree["left"], dtype=int),
            np.asarray(tree["right"], dtype=int),
        )
        missing_left = np.asarray(tree["missing_go_to_left"], dtype=bool)
        for _ in range(len(leaf) + 1):
            active = np.flatnonzero(~leaf[node])
            if not len(active):
                break
            current = node[active]
            value = features[active, feature[current]]
            go_left = np.where(
                np.isnan(value), missing_left[current], value <= threshold[current]
            )
            node[active] = np.where(go_left, left[current], right[current])
        else:
            raise ValueError("Specialist tree traversal did not reach a leaf.")
        score += np.asarray(tree["value"])[node]
    return 1.0 / (1.0 + np.exp(-np.clip(score, -700, 700)))


def predict_specialists(features, model):
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != model["feature_dim"]:
        raise ValueError("Specialist feature dimensions differ from training.")
    return np.column_stack([_predict_one(x, item) for item in model["labels"]])


def calibrate_blend(base_prob, specialist_prob, target, labels):
    from cdskit.localize_model import _tune_binary_threshold
    from cdskit.localize_evaluation import average_precision

    base_prob, specialist_prob, target = map(
        np.asarray, (base_prob, specialist_prob, target)
    )
    if base_prob.shape != specialist_prob.shape or base_prob.shape != target.shape:
        raise ValueError("Blend arrays must have identical shapes.")
    if target.ndim != 2 or not len(target) or target.shape[1] != len(labels):
        raise ValueError("Invalid blend calibration labels.")
    observed_targets(target)
    blend_probabilities(base_prob, specialist_prob, np.zeros(len(labels)))
    weights, thresholds = [], {}
    for i, name in enumerate(labels):
        observed = np.isfinite(target[:, i])
        column = target[observed, i]
        if len(np.unique(column)) < 2:
            weights.append(0.0)
            thresholds[name] = 0.5
            continue
        best = (-1.0, -1.0, 0.0)
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            probability = (1 - weight) * base_prob[
                observed, i
            ] + weight * specialist_prob[observed, i]
            threshold = _tune_binary_threshold(probability, column, objective="f1")
            prediction = probability >= threshold
            tp = np.sum(prediction & (column == 1))
            f1 = 2 * tp / max(1, prediction.sum() + column.sum())
            key = (f1, average_precision(column, probability), -weight)
            if key > best:
                best = key
                chosen_weight, chosen_threshold = weight, threshold
        weights.append(chosen_weight)
        thresholds[name] = float(chosen_threshold)
    return weights, thresholds


def blend_probabilities(base, specialists, weights):
    base, specialists, weights = map(np.asarray, (base, specialists, weights))
    if (
        base.ndim != 2
        or base.shape != specialists.shape
        or weights.shape != (base.shape[1],)
    ):
        raise ValueError("Invalid specialist blend dimensions.")
    for array in (base, specialists, weights):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(
                "Specialist probabilities and weights must be finite values in [0, 1]."
            )
    return (1 - weights) * base + weights * specialists


def threshold_predictions(probability, thresholds, labels, ensure_one_label=True):
    from cdskit.localize_decision import threshold_decisions

    return threshold_decisions(probability, thresholds, labels, ensure_one_label)[
        "prediction_matrix"
    ]


def sequence_features(sequences, feature_schema=None):
    from cdskit.localize_model import (
        extract_broad_localize_features,
        BROAD_FEATURE_NAMES,
    )

    if not sequences:
        return np.zeros((0, len(BROAD_FEATURE_NAMES)))
    return np.asarray(
        [
            extract_broad_localize_features(
                seq, kingdom="", feature_schema=feature_schema
            )[0]
            for seq in sequences
        ]
    )


@with_model_feature_schema("model")
def apply_specialists(sequences, model, probabilities):
    if "specialist_head" not in model:
        return probabilities
    schema = model["specialist_head"].get("feature_schema", extraction_schema())
    expert = predict_specialists(
        sequence_features(sequences, feature_schema=schema), model["specialist_head"]
    )
    return blend_probabilities(probabilities, expert, model["specialist_weights"])
