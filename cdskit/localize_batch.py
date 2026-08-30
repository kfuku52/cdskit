"""Shared, bounded batch inference for CLI output and cross validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cdskit.localize_runtime import current_prediction_runtime
from cdskit.localize_model import (
    FEATURE_NAMES,
    LOCALIZATION_CLASSES,
    apply_organism_group_constraints,
    extract_localize_features,
    normalize_organism_group,
    normalize_localization_probability_matrix,
    postprocess_localization_probabilities,
    predict_perox_batch,
    apply_targetp_feature_ltp_specialist_postprocess_batch,
    apply_targetp_specialist_postprocess_batch,
    compose_two_stage_ctp_ltp_probabilities,
    _normalize_ctp_ltp_probs,
)


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def predict_model_probability_matrix(
    aa_sequences: Sequence[str],
    feature_matrix: np.ndarray,
    model_type: str,
    localization_model: dict[str, Any],
    organism_group: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Batch base predictors and recursively batch TargetP blend models."""
    num_rows = len(aa_sequences)
    strategy = str(localization_model.get("strategy", "single_stage")).strip().lower()
    if strategy in {"two_stage", "two_stage_ctp_ltp"}:
        stage1, stage1_order, _ = predict_model_probability_matrix(
            aa_sequences=aa_sequences,
            feature_matrix=feature_matrix,
            model_type=model_type,
            localization_model=localization_model.get("stage1_model", {}),
            organism_group=organism_group,
        )
        stage2, stage2_order, _ = predict_model_probability_matrix(
            aa_sequences=aa_sequences,
            feature_matrix=feature_matrix,
            model_type=model_type,
            localization_model=localization_model.get("stage2_model", {}),
            organism_group=organism_group,
        )
        base_matrix = np.zeros((num_rows, len(LOCALIZATION_CLASSES)), dtype=np.float64)
        no_tp_col = stage1_order.index("noTP")
        tp_col = stage1_order.index("TP") if "TP" in stage1_order else None
        base_matrix[:, 0] = stage1[:, no_tp_col]
        tp_probs = (
            stage1[:, tp_col]
            if tp_col is not None
            else np.maximum(0.0, 1.0 - base_matrix[:, 0])
        )
        for class_i, class_name in enumerate(LOCALIZATION_CLASSES[1:], start=1):
            if class_name in stage2_order:
                base_matrix[:, class_i] = (
                    tp_probs * stage2[:, stage2_order.index(class_name)]
                )
        totals = np.sum(base_matrix, axis=1, keepdims=True)
        base_matrix[totals[:, 0] <= 0.0, 0] = 1.0
        totals[totals <= 0.0] = 1.0
        base_matrix /= totals
        if strategy == "two_stage":
            return base_matrix, list(LOCALIZATION_CLASSES), {}
        stage3_model = localization_model.get("stage3_model", None)
        if not isinstance(stage3_model, dict) or len(stage3_model) == 0:
            details = [
                {
                    "base_class_probabilities": dict(
                        zip(LOCALIZATION_CLASSES, row.tolist(), strict=True)
                    ),
                    "stage3_ctp_ltp_probabilities": {"cTP": 0.5, "lTP": 0.5},
                    "gate_threshold": 0.0,
                    "blend_beta": 1.0,
                    "ltp_threshold": 0.5,
                    "ctp_ltp_mass": float(row[3] + row[4]),
                    "gate_active": False,
                }
                for row in base_matrix
            ]
            return (
                base_matrix,
                list(LOCALIZATION_CLASSES),
                {"two_stage_ctp_ltp_details": details},
            )
        stage3, stage3_order, _ = predict_model_probability_matrix(
            aa_sequences=aa_sequences,
            feature_matrix=feature_matrix,
            model_type=model_type,
            localization_model=stage3_model,
            organism_group=organism_group,
        )
        out_matrix = np.zeros_like(base_matrix)
        details = []
        for row_i in range(num_rows):
            base_probs = {
                class_name: float(base_matrix[row_i, class_i])
                for class_i, class_name in enumerate(LOCALIZATION_CLASSES)
            }
            stage3_probs = {
                class_name: float(stage3[row_i, class_i])
                for class_i, class_name in enumerate(stage3_order)
            }
            stage3_probs = _normalize_ctp_ltp_probs(stage3_probs)
            out_probs, detail = compose_two_stage_ctp_ltp_probabilities(
                base_class_probs=base_probs,
                stage3_ctp_ltp_probs=stage3_probs,
                stage3_gate_threshold=localization_model.get(
                    "stage3_gate_threshold", 0.0
                ),
                stage3_blend_beta=localization_model.get("stage3_blend_beta", 1.0),
                stage3_ltp_threshold=localization_model.get(
                    "stage3_ltp_threshold", 0.5
                ),
            )
            detail["base_class_probabilities"] = base_probs
            detail["stage3_ctp_ltp_probabilities"] = stage3_probs
            details.append(detail)
            out_matrix[row_i] = [out_probs[name] for name in LOCALIZATION_CLASSES]
        return (
            out_matrix,
            list(LOCALIZATION_CLASSES),
            {"two_stage_ctp_ltp_details": details},
        )
    if str(localization_model.get("mode", "")).strip().lower() == "constant":
        class_order = list(localization_model.get("class_order", LOCALIZATION_CLASSES))
        class_label = str(localization_model.get("class_label", "")).strip()
        if class_label == "" and len(class_order) == 1:
            class_label = class_order[0]
        matrix = np.zeros((num_rows, len(class_order)), dtype=np.float64)
        matrix[:, class_order.index(class_label)] = 1.0
        return matrix, class_order, {}
    if model_type == "bilstm_attention_v1":
        from cdskit.localize_bilstm import predict_bilstm_attention_batch

        prob_matrix = predict_bilstm_attention_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            device=current_prediction_runtime().device,
            batch_size=current_prediction_runtime().batch_size,
            feature_matrix=feature_matrix,
        )
        return np.asarray(prob_matrix), list(localization_model["class_order"]), {}
    elif model_type == "esm_head_v1":
        from cdskit.localize_esm_head import predict_esm_head_batch

        prob_matrix = predict_esm_head_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            device=current_prediction_runtime().device,
            batch_size=current_prediction_runtime().esm_batch_size,
        )
        return np.asarray(prob_matrix), list(localization_model["class_order"]), {}
    elif model_type == "targetp_torch_v1":
        from cdskit.targetp_torch import predict_targetp2_torch_batch

        prob_matrix = predict_targetp2_torch_batch(
            aa_sequences=aa_sequences,
            organism_groups=[organism_group] * len(aa_sequences),
            localization_model=localization_model,
            device=current_prediction_runtime().device,
            batch_size=current_prediction_runtime().batch_size,
        )
        return np.asarray(prob_matrix), list(LOCALIZATION_CLASSES), {}
    elif model_type == "targetp_feature_ensemble_v1":
        from cdskit.localize_model import predict_targetp_feature_ensemble_batch

        prob_matrix = predict_targetp_feature_ensemble_batch(
            aa_sequences=aa_sequences,
            localization_model=localization_model,
            organism_group=organism_group,
        )
        return prob_matrix, list(LOCALIZATION_CLASSES), {}
    elif model_type == "nearest_centroid_v1":
        mean = np.asarray(localization_model["mean"], dtype=np.float64)
        std = np.asarray(localization_model["std"], dtype=np.float64)
        centroids = np.asarray(localization_model["centroids"], dtype=np.float64)
        log_priors = np.asarray(localization_model["log_priors"], dtype=np.float64)
        z = (np.asarray(feature_matrix, dtype=np.float64) - mean) / std
        logits = (
            -0.5
            * np.sum(
                (z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                axis=2,
            )
            + log_priors[np.newaxis, :]
        )
        logits -= np.max(logits, axis=1, keepdims=True)
        prob_matrix = np.exp(logits)
        prob_matrix /= np.sum(prob_matrix, axis=1, keepdims=True)
        return prob_matrix, list(localization_model["class_order"]), {}
    elif model_type == "targetp_blend_v1":
        base_models = localization_model.get("base_models", [])
        if not isinstance(base_models, list) or len(base_models) != 2:
            raise ValueError("targetp_blend_v1 requires exactly two base_models.")
        base_matrices = []
        for base_model in base_models:
            base_matrix, base_order, _ = predict_model_probability_matrix(
                aa_sequences=aa_sequences,
                feature_matrix=feature_matrix,
                model_type=str(base_model.get("model_type", "")).strip(),
                localization_model=base_model.get("localization_model", {}),
                organism_group=organism_group,
            )
            reordered = np.zeros(
                (num_rows, len(LOCALIZATION_CLASSES)),
                dtype=np.float64,
            )
            for class_i, class_name in enumerate(LOCALIZATION_CLASSES):
                if class_name in base_order:
                    reordered[:, class_i] = base_matrix[:, base_order.index(class_name)]
            reordered = normalize_localization_probability_matrix(
                probability_matrix=reordered,
                organism_group=organism_group,
            )
            base_matrices.append(reordered)
        alpha_by_class = localization_model.get("alpha_by_class", 1.0)
        alpha = np.ones((len(LOCALIZATION_CLASSES),), dtype=np.float64)
        if isinstance(alpha_by_class, dict):
            alpha[:] = [
                _float_or_default(alpha_by_class.get(class_name, 1.0), 1.0)
                for class_name in LOCALIZATION_CLASSES
            ]
        else:
            try:
                alpha[:] = float(alpha_by_class)
            except (TypeError, ValueError):
                alpha[:] = 1.0
        alpha = np.clip(alpha, 0.0, 1.0)
        prob_matrix = (
            alpha[np.newaxis, :] * base_matrices[0]
            + (1.0 - alpha[np.newaxis, :]) * base_matrices[1]
        )
        totals = np.sum(prob_matrix, axis=1, keepdims=True)
        totals[totals <= 0.0] = 1.0
        prob_matrix /= totals
        return (
            prob_matrix,
            list(LOCALIZATION_CLASSES),
            {
                "base_probabilities": base_matrices,
            },
        )
    raise ValueError("Unsupported batched localize model_type: {}".format(model_type))


def _predict_same_organism(
    aa_sequences: Sequence[str],
    model: dict[str, Any],
    organism_group: str,
    feature_matrix: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    model_type = str(model.get("model_type", ""))
    localization_model = model["localization_model"]
    feature_rows = [extract_localize_features(seq) for seq in aa_sequences]
    if feature_matrix is None:
        feature_matrix = np.asarray([row[0] for row in feature_rows], dtype=np.float64)
    matrix, class_order, details = predict_model_probability_matrix(
        aa_sequences,
        feature_matrix,
        model_type,
        localization_model,
        organism_group,
    )
    perox_yes = predict_perox_batch(
        feature_matrix=feature_matrix,
        perox_model=model["perox_model"],
        aa_sequences=aa_sequences,
        organism_group=organism_group,
    )
    probabilities = []
    predictions = []
    for row in matrix:
        class_probs = {name: float(row[i]) for i, name in enumerate(class_order)}
        if model_type != "targetp_blend_v1":
            class_probs = apply_organism_group_constraints(class_probs, organism_group)
        prediction, class_probs = postprocess_localization_probabilities(
            class_probs,
            localization_model,
        )
        probabilities.append(class_probs)
        predictions.append(prediction)
    processed = np.asarray(
        [
            [row.get(name, 0.0) for name in LOCALIZATION_CLASSES]
            for row in probabilities
        ],
        dtype=np.float64,
    )
    if model_type == "targetp_blend_v1":
        bases = details["base_probabilities"]
        predictions = apply_targetp_specialist_postprocess_batch(
            aa_sequences=aa_sequences,
            base_prob_matrix=processed,
            prob_a_matrix=bases[0],
            prob_b_matrix=bases[1],
            localization_model=localization_model,
            organism_group=organism_group,
        )
    elif model_type == "targetp_feature_ensemble_v1":
        predictions = apply_targetp_feature_ltp_specialist_postprocess_batch(
            aa_sequences=aa_sequences,
            base_prob_matrix=processed,
            pred_classes=predictions,
            localization_model=localization_model,
            organism_group=organism_group,
        )
    results = []
    for index, (features, signals) in enumerate(feature_rows):
        result = {
            "predicted_class": predictions[index],
            "class_probabilities": probabilities[index],
            "perox_probability_yes": float(perox_yes[index]),
            "perox_signal_type": signals["signal_type"],
            "feature_values": features,
            "feature_names": list(FEATURE_NAMES),
            "pts1_match": bool(signals["pts1_match"]),
            "pts2_match": bool(signals["pts2_match"]),
        }
        if "two_stage_ctp_ltp_details" in details:
            result["two_stage_ctp_ltp_details"] = details["two_stage_ctp_ltp_details"][
                index
            ]
        results.append(result)
    return results


def predict_localization_batch(
    aa_sequences: Sequence[str],
    model: dict[str, Any],
    organism_groups: Sequence[str] | None = None,
    feature_matrix: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Preserve input order while batching records with the same organism rules."""
    count = len(aa_sequences)
    groups = [""] * count if organism_groups is None else list(organism_groups)
    if len(groups) != count:
        raise ValueError("Sequence and organism-group counts must match.")
    if feature_matrix is not None and len(feature_matrix) != count:
        raise ValueError("Sequence and feature counts must match.")
    results: list[dict[str, Any]] = [{} for _ in range(count)]
    batch_size = current_prediction_runtime().batch_size
    for start in range(0, count, batch_size):
        grouped: dict[str, list[int]] = {}
        for index in range(start, min(start + batch_size, count)):
            group = normalize_organism_group(groups[index])
            grouped.setdefault(group, []).append(index)
        for group, indices in grouped.items():
            batch = _predict_same_organism(
                [aa_sequences[index] for index in indices],
                model,
                group,
                None if feature_matrix is None else feature_matrix[indices],
            )
            for index, prediction in zip(indices, batch, strict=True):
                results[index] = prediction
    return results
