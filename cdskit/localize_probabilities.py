"""Probability validation, normalization and calibration shared by inference paths.

Invalid numerical results must never become a confident biological prediction.
Zero mass after an explicit class restriction retains the legacy noTP fallback.
"""

import re
import numpy as np

LOCALIZATION_CLASSES = ("noTP", "SP", "mTP", "cTP", "lTP")


def validate_probability_values(values):
    try:
        probability = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Localization scores must be finite values in [0, 1]."
        ) from exc
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("Localization scores must be finite values in [0, 1].")
    return probability


def validate_scores(probability, labels, rows=None):
    probability = validate_probability_values(probability)
    if (
        probability.ndim != 2
        or probability.shape[1] != len(labels)
        or (rows is not None and probability.shape[0] != rows)
    ):
        raise ValueError(
            "Localization score dimensions differ from sequences or labels."
        )
    return probability


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0:
        return logits
    if not np.isfinite(logits).all():
        raise ValueError("Localization logits must be finite.")
    max_logit = float(np.max(logits))
    shifted = logits - max_logit
    exp_vals = np.exp(shifted)
    denom = float(np.sum(exp_vals))
    if denom <= 0:
        return np.zeros_like(exp_vals)
    return exp_vals / denom


def _sanitize_probability(value):
    return float(validate_probability_values([value])[0])


def normalize_class_probabilities(class_probs):
    out_probs = {class_name: 0.0 for class_name in LOCALIZATION_CLASSES}
    if isinstance(class_probs, dict):
        for class_name in LOCALIZATION_CLASSES:
            out_probs[class_name] = _sanitize_probability(
                class_probs.get(class_name, 0.0)
            )
    total = float(sum(out_probs.values()))
    if total <= 0.0:
        out_probs["noTP"] = 1.0
        return out_probs
    for class_name in LOCALIZATION_CLASSES:
        out_probs[class_name] = out_probs[class_name] / total
    return out_probs


def normalize_localization_probability_matrix(probability_matrix, organism_group=""):
    """Normalize localization rows with the same fallbacks as scalar inference."""
    probabilities = np.asarray(probability_matrix, dtype=np.float64).copy()
    if probabilities.ndim != 2 or probabilities.shape[1] != len(LOCALIZATION_CLASSES):
        raise ValueError(
            "Localization probability matrix should have {} columns.".format(
                len(LOCALIZATION_CLASSES)
            )
        )
    validate_probability_values(probabilities)
    if normalize_organism_group(organism_group) == "non_plant":
        probabilities[:, LOCALIZATION_CLASSES.index("cTP")] = 0.0
        probabilities[:, LOCALIZATION_CLASSES.index("lTP")] = 0.0
    totals = np.sum(probabilities, axis=1, keepdims=True)
    nonempty = totals[:, 0] > 0.0
    probabilities[nonempty] /= totals[nonempty]
    probabilities[~nonempty, :] = 0.0
    probabilities[~nonempty, LOCALIZATION_CLASSES.index("noTP")] = 1.0
    return probabilities


def normalize_organism_group(value):
    txt = str(value or "").strip().lower()
    txt = re.sub(r"[\s\-]+", "_", txt)
    mapping = {
        "": "",
        "unknown": "",
        "auto": "",
        "plant": "plant",
        "plants": "plant",
        "viridiplantae": "plant",
        "nonplant": "non_plant",
        "non_plant": "non_plant",
        "non_plants": "non_plant",
        "other": "non_plant",
        "metazoa": "non_plant",
        "fungi": "non_plant",
        "animal": "non_plant",
        "animals": "non_plant",
    }
    if txt in mapping:
        return mapping[txt]
    raise ValueError("Unsupported organism_group: {}".format(value))


def apply_organism_group_constraints(class_probs, organism_group=""):
    group = normalize_organism_group(organism_group)
    probs = normalize_class_probabilities(class_probs=class_probs)
    if group == "non_plant":
        probs["cTP"] = 0.0
        probs["lTP"] = 0.0
        probs = normalize_class_probabilities(class_probs=probs)
    return probs


def apply_temperature_scaling(class_probs, temperature):
    probs = normalize_class_probabilities(class_probs=class_probs)
    try:
        temp = float(temperature)
    except Exception:
        temp = 1.0
    if (not np.isfinite(temp)) or (temp <= 0.0) or (abs(temp - 1.0) < 1.0e-12):
        return probs
    vec = np.asarray(
        [probs[class_name] for class_name in LOCALIZATION_CLASSES], dtype=np.float64
    )
    vec = np.clip(vec, 1.0e-12, 1.0)
    logits = np.log(vec) / temp
    scaled = softmax(logits)
    return {
        LOCALIZATION_CLASSES[i]: float(scaled[i])
        for i in range(len(LOCALIZATION_CLASSES))
    }


def _predict_class_with_thresholds(class_probs, class_thresholds):
    probs = normalize_class_probabilities(class_probs=class_probs)
    scores = list()
    for class_name in LOCALIZATION_CLASSES:
        threshold = 1.0
        if isinstance(class_thresholds, dict):
            threshold = class_thresholds.get(class_name, 1.0)
        try:
            threshold = float(threshold)
        except Exception:
            threshold = 1.0
        if (not np.isfinite(threshold)) or (threshold <= 0.0):
            threshold = 1.0
        scores.append(float(probs[class_name]) / threshold)
    pred_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return LOCALIZATION_CLASSES[pred_idx], probs


def postprocess_localization_probabilities(class_probs, localization_model):
    probs = normalize_class_probabilities(class_probs=class_probs)
    calibration = localization_model.get("probability_calibration", {})
    if isinstance(calibration, dict):
        method = str(calibration.get("method", "")).strip().lower()
        if method == "temperature":
            probs = apply_temperature_scaling(
                class_probs=probs,
                temperature=calibration.get("temperature", 1.0),
            )
    pred_class, probs = _predict_class_with_thresholds(
        class_probs=probs,
        class_thresholds=localization_model.get("class_thresholds", None),
    )
    return pred_class, probs
