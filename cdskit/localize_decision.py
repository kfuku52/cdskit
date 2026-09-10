"""Explicit inference decisions; numerical scores are not evidence of localization.

Safe inference skips empty, all-unknown and single-residue sequences. Numeric batch
APIs use zero placeholders for skipped rows, accompanied by score_available=False;
the detailed report serializes their scores as null. Legacy inference is unchanged.
"""

from functools import wraps
from inspect import signature
import json

import numpy as np

from cdskit.localize_runtime import current_prediction_runtime


LEGACY_POLICY = "legacy"
SAFE_POLICY = "safe-v1"
TAXONOMY_RULES_VERSION = "human-plastid-v1"


def decision_policy(model):
    override = current_prediction_runtime().decision_policy
    head = model.get("localization_model", model)
    value = (
        head.get("decision_policy", LEGACY_POLICY) if override == "model" else override
    )
    if value not in (LEGACY_POLICY, SAFE_POLICY):
        raise ValueError("Unsupported localization decision policy: {}".format(value))
    return value


def sequence_quality(sequence):
    from cdskit.localize_model import to_canonical_aa_sequence

    seq = to_canonical_aa_sequence(sequence)
    if not seq:
        return "empty_sequence"
    if set(seq) == {"X"}:
        return "all_unknown"
    if len(seq) < 2:
        return "single_residue"
    return ""


def taxonomy_allowed(labels):
    # Explicit organism metadata only. `non_plant` is never interpreted here.
    # This first rule is intentionally limited to Homo sapiens (NCBI 9606).
    taxon = current_prediction_runtime().taxonomy_id
    forbidden = {"chloroplast"} if taxon == "9606" else set()
    return np.asarray([name not in forbidden for name in labels], dtype=bool)


def validate_scores(probability, labels, rows=None):
    probability = np.asarray(probability)
    if (
        probability.ndim != 2
        or probability.shape[1] != len(labels)
        or (rows is not None and probability.shape[0] != rows)
    ):
        raise ValueError(
            "Localization score dimensions differ from sequences or labels."
        )
    if not np.isfinite(probability).all() or np.any(
        (probability < 0) | (probability > 1)
    ):
        raise ValueError("Localization scores must be finite values in [0, 1].")
    return probability


def threshold_decisions(probability, thresholds, labels, ensure_one_label=True):
    probability = validate_scores(probability, labels)
    vector = np.asarray([thresholds.get(name, 0.5) for name in labels], dtype=float)
    vector[~np.isfinite(vector) | (vector <= 0)] = 0.5
    allowed = taxonomy_allowed(labels)
    prediction = (probability >= vector).astype(np.int64)
    before_mask = prediction.any(axis=1)
    prediction[:, ~allowed] = 0
    forced = np.zeros(len(probability), dtype=bool)
    if ensure_one_label and allowed.any():
        empty = np.flatnonzero(~prediction.any(axis=1))
        scores = probability[empty] / vector
        scores[:, ~allowed] = -np.inf
        prediction[empty, scores.argmax(axis=1)] = 1
        forced[empty] = True
    status = [
        "forced_label"
        if forced[i]
        else "predicted"
        if row.any()
        else "taxonomy_excluded"
        if before_mask[i] or not allowed.any()
        else "below_threshold"
        for i, row in enumerate(prediction)
    ]
    return {
        "prediction_matrix": prediction,
        "decision_status": status,
        "forced_label": forced,
    }


def guard_multilabel_inputs(sequence_argument, model_argument):
    """Apply the same pre-inference gate to CNN and PLM, including direct APIs."""

    def decorate(function):
        spec = signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = spec.bind(*args, **kwargs)
            bound.apply_defaults()
            from cdskit.localize_model import to_canonical_aa_sequence

            sequences = [
                to_canonical_aa_sequence(seq)
                for seq in bound.arguments[sequence_argument]
            ]
            bound.arguments[sequence_argument] = sequences
            model = bound.arguments[model_argument]
            batch_size = bound.arguments["batch_size"]
            if (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, (int, np.integer))
                or batch_size < 1
            ):
                raise ValueError("batch_size must be positive.")
            features = bound.arguments.get("feature_matrix")
            if features is not None:
                features = np.asarray(features)
                if features.ndim != 2 or len(features) != len(sequences):
                    raise ValueError(
                        "Sequence and feature counts must match in a 2D feature matrix."
                    )
            policy = decision_policy(model)
            reasons = [sequence_quality(seq) for seq in sequences]
            keep = [
                i
                for i, reason in enumerate(reasons)
                if not reason or policy == LEGACY_POLICY
            ]
            if len(keep) == len(sequences) and sequences:
                result = function(*bound.args, **bound.kwargs)
                validate_scores(
                    result["prob_matrix"], model["class_order"], len(sequences)
                )
                result["score_available"] = np.ones(len(sequences), dtype=bool)
                result["quality_reason"] = reasons
                return result
            n, width = len(sequences), len(model["class_order"])
            result = {
                "prob_matrix": np.zeros((n, width), dtype=float),
                "score_available": np.zeros(n, dtype=bool),
                "quality_reason": reasons,
            }
            if bound.arguments["apply_thresholds"]:
                result.update(
                    prediction_matrix=np.zeros((n, width), dtype=np.int64),
                    decision_status=[
                        "invalid_input" if r == "empty_sequence" else "abstained"
                        for r in reasons
                    ],
                    forced_label=np.zeros(n, dtype=bool),
                )
            if keep:
                bound.arguments[sequence_argument] = [sequences[i] for i in keep]
                features = bound.arguments.get("feature_matrix")
                if features is not None:
                    bound.arguments["feature_matrix"] = np.asarray(features)[keep]
                valid = function(*bound.args, **bound.kwargs)
                validate_scores(valid["prob_matrix"], model["class_order"], len(keep))
                result["prob_matrix"][keep] = valid["prob_matrix"]
                result["score_available"][keep] = True
                if "prediction_matrix" in result:
                    result["prediction_matrix"][keep] = valid["prediction_matrix"]
                    result["forced_label"][keep] = valid["forced_label"]
                    for i, status in zip(keep, valid["decision_status"], strict=True):
                        result["decision_status"][i] = status
            return result

        return wrapped

    return decorate


def single_decision(sequence, model, result):
    reason = sequence_quality(sequence)
    available = not reason or decision_policy(model) == LEGACY_POLICY
    result.update(
        decision_status="predicted"
        if available
        else "invalid_input"
        if reason == "empty_sequence"
        else "abstained",
        quality_reason=reason,
        score_available=available,
        forced_label=False,
    )
    if not available:
        result["predicted_class"] = ""
    return result


DETAIL_FIELDS = [
    "report_schema",
    "source_model_sha256",
    "pts2_annotation_match",
    "decision_status",
    "quality_reason",
    "score_available",
    "forced_label",
    "decision_policy",
    "feature_schema",
    "specialist_feature_schema",
    "perox_feature_schema",
    "signal_schema",
    "taxonomy_id",
    "taxonomy_rules",
    "score_kind",
    "head_status",
    "perox_head_status",
    "label_head_status",
    "calibration_status",
]


def report_details(model, prediction):
    from cdskit.localize_schema import model_feature_schema

    head = model["localization_model"]
    model_type = model["model_type"]
    multilabel = model_type.startswith("multilabel_")
    if "specialist_head" in head:
        kind = "specialist_blend_score"
    elif "centroid" in model_type:
        kind = "centroid_score"
    elif multilabel:
        kind = "sigmoid_score"
    else:
        kind = "multiclass_model_score"
    constant = head.get("mode") == "constant"
    perox = model.get("perox_model", {})
    perox_status = (
        "embedded_multilabel"
        if multilabel and "peroxisome" in head["class_order"]
        else "unavailable"
        if multilabel
        else "constant"
        if perox.get("mode") == "constant"
        else "trained"
        if perox.get("mode") in ("centroid", "sklearn_binary")
        else "unavailable"
    )
    label_status = {}
    for name, label in zip(
        head.get("class_order", []), head.get("label_models", []), strict=False
    ):
        label_status[name] = (
            "constant" if label.get("mode") == "constant" else "trained"
        )
    specialist = head.get("specialist_head")
    if specialist:
        for name, expert, weight in zip(
            head["class_order"],
            specialist["labels"],
            head["specialist_weights"],
            strict=True,
        ):
            base_constant = constant or label_status.get(name) == "constant"
            expert_constant = "constant" in expert
            label_status[name] = (
                "constant"
                if (base_constant or weight == 1) and (expert_constant or weight == 0)
                else "trained"
            )
    if multilabel and label_status.get("peroxisome") == "constant":
        perox_status = "constant"
    head_status = "constant" if constant else "trained"
    if label_status:
        statuses = set(label_status.values())
        head_status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
    schema = model_feature_schema(model)
    return {
        "report_schema": "v2",
        "source_model_sha256": model.get("_artifact_sha256", ""),
        "decision_status": prediction["decision_status"],
        "quality_reason": prediction.get("quality_reason", ""),
        "score_available": bool(prediction["score_available"]),
        "forced_label": bool(prediction["forced_label"]),
        "decision_policy": decision_policy(model),
        "feature_schema": schema,
        "specialist_feature_schema": head.get("specialist_head", {}).get(
            "feature_schema", schema
        )
        if "specialist_head" in head
        else "none",
        "perox_feature_schema": perox.get("feature_schema", schema),
        "signal_schema": schema,
        "taxonomy_id": current_prediction_runtime().taxonomy_id,
        "taxonomy_rules": TAXONOMY_RULES_VERSION,
        "score_kind": "constant" if constant else kind,
        "head_status": head_status,
        "perox_head_status": perox_status,
        "label_head_status": json.dumps(label_status, sort_keys=True),
        "calibration_status": "unverified",
    }


def detailed_row(row, model, prediction, sequence):
    from cdskit.localize_model import detect_perox_signals
    from cdskit.localize_schema import CURRENT_FEATURE_SCHEMA

    row.update(report_details(model, prediction))
    signals = detect_perox_signals(sequence, feature_schema=CURRENT_FEATURE_SCHEMA)
    row["signal_schema"] = CURRENT_FEATURE_SCHEMA
    row["pts2_annotation_match"] = signals["pts2_match"]
    if "perox_signal_type" in row:
        row["perox_signal_type"] = signals["signal_type"]
    if not prediction["score_available"]:
        for key in row:
            if key.startswith("p_"):
                row[key] = None
    return row
