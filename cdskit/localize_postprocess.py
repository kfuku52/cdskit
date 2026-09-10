"""Fit postprocessing only on an inner development partition."""


def fit_postprocess(rows, options):
    from cdskit.localize_learn import (
        apply_two_stage_ctp_ltp_params_to_oof_rows,
        fit_temperature_from_oof,
        optimize_class_thresholds_from_oof,
        optimize_two_stage_ctp_ltp_from_oof,
    )

    from typing import Any

    params: dict[str, Any] = {}
    if options.get("two_stage"):
        tuned = optimize_two_stage_ctp_ltp_from_oof(rows, min_ltp_precision=0.0)
        params["two_stage"] = {
            key: tuned[key]
            for key in (
                "stage3_gate_threshold",
                "stage3_blend_beta",
                "stage3_ltp_threshold",
            )
        }
        rows = apply_two_stage_ctp_ltp_params_to_oof_rows(rows, **params["two_stage"])
    params["temperature"] = (
        fit_temperature_from_oof(rows) if options.get("temperature") else 1.0
    )
    params["thresholds"] = (
        optimize_class_thresholds_from_oof(
            rows,
            temperature=params["temperature"],
            objective=options.get("objective", "macro"),
        )
        if options.get("thresholds")
        else None
    )
    return params


def apply_postprocess(rows, params):
    from cdskit.localize_learn import apply_two_stage_ctp_ltp_params_to_oof_rows
    from cdskit.localize_model import postprocess_localization_probabilities

    if params.get("two_stage"):
        rows = apply_two_stage_ctp_ltp_params_to_oof_rows(rows, **params["two_stage"])
    out = []
    for row in rows:
        predicted, probabilities = postprocess_localization_probabilities(
            row["class_probabilities"],
            {
                "probability_calibration": {
                    "method": "temperature",
                    "temperature": params["temperature"],
                },
                "class_thresholds": params["thresholds"],
            },
        )
        out.append(
            dict(row, class_probabilities=probabilities, predicted_class=predicted)
        )
    return out


def evaluate_postprocessed_rows(rows):
    """Score frozen decisions; preserve calibrated probabilities in saved rows."""
    from cdskit.localize_learn import evaluate_oof_postprocess

    return evaluate_oof_postprocess(
        [dict(row, class_probabilities={row["predicted_class"]: 1.0}) for row in rows]
    )
