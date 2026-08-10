"""Pure configuration and scheduling helpers for TargetP model training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TargetPTrainingOptions:
    batch_size: int
    epochs: int
    balanced_batch: bool
    selection_metric: str
    selection_threshold_grid: list[float]
    resume_state: str
    resume_learning_rate: float | None
    resume_reset_optimizer: bool
    resume_reset_scheduler: bool
    resume_reset_best_metrics: bool


def merge_targetp_training_config(
    *,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
    resume_payload: object,
    resume_state_impl: str,
) -> dict[str, Any]:
    """Merge defaults, explicit options, and architecture metadata for resume."""

    config = dict(defaults)
    config.update({key: value for key, value in overrides.items() if value is not None})
    if not isinstance(resume_payload, Mapping):
        return config

    resume_config = resume_payload.get("config", {})
    if resume_state_impl:
        config["rnn_impl"] = resume_state_impl
    elif (
        isinstance(resume_config, Mapping)
        and "rnn_impl" in resume_config
        and "rnn_impl" not in overrides
    ):
        config["rnn_impl"] = resume_config["rnn_impl"]
    return config


def _enabled(value: object) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1"}


def parse_targetp_training_options(
    config: Mapping[str, Any],
    *,
    default_threshold_grid: list[float],
) -> TargetPTrainingOptions:
    """Validate options used by the training loop and convert them once."""

    selection_metric = str(config.get("selection_metric", "val_loss")).strip().lower()
    valid_metrics = {"val_loss", "val_macro_f1", "val_threshold_macro_f1"}
    if selection_metric not in valid_metrics:
        raise ValueError(
            "selection_metric should be val_loss, val_macro_f1, "
            "or val_threshold_macro_f1."
        )

    raw_grid = config.get("selection_threshold_grid", default_threshold_grid)
    if raw_grid is None:
        raw_grid = default_threshold_grid

    resume_state = str(config.get("resume_state", "latest")).strip().lower()
    if resume_state not in {"latest", "best"}:
        raise ValueError("resume_state should be latest or best.")

    raw_resume_learning_rate = config.get("resume_learning_rate")
    resume_learning_rate = (
        None if raw_resume_learning_rate is None else float(raw_resume_learning_rate)
    )
    return TargetPTrainingOptions(
        batch_size=int(config["batch_size"]),
        epochs=int(config["epochs"]),
        balanced_batch=_enabled(config.get("balanced_batch", "no")),
        selection_metric=selection_metric,
        selection_threshold_grid=[float(value) for value in raw_grid],
        resume_state=resume_state,
        resume_learning_rate=resume_learning_rate,
        resume_reset_optimizer=_enabled(config.get("resume_reset_optimizer", "no")),
        resume_reset_scheduler=_enabled(config.get("resume_reset_scheduler", "no")),
        resume_reset_best_metrics=_enabled(
            config.get("resume_reset_best_metrics", "no")
        ),
    )


def targetp_metric_is_better(
    *,
    row: Mapping[str, Any],
    best_metrics: Mapping[str, Any] | None,
    selection_metric: str,
) -> bool:
    """Return whether an epoch should replace the current best checkpoint."""

    if best_metrics is None:
        return True
    if selection_metric == "val_loss":
        return float(row["val_loss"]) < float(best_metrics["val_loss"])
    if selection_metric == "val_threshold_macro_f1":
        return float(row["val_threshold_macro_f1"]) > float(
            best_metrics["val_threshold_macro_f1"]
        )
    return float(row["val_macro_f1"]) > float(best_metrics["val_macro_f1"])


def should_reduce_learning_rate(
    *, epoch: int, lr_reference_epoch: int, patience_epochs: int
) -> bool:
    """Match the legacy warm-up and exact-patience scheduler behavior."""

    return epoch > 10 and epoch - lr_reference_epoch == patience_epochs
