"""Scoped prediction settings, kept out of serialized model payloads."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRuntime:
    device: str = "cpu"
    offline: bool = False
    batch_size: int = 512
    esm_batch_size: int = 128
    decision_policy: str = "model"
    report_schema: str = "model"
    taxonomy_id: str = ""

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.esm_batch_size < 1:
            raise ValueError("Prediction batch sizes must be positive.")
        if self.decision_policy not in ("model", "legacy", "safe-v1"):
            raise ValueError("Unsupported localization decision policy.")
        if self.report_schema not in ("model", "legacy", "v2"):
            raise ValueError("Unsupported localization report schema.")
        if self.taxonomy_id and (
            not self.taxonomy_id.isdigit() or int(self.taxonomy_id) < 1
        ):
            raise ValueError("taxonomy_id must be a positive NCBI taxonomy ID.")


_DEFAULT_RUNTIME = PredictionRuntime()
_RUNTIME: ContextVar[PredictionRuntime] = ContextVar(
    "cdskit_prediction_runtime", default=_DEFAULT_RUNTIME
)


def current_prediction_runtime() -> PredictionRuntime:
    return _RUNTIME.get()


def offline_requested() -> bool:
    return current_prediction_runtime().offline or (
        os.environ.get("CDSKIT_OFFLINE", "").strip().lower()
        in {"1", "true", "t", "yes", "y", "on"}
    )


@contextmanager
def prediction_runtime(settings: PredictionRuntime) -> Iterator[None]:
    """Share device/offline/batch settings across every nested predictor."""
    token = _RUNTIME.set(settings)
    try:
        yield
    finally:
        _RUNTIME.reset(token)
