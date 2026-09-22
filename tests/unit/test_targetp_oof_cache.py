"""Reject stale or malformed cached predictions before resumed evaluation."""

import numpy as np
import pytest

from cdskit.targetp_blend import _load_oof_npz, _load_oof_fold_npz


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize(
    "corruption",
    ["nan", "range", "shape", "targets", "fractional_target", "provenance", "mass"],
)
def test_invalid_cached_predictions_are_rejected(tmp_path, fold, corruption):
    path = tmp_path / "cache.npz"
    data = dict(
        prob_matrix=np.eye(2),
        true_idx=np.array([0, 1]),
        class_names=np.array(["noTP", "SP"]),
        cache_key="expected",
        row_index=np.array([4, 7]),
    )
    if corruption == "nan":
        data["prob_matrix"][0, 0] = np.nan
    elif corruption == "range":
        data["prob_matrix"][0] = [-0.1, 1.1]
    elif corruption == "shape":
        data["prob_matrix"] = np.ones((2, 3)) / 3
    elif corruption == "targets":
        data["true_idx"] = np.array([0, 2])
    elif corruption == "fractional_target":
        data["true_idx"] = np.array([0, 0.5])
    elif corruption == "provenance":
        data["cache_key"] = "other dataset or training settings"
    else:
        data["prob_matrix"][0] = [0.1, 0.1]
    np.savez(path, **data)
    with pytest.raises(ValueError):
        if fold:
            _load_oof_fold_npz(path, ["noTP", "SP"], cache_key="expected")
        else:
            _load_oof_npz(path, cache_key="expected")


def test_duplicate_fold_row_indices_are_rejected(tmp_path):
    path = tmp_path / "fold.npz"
    np.savez(
        path,
        prob_matrix=np.eye(2),
        true_idx=np.array([0, 1]),
        class_names=np.array(["noTP", "SP"]),
        row_index=np.array([4, 4]),
    )
    with pytest.raises(ValueError, match="row indices"):
        _load_oof_fold_npz(path, ["noTP", "SP"])
