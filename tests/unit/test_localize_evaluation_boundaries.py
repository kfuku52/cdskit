"""Reject incomplete evaluation populations and preserve score ordering."""

import numpy as np
import pytest

from cdskit.deeploc_benchmark import compute_multilabel_metrics
from cdskit.localize_evaluation import (
    assert_disjoint,
    average_precision,
    cluster_bootstrap,
    grouped_folds,
    paired_cluster_bootstrap,
)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64, np.int64, bool, float])
def test_ap_preserves_perfect_ranking_with_zero(dtype):
    assert average_precision([1, 0], np.array([1, 0], dtype=dtype)) == 1.0


def test_ap_orders_large_integer_scores_without_float_rounding():
    scores = np.array([2**63, 2**63 + 1, 0], dtype=np.uint64)
    assert average_precision([0, 1, 0], scores) == 1.0


@pytest.mark.parametrize(
    "target,scores",
    [
        ([1, 0], [0.9]),
        ([1, 2], [0.9, 0.1]),
        ([0], [np.nan]),
        ([1], [np.inf]),
        ([[1]], [[0.5]]),
    ],
)
def test_ap_rejects_invalid_inputs_even_without_positives(target, scores):
    with pytest.raises(ValueError, match="AP requires"):
        average_precision(target, scores)


@pytest.mark.parametrize(
    "groups",
    [
        ["a", "b"],
        ["a", "b", "c", "d"],
        ["a", "b", None],
        ["a", "b", " "],
        ["a", "b", np.nan],
    ],
)
def test_grouped_folds_rejects_missing_or_extra_groups(groups):
    with pytest.raises(ValueError, match="groups must match"):
        grouped_folds([{}, {}, {}], groups, n_folds=2)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize(
    "groups", [["a", "b"], ["a", "b", "c", "d"], ["a", "b", None], ["a", "b", np.nan]]
)
def test_bootstrap_cannot_silently_drop_rows(paired, groups):
    target = np.array([[1], [0], [1]])
    prediction = np.array([[1], [0], [0]])
    with pytest.raises(ValueError, match="groups must match"):
        if paired:
            paired_cluster_bootstrap(
                target,
                prediction,
                prediction,
                groups,
                ["x"],
                compute_multilabel_metrics,
            )
        else:
            cluster_bootstrap(
                target, prediction, groups, ["x"], compute_multilabel_metrics
            )


@pytest.mark.parametrize("paired", [False, True])
def test_bootstrap_validates_unsampled_extra_prediction_row(paired):
    target = [[1], [0]]
    prediction = [[1], [0], [1]]
    with pytest.raises(ValueError, match="dimensions"):
        if paired:
            paired_cluster_bootstrap(
                target, target, prediction, [0, 1], ["x"], compute_multilabel_metrics
            )
        else:
            cluster_bootstrap(
                target, prediction, [0, 1], ["x"], compute_multilabel_metrics
            )


@pytest.mark.parametrize("iterations", [0, -1, 1.5, True])
def test_bootstrap_requires_positive_integer_iterations(iterations):
    with pytest.raises(ValueError, match="positive integer"):
        cluster_bootstrap(
            [[1], [0]],
            [[1], [0]],
            [0, 1],
            ["x"],
            compute_multilabel_metrics,
            iterations=iterations,
        )


@pytest.mark.parametrize("field", ["cluster_id", "accession"])
def test_partition_overlap_detects_zero_identifier(field):
    with pytest.raises(ValueError, match="Partition overlap"):
        assert_disjoint(
            [{"sequence": "MAAAA", field: 0}], [{"sequence": "MCCCC", field: "0"}]
        )


def test_missing_optional_identifiers_do_not_create_false_overlap():
    assert_disjoint(
        [{"sequence": "MAAAA", "cluster_id": None}],
        [{"sequence": "MCCCC", "cluster_id": np.nan}],
    )


def test_valid_grouping_and_paired_bootstrap_keep_all_rows():
    groups = [0, 0, 1, 2]
    folds = grouped_folds([{}] * 4, groups, 3)
    assert len(set(folds)) == 3 and folds[0] == folds[1]
    result = paired_cluster_bootstrap(
        [[1], [0], [1], [0]],
        [[1], [0], [1], [0]],
        [[1], [0], [1], [0]],
        groups,
        ["x"],
        compute_multilabel_metrics,
        iterations=12,
    )
    assert result["percentile_95"]["micro_f1"] == [0.0, 0.0]
    assert result["valid_iterations"]["micro_f1"] == 12


def test_ap_finite_extreme_scores_do_not_overflow():
    with np.errstate(over="raise"):
        assert (
            average_precision([1, 0], [np.finfo(float).max, -np.finfo(float).max])
            == 1.0
        )


def test_calibration_keeps_numeric_zero_cluster_together():
    from cdskit.targetp_external_aug import split_external_train_calibration_rows

    rows = [
        {
            "accession": str(i),
            "sequence": "M" + "A" * (i + 1),
            "localization": "SP",
            "cluster_id": group,
        }
        for i, group in enumerate([0, 0, 1, 1, 2, 2, 3, 3])
    ]
    train, validation, _ = split_external_train_calibration_rows(rows, 0.25, 9)
    assert len(train) + len(validation) == 8
    assert {r["cluster_id"] for r in train}.isdisjoint(
        r["cluster_id"] for r in validation
    )


def test_all_unknown_bootstrap_does_not_claim_success():
    result = cluster_bootstrap(
        np.full((2, 1), np.nan),
        [[0], [0]],
        [0, 1],
        ["x"],
        compute_multilabel_metrics,
        iterations=3,
    )
    assert result["status"] == "insufficient_observations"
    assert result["percentile_95"]["macro_f1"] is None
