"""Preserve calibration-bin boundaries, reduction order and direct API behavior."""

import json
import warnings

import numpy as np
import pytest

from cdskit.localize_evaluation import reliability_bins

pytestmark = pytest.mark.unit


def _original_bins(target, probability, bins=10):
    target, probability = np.asarray(target), np.asarray(probability)
    result = []
    for i in range(bins):
        selected = (probability >= i / bins) & (
            (probability < (i + 1) / bins) if i + 1 < bins else (probability <= 1)
        )
        result.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": int(selected.sum()),
                "mean_probability": float(probability[selected].mean())
                if selected.any()
                else None,
                "positive_fraction": float(target[selected].mean())
                if selected.any()
                else None,
            }
        )
    return result


def _outcome(function, target, probability, bins=10):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = json.dumps(function(target, probability, bins), sort_keys=True)
            error = None
        except (IndexError, TypeError, ValueError, OverflowError) as exception:
            result = None
            error = (type(exception), str(exception))
    return (
        result,
        error,
        [(warning.category, str(warning.message)) for warning in caught],
    )


@pytest.mark.parametrize("bins", [1, 2, 3, 7, 10, 31])
def test_bin_boundaries_include_lower_and_only_final_upper(bins):
    probability = np.concatenate(
        [
            [np.nextafter(0.0, -np.inf), 0.0, 1.0, np.nextafter(1.0, np.inf)],
            np.asarray(
                [
                    neighbor
                    for i in range(1, bins)
                    for neighbor in (
                        np.nextafter(i / bins, -np.inf),
                        i / bins,
                        np.nextafter(i / bins, np.inf),
                    )
                ]
            ),
            [np.nan, -np.inf, np.inf],
        ]
    )
    target = np.arange(len(probability), dtype=float)
    actual = reliability_bins(target, probability, bins)
    for i, entry in enumerate(actual):
        # Calculate membership individually, without the bulk-mask/index path.
        members = [
            j
            for j, value in enumerate(probability)
            if value >= i / bins
            and (value < (i + 1) / bins if i < bins - 1 else value <= 1)
        ]
        assert entry == {
            "lower": i / bins,
            "upper": (i + 1) / bins,
            "count": len(members),
            "mean_probability": float(probability[members].mean()) if members else None,
            "positive_fraction": float(target[members].mean()) if members else None,
        }
    assert sum(entry["count"] for entry in actual) == 2 + 3 * (bins - 1)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("bins", [3, 7, 10, 31])
def test_dtype_specific_boundary_comparisons_are_unchanged(dtype, bins):
    edges = np.asarray([i / bins for i in range(bins + 1)], dtype=dtype)
    probability = np.concatenate(
        [np.nextafter(edges, -np.inf), edges, np.nextafter(edges, np.inf)]
    )
    target = np.arange(probability.size, dtype=dtype)
    assert _outcome(reliability_bins, target, probability, bins) == _outcome(
        _original_bins, target, probability, bins
    )
    for value in edges:
        assert _outcome(reliability_bins, 1, value, bins) == _outcome(
            _original_bins, 1, value, bins
        )


@pytest.mark.parametrize(
    "probability_dtype",
    [bool, np.int8, np.uint64, np.float16, np.float32, np.float64, np.longdouble],
)
@pytest.mark.parametrize(
    "target_dtype", [bool, np.int8, np.uint64, np.float16, np.float32, np.float64]
)
@pytest.mark.parametrize("bins", [1, 3, 10, 50])
def test_bins_preserve_dtype_and_numpy_reduction_order(
    probability_dtype, target_dtype, bins
):
    rng = np.random.default_rng(532)
    probability = rng.random((29, 17)).astype(probability_dtype)
    target = rng.integers(0, 2, probability.shape).astype(target_dtype)
    if np.issubdtype(probability_dtype, np.floating):
        probability.flat[:3] = [0, 1, 0.5]
    assert _outcome(reliability_bins, target, probability, bins) == _outcome(
        _original_bins, target, probability, bins
    )


@pytest.mark.parametrize("order", ["C", "F", "slice", "reverse"])
@pytest.mark.parametrize("bins", [3, 10, 50])
def test_bins_preserve_noncontiguous_and_nonbinary_target_values(order, bins):
    rng = np.random.default_rng(105)
    probability = rng.random((60, 40))
    target = rng.normal(size=probability.shape)
    target.flat[::19] = np.nan
    if order == "F":
        target, probability = np.asfortranarray(target), np.asfortranarray(probability)
    elif order == "slice":
        target, probability = target[::2, ::3], probability[::2, ::3]
    elif order == "reverse":
        target, probability = target[::-1, ::-1], probability[::-1, ::-1]
    assert _outcome(reliability_bins, target, probability, bins) == _outcome(
        _original_bins, target, probability, bins
    )


@pytest.mark.parametrize(
    "target,probability,bins",
    [
        ([], [], 10),
        (np.empty((2, 0)), np.empty((2, 0)), 10),
        (1, 0.3, 10),
        ([1], [0.3, 0.7], 10),
        ([[1]], [0.3], 10),
        ([], [np.nan, np.inf, -np.inf, -0.1, 1.1], 10),
        ([0, 1, 1], [0.2, 0.8], 0),
        ([0, 1], [0.2, 0.8], -2),
        ([0, 1], [0.2, 0.8], 2.5),
        ([0, 1], [0.2, 0.8], True),
        ([0, 1], [0.2, 0.8], np.int64(10)),
        (["bad", "target"], [0.2, 0.8], 10),
        ([1, 0], ["bad", "probability"], 10),
        ([np.nan, np.inf, -np.inf], [0.1, 0.1, 0.1], 10),
        (np.array([1 + 1j, 0]), [0.1, 0.8], 10),
        (np.array([1, None], dtype=object), [0.1, 0.8], 10),
    ],
)
def test_direct_inputs_keep_results_errors_and_warning_order(target, probability, bins):
    assert _outcome(reliability_bins, target, probability, bins) == _outcome(
        _original_bins, target, probability, bins
    )
