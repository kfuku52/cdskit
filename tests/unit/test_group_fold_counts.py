"""Check fold assignments against the pre-optimization comparison algorithm."""

import numpy as np
import pytest

from cdskit.localize_splits import stratified_group_ids, valid_partition_id


def _comparison_reference(labels, groups, n_folds, seed):
    """Frozen original algorithm; deliberately do not use hashed label counts."""
    if (
        len(labels) != len(groups)
        or n_folds < 2
        or any(not valid_partition_id(g) for g in groups)
    ):
        raise ValueError(
            "Matching labels and complete groups, and at least two folds, are required."
        )
    classes = sorted(set(labels))
    members = {}
    for i, group in enumerate(groups):
        members.setdefault(str(group), []).append(i)
    if len(members) < 2:
        raise ValueError("At least two independent groups are required for evaluation.")
    n_folds = min(n_folds, len(members))
    counts = np.zeros((n_folds, len(classes)))
    sizes = np.zeros(n_folds)
    rng = np.random.default_rng(seed)
    keys = list(members)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(members[key]), reverse=True)
    result = np.empty(len(labels), dtype=object)
    for key in keys:
        ids = members[key]
        vector = np.asarray([sum(labels[i] == name for i in ids) for name in classes])
        costs = (counts * vector).sum(axis=1)
        fold = min(range(n_folds), key=lambda i: (costs[i], sizes[i]))
        result[ids] = str(fold)
        counts[fold] += vector
        sizes[fold] += len(ids)
    return result


@pytest.mark.parametrize("seed", [0, 1, 17, 20261023])
@pytest.mark.parametrize("n_folds", [2, 3, 9])
def test_group_fold_assignments_match_original_across_group_shapes(seed, n_folds):
    rng = np.random.default_rng(109)
    for rows, n_groups, n_labels in [
        (2, 2, 1),
        (20, 20, 5),
        (120, 3, 20),
        (300, 40, 5),
    ]:
        labels = ["class_{}".format(i) for i in rng.integers(n_labels, size=rows)]
        groups = ["group_{}".format(i) for i in rng.integers(n_groups, size=rows)]
        groups[:2] = ["group_0", "group_1"]
        np.testing.assert_array_equal(
            stratified_group_ids(labels, groups, n_folds, seed),
            _comparison_reference(labels, groups, n_folds, seed),
        )


@pytest.mark.parametrize("n_labels", [1, 2, 3, 4, 5])
def test_group_fold_small_class_count_boundary_preserves_assignments(n_labels):
    labels = ["class_{}".format(i % n_labels) for i in range(100)]
    groups = [str(i // 5) for i in range(len(labels))]
    np.testing.assert_array_equal(
        stratified_group_ids(labels, groups, 5, 17),
        _comparison_reference(labels, groups, 5, 17),
    )


@pytest.mark.parametrize(
    "labels",
    [
        ["SP", "mTP", "noTP", "cTP", "lTP", "cTP"],
        np.array(["SP", "mTP", "noTP", "cTP", "lTP", "cTP"]),
        [np.str_("SP"), "mTP", np.str_("noTP"), "cTP", "lTP", "cTP"],
        [0, 1, 0, 2, 1, 2],
        np.array([0, 1, 0, 2, 1, 2], dtype=np.int32),
        [False, True, 0, 2, 1, 2.0],
        [float("nan"), 1.0, 2.0, float("nan"), 1.0, 2.0],
        [np.float64("nan"), 1, 2, np.float64("nan"), 1, 2],
        [b"SP", b"mTP", b"SP", b"cTP", b"mTP", b"cTP"],
        [(0, 1), (1, 0), (0, 1), (2, 0), (1, 0), (2, 0)],
    ],
)
def test_group_fold_labels_and_stringified_groups_keep_existing_meaning(labels):
    # Numeric IDs collide only through str(), and whitespace is retained.
    groups = [1, "1", np.int64(2), "2", " 2 ", 0]
    np.testing.assert_array_equal(
        stratified_group_ids(labels, groups, 4, 11),
        _comparison_reference(labels, groups, 4, 11),
    )


def test_group_fold_ties_preserve_shuffle_and_rng_consumption():
    labels = ["SP", "mTP", "cTP", "noTP"] * 25
    groups = [str(i // 4) for i in range(len(labels))]
    actual_rng = np.random.default_rng(551)
    reference_rng = np.random.default_rng(551)
    np.testing.assert_array_equal(
        stratified_group_ids(labels, groups, np.int64(7), actual_rng),
        _comparison_reference(labels, groups, np.int64(7), reference_rng),
    )
    np.testing.assert_array_equal(
        actual_rng.integers(1000000, size=100),
        reference_rng.integers(1000000, size=100),
    )


class _StringSubclass(str):
    """String subclasses can supply different equality and must keep comparisons."""

    __hash__ = str.__hash__

    def __eq__(self, other):
        return str(self).lower() == str(other).lower()


def test_group_fold_custom_string_equality_uses_original_comparisons():
    labels = [_StringSubclass("SP"), "sp", _StringSubclass("MT"), "mt"] * 5
    groups = [str(i // 2) for i in range(len(labels))]
    np.testing.assert_array_equal(
        stratified_group_ids(labels, groups, 3, 2),
        _comparison_reference(labels, groups, 3, 2),
    )


@pytest.mark.parametrize(
    "labels,groups,n_folds,seed",
    [
        ([], [], 2, 1),
        (["SP"], ["a"], 2, 1),
        (["SP", "SP"], ["a", "a"], 2, 1),
        (["SP", "mTP"], ["a"], 2, 1),
        (["SP", "mTP"], ["a", "b"], 1, 1),
        (["SP", "mTP", "cTP"], ["a", "b", "c"], 2.5, 1),
        (["SP", "mTP"], ["a", None], 2, 1),
        (["SP", "mTP"], ["a", "nan"], 2, 1),
        (["SP", "mTP"], ["a", np.inf], 2, 1),
        (["SP", "mTP"], ["a", " "], 2, 1),
        (["SP", "mTP"], ["a", "b"], 2, -1),
        (["SP", "mTP"], ["a", "b"], 2, "invalid"),
        (["SP", 0], ["a", "b"], 2, 1),
        ([["SP"], ["mTP"]], ["a", "b"], 2, 1),
    ],
)
def test_group_fold_invalid_inputs_preserve_exception(labels, groups, n_folds, seed):
    try:
        _comparison_reference(labels, groups, n_folds, seed)
    except Exception as reference_error:
        with pytest.raises(type(reference_error)) as actual:
            stratified_group_ids(labels, groups, n_folds, seed)
        assert str(actual.value) == str(reference_error)
    else:
        pytest.fail("Invalid-input reference unexpectedly succeeded")
