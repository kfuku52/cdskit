"""Independent coordinate oracles for shared endpoint edit preparation."""

import random

import numpy as np
import pytest

from cdskit.gapjust import vectorized_coordinate_update


def _surviving_base_coordinates(length, edits):
    """Track base identities through sequence slices, independent of prefix sums."""
    bases = list(range(1, length + 1))
    for edit in sorted(edits, key=lambda e: e["original_gap_start"], reverse=True):
        start = edit["original_gap_start"]
        old_length = edit["original_gap_length"]
        target = edit["target_gap_length"]
        bases[start : start + old_length] = bases[
            start : start + min(old_length, target)
        ] + [None] * max(0, target - old_length)
    return {old: new for new, old in enumerate(bases, 1) if old is not None}


def _replay_legacy_coordinates(coords, edits):
    """Replay old point shifts in edited coordinates, retaining site tie order."""
    values = np.asarray(coords).copy()
    offset = 0
    for original_start, delta in sorted(edits):
        actual_start = original_start + 1 + offset
        for index in np.ndindex(values.shape):
            if values[index] > actual_start:
                values[index] += delta
        offset += delta
    return values


def test_range_endpoints_match_base_identity_oracle():
    rng = random.Random(56401)
    for _ in range(100):
        edits = [
            {
                "original_gap_start": 10 * i + rng.randrange(3),
                "original_gap_length": rng.randrange(1, 7),
                "target_gap_length": rng.randrange(9),
            }
            for i in range(6)
        ]
        expected = _surviving_base_coordinates(65, edits)
        coords = np.array(sorted(expected))
        rng.shuffle(edits)
        coords.flags.writeable = False
        starts, ends = vectorized_coordinate_update(coords, coords[::-1], edits)
        np.testing.assert_array_equal(starts, [expected[old] for old in coords])
        np.testing.assert_array_equal(ends, [expected[old] for old in coords[::-1]])
        deleted = sorted(set(range(1, 66)) - expected.keys())
        if deleted:
            with pytest.raises(ValueError, match="Deleted feature endpoint"):
                vectorized_coordinate_update(coords[:1], np.array(deleted), edits)


@pytest.mark.parametrize("as_dict", [False, True])
def test_legacy_endpoints_match_sequential_oracle(as_dict):
    rng = random.Random(70713)
    cases = [[(0, 1), (0, -1)], [(10, -15), (20, 5)]]
    cases.extend(
        [(rng.randrange(-10, 40), rng.randrange(-15, 16)) for _ in range(6)]
        for _ in range(100)
    )
    original = np.array([[-10, 0, 1, 8], [12, 30, 45, 80]])
    original.flags.writeable = False
    for edits in cases:
        formatted = (
            [
                {"original_edit_start": start, "edit_length": delta}
                for start, delta in edits
            ]
            if as_dict
            else edits
        )
        starts, ends = vectorized_coordinate_update(original, original + 1, formatted)
        np.testing.assert_array_equal(
            starts, _replay_legacy_coordinates(original, edits)
        )
        np.testing.assert_array_equal(
            ends, _replay_legacy_coordinates(original + 1, edits)
        )


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int32, np.int64, np.uint64, np.float32, np.float64, object]
)
@pytest.mark.parametrize("shape", [(0,), (2, 0), (4,), (2, 2), ()])
@pytest.mark.parametrize("range_aware", [False, True])
def test_coordinate_dtype_shape_and_empty_arrays(dtype, shape, range_aware):
    original = np.array(
        12 if not shape else [2, 3, 12, 20][: np.prod(shape)], dtype=dtype
    ).reshape(shape)
    edits = (
        [{"original_gap_start": 4, "original_gap_length": 3, "target_gap_length": 5}]
        if range_aware
        else [(4, 2)]
    )
    starts, ends = vectorized_coordinate_update(original, original, edits)
    expected = np.asarray(
        original + (original > 7) * 2, dtype=np.result_type(dtype, np.int64)
    )
    for updated in (starts, ends):
        if dtype is object and not shape:
            assert isinstance(updated, int)
            assert updated == 14
            continue
        assert updated.shape == shape
        assert updated.dtype == expected.dtype
        np.testing.assert_array_equal(updated, expected)


def test_no_edits_preserves_input_identity():
    starts, ends = [1, 2], np.array([[3, 4]])
    new_starts, new_ends = vectorized_coordinate_update(starts, ends, [])
    assert new_starts is starts
    assert new_ends is ends


@pytest.mark.parametrize("in_end", [False, True])
def test_deleted_endpoint_raises_in_either_coordinate_array(in_end):
    safe, deleted = np.array([2, 5, 12]), np.array([7])
    starts, ends = (safe, deleted) if in_end else (deleted, safe)
    edits = [
        {"original_gap_start": 3, "original_gap_length": 5, "target_gap_length": 2}
    ]
    with pytest.raises(ValueError, match="Deleted feature endpoint"):
        vectorized_coordinate_update(starts, ends, edits)
    np.testing.assert_array_equal(safe, [2, 5, 12])
    np.testing.assert_array_equal(deleted, [7])
