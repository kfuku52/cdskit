"""Compact alignment scoring preserves weighted columns and exact ties."""

import random

import numpy as np
import pytest

from cdskit import maxalign as mx


def _case(variables=10):
    return dict(
        subset_start=0,
        subset_end=1 << variables,
        subset_variable_masks=mx.build_variable_subset_masks(list(range(variables))),
        required_mask=0,
        required_count=0,
        support_counts={(1 << variables) - 1: 5},
        max_removed=None,
        total_sequences=variables,
        num_sequences=variables,
    )


def _brute(kwargs):
    best = None
    for subset in range(kwargs["subset_start"], kwargs["subset_end"]):
        kept = kwargs["required_count"] + subset.bit_count()
        if not kept or (
            kwargs["max_removed"] is not None
            and kwargs["total_sequences"] - kept > kwargs["max_removed"]
        ):
            continue
        mask = kwargs["required_mask"] | kwargs["subset_variable_masks"][subset]
        columns = (
            int(kwargs["complete_counts"][subset])
            if kwargs.get("complete_counts") is not None
            else sum(
                value
                for support, value in kwargs["support_counts"].items()
                if support & mask == mask
            )
        )
        indices = [i for i in range(kwargs["num_sequences"]) if mask & (1 << i)]
        candidate = dict(
            area=kept * columns,
            num_kept=kept,
            complete_codon_columns=columns,
            kept_indices=indices,
            steps=[],
        )
        score = (candidate["area"], kept)
        if (
            best is None
            or score > (best["area"], best["num_kept"])
            or (
                score == (best["area"], best["num_kept"])
                and indices < best["kept_indices"]
            )
        ):
            best = candidate
    return best


@pytest.mark.parametrize("maximum_removed", [None, 0, 7, 18])
def test_blocked_partial_range_matches_independent_brute_force(
    monkeypatch, maximum_removed
):
    rng = random.Random(7301)
    variable = [0, 2, 4, 5, 7, 8, 10, 12, 13, 15, 17]
    kwargs = _case(11)
    kwargs.update(
        subset_start=13,
        subset_end=2031,
        subset_variable_masks=mx.build_variable_subset_masks(variable),
        required_mask=(1 << 3) | (1 << 14),
        required_count=2,
        support_counts={rng.getrandbits(18): rng.randrange(1, 8) for _ in range(40)},
        max_removed=maximum_removed,
        total_sequences=18,
        num_sequences=18,
    )
    kwargs["support_counts"][(1 << 18) - 1] = 7
    expected = _brute(kwargs)
    monkeypatch.setattr(mx, "_EXACT_SUBSET_BLOCK_SIZE", 37)
    assert mx.evaluate_exact_subset_range(**kwargs) == expected


def test_global_tie_uses_indices_lexicographically_across_blocks(monkeypatch):
    kwargs = _case()
    counts = np.zeros(1024, dtype=np.int64)
    counts[6] = counts[9] = 10
    kwargs["complete_counts"] = counts
    monkeypatch.setattr(mx, "_EXACT_SUBSET_BLOCK_SIZE", 8)
    result = mx.evaluate_exact_subset_range(**kwargs)
    assert result == _brute(kwargs)
    assert result["kept_indices"] == [0, 3]  # mask 9 beats mask 6's [1, 2].


def test_no_solution_when_candidates_cannot_meet_removal_limit():
    kwargs = _case()
    kwargs.update(total_sequences=11, max_removed=0)
    assert mx.evaluate_exact_subset_range(**kwargs) is None


def test_vector_candidate_arrays_are_bounded_by_block_size(monkeypatch):
    kwargs = _case(17)
    original = np.arange
    lengths = []

    def bounded_arange(start, end, **options):
        lengths.append(end - start)
        return original(start, end, **options)

    monkeypatch.setattr(mx.np, "arange", bounded_arange)
    result = mx.evaluate_exact_subset_range(**kwargs)
    assert result["num_kept"] == 17
    assert len(lengths) > 1
    assert max(lengths) <= mx._EXACT_SUBSET_BLOCK_SIZE


def test_small_and_empty_ranges_preserve_scalar_behavior():
    kwargs = _case(6)
    assert mx.evaluate_exact_subset_range(**kwargs) == _brute(kwargs)
    kwargs["subset_end"] = 0
    assert mx.evaluate_exact_subset_range(**kwargs) is None


def test_compact_exact_matches_brute_force_with_sparse_sequence_indices():
    rng = random.Random(316)
    variable = list(range(0, 20, 2))
    required = [5, 65]
    required_mask = mx.indices_to_bitmask(required)
    support = {required_mask | rng.getrandbits(21): 180 for _ in range(48)}
    kwargs = dict(
        subset_start=0,
        subset_end=1 << len(variable),
        subset_variable_masks=mx.build_variable_subset_masks(variable),
        required_mask=required_mask,
        required_count=len(required),
        support_counts=support,
        max_removed=6,
        total_sequences=12,
        num_sequences=66,
    )
    assert mx.solve_exact(
        None,
        candidate_indices=variable + required,
        required_indices=required,
        max_removed=kwargs["max_removed"],
        total_sequences=kwargs["total_sequences"],
        support_counts=support,
        num_sequences=kwargs["num_sequences"],
    ) == _brute(kwargs)


def test_greedy_index_matches_direct_weighted_counts_across_row_groups():
    rng = random.Random(722)
    active = mx.indices_to_bitmask([0, 1, 6, 7, 10, 14, 22, 64, 67, 72])
    support = {
        rng.getrandbits(73): weight for weight in (0, 1, 2, 19, 2**32, 2**63 + 1) * 20
    }
    count = mx._indexed_column_counter(support, active)
    assert count is not None
    for kept in [0, active, *[rng.getrandbits(73) & active for _ in range(256)]]:
        assert count(kept) == mx.count_complete_columns_with_support(support, kept)
