"""Acceptance tests for src.validation.cv_purged.CombinatorialPurgedKFold.

Three primary acceptance tests from audit Brief 1.1:
  (a) every train/test pair is disjoint;
  (b) no train index t satisfies test_idx_min <= t + label_horizons[t] <= test_idx_max;
  (c) the number of paths returned equals comb(n_splits, n_test_groups).

Plus extras that catch regressions a reviewer would notice:
  - embargo correctly drops bars after every contiguous test block;
  - groups partition exactly into [0, n);
  - constructor argument validation;
  - vectorised label_horizons path matches a brute-force ground truth.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.cv_purged import (  # noqa: E402
    CombinatorialPurgedKFold,
    _contiguous_blocks,
)


# ---------------------------------------------------------------------------
# (a) Disjointness
# ---------------------------------------------------------------------------


def test_train_test_disjoint_default():
    """Acceptance (a): every train/test pair must be disjoint."""
    n = 1000
    cv = CombinatorialPurgedKFold(n_splits=10, n_test_groups=2, purge=5)
    X = np.zeros((n, 3))
    n_paths = 0
    for train_idx, test_idx in cv.split(X):
        n_paths += 1
        assert set(train_idx).isdisjoint(set(test_idx)), (
            f"train and test overlap on path {n_paths}"
        )
    assert n_paths == comb(10, 2)


def test_train_test_disjoint_with_label_horizons():
    """Disjointness must also hold when per-sample label horizons are used."""
    n = 600
    horizons = np.full(n, 10, dtype=np.int64)
    cv = CombinatorialPurgedKFold(
        n_splits=10, n_test_groups=2, embargo_pct=0.02,
        label_horizons=horizons,
    )
    X = np.zeros((n, 2))
    for train_idx, test_idx in cv.split(X):
        assert set(train_idx).isdisjoint(set(test_idx))


# ---------------------------------------------------------------------------
# (b) No leakage from forward-looking label horizon
# ---------------------------------------------------------------------------


def test_no_label_horizon_leakage():
    """Acceptance (b): for any train index t with horizon h_t, the closed
    interval [t, t + h_t] must not intersect the test set on that path.
    """
    n = 800
    rng = np.random.default_rng(42)
    horizons = rng.integers(low=5, high=15, size=n)
    cv = CombinatorialPurgedKFold(
        n_splits=8, n_test_groups=2, embargo_pct=0.0,
        label_horizons=horizons,
    )
    X = np.zeros((n, 2))

    leaks = []
    for path_id, (train_idx, test_idx) in enumerate(cv.split(X)):
        test_set = set(test_idx.tolist())
        for t in train_idx:
            end = min(t + int(horizons[t]), n - 1)
            for j in range(t, end + 1):
                if j in test_set:
                    leaks.append((path_id, int(t), j))
                    break
    assert not leaks, f"Found {len(leaks)} label-horizon leaks: {leaks[:3]}"


# ---------------------------------------------------------------------------
# (c) Path count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_splits,k_test,expected",
    [(10, 2, 45), (8, 2, 28), (6, 1, 6), (12, 3, 220)],
)
def test_n_paths_equals_combinatorial(n_splits, k_test, expected):
    """Acceptance (c): C(n_splits, n_test_groups) paths exactly."""
    cv = CombinatorialPurgedKFold(n_splits=n_splits, n_test_groups=k_test)
    X = np.zeros((n_splits * 50, 2))
    paths = list(cv.split(X))
    assert len(paths) == expected
    assert cv.get_n_splits(X) == expected


# ---------------------------------------------------------------------------
# Embargo behaviour
# ---------------------------------------------------------------------------


def test_embargo_drops_bars_after_each_test_block():
    """Embargo of E bars must remove [end+1, end+E] from the train set
    after every contiguous test block on every path.
    """
    n = 1000
    embargo_pct = 0.02
    cv = CombinatorialPurgedKFold(
        n_splits=10, n_test_groups=2, embargo_pct=embargo_pct, purge=0,
    )
    X = np.zeros((n, 1))
    embargo = int(np.ceil(embargo_pct * n))

    for train_idx, test_idx in cv.split(X):
        train_set = set(train_idx.tolist())
        blocks = _contiguous_blocks(sorted(test_idx.tolist()))
        for _, end in blocks:
            for offset in range(1, embargo + 1):
                bar = end + offset
                if bar < n:
                    assert bar not in train_set, (
                        f"embargo bar {bar} (offset {offset} after test "
                        f"block ending at {end}) leaked into train set"
                    )


def test_groups_partition_full_range():
    """Union of test sets across all paths must cover [0, n)."""
    n = 500
    cv = CombinatorialPurgedKFold(n_splits=10, n_test_groups=2, purge=0,
                                   embargo_pct=0.0)
    X = np.zeros((n, 1))
    seen = set()
    for _, test_idx in cv.split(X):
        seen.update(test_idx.tolist())
    assert seen == set(range(n))


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,err_substring",
    [
        ({"n_splits": 1}, "n_splits"),
        ({"n_splits": 5, "n_test_groups": 5}, "n_test_groups"),
        ({"n_splits": 5, "n_test_groups": 0}, "n_test_groups"),
        ({"embargo_pct": 1.0}, "embargo_pct"),
        ({"embargo_pct": -0.1}, "embargo_pct"),
        ({"purge": -1}, "purge"),
    ],
)
def test_invalid_constructor_args_raise(kwargs, err_substring):
    with pytest.raises(ValueError, match=err_substring):
        CombinatorialPurgedKFold(**kwargs)


# ---------------------------------------------------------------------------
# Vectorised path correctness vs brute force
# ---------------------------------------------------------------------------


def test_uniform_horizons_purge_correctness():
    """With uniform horizon h, the vectorised purge must drop exactly those
    train indices i where some test index lies in [i, i+h]. Verified by
    brute force on a small problem.
    """
    n = 100
    h = 7
    horizons = np.full(n, h, dtype=np.int64)
    cv = CombinatorialPurgedKFold(
        n_splits=5, n_test_groups=2, embargo_pct=0.0,
        label_horizons=horizons,
    )
    X = np.zeros((n, 1))

    for train_idx, test_idx in cv.split(X):
        test_set = set(test_idx.tolist())
        train_set = set(train_idx.tolist())
        forbidden_ground_truth = set(test_idx.tolist())
        for i in range(n):
            for j in range(i, min(i + h, n - 1) + 1):
                if j in test_set:
                    forbidden_ground_truth.add(i)
                    break
        expected_train = set(range(n)) - forbidden_ground_truth
        assert train_set == expected_train


# ---------------------------------------------------------------------------
# Helper sanity
# ---------------------------------------------------------------------------


def test_contiguous_blocks_helper():
    assert _contiguous_blocks([]) == []
    assert _contiguous_blocks([5]) == [(5, 5)]
    assert _contiguous_blocks([0, 1, 2, 5, 6, 9]) == [(0, 2), (5, 6), (9, 9)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
