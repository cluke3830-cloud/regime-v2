"""Combinatorial Purged K-Fold cross-validation with embargo.

Implementation of López de Prado's CPCV (Advances in Financial Machine
Learning, 2018, §7.4) — the gold-standard cross-validator for financial
time-series with non-trivial label horizons.

Why this replaces walk-forward:

    Walk-forward gives a single OOS path. CPCV with N=10, k=2 gives
    C(10,2)=45 OOS paths — a *distribution* of out-of-sample Sharpes
    instead of a point estimate. That distribution is what the Deflated
    Sharpe Ratio and Probability of Backtest Overfitting consume.

Why purge AND embargo:

    Purge handles forward-looking labels: a training sample at time t
    whose label depends on prices through t + h is non-iid with any test
    sample in [t, t+h]. Drop it.

    Embargo handles serial correlation in the *features*: even after the
    label window ends, autocorrelation in prices means a training sample
    immediately after a test block has leakage from the test region.
    Drop a fixed embargo of bars after every test block.

Brief 1.1 of the regime_dashboard upgrade plan. See the audit PDF §8.1.1
for the prescribed signature and acceptance tests.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import BaseCrossValidator


def _contiguous_blocks(sorted_indices: List[int]) -> List[Tuple[int, int]]:
    """Group sorted integer indices into (start, end_inclusive) blocks.

    Used to find each contiguous test region so purge and embargo can be
    applied at block boundaries rather than at every individual test index.
    """
    if not sorted_indices:
        return []
    blocks: List[Tuple[int, int]] = []
    start = sorted_indices[0]
    prev = sorted_indices[0]
    for cur in sorted_indices[1:]:
        if cur != prev + 1:
            blocks.append((start, prev))
            start = cur
        prev = cur
    blocks.append((start, prev))
    return blocks


class CombinatorialPurgedKFold(BaseCrossValidator):
    """CPCV with purge + embargo for financial time-series.

    Splits ``n`` contiguous time-ordered samples into ``n_splits`` groups.
    For each combination of ``n_test_groups`` test groups, yields one
    ``(train_idx, test_idx)`` pair where:

      * Test indices are the union of the chosen test groups.
      * Training indices are everything else, MINUS:
          - any index ``i`` whose label horizon ``[i, i + label_horizons[i]]``
            overlaps any test index (when ``label_horizons`` is given);
          - or ``purge`` bars immediately before each contiguous test block
            (fallback when ``label_horizons`` is None);
          - PLUS an embargo of ``ceil(embargo_pct * n)`` bars immediately
            after each contiguous test block.

    Total paths yielded: ``C(n_splits, n_test_groups)``. Default 10/2 → 45.

    Parameters
    ----------
    n_splits : int, default=10
        Number of contiguous time-ordered groups.
    n_test_groups : int, default=2
        How many groups (out of ``n_splits``) each path uses for testing.
    embargo_pct : float, default=0.01
        Embargo length as fraction of total samples, applied AFTER each test
        block. With 5000 bars this is 50 bars (~2 trading months).
    label_horizons : array-like of int, optional
        Per-sample label horizon. For triple-barrier labels, pass
        ``t1[i] - i`` for each sample. When provided, purge is computed
        per-sample (correct, exact). When None, falls back to the fixed
        ``purge`` integer.
    purge : int, default=10
        Bars purged before each test block when ``label_horizons`` is None.
        For triple-barrier labels prefer passing ``label_horizons``.

    References
    ----------
    López de Prado, M. (2018). *Advances in Financial Machine Learning*.
        Wiley. §7.4 (CPCV), §7.5 (purge & embargo).

    Examples
    --------
    >>> import numpy as np
    >>> from src.validation.cv_purged import CombinatorialPurgedKFold
    >>> X = np.zeros((1000, 5))
    >>> cv = CombinatorialPurgedKFold(n_splits=10, n_test_groups=2,
    ...                                embargo_pct=0.01, purge=5)
    >>> n_paths = sum(1 for _ in cv.split(X))
    >>> assert n_paths == 45
    """

    def __init__(
        self,
        n_splits: int = 10,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        label_horizons: Optional[np.ndarray] = None,
        purge: int = 10,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not 1 <= n_test_groups < n_splits:
            raise ValueError(
                f"n_test_groups must be in [1, n_splits-1], got {n_test_groups}"
            )
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")
        if purge < 0:
            raise ValueError(f"purge must be >= 0, got {purge}")

        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.label_horizons = (
            np.asarray(label_horizons, dtype=np.int64)
            if label_horizons is not None
            else None
        )
        self.purge = int(purge)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Number of CPCV paths = C(n_splits, n_test_groups)."""
        return comb(self.n_splits, self.n_test_groups)

    def _group_boundaries(self, n: int) -> List[Tuple[int, int]]:
        """Return [(start, end_exclusive)] for each of the n_splits groups.

        Any leftover from ``n // n_splits`` integer division is absorbed into
        the final group, so the union of groups is exactly ``range(n)``.
        """
        if n < self.n_splits:
            raise ValueError(
                f"n={n} samples cannot be split into {self.n_splits} groups"
            )
        group_size = n // self.n_splits
        boundaries = [
            (i * group_size, (i + 1) * group_size) for i in range(self.n_splits)
        ]
        last_start, _ = boundaries[-1]
        boundaries[-1] = (last_start, n)
        return boundaries

    def _compute_forbidden(
        self,
        n: int,
        test_idx_set: set,
        sorted_test: List[int],
        embargo: int,
    ) -> set:
        """Indices that must NOT appear in the training set.

        Always includes the test indices themselves. Then adds:
          - per-sample purge from ``label_horizons`` (vectorised), OR
          - fixed-purge fallback before each contiguous test block;
          - embargo bars after each contiguous test block.
        """
        forbidden = set(test_idx_set)
        blocks = _contiguous_blocks(sorted_test)

        if self.label_horizons is not None:
            horizons = self.label_horizons
            if len(horizons) != n:
                raise ValueError(
                    f"label_horizons length {len(horizons)} != n {n}"
                )
            test_mask = np.zeros(n, dtype=np.int64)
            test_mask[sorted_test] = 1
            cum_test = np.concatenate(([0], np.cumsum(test_mask)))
            starts = np.arange(n)
            ends = np.minimum(starts + np.maximum(horizons, 0), n - 1)
            overlap_count = cum_test[ends + 1] - cum_test[starts]
            overlapping = np.where(overlap_count > 0)[0]
            forbidden.update(int(i) for i in overlapping)
        else:
            for start, _ in blocks:
                for x in range(max(0, start - self.purge), start):
                    forbidden.add(x)

        for _, end in blocks:
            for x in range(end + 1, min(n, end + 1 + embargo)):
                forbidden.add(x)

        return forbidden

    def split(
        self, X, y=None, groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for each of the C(n_splits, k_test) paths.

        Parameters
        ----------
        X : array-like
            Used only for its length. The CV does not look at feature values.
        y : ignored
        groups : ignored
        """
        n = len(X)
        boundaries = self._group_boundaries(n)
        embargo = int(np.ceil(self.embargo_pct * n))

        for test_groups in combinations(range(self.n_splits), self.n_test_groups):
            test_idx_list: List[int] = []
            for g in test_groups:
                t0, t1 = boundaries[g]
                test_idx_list.extend(range(t0, t1))
            test_idx_list.sort()
            test_idx_set = set(test_idx_list)

            forbidden = self._compute_forbidden(
                n, test_idx_set, test_idx_list, embargo
            )
            train_idx = np.fromiter(
                (i for i in range(n) if i not in forbidden),
                dtype=np.int64,
            )
            test_idx = np.array(test_idx_list, dtype=np.int64)
            yield train_idx, test_idx


__all__ = ["CombinatorialPurgedKFold"]
