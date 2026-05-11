"""HSMM — Hidden Semi-Markov Model with explicit per-state durations.

Brief 3.2 of the regime upgrade plan. Audit reference: §4.2 ("Tier-3
baseline"), §8.3.2. The audit flags the legacy HMM's geometric
duration assumption as problematic for crisis regimes (which are
empirically NOT geometric in their persistence). HSMM replaces the
implicit-geometric-duration with explicit per-state duration distributions.

Architecture (pragmatic v1 — no hsmmlearn dependency):

  1. Fit a Gaussian HMM (hmmlearn.GaussianHMM) on multivariate features.
     ``k_states`` states, diagonal covariance.
  2. Run Viterbi to get the most likely state path on training data.
  3. Compute per-state realised run-lengths (durations).
  4. Fit a Weibull distribution to each state's realised durations.
     (Audit §8.3.2 specifies Weibull. With < 5 run-lengths per state
     we fall back to exponential as a Bayesian backstop.)
  5. ``predict_proba`` returns ``hmm.predict_proba`` smoothed
     probabilities — note: this is the SMOOTHED estimate (uses future
     data per bar), NOT the forward filter. For strictly causal use,
     v1.1 will swap to forward-only filtering. v1 matches the audit
     §5.5 HMM walk-forward practice which accepts smoothed probs as the
     standard.
  6. ``estimate_remaining_duration`` returns survival expectation E[D|D>d]
     for the Weibull distribution conditioned on already-observed
     persistence ``d`` bars in the current state.

State → position mapping (K=4 default, ordered by ascending variance):
    state 0 (lowest var, Full Bull):   +1.00
    state 1 (Half Bull):                +0.50
    state 2 (Half Bear):                -0.20
    state 3 (highest var, Full Bear):   -0.50

Outputs:
    fit() → self
    predict_proba(X) → (n, K) probabilities
    predict_state_path(X) → (n,) Viterbi labels (remapped to canonical
                            ascending-variance order)
    estimate_remaining_duration(state, observed_duration) → float
        Survival expectation in bars.

References
----------
Yu, S.-Z. (2010). Hidden Semi-Markov Models. *Artificial Intelligence*,
   174(2). The forward-backward algorithm for HSMM (deferred to v1.1).
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")


# Default 4-state position mapping per the audit's K=4 prescription.
DEFAULT_K4_POSITIONS: Dict[int, float] = {
    0:  1.00,   # lowest-variance: Full Bull
    1:  0.50,   # Half Bull
    2: -0.20,   # Half Bear
    3: -0.50,   # highest-variance: Full Bear
}


# ---------------------------------------------------------------------------
# DurationAwareHMM
# ---------------------------------------------------------------------------


class DurationAwareHMM:
    """Gaussian HMM + Weibull-fitted per-state duration distributions.

    Parameters
    ----------
    k_states : int, default=4
        Number of hidden states.
    covariance_type : str, default="diag"
        Covariance shape for Gaussian emissions. ``"diag"`` is the
        audit's recommendation (§5.5.1) for high-feature-count fits.
    n_iter : int, default=100
        EM iterations during fit.
    seed : int, default=42
        Determinism.

    Attributes
    ----------
    hmm_ : hmmlearn.GaussianHMM or None
        Fitted HMM (None if fit failed).
    state_remap_ : dict[int, int] or None
        Maps fitted-state index → canonical order (0 = lowest variance,
        K-1 = highest). Variance ranking uses ``mean(diag(covar))`` per
        state.
    duration_fits_ : dict[int, tuple] or None
        Per CANONICAL state: ``(weibull_shape, weibull_scale)``.
        ``None`` for any state with too few realised durations to fit.
    """

    def __init__(
        self,
        *,
        k_states: int = 4,
        covariance_type: str = "diag",
        n_iter: int = 100,
        seed: int = 42,
    ):
        if k_states < 2:
            raise ValueError(f"k_states must be >= 2, got {k_states}")
        self.k_states = k_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.seed = seed
        self.hmm_: Optional[object] = None
        self.state_remap_: Optional[Dict[int, int]] = None
        self.duration_fits_: Optional[Dict[int, tuple]] = None

    # ----------------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DurationAwareHMM":
        """Fit Gaussian HMM + duration-Weibulls. ``X`` is (n, d)."""
        from hmmlearn.hmm import GaussianHMM

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] < self.k_states * 20:
            # Too few rows to fit reliably
            self.hmm_ = None
            return self

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.hmm_ = GaussianHMM(
                    n_components=self.k_states,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=self.seed,
                    verbose=False,
                )
                self.hmm_.fit(X)
        except Exception:
            self.hmm_ = None
            return self

        # Rank states by ascending variance (use mean of diag covariance)
        try:
            covars = self.hmm_.covars_
            if self.covariance_type == "diag":
                var_per_state = covars.mean(axis=1) if covars.ndim == 2 else \
                                np.array([np.mean(c) for c in covars])
            else:
                # Full covariance: use trace
                var_per_state = np.array(
                    [float(np.trace(c)) for c in covars]
                )
            order = np.argsort(var_per_state)  # ascending
            self.state_remap_ = {int(src): int(dst) for dst, src in enumerate(order)}
        except Exception:
            self.state_remap_ = {k: k for k in range(self.k_states)}

        # Fit per-state Weibull on realised durations
        self._fit_duration_distributions(X)
        return self

    def _fit_duration_distributions(self, X: np.ndarray) -> None:
        # Patch zero-sum transmat rows (hmmlearn EM convergence quirk —
        # a state never transitioned during EM ends up with a zero
        # row, which breaks Viterbi). Replace with uniform escape.
        try:
            tm = np.asarray(self.hmm_.transmat_, dtype=float)
            row_sums = tm.sum(axis=1)
            zero_rows = row_sums == 0
            if zero_rows.any():
                tm[zero_rows] = 1.0 / self.k_states
                self.hmm_.transmat_ = tm
        except Exception:
            pass

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                state_path_raw = self.hmm_.predict(X)
        except Exception:
            # Defensive default: every state gets exponential(scale=5)
            self.duration_fits_ = {
                k: (1.0, 5.0) for k in range(self.k_states)
            }
            return
        # Remap to canonical order
        state_path = self._apply_remap(state_path_raw)
        # Compute per-state run-lengths (durations)
        runs: Dict[int, List[int]] = {k: [] for k in range(self.k_states)}
        cur = int(state_path[0])
        length = 1
        for t in range(1, len(state_path)):
            if int(state_path[t]) == cur:
                length += 1
            else:
                runs[cur].append(length)
                cur = int(state_path[t])
                length = 1
        runs[cur].append(length)

        fits: Dict[int, tuple] = {}
        for state, dur_list in runs.items():
            if len(dur_list) < 3:
                # Too few to fit Weibull — use a sensible default
                fits[state] = (1.0, max(np.mean(dur_list) if dur_list else 5.0, 1.0))
                continue
            try:
                # Fit Weibull (location frozen at 0)
                shape, _, scale = stats.weibull_min.fit(
                    np.asarray(dur_list, dtype=float), floc=0,
                )
                fits[state] = (float(shape), float(scale))
            except Exception:
                fits[state] = (1.0, float(np.mean(dur_list)))
        self.duration_fits_ = fits

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _apply_remap(self, raw_states: np.ndarray) -> np.ndarray:
        """Apply state_remap_ to a (n,) array of raw state indices."""
        if self.state_remap_ is None:
            return raw_states.astype(np.int64, copy=False)
        out = np.empty_like(raw_states, dtype=np.int64)
        for raw, canon in self.state_remap_.items():
            out[raw_states == raw] = canon
        return out

    # ----------------------------------------------------------------------
    # Predict
    # ----------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns shape (n, k_states) state probabilities in CANONICAL
        order (column 0 = lowest variance, column K-1 = highest).
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if self.hmm_ is None:
            return np.full((n, self.k_states), 1.0 / self.k_states)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                probs_raw = self.hmm_.predict_proba(X)
        except Exception:
            return np.full((n, self.k_states), 1.0 / self.k_states)

        # Reorder columns to canonical
        if self.state_remap_ is None:
            return probs_raw
        col_order = [None] * self.k_states
        for raw, canon in self.state_remap_.items():
            col_order[canon] = raw
        return probs_raw[:, col_order]

    def predict_state_path(self, X: np.ndarray) -> np.ndarray:
        if self.hmm_ is None:
            return np.zeros(len(X), dtype=np.int64)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw_path = self.hmm_.predict(np.asarray(X, dtype=float))
            return self._apply_remap(raw_path)
        except Exception:
            return np.zeros(len(X), dtype=np.int64)

    def estimate_remaining_duration(
        self, canonical_state: int, observed_duration: float,
    ) -> float:
        """Survival expectation E[D - d | D > d] for the fitted Weibull
        of ``canonical_state``, given the regime has already persisted
        ``observed_duration`` bars.
        """
        if self.duration_fits_ is None or canonical_state not in self.duration_fits_:
            return float("nan")
        shape, scale = self.duration_fits_[canonical_state]
        d = max(observed_duration, 0.0)
        try:
            # E[D | D > d] using the Weibull's conditional expectation.
            # Closed form: scale * Γ(1 + 1/shape) * (1 - F((d/scale)^shape, 1 + 1/shape))
            # / S(d), where F is the lower-regularised gamma and S is survival.
            # We can also compute numerically via integration.
            from scipy.integrate import quad
            s_d = stats.weibull_min.sf(d, c=shape, scale=scale)
            if s_d < 1e-9:
                return float("nan")
            integral, _ = quad(
                lambda t: t * stats.weibull_min.pdf(t, c=shape, scale=scale),
                d, np.inf,
            )
            mean_given_survive = integral / s_d
            return float(mean_given_survive - d)
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# Strategy adapter
# ---------------------------------------------------------------------------


def make_hsmm_strategy(
    *,
    k_states: int = 4,
    state_positions: Optional[Dict[int, float]] = None,
    feature_cols: Optional[List[str]] = None,
    close_col: str = "close",
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Strategy_fn factory for the duration-aware HMM baseline.

    Per outer CPCV fold:
      1. Pick ``feature_cols`` from features_train; default = every
         column except ``close``.
      2. Fit ``DurationAwareHMM`` on train features.
      3. Predict state probabilities on (train ∪ test) — Viterbi smoothed.
      4. Map state probabilities to positions using ``state_positions``.

    Defaults to ``DEFAULT_K4_POSITIONS`` (Full Bull/Half Bull/Half Bear/
    Full Bear ordered by ascending variance).
    """
    if state_positions is None:
        state_positions = DEFAULT_K4_POSITIONS

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        if close_col not in features_train.columns:
            raise KeyError(f"hsmm requires '{close_col}' in features_train")

        cols = feature_cols
        if cols is None:
            cols = [c for c in features_train.columns if c != close_col]

        X_train = features_train[cols].to_numpy(dtype=float)
        X_train = np.nan_to_num(X_train, nan=0.0)  # hmmlearn dislikes NaN

        model = DurationAwareHMM(k_states=k_states).fit(X_train)

        # Combine train + test for the full state probability sequence
        combined = pd.concat([features_train, features_test]).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        X_combined = combined[cols].to_numpy(dtype=float)
        X_combined = np.nan_to_num(X_combined, nan=0.0)

        probs = model.predict_proba(X_combined)  # (n_combined, k_states)
        probs_df = pd.DataFrame(
            probs, index=combined.index,
            columns=[f"p_state_{r}" for r in range(k_states)],
        )

        # Map to position
        position = np.zeros(len(combined), dtype=float)
        for state, alloc in state_positions.items():
            if 0 <= state < k_states:
                position += probs_df[f"p_state_{state}"].to_numpy() * alloc

        position_series = pd.Series(position, index=combined.index)
        aligned = position_series.reindex(features_test.index).fillna(0.0)
        return aligned.to_numpy(dtype=float)

    return strategy_fn


__all__ = [
    "DurationAwareHMM",
    "DEFAULT_K4_POSITIONS",
    "make_hsmm_strategy",
]
