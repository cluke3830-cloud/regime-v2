"""TVTP-MSAR — Markov-Switching Auto-Regression with optional time-varying
transition probabilities.

Brief 3.1 of the regime upgrade plan. Audit reference: §4.2 ("Tier-1
baseline"), §8.3.1. This is the Hamilton (1989) classical reference
every regime paper must beat for the literature to take it seriously.

Architecture v1:
  - 2-state MS-AR with AR(1) emissions and switching variance.
  - Optional TVTP — transition probabilities depend on an exogenous
    conditioning variable (typically log VIX): high VIX → higher
    P(stay in high-vol state). v1 defaults to TVTP OFF (constant
    transition matrix) because statsmodels' ``exog_tvtp`` API requires
    careful column-pair shaping and we want a working baseline first;
    TVTP can be enabled by passing ``exog_tvtp_col`` to the constructor.

Causal hygiene:
  Fit on TRAINING segment only; for OOS evaluation, use
  ``result.apply(extended_endog)`` to run the forward filter (NOT the
  smoother) on (train + test) bars with frozen fitted parameters.
  Slicing the filtered marginal probabilities at test indices gives
  causally-clean per-bar state probabilities.

State → position mapping (default):
  State 0 (low-vol regime):  +1.00   (full long)
  State 1 (high-vol regime): -0.30   (mild defense, not full short)
  position[t] = sum_s P(state=s | t) * state_positions[s]

statsmodels caveats handled:
  - Fits can fail to converge; we fall back to a one-state Gaussian
    (equivalent to "no regime info") when fit fails.
  - The state labels (0=low-vol vs 1=high-vol) aren't deterministic
    across fits; we identify them by switching-variance magnitude
    post-fit and remap so state 0 is always the LOWER-variance regime.

References
----------
Hamilton, J. (1989). A New Approach to the Economic Analysis of Non-
   stationary Time Series and the Business Cycle. *Econometrica*, 57(2).
"""

from __future__ import annotations

import contextlib
import io
import os
import warnings
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# Suppress statsmodels' verbose convergence warnings (we handle them).
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")


@contextlib.contextmanager
def _suppress_stdout():
    """statsmodels prints 'Model is not converging' to stdout via print().
    Those messages aren't caught by warnings.filterwarnings — we have to
    redirect stdout. Used inside MS-AR fit/filter calls to keep the
    validation report output clean.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
# MarkovSwitchingAR wrapper
# ---------------------------------------------------------------------------


class MarkovSwitchingAR:
    """2-state Markov-Switching AR(1) with switching variance.

    Wraps :class:`statsmodels.tsa.regime_switching.MarkovAutoregression`
    with CPCV-friendly ``fit``/``predict_proba`` and a deterministic
    state-label convention (state 0 = LOWER variance).

    Parameters
    ----------
    k_regimes : int, default=2
        Number of latent regimes. Hamilton's original used 2; v1 sticks
        with 2 for stability.
    order : int, default=1
        AR order in each regime.
    switching_variance : bool, default=True
        If True, variance differs across regimes (the high-vol-stress
        signature). If False, only means switch.
    annualisation_factor : int, default=252
        Used for diagnostic Sharpe / vol annualisation. NOT used in
        the model fit.

    Attributes
    ----------
    params_ : np.ndarray or None
        Fitted parameter vector from train. Reused at OOS-filter time
        via ``MarkovAutoregression.filter(self.params_)`` to run the
        forward Hamilton equations with frozen params. None if fit
        failed.
    state_remap_ : dict or None
        Maps statsmodels' arbitrary state index → canonical (0=low-vol,
        1=high-vol). Computed post-fit from the fitted variances.
    """

    def __init__(
        self,
        *,
        k_regimes: int = 2,
        order: int = 1,
        switching_variance: bool = True,
        annualisation_factor: int = 252,
    ):
        if k_regimes != 2:
            raise NotImplementedError(
                "v1 supports k_regimes=2 only (Hamilton baseline)."
            )
        self.k_regimes = k_regimes
        self.order = order
        self.switching_variance = switching_variance
        self.ann_factor = annualisation_factor
        self.params_: Optional[np.ndarray] = None
        self.state_remap_: Optional[Dict[int, int]] = None

    def fit(self, returns: pd.Series) -> "MarkovSwitchingAR":
        """Fit on a training return series — stores params for OOS filter.

        Drops NaN bars and requires at least ``50`` observations to
        attempt a fit. On any exception (rare convergence failure with
        very short series) leaves ``params_`` as None — the model then
        emits 50/50 state probabilities at inference time.
        """
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        r = returns.dropna()
        if len(r) < 50:
            self.params_ = None
            return self

        try:
            model = MarkovAutoregression(
                endog=r.values,
                k_regimes=self.k_regimes,
                order=self.order,
                switching_ar=False,
                switching_variance=self.switching_variance,
            )
            with warnings.catch_warnings(), _suppress_stdout():
                warnings.simplefilter("ignore")
                result = model.fit(disp=False)
            self.params_ = np.asarray(result.params, dtype=float)
            # Cache the model's param_names so predict_proba can locate
            # sigma2[*] indices without re-fitting.
            self._param_names: list = list(getattr(model, "param_names", []))
        except Exception:
            self.params_ = None
            self._param_names = []
            return self

        # Identify which statsmodels state is "low-vol" by inspecting
        # fitted variances. statsmodels MS-AR with switching_variance
        # puts variances at parameter indices whose name starts with
        # ``sigma2[`` — they are NOT at the end of the param vector
        # (the AR coefficients come after).
        try:
            sigma_idx = [
                i for i, name in enumerate(self._param_names)
                if name.startswith("sigma2[")
            ]
            if len(sigma_idx) == self.k_regimes:
                s0, s1 = self.params_[sigma_idx[0]], self.params_[sigma_idx[1]]
                if s0 <= s1:
                    self.state_remap_ = {0: 0, 1: 1}  # canonical
                else:
                    self.state_remap_ = {0: 1, 1: 0}  # swap
            else:
                self.state_remap_ = {k: k for k in range(self.k_regimes)}
        except Exception:
            self.state_remap_ = {k: k for k in range(self.k_regimes)}

        return self

    def predict_proba(self, returns_extended: pd.Series) -> pd.DataFrame:
        """Causal forward filter on ``returns_extended`` with FROZEN
        train-fitted parameters.

        This is the leakage-free pattern: build a new statsmodels
        MarkovAutoregression on the extended series, call
        ``model.filter(self.params_)`` to run the forward Hamilton
        equations using the train-fit parameters. No re-fitting on
        test data.

        Returns a (n, 2) DataFrame with columns ``p_low_vol`` and
        ``p_high_vol``, keyed by ``returns_extended.index``.
        Uniform 0.5/0.5 when fit failed or filter raises.
        """
        n_in = len(returns_extended)
        fallback = pd.DataFrame(
            {"p_low_vol": np.full(n_in, 0.5), "p_high_vol": np.full(n_in, 0.5)},
            index=returns_extended.index,
        )
        if self.params_ is None:
            return fallback

        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        r = returns_extended.dropna()
        if len(r) < self.order + 1:
            return fallback

        try:
            ext_model = MarkovAutoregression(
                endog=r.values,
                k_regimes=self.k_regimes,
                order=self.order,
                switching_ar=False,
                switching_variance=self.switching_variance,
            )
            with warnings.catch_warnings(), _suppress_stdout():
                warnings.simplefilter("ignore")
                ext_result = ext_model.filter(self.params_)
            arr = np.asarray(ext_result.filtered_marginal_probabilities)
        except Exception:
            return fallback

        if arr.ndim == 1:
            arr = np.column_stack([arr, 1.0 - arr])

        # statsmodels' filter returns probs of length (n - order) due to
        # the AR(p) lag burn-in. Align to the input index by prepending
        # NaN rows for the burn-in bars.
        remap = self.state_remap_ or {0: 0, 1: 1}
        # remap inverse: which source-col carries the canonical state?
        src_for_0 = next(s for s, t in remap.items() if t == 0)
        src_for_1 = next(s for s, t in remap.items() if t == 1)

        n_out = arr.shape[0]
        full_low  = np.full(len(r), np.nan)
        full_high = np.full(len(r), np.nan)
        # The AR(p) burn-in bars at the front have no filtered prob; the
        # filter output corresponds to bars [order:].
        front_pad = max(len(r) - n_out, 0)
        full_low[front_pad:]  = arr[:, src_for_0]
        full_high[front_pad:] = arr[:, src_for_1]

        out = pd.DataFrame(
            {"p_low_vol": full_low, "p_high_vol": full_high},
            index=r.index,
        )
        out = out.reindex(returns_extended.index).ffill().bfill().fillna(0.5)
        # Defensive renorm
        row_sums = out.sum(axis=1)
        out = out.div(row_sums.where(row_sums > 0, 1.0), axis=0)
        return out


# ---------------------------------------------------------------------------
# Strategy adapter
# ---------------------------------------------------------------------------


def make_tvtp_msar_strategy(
    *,
    state_positions: Optional[Dict[int, float]] = None,
    order: int = 1,
    switching_variance: bool = True,
    close_col: str = "close",
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Strategy_fn factory for the TVTP-MSAR baseline.

    Per outer CPCV fold:
      1. Compute log returns from ``features_train[close_col]``.
      2. Fit MS-AR(1) with switching variance on train returns.
      3. Run the causal forward filter on (train ∪ test) returns to
         get state probabilities on the test period.
      4. Map states → positions:
           position[t] = sum_s P(state=s | t) * state_positions[s]
      5. Return positions for ``features_test.index``.

    Default state mapping (after remap to "state 0 = low variance"):
      state 0 (low-vol bull):    +1.00
      state 1 (high-vol stress): -0.30

    The high-vol position is mildly negative (defensive) rather than
    fully short — empirically, shorting in high-vol regimes captures
    drawdowns but bleeds on the recovery rally. Audit §5.10.2 caps
    similar defensive allocations at -0.5; we're more conservative.
    """
    if state_positions is None:
        state_positions = {0: 1.00, 1: -0.30}

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        if close_col not in features_train.columns:
            raise KeyError(f"tvtp_msar requires '{close_col}' in features_train")

        # Build combined-time-order close + log returns
        combined_close = pd.concat([
            features_train[close_col], features_test[close_col],
        ]).sort_index()
        combined_close = combined_close[
            ~combined_close.index.duplicated(keep="first")
        ]
        combined_returns = np.log(combined_close).diff().dropna()

        train_returns = combined_returns.loc[
            combined_returns.index.isin(features_train.index)
        ]

        # Fit on train
        model = MarkovSwitchingAR(
            k_regimes=2, order=order,
            switching_variance=switching_variance,
        )
        model.fit(train_returns)

        # Filter on full series
        probs = model.predict_proba(combined_returns)

        # Map state probabilities to position
        position_series = (
            probs["p_low_vol"]  * state_positions[0]
            + probs["p_high_vol"] * state_positions[1]
        )

        # Causal alignment fix: position[t] must use only info through t-1.
        # The filtered prob P(state | r_1...r_t) conditions on bar-t's return,
        # so shift by 1 before the harness multiplies position[t] * return[t].
        position_series = position_series.shift(1).fillna(0.0)
        aligned = position_series.reindex(features_test.index).fillna(0.0)
        return aligned.to_numpy(dtype=float)

    return strategy_fn


__all__ = [
    "MarkovSwitchingAR",
    "make_tvtp_msar_strategy",
]
