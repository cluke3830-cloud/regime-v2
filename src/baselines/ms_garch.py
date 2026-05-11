"""MS-GARCH baseline (v1: GARCH(1,1) volatility-conditional strategy).

Brief 3.3 of the regime upgrade plan. Audit reference: §4.2 ("Tier-3
baseline"), §8.3.3. Full MS-GARCH (Haas-Mittnik-Paolella 2004 w/ Klaassen
2002 collapsing) requires either custom Markov-switching estimation or
an R bridge (MSGARCH package). v1 ships a pragmatic vol-conditional
GARCH(1,1) baseline that captures the volatility-clustering dynamics
the full MS-GARCH would model.

Architecture:

  1. Fit GARCH(1,1) on the TRAIN returns segment (omega, alpha, beta).
  2. Run the GARCH(1,1) variance recursion FORWARD on the test segment
     using train-fitted parameters — causal, no test-data leakage.
     sigma²[t] = ω + α · r[t-1]² + β · sigma²[t-1]
  3. Convert conditional volatility → position via inverse-vol scaling:
     position[t] = clip(target_ann_vol / cond_vol_ann[t], 0, 1)
     This is the audit §5.10 vol-targeting overlay applied as the
     strategy itself: high vol → de-risk, low vol → full long.

Acceptance gate (audit Brief 3.3):
    OOS volatility-forecast RMSE strictly better than rolling 21-day RV.
    v1 implements the GARCH forecast; gate verification is in the
    ``evaluate_forecast_rmse_vs_rolling`` helper.

References
----------
Engle, R. (1982). Autoregressive Conditional Heteroskedasticity.
   *Econometrica*, 50(4).
Bollerslev, T. (1986). Generalized Autoregressive Conditional
   Heteroskedasticity. *Journal of Econometrics*, 31(3).
Haas, M., Mittnik, S., Paolella, M. (2004). A New Approach to
   Markov-Switching GARCH Models. *J. Financial Econometrics*. (v1.1 spec)
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=Warning, module="arch")


# ---------------------------------------------------------------------------
# GARCH(1,1) volatility model
# ---------------------------------------------------------------------------


class GARCHVolatilityModel:
    """GARCH(1,1) with manual causal recursion on out-of-sample data.

    Fit on train returns to estimate (omega, alpha, beta). Apply the
    forward recursion on test returns to get bar-by-bar conditional
    volatility — STRICTLY CAUSAL (each forecast uses only data through
    the previous bar with frozen train-fit params).

    Parameters
    ----------
    scale_factor : float, default=100.0
        Multiplier applied to returns before fitting. The ``arch``
        package prefers returns in percentage units (~ -3 to +3) for
        numerical stability; raw daily log returns are ~0.01 magnitude.

    Attributes
    ----------
    omega, alpha, beta : float or None
        Fitted GARCH(1,1) parameters. None when fit failed.
    last_sigma2_train : float or None
        Conditional variance at the last train bar. Used as the seed
        for the forward recursion on test data.
    """

    def __init__(self, scale_factor: float = 100.0):
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
        self.scale_factor = scale_factor
        self.omega: Optional[float] = None
        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        self.last_sigma2_train: Optional[float] = None
        self.unconditional_var: Optional[float] = None

    def fit(self, returns: pd.Series) -> "GARCHVolatilityModel":
        """Fit GARCH(1,1) on a return series."""
        from arch import arch_model

        r = returns.dropna()
        if len(r) < 50:
            return self
        r_scaled = r * self.scale_factor

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(
                    r_scaled, vol="Garch", p=1, q=1,
                    mean="Zero", dist="normal", rescale=False,
                )
                result = am.fit(disp="off", show_warning=False)
            params = result.params
            self.omega = float(params["omega"])
            self.alpha = float(params["alpha[1]"])
            self.beta  = float(params["beta[1]"])
            # Unconditional variance = omega / (1 - alpha - beta) if stationary
            persistence = self.alpha + self.beta
            if 0 < persistence < 1:
                self.unconditional_var = self.omega / (1.0 - persistence)
            else:
                self.unconditional_var = float(
                    result.conditional_volatility.iloc[-1] ** 2
                )
            self.last_sigma2_train = float(
                result.conditional_volatility.iloc[-1] ** 2
            )
        except Exception:
            self.omega = self.alpha = self.beta = None
            self.last_sigma2_train = None
            self.unconditional_var = None
        return self

    def predict_volatility(self, returns: pd.Series) -> pd.Series:
        """Forward-recurse the GARCH(1,1) on ``returns`` using fitted
        params; returns conditional vol at each bar in the original
        (un-scaled) units.

        Falls back to a 21-day rolling vol when the model failed to fit.
        """
        r = returns.dropna()
        if len(r) == 0:
            return returns.copy() * 0.0

        if self.omega is None:
            return r.rolling(21, min_periods=5).std().bfill().ffill().reindex(
                returns.index
            )

        r_scaled = r.to_numpy(dtype=float) * self.scale_factor
        n = len(r_scaled)
        sigma2 = np.zeros(n)
        sigma2[0] = (
            self.last_sigma2_train
            if self.last_sigma2_train is not None
            else (self.unconditional_var or self.omega)
        )
        for t in range(1, n):
            sigma2[t] = (
                self.omega
                + self.alpha * r_scaled[t - 1] ** 2
                + self.beta  * sigma2[t - 1]
            )
        sigma = np.sqrt(np.maximum(sigma2, 0.0)) / self.scale_factor
        return pd.Series(sigma, index=r.index).reindex(returns.index).ffill().bfill()


# ---------------------------------------------------------------------------
# Acceptance-gate helper
# ---------------------------------------------------------------------------


def evaluate_forecast_rmse_vs_rolling(
    actual_returns: pd.Series,
    forecast_vol: pd.Series,
    rolling_window: int = 21,
) -> dict:
    """Audit Brief 3.3 gate: GARCH RMSE strictly < rolling-21-day RMSE.

    Compares the GARCH conditional-volatility forecast vs a rolling-21-
    day realised-vol baseline as predictors of NEXT-bar absolute return.

    Returns
    -------
    dict with ``"garch_rmse"``, ``"rolling_rmse"``, ``"passes_gate"``
    (True iff garch_rmse < rolling_rmse).
    """
    # Forecast target = next-bar |return| (a noisy but unbiased proxy)
    target = actual_returns.shift(-1).abs().dropna()
    garch_pred = forecast_vol.reindex(target.index).ffill()
    rolling_pred = (
        actual_returns.rolling(rolling_window).std()
        .shift(1).reindex(target.index).ffill()
    )
    mask = garch_pred.notna() & rolling_pred.notna() & target.notna()
    if mask.sum() < 30:
        return {
            "garch_rmse": float("nan"),
            "rolling_rmse": float("nan"),
            "passes_gate": False,
        }
    garch_rmse = float(np.sqrt(((target[mask] - garch_pred[mask]) ** 2).mean()))
    rolling_rmse = float(np.sqrt(((target[mask] - rolling_pred[mask]) ** 2).mean()))
    return {
        "garch_rmse": garch_rmse,
        "rolling_rmse": rolling_rmse,
        "passes_gate": garch_rmse < rolling_rmse,
    }


# ---------------------------------------------------------------------------
# Strategy adapter
# ---------------------------------------------------------------------------


def make_ms_garch_strategy(
    *,
    target_ann_vol: float = 0.14,
    max_position: float = 1.0,
    min_position: float = 0.0,
    close_col: str = "close",
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Strategy_fn factory for the GARCH vol-conditional baseline.

    Per outer CPCV fold:
      1. Compute log returns from ``features_train[close_col]``.
      2. Fit GARCH(1,1) on train returns.
      3. Run causal forward recursion on (train ∪ test) returns.
      4. Convert conditional vol → position:
         position[t] = clip(target_ann_vol / cond_vol_ann[t],
                            min_position, max_position)
      5. Return positions for ``features_test.index``.

    Always-long (default ``min_position=0``) by design — this is a
    vol-targeting baseline, not a directional bet. Audit §5.10 caps the
    vol-target overlay at ``VOL_SCALE_CAP=1.0`` (max_position=1.0) so the
    strategy never levers above 1×.
    """

    def strategy_fn(
        features_train: pd.DataFrame, features_test: pd.DataFrame
    ) -> np.ndarray:
        if close_col not in features_train.columns:
            raise KeyError(f"ms_garch requires '{close_col}' in features_train")

        # Build the combined close series + log returns
        combined_close = pd.concat(
            [features_train[close_col], features_test[close_col]]
        ).sort_index()
        combined_close = combined_close[
            ~combined_close.index.duplicated(keep="first")
        ]
        combined_returns = np.log(combined_close).diff().dropna()
        train_returns = combined_returns.loc[
            combined_returns.index.isin(features_train.index)
        ]

        # Fit GARCH on train
        model = GARCHVolatilityModel().fit(train_returns)

        # Forward recursion on combined → vol at every bar
        cond_vol = model.predict_volatility(combined_returns)
        cond_vol_ann = cond_vol * np.sqrt(252)
        # Avoid div-by-zero
        cond_vol_ann = cond_vol_ann.where(cond_vol_ann > 1e-9, np.nan)

        # Position = target / forecast vol, clipped
        position = (target_ann_vol / cond_vol_ann).clip(
            lower=min_position, upper=max_position
        ).fillna(0.0)

        aligned = position.reindex(features_test.index).fillna(0.0)
        return aligned.to_numpy(dtype=float)

    return strategy_fn


__all__ = [
    "GARCHVolatilityModel",
    "evaluate_forecast_rmse_vs_rolling",
    "make_ms_garch_strategy",
]
