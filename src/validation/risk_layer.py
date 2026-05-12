"""Portfolio risk layer — drawdown circuit-breaker + VaR gate.

Two causal (no lookahead) controls applied to raw strategy positions
inside the CPCV per-path loop and in the live regime classifier:

1. **Drawdown circuit-breaker**
   Track the equity curve bar-by-bar. When the rolling drawdown exceeds
   ``dd_limit`` (default 15%), zero all positions (circuit open). Re-enter
   when the drawdown has recovered above ``dd_reentry`` (default 7.5%).

2. **VaR gate**
   Estimate the 1-day 95% historical VaR from the last ``var_window``
   bars of realised returns. Scale the position so that the expected
   loss at the VaR confidence level does not exceed ``var_nav_pct``
   (default 2%) of NAV. Scaling only reduces positions — never levers up.

Both controls are applied bar-by-bar in a single forward pass. The caller
supplies ``init_returns`` (the in-sample returns on the same CPCV path)
so the VaR window is warm from bar 0 of the test period.

Usage
-----
    from src.validation.risk_layer import RiskControls
    rc = RiskControls(dd_limit=0.15, var_nav_pct=0.02)
    safe_positions = rc.apply_risk_controls(raw_pos, test_returns, init_returns=is_returns)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RiskControls:
    """Causal portfolio risk filter.

    Parameters
    ----------
    dd_limit : float
        Drawdown threshold that triggers the circuit-breaker (default 0.15 = 15%).
    dd_reentry : float
        Drawdown level at which positions are allowed back in after a
        circuit-open event (default 0.075 = 7.5% — half the limit).
    var_conf : float
        VaR confidence level (default 0.95 = 95th percentile left tail).
    var_window : int
        Rolling lookback for historical VaR estimation in bars (default 60).
    var_nav_pct : float
        Maximum allowed 1-day VaR as a fraction of NAV (default 0.02 = 2%).

    Examples
    --------
    >>> import numpy as np
    >>> rc = RiskControls(dd_limit=0.10)
    >>> pos = np.ones(200)
    >>> ret = np.full(200, -0.002)   # constant 0.2%/day loss → 20%+ DD
    >>> filtered = rc.apply_risk_controls(pos, ret)
    >>> # Positions go to 0 once 10% DD is breached
    >>> assert filtered[150:].sum() == 0.0
    """

    dd_limit: float = 0.15
    dd_reentry: float = 0.075
    var_conf: float = 0.95
    var_window: int = 60
    var_nav_pct: float = 0.02

    def apply_risk_controls(
        self,
        positions: np.ndarray,
        bar_returns: np.ndarray,
        init_returns: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply DD circuit-breaker and VaR gate to *positions*.

        Parameters
        ----------
        positions : np.ndarray, shape (n,)
            Raw strategy positions from the regime classifier.
        bar_returns : np.ndarray, shape (n,)
            OOS bar returns for this CPCV path (used to track equity and
            warm the VaR window after ``init_returns``).
        init_returns : np.ndarray, optional
            In-sample returns preceding this test path. Prepended to
            ``bar_returns`` to pre-fill the VaR rolling window so estimates
            are not cold at the start of each OOS path. When None, the
            window is filled progressively from bar 0.

        Returns
        -------
        np.ndarray, shape (n,)
            Modified positions (same shape as input). Values can only
            decrease in absolute magnitude relative to *positions*.
        """
        positions = np.asarray(positions, dtype=float)
        bar_returns = np.asarray(bar_returns, dtype=float)
        n = len(positions)

        if n == 0:
            return positions.copy()

        # Build history buffer: init_returns (IS) + test returns seen so far.
        # We use realised bar returns (not strategy returns) for VaR — the
        # underlying asset volatility sets the distributional risk, independent
        # of our position sizing.
        if init_returns is not None and len(init_returns) > 0:
            history_buf = list(init_returns[-self.var_window:])
        else:
            history_buf = []

        out = positions.copy()
        equity = 1.0
        running_max = 1.0
        circuit_open = False  # True = no positions allowed

        left_tail_quantile = 1.0 - self.var_conf  # e.g. 0.05 for 95% VaR

        for t in range(n):
            raw_pos = positions[t]

            # ---- 1. VaR gate -----------------------------------------------
            # Estimate 1-day VaR from history_buf (causal: excludes bar t).
            if len(history_buf) >= 5:
                hist = np.asarray(history_buf, dtype=float)
                var_t = float(np.quantile(hist, left_tail_quantile))
                # var_t is negative (a loss). Expected loss at position p is
                # |p × var_t|. We scale so |p × var_t| ≤ var_nav_pct.
                if var_t < 0 and abs(raw_pos) > 1e-12:
                    max_pos_by_var = self.var_nav_pct / abs(var_t)
                    if abs(raw_pos) > max_pos_by_var:
                        # Scale preserving sign
                        raw_pos = np.sign(raw_pos) * max_pos_by_var

            # ---- 2. DD circuit-breaker -------------------------------------
            if circuit_open:
                # Drawdown has been breached; stay flat until recovery.
                dd_now = equity / running_max - 1.0
                if dd_now > -self.dd_reentry:
                    circuit_open = False   # recovered — allow positions again
                else:
                    raw_pos = 0.0

            if not circuit_open:
                # Check if this bar's cumulative drawdown breaches the limit.
                # We evaluate BEFORE applying the position for this bar —
                # the decision uses only information through t-1.
                dd_now = equity / running_max - 1.0
                if dd_now < -self.dd_limit:
                    circuit_open = True
                    raw_pos = 0.0

            out[t] = raw_pos

            # ---- Update equity curve and history ---------------------------
            # Equity advances by the *underlying* return (not the strategy
            # return) so the DD tracks the drawdown of the asset we're trading,
            # which is what the risk limit is expressed in.
            equity *= (1.0 + bar_returns[t])
            running_max = max(running_max, equity)
            history_buf.append(bar_returns[t])
            if len(history_buf) > self.var_window:
                history_buf.pop(0)

        return out
