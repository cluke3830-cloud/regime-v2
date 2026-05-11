"""Triple-barrier labels (López de Prado AFML §3.2).

Brief 1.3 of the regime_dashboard upgrade plan.

Why triple-barrier replaces forward-window labels:

    The existing LSTM teacher signal (regime_dashboard.py:1501) classifies
    each bar by the *fixed* h-bar forward window's realised return, vol,
    and drawdown. Two failure modes:

      1. Coupling — the rule features and the LSTM target both read from
         the same h-bar forward window of the same price process, so the
         LSTM ends up memorising the rule layer rather than adding
         independent information (audit §5.6.7 "right pattern, wrong size").
      2. Horizon mismatch — a fixed h applies the same window to a calm
         range and to a violent breakout. Triple-barrier defines the label
         by which barrier hits first, so high-vol regimes resolve faster
         and low-vol regimes hold longer.

How triple-barrier works:

    For each bar t with close C_t and volatility σ_t:
      - profit barrier   = C_t · (1 + π_up · σ_t)
      - stop barrier     = C_t · (1 - π_down · σ_t)
      - vertical barrier = bar t + h
    Walk forward bar by bar from t+1. The first barrier touched defines
    the label:
      +1 if profit barrier touched first   (long edge)
      -1 if stop barrier touched first     (drawdown)
       0 if vertical (time) barrier hit first (no edge)

Outputs a DataFrame indexed by the input close index with columns:
    t1     — int positional index of the bar where the label resolved
    ret    — realised log-return from t to t1 (close-to-close)
    label  — int in {-1, 0, +1}

The function is causally clean: label[i] depends only on prices in
[i+1, i + h]. Verified by the test suite via brute-force ground-truth
comparison and by an explicit "perturb future" test that mutates prices
beyond t1[i] and asserts label[i] is unchanged.

References
----------
López de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. §3.2 (Triple-Barrier Method), §3.3 (Meta-Labels), §3.4
    (Path-Dependent Labels).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    vol: pd.Series,
    pi_up: float = 2.0,
    pi_down: Optional[float] = None,
    horizon: int = 10,
) -> pd.DataFrame:
    """Compute triple-barrier labels for an aligned (close, vol) pair.

    Parameters
    ----------
    close : pd.Series
        Bar-close prices, monotonically time-ordered. The index of ``close``
        becomes the index of the returned DataFrame.
    vol : pd.Series
        Per-bar volatility estimate, on the same index as ``close``. Must
        be in *return* units (e.g., 0.01 for 1% daily) — NOT annualised.
        Pass the EWMA-vol column from regime_dashboard.compute_features
        directly.
    pi_up : float, default=2.0
        Profit barrier multiplier. Profit barrier = ``C_t · (1 + pi_up · σ_t)``.
    pi_down : float, optional
        Stop barrier multiplier. Stop barrier = ``C_t · (1 - pi_down · σ_t)``.
        Defaults to ``pi_up`` (symmetric barriers).
    horizon : int, default=10
        Number of bars in the vertical (time) barrier. Label resolves at
        ``min(t + horizon, len-1)`` if neither price barrier is hit first.

    Returns
    -------
    pd.DataFrame
        Indexed identically to ``close``, with columns
        ``[t1, ret, label]``:
          - ``t1`` (int): positional index of the bar where the label
            resolved (when label = 0, this is t + horizon clamped to
            len-1).
          - ``ret`` (float): log-return ``log(close[t1] / close[t])``.
          - ``label`` (int): {-1, 0, +1}.

        Bars within ``horizon`` of the end of the series cannot resolve
        cleanly through the time barrier — they receive
        ``label = 0, t1 = len-1, ret = log(close[-1]/close[t])`` and the
        caller is expected to drop them or treat them as held-position
        bars.

    Raises
    ------
    ValueError
        If ``close`` and ``vol`` have mismatched lengths or indices, if
        ``pi_up <= 0`` or ``pi_down <= 0``, or if ``horizon < 1``.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> close = pd.Series(np.linspace(100.0, 110.0, 50))  # smooth uptrend
    >>> vol = pd.Series(np.full(50, 0.01))                # 1% daily vol
    >>> labels = triple_barrier_labels(close, vol, pi_up=2.0, horizon=10)
    >>> # uptrend with 2% barriers: most bars hit profit barrier (+1)
    >>> assert (labels["label"] == 1).sum() > 0
    """
    if not isinstance(close, pd.Series):
        raise TypeError(f"close must be pd.Series, got {type(close).__name__}")
    if not isinstance(vol, pd.Series):
        raise TypeError(f"vol must be pd.Series, got {type(vol).__name__}")
    if len(close) != len(vol):
        raise ValueError(
            f"close len {len(close)} != vol len {len(vol)}"
        )
    if not close.index.equals(vol.index):
        raise ValueError("close and vol must share an index")
    if pi_up <= 0:
        raise ValueError(f"pi_up must be > 0, got {pi_up}")
    pi_down = pi_up if pi_down is None else pi_down
    if pi_down <= 0:
        raise ValueError(f"pi_down must be > 0, got {pi_down}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    n = len(close)
    close_arr = close.to_numpy(dtype=np.float64)
    vol_arr = vol.to_numpy(dtype=np.float64)

    t1 = np.full(n, n - 1, dtype=np.int64)
    label = np.zeros(n, dtype=np.int64)
    ret = np.zeros(n, dtype=np.float64)

    for t in range(n):
        c_t = close_arr[t]
        sigma = vol_arr[t]
        if not np.isfinite(c_t) or not np.isfinite(sigma) or sigma <= 0:
            # Cannot define barriers — emit label 0 with horizon endpoint.
            end = min(t + horizon, n - 1)
            t1[t] = end
            ret[t] = (
                float(np.log(close_arr[end] / c_t))
                if np.isfinite(c_t) and c_t > 0 and np.isfinite(close_arr[end])
                else float("nan")
            )
            continue

        upper = c_t * (1.0 + pi_up * sigma)
        lower = c_t * (1.0 - pi_down * sigma)
        end = min(t + horizon, n - 1)

        resolved = False
        for j in range(t + 1, end + 1):
            c_j = close_arr[j]
            if not np.isfinite(c_j):
                continue
            hit_upper = c_j >= upper
            hit_lower = c_j <= lower
            if hit_upper and hit_lower:
                # Same bar touches both — break ties toward the more
                # extreme move. A single bar with C_j >= upper AND
                # C_j <= lower is impossible by construction (upper > lower);
                # this branch only fires under data corruption.
                label[t] = 1 if (c_j - c_t) >= 0 else -1
                t1[t] = j
                ret[t] = float(np.log(c_j / c_t))
                resolved = True
                break
            if hit_upper:
                label[t] = 1
                t1[t] = j
                ret[t] = float(np.log(c_j / c_t))
                resolved = True
                break
            if hit_lower:
                label[t] = -1
                t1[t] = j
                ret[t] = float(np.log(c_j / c_t))
                resolved = True
                break

        if not resolved:
            # Vertical barrier — clamp at series end.
            label[t] = 0
            t1[t] = end
            c_end = close_arr[end]
            ret[t] = (
                float(np.log(c_end / c_t))
                if np.isfinite(c_end) and c_end > 0
                else float("nan")
            )

    return pd.DataFrame(
        {"t1": t1, "ret": ret, "label": label},
        index=close.index,
    )


__all__ = ["triple_barrier_labels"]