"""
decomposition_terms.py
======================
Efficient computation of the three additive terms in the multiplicative
decomposition of a diversity-p portfolio's log relative wealth.

In discrete-time price-based SPT, the log wealth of the diversity portfolio
pi_i = w^diamond_i^p / D_p(w^diamond)^p relative to the benchmark I(t)
satisfies the approximate identity:

    log[V_pi(t) / I(t)] ≈ C_p(t) + E_p(t) + J_p(t)

where:

    C_p(t)  adjusted-state contribution:
            cumulative sum of log[D_p(w(s)) / D_p(w^diamond(s))]
            Tracks how the diversity function value moves as prices evolve
            during each trading day.

    E_p(t)  excess-growth contribution:
            -(1/2) cumulative sum of the Hessian quadratic form
            H_Dp(w^diamond(s))[delta_w(s), delta_w(s)]
            Reflects the convexity of D_p that generates positive excess
            growth for p < 1.

    J_p(t)  split-jump contribution (price-weight basis only):
            cumulative sum over split dates of
            log[D_p(nu^diamond(s)) / D_p(nu(s-1))]
            Captures the discontinuous jump in price weights when a split
            resets a stock's price without changing its economic value.

All functions below operate on pre-aligned pandas Series / DataFrames that
share a common DatetimeIndex of trading dates.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Hessian quadratic form – O(d) vectorised implementation
# ---------------------------------------------------------------------------

def hessian_quadratic_form_Dp(
    x: np.ndarray,
    delta_x: np.ndarray,
    p: float,
) -> float:
    """Compute (1/2) delta_x^T H_{D_p}(x) delta_x without forming H.

    The Hessian of D_p at x has the factored form:

        H_{ii} = (p-1) * A^{1/p-1} * x_i^{p-2}  +  (1-p) * A^{1/p-2} * x_i^{2p-2}
        H_{ij} = (1-p) * A^{1/p-2} * x_i^{p-1} * x_j^{p-1}   (i ≠ j)

    The quadratic form sum_{ij} H_{ij} delta_i delta_j simplifies to:

        QF = (p-1)*A^{1/p-1} * sum_i(x_i^{p-2} * delta_i^2)
           + (1-p)*A^{1/p-2} * (sum_i(x_i^{p-1} * delta_i))^2

    This avoids the O(d^2) cost of forming the full matrix and is exact.
    The 1/2 prefactor is included in the return value to match the convention
    used in the excess-growth integral.

    Parameters
    ----------
    x : np.ndarray, shape (d,)
        Weight vector (all entries > 0).
    delta_x : np.ndarray, shape (d,)
        Weight change vector.
    p : float
        Diversity parameter, typically in (0, 1).

    Returns
    -------
    float
        (1/2) * delta_x^T H_{D_p}(x) delta_x.
    """
    x = np.asarray(x, dtype=float)
    delta_x = np.asarray(delta_x, dtype=float)

    # Guard against non-positive weights which cause numerical issues
    mask = x > 0
    if not mask.any():
        return 0.0

    x_safe = np.where(mask, x, 1e-30)

    A = np.sum(x_safe ** p)
    if A <= 0:
        return 0.0

    # Paper formula: (1/2) * delta^T [H_{D_p}(x) / D_p(x)] delta
    # where D_p(x) = A^{1/p}.  Dividing H by D_p removes the ill-conditioned
    # A^{1/p-1} and A^{1/p-2} factors, replacing them with A^{-1} and A^{-2}.
    #
    # Normalised Hessian entries (H_ij / D_p):
    #   diagonal: (p-1)*A^{-1}*x_i^{p-2}  +  (1-p)*A^{-2}*x_i^{2p-2}
    #   off-diag: (1-p)*A^{-2}*x_i^{p-1}*x_j^{p-1}
    #
    # Quadratic form (vectorised):
    #   sum_ij (H_ij/Dp) di dj
    #   = (p-1)*A^{-1} * sum_i(x_i^{p-2} di^2)
    #   + (1-p)*A^{-2} * (sum_i x_i^{p-1} di)^2

    # Term 1: diagonal contribution (normalised by D_p)
    term1 = (p - 1.0) / A * np.sum(
        np.where(mask, x_safe ** (p - 2.0), 0.0) * delta_x ** 2
    )

    # Term 2: rank-1 outer-product contribution (normalised by D_p)
    inner = np.sum(np.where(mask, x_safe ** (p - 1.0), 0.0) * delta_x)
    term2 = (1.0 - p) / (A ** 2) * inner ** 2

    # Full quadratic form; multiply by 1/2 for the excess-growth convention
    return 0.5 * (term1 + term2)


# ---------------------------------------------------------------------------
# C_p – adjusted-state contribution
# ---------------------------------------------------------------------------

def compute_C_p(
    D_G_w: pd.Series,
    D_G_w_adj: pd.Series,
) -> pd.Series:
    """C_p(t) = sum_{s=1}^t log[D_p(w(s)) / D_p(w^diamond(s))].

    Each one-period contribution is the log ratio of the diversity function
    value at end-of-day weights to the diversity function value at the
    adjusted start-of-day weights.  Summing cumulatively gives the total
    adjusted-state contribution over [1, t].

    Parameters
    ----------
    D_G_w : pd.Series
        D_p(w(t)) at end-of-day weights for each date.
    D_G_w_adj : pd.Series
        D_p(w^diamond(t)) at adjusted start-of-day weights for each date.

    Returns
    -------
    pd.Series
        C_p cumulated over trading dates, starting at 0.
    """
    # Log ratio per period
    log_ratio = np.log(D_G_w.values / D_G_w_adj.values)
    # Replace NaN/Inf from degenerate days with 0
    log_ratio = np.where(np.isfinite(log_ratio), log_ratio, 0.0)
    C_p = np.cumsum(log_ratio)
    return pd.Series(C_p, index=D_G_w.index, name="C_p")


# ---------------------------------------------------------------------------
# E_p – excess-growth contribution
# ---------------------------------------------------------------------------

def compute_E_p(
    w_end: pd.DataFrame,
    w_adj: pd.DataFrame,
    p: float,
) -> pd.Series:
    """E_p(t) = -(1/2) * cumsum of Hessian quadratic forms over [1, t].

    For each trading day s, the Hessian quadratic form measures the curvature
    of D_p evaluated at the adjusted start-of-day weights w^diamond(s),
    applied to the intraday weight move delta_w(s) = w_end(s) - w_adj(s).
    The negative sign reflects the sign convention in the SPT excess-growth
    formula.

    Parameters
    ----------
    w_end : pd.DataFrame
        End-of-day weights, shape (T, d), aligned with w_adj.
    w_adj : pd.DataFrame
        Adjusted start-of-day weights, shape (T, d).
    p : float
        Diversity parameter.

    Returns
    -------
    pd.Series
        E_p cumulated over trading dates.
    """
    n_dates = len(w_end)
    qf_vals = np.zeros(n_dates)

    w_end_arr = w_end.values.astype(float)
    w_adj_arr = w_adj.values.astype(float)

    for t in range(n_dates):
        x = w_adj_arr[t]
        delta = w_end_arr[t] - x
        # Replace NaN with 0 for missing stocks (not in universe)
        x = np.nan_to_num(x, nan=0.0)
        delta = np.nan_to_num(delta, nan=0.0)
        if x.sum() > 0:
            qf_vals[t] = hessian_quadratic_form_Dp(x, delta, p)

    # E_p is *negative* of cumulative Hessian form (excess growth)
    E_p = -np.cumsum(qf_vals)
    return pd.Series(E_p, index=w_end.index, name="E_p")


# ---------------------------------------------------------------------------
# J_p – split-jump contribution (price-weight basis only)
# ---------------------------------------------------------------------------

def compute_J_p(
    nu_adj: pd.DataFrame,
    nu_prev: pd.DataFrame,
    split_dates: pd.DatetimeIndex,
    p: float,
    all_dates: pd.DatetimeIndex,
) -> pd.Series:
    """J_p(t) = sum_{s in S, s<=t} log[D_p(nu^diamond(s)) / D_p(nu(s-1))].

    On a split date s, the price weights jump discontinuously: nu^diamond(s)
    is the weight vector after the split adjustment, while nu(s-1) is the
    weight vector at the close of the previous trading day.  This term
    captures the induced discontinuity in the diversity function.

    For the cap-weight basis (mu), splits do not affect market caps, so J_p
    is identically zero and need not be computed.

    Parameters
    ----------
    nu_adj : pd.DataFrame
        Adjusted start-of-day price weights on *split_dates*, shape (|S|, d).
    nu_prev : pd.DataFrame
        End-of-day price weights on the day before each split date, same shape.
    split_dates : pd.DatetimeIndex
        Dates in S (splits not coinciding with rebalance dates).
    p : float
        Diversity parameter.
    all_dates : pd.DatetimeIndex
        Full trading-date index (J_p is zero on non-split dates).

    Returns
    -------
    pd.Series
        J_p(t) cumulated over all_dates.
    """
    # Build a per-date increment series (0 on non-split days)
    increments = pd.Series(0.0, index=all_dates, name="J_p_inc")

    for date in split_dates:
        if date not in all_dates:
            continue
        x_adj = nu_adj.loc[date].values.astype(float)
        x_prev = nu_prev.loc[date].values.astype(float)
        x_adj = np.nan_to_num(x_adj, nan=0.0)
        x_prev = np.nan_to_num(x_prev, nan=0.0)

        # D_p of each weight vector
        A_adj = np.sum(x_adj ** p)
        A_prev = np.sum(x_prev ** p)
        if A_adj > 0 and A_prev > 0:
            Dp_adj = A_adj ** (1.0 / p)
            Dp_prev = A_prev ** (1.0 / p)
            ratio = Dp_adj / Dp_prev
            if ratio > 0:
                increments.loc[date] = np.log(ratio)

    J_p = increments.cumsum()
    J_p.name = "J_p"
    return J_p
