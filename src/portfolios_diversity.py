"""
portfolios_diversity.py
=======================
Diversity-weighted portfolio construction and multiplicative decomposition for
price-based Stochastic Portfolio Theory (SPT).

The diversity portfolio of order p with respect to a benchmark weight vector
w = (w_1, ..., w_d) allocates:

    pi_i = w_i^p / sum_j w_j^p

When p < 1 the portfolio overweights small constituents and underweights large
ones.  Fernholz (2002) shows that the portfolio earns a positive long-run
excess growth rate proportional to (1-p) times the portfolio's excess growth
function.

Here the weights w are either
    * price weights  nu_i = P_i / sum_j P_j   (basis='price')
    * cap weights    mu_i = S_i / sum_j S_j   (basis='cap')
computed from *adjusted* start-of-day prices P^diamond, which makes the daily
return invariant to within-day split announcements.

Log relative wealth decomposition
----------------------------------
For a diversity-p portfolio relative to its own benchmark:

    log[V_pi(t) / I(t)] ≈ C_p(t) + E_p(t) + J_p(t)

See :mod:`src.decomposition_terms` for the individual term definitions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .generators import diversity_function, diversity_gradient, diversity_hessian
from .benchmarks import compute_adjusted_start_of_day_universe, _precompute_active_universes
from .decomposition_terms import compute_C_p, compute_E_p, compute_J_p


# ---------------------------------------------------------------------------
# Core weighting function
# ---------------------------------------------------------------------------

def compute_diversity_weights(
    w: np.ndarray,
    p: float,
) -> np.ndarray:
    """Compute diversity-p portfolio weights from benchmark weights w.

    Allocation rule:

        pi_i = w_i^p / sum_j w_j^p

    Parameters
    ----------
    w : np.ndarray, shape (d,)
        Benchmark weights (price or cap), must be non-negative.
    p : float
        Diversity parameter.  p=1 returns equal-benchmark; p→0 approaches
        equal-weight; p<1 overweights small stocks.

    Returns
    -------
    np.ndarray, shape (d,)
        Portfolio weights that sum to 1.
    """
    w = np.asarray(w, dtype=float)
    # Replace non-positive weights with zero (zero-weight stocks excluded)
    w_pos = np.maximum(w, 0.0)
    powered = w_pos ** p
    total = powered.sum()
    if total <= 0:
        # Degenerate: return equal weight
        d = len(w)
        return np.ones(d) / d if d > 0 else np.array([])
    return powered / total


# ---------------------------------------------------------------------------
# Daily-rebalanced diversity portfolio
# ---------------------------------------------------------------------------

def run_diversity_portfolio(
    P: pd.DataFrame,
    N: pd.DataFrame,
    m: pd.DataFrame,
    quarterly_universes: Dict[pd.Timestamp, List[str]],
    trading_dates: pd.DatetimeIndex,
    p: float,
    basis: str,
    V0: float = 1.0,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run a daily-rebalanced diversity-p portfolio.

    On each trading day t the portfolio:
    1. Determines the active universe U_t from the quarterly calendar.
    2. Computes adjusted start-of-day weights (nu^diamond or mu^diamond).
    3. Forms diversity-p weights: pi_i = (w^diamond_i)^p / sum(w^diamond_j^p).
    4. Marks the portfolio return as sum_i pi_i * P_i(t) / P^diamond_i(t).

    Parameters
    ----------
    P, N, m : pd.DataFrame
        Raw prices, shares outstanding, split multipliers.
    quarterly_universes : dict
        {rebalance_date: [tickers]} mapping.
    trading_dates : pd.DatetimeIndex
        Ordered sequence of trading dates.
    p : float
        Diversity exponent (0 < p <= 1 in typical SPT usage).
    basis : str
        'price' to base weights on nu (price weights) or
        'cap'   to base weights on mu (cap weights).
    V0 : float
        Starting wealth (default 1.0).

    Returns
    -------
    wealth : pd.Series
        Portfolio wealth indexed by trading_dates.  wealth[0] = V0.
    weights_df : pd.DataFrame
        Portfolio weights (pi) on each trading date and stock.  NaN for
        stocks outside the active universe.
    """
    if basis not in ("price", "cap"):
        raise ValueError(f"basis must be 'price' or 'cap'; got {basis!r}")

    active_universes = _precompute_active_universes(trading_dates, quarterly_universes)

    T = len(trading_dates)
    wealth = np.empty(T)
    wealth[0] = V0

    # We store weights in a dict (date → dict of ticker weights) to later
    # assemble a DataFrame efficiently.
    weight_records: List[Dict[str, float]] = [{}] * T

    for t in range(1, T):
        date = trading_dates[t]
        date_prev = trading_dates[t - 1]
        universe = active_universes[t]
        cols = [c for c in universe if c in P.columns]
        if not cols:
            wealth[t] = wealth[t - 1]
            continue

        # --- Adjusted start-of-day prices ---
        P_prev = P.loc[date_prev, cols].values.astype(float)
        m_t = m.loc[date, cols].values.astype(float)
        P_dmd = P_prev / m_t  # P^diamond

        # Guard against zero or nan adjusted prices
        valid = np.isfinite(P_dmd) & (P_dmd > 0)
        if valid.sum() == 0:
            wealth[t] = wealth[t - 1]
            continue

        cols_v = [cols[i] for i in range(len(cols)) if valid[i]]
        P_dmd_v = P_dmd[valid]
        m_t_v = m_t[valid]
        P_t_v = P.loc[date, cols_v].values.astype(float)
        N_t_v = N.loc[date, cols_v].values.astype(float)

        # --- Benchmark weights w^diamond ---
        if basis == "price":
            sum_P = P_dmd_v.sum()
            if sum_P <= 0:
                wealth[t] = wealth[t - 1]
                continue
            w_dmd = P_dmd_v / sum_P
        else:  # cap basis
            S_dmd_v = P_dmd_v * N_t_v
            sum_S = S_dmd_v.sum()
            if sum_S <= 0:
                wealth[t] = wealth[t - 1]
                continue
            w_dmd = S_dmd_v / sum_S

        # --- Diversity-p portfolio weights ---
        pi = compute_diversity_weights(w_dmd, p)

        # --- Portfolio gross return: sum_i pi_i * P_i(t) / P^diamond_i(t) ---
        P_t_v_safe = np.where(np.isfinite(P_t_v) & (P_t_v > 0), P_t_v, P_dmd_v)
        gross_ret = np.sum(pi * P_t_v_safe / P_dmd_v)
        wealth[t] = wealth[t - 1] * gross_ret

        # Store portfolio weights for this day
        weight_records[t] = dict(zip(cols_v, pi))

    wealth_series = pd.Series(wealth, index=trading_dates, name=f"V_pi_p{p}_{basis}")

    # Assemble weights DataFrame (sparse: most dates have only 500 stocks)
    weights_df = pd.DataFrame(weight_records, index=trading_dates).fillna(0.0)

    return wealth_series, weights_df


# ---------------------------------------------------------------------------
# Multiplicative decomposition of log relative wealth
# ---------------------------------------------------------------------------

def compute_decomposition_terms(
    w_end: pd.DataFrame,
    w_adj: pd.DataFrame,
    m_events: pd.DataFrame,
    bench_log: pd.Series,
    p: float,
    split_dates: pd.DatetimeIndex,
    rebal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Compute the C_p, E_p, J_p decomposition of log relative wealth.

    The three additive terms approximate the log relative wealth of the
    diversity-p portfolio versus its benchmark:

        log[V_pi(t) / I(t)] ≈ C_p(t) + E_p(t) + J_p(t)

    Definitions (see :mod:`src.decomposition_terms` for implementation details):

        C_p(t) = sum_{s=1}^t log[D_p(w(s)) / D_p(w^diamond(s))]
        E_p(t) = -(1/2) sum_{s=1}^t H_{D_p}(w^diamond(s))[delta_w(s), delta_w(s)]
        J_p(t) = sum_{s in S_price, s<=t} log[D_p(nu^diamond(s)) / D_p(nu(s-1))]
                 (price basis only; zero for cap basis)

    Parameters
    ----------
    w_end : pd.DataFrame
        End-of-day benchmark weights (nu or mu) indexed by trading dates.
    w_adj : pd.DataFrame
        Adjusted start-of-day weights (nu^diamond or mu^diamond).
    m_events : pd.DataFrame
        Full split-multiplier matrix (to identify split dates).
    bench_log : pd.Series
        Log benchmark level log(I(t)).
    p : float
        Diversity parameter.
    split_dates : pd.DatetimeIndex
        Trading dates on which at least one split occurred (in the universe).
    rebal_dates : pd.DatetimeIndex
        Quarterly rebalance dates (excluded from J_p to avoid double-counting).

    Returns
    -------
    pd.DataFrame
        Columns: C_p, E_p, J_p, C_p_plus_E_p.  Indexed by trading_dates.
    """
    # Align both weight DataFrames to the same date index and column set
    common_cols = w_end.columns.intersection(w_adj.columns)
    w_end_a = w_end[common_cols].fillna(0.0)
    w_adj_a = w_adj[common_cols].fillna(0.0)

    # --- D_p values at end-of-day and adjusted start-of-day weights ---
    def _batch_Dp(df: pd.DataFrame, p: float) -> pd.Series:
        arr = df.values.astype(float)
        # Clip negatives; normalise rows that don't sum to ~1
        arr = np.clip(arr, 0.0, None)
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = np.where(row_sums > 0, arr / row_sums, arr)
        powered = arr ** p
        A = powered.sum(axis=1)
        Dp = np.where(A > 0, A ** (1.0 / p), 0.0)
        return pd.Series(Dp, index=df.index)

    D_w_end = _batch_Dp(w_end_a, p)
    D_w_adj = _batch_Dp(w_adj_a, p)

    # --- C_p ---
    # Guard against zero diversity values on days with degenerate weights
    D_ratio_safe = np.where(
        (D_w_adj.values > 0) & (D_w_end.values > 0),
        D_w_end.values / D_w_adj.values,
        1.0,
    )
    D_w_end_safe = pd.Series(
        np.where(D_w_end.values > 0, D_w_end.values, 1.0),
        index=w_end.index,
    )
    D_w_adj_safe = pd.Series(
        np.where(D_w_adj.values > 0, D_w_adj.values, 1.0),
        index=w_adj.index,
    )
    C_p = compute_C_p(D_w_end_safe, D_w_adj_safe)

    # --- E_p ---
    E_p = compute_E_p(w_end_a, w_adj_a, p)

    # --- J_p (price-basis split-jump term) ---
    # Only for split dates that do not coincide with rebalance dates
    pure_split_dates = split_dates.difference(rebal_dates)
    if len(pure_split_dates) > 0:
        # nu^diamond on split dates
        nu_adj_splits = w_adj_a.loc[w_adj_a.index.isin(pure_split_dates)]
        # nu(s-1): end-of-day weights on the day before each split date
        prev_idx = [w_end_a.index.get_loc(d) - 1 for d in pure_split_dates if d in w_end_a.index]
        valid_prev = [i for i in prev_idx if i >= 0]
        prev_dates = w_end_a.index[valid_prev]
        nu_prev_splits = w_end_a.iloc[valid_prev].set_index(
            pd.DatetimeIndex([pure_split_dates[k] for k, i in enumerate(prev_idx) if i >= 0])
        )
        J_p = compute_J_p(
            nu_adj=nu_adj_splits,
            nu_prev=nu_prev_splits,
            split_dates=pd.DatetimeIndex([pure_split_dates[k] for k, i in enumerate(prev_idx) if i >= 0]),
            p=p,
            all_dates=w_end.index,
        )
    else:
        J_p = pd.Series(0.0, index=w_end.index, name="J_p")

    # Align all series to common index
    idx = w_end.index
    C_p = C_p.reindex(idx).fillna(method="ffill").fillna(0.0)
    E_p = E_p.reindex(idx).fillna(method="ffill").fillna(0.0)
    J_p = J_p.reindex(idx).fillna(method="ffill").fillna(0.0)

    return pd.DataFrame({
        "C_p": C_p.values,
        "E_p": E_p.values,
        "J_p": J_p.values,
        "C_p_plus_E_p": C_p.values + E_p.values,
    }, index=idx)
