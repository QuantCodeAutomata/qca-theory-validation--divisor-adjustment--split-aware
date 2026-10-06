"""
benchmarks.py
=============
Construction of chain-linked price-weighted and cap-weighted benchmark
indices for price-based Stochastic Portfolio Theory (SPT).

The key object is the *adjusted start-of-day* price

    P^diamond_i(t) = P_i(t-1) / m_i(t)

which removes the mechanical price drop caused by a split of multiplier
m_i(t).  Using P^diamond instead of P(t-1) makes benchmark returns
invariant to splits within a holding period.

Chain-linked indices are constructed so that each quarterly universe
reconstitution does not introduce a level jump in the index value:

    I(t) = I(t-1) * [gross return of new universe from adjusted start]

This mirrors the standard chain-linking used for major equity indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Adjusted start-of-day quantities
# ---------------------------------------------------------------------------

def compute_adjusted_start_of_day_universe(
    P: pd.DataFrame,
    N: pd.DataFrame,
    m: pd.DataFrame,
    universe: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute adjusted start-of-day values for a given universe.

    For each stock i in *universe* and each trading date t:

        P^diamond_i(t) = P_i(t-1) / m_i(t)     [split-adjusted carry-forward price]
        S^diamond_i(t) = P^diamond_i(t) * N_i(t) [start-of-day market cap]
        nu^diamond_i(t) = P^diamond_i(t) / sum_j P^diamond_j(t)  [price weights]
        mu^diamond_i(t) = S^diamond_i(t) / sum_j S^diamond_j(t)  [cap weights]

    The first row (t=0) is set to NaN because there is no previous day.

    Parameters
    ----------
    P : pd.DataFrame
        Raw closing prices, shape (T, n_stocks).
    N : pd.DataFrame
        Shares outstanding, shape (T, n_stocks).
    m : pd.DataFrame
        Split multipliers (1.0 on non-split days), shape (T, n_stocks).
    universe : list of str
        Active ticker symbols for this quarter.

    Returns
    -------
    (P_diamond, S_diamond, nu_diamond, mu_diamond), each a DataFrame
    restricted to *universe* columns and indexed by P.index.
    """
    # Restrict to universe columns that exist in data
    cols = [c for c in universe if c in P.columns]
    P_u = P[cols]
    N_u = N[cols]
    m_u = m[cols]

    # P^diamond_i(t) = P_i(t-1) / m_i(t)  — shift prices back by one day
    P_prev = P_u.shift(1)             # P_i(t-1) for all t
    P_dmd = P_prev / m_u             # divide by today's split multiplier

    # Market-cap adjusted start-of-day: cap = price * shares outstanding today
    S_dmd = P_dmd * N_u

    # Normalised price weights (nu^diamond)
    nu_dmd = P_dmd.div(P_dmd.sum(axis=1), axis=0)

    # Normalised cap weights (mu^diamond)
    mu_dmd = S_dmd.div(S_dmd.sum(axis=1), axis=0)

    return P_dmd, S_dmd, nu_dmd, mu_dmd


# ---------------------------------------------------------------------------
# Helper: map each trading date to the active rebalance entry
# ---------------------------------------------------------------------------

def _active_universe(
    date: pd.Timestamp,
    rebal_dates: List[pd.Timestamp],
    quarterly_universes: Dict[pd.Timestamp, List[str]],
) -> List[str]:
    """Return the universe active on *date* (most recent rebal date <= date)."""
    active = [d for d in rebal_dates if d <= date]
    if not active:
        return quarterly_universes[rebal_dates[0]]
    return quarterly_universes[max(active)]


def _precompute_active_universes(
    trading_dates: pd.DatetimeIndex,
    quarterly_universes: Dict[pd.Timestamp, List[str]],
) -> List[List[str]]:
    """Precompute the active universe for every trading date."""
    rebal_dates = sorted(quarterly_universes.keys())
    result = []
    active_idx = 0
    for date in trading_dates:
        # Advance pointer while there is a later rebal date still <= date
        while (active_idx + 1 < len(rebal_dates)
               and rebal_dates[active_idx + 1] <= date):
            active_idx += 1
        result.append(quarterly_universes[rebal_dates[active_idx]])
    return result


# ---------------------------------------------------------------------------
# Cap-weighted benchmark
# ---------------------------------------------------------------------------

def build_cap_weighted_benchmark(
    P: pd.DataFrame,
    N: pd.DataFrame,
    m: pd.DataFrame,
    quarterly_universes: Dict[pd.Timestamp, List[str]],
    trading_dates: pd.DatetimeIndex,
    I0: float = 1.0,
) -> pd.Series:
    """Build chain-linked cap-weighted benchmark I_mu(t).

    On each trading day t the benchmark grows by the gross return of the
    active universe under cap weights, using adjusted start-of-day caps:

        I_mu(t) = I_mu(t-1) * [sum_{i in U} S_i(t)] / [sum_{i in U} S^diamond_i(t)]

    where S^diamond_i(t) = P^diamond_i(t) * N_i(t) is the start-of-day cap.

    Parameters
    ----------
    P, N, m : pd.DataFrame
        Raw prices, shares outstanding, and split multipliers.
    quarterly_universes : dict
        {rebalance_date: [tickers]} mapping.
    trading_dates : pd.DatetimeIndex
        Ordered sequence of trading dates; must match P.index.
    I0 : float
        Initial index level (default 1.0).

    Returns
    -------
    pd.Series
        Index level I_mu, indexed by *trading_dates*.
    """
    active_universes = _precompute_active_universes(trading_dates, quarterly_universes)

    values = np.empty(len(trading_dates))
    values[0] = I0

    for t in range(1, len(trading_dates)):
        date = trading_dates[t]
        date_prev = trading_dates[t - 1]
        universe = active_universes[t]
        cols = [c for c in universe if c in P.columns]

        # Start-of-day adjusted prices: P_i(t-1) / m_i(t)
        P_prev = P.loc[date_prev, cols].values
        m_t = m.loc[date, cols].values
        P_dmd = P_prev / m_t  # shape (d,)

        # Today's end-of-day prices and shares outstanding
        P_t = P.loc[date, cols].values
        N_t = N.loc[date, cols].values

        S_dmd = P_dmd * N_t     # start-of-day caps
        S_t = P_t * N_t         # end-of-day caps

        sum_S_dmd = S_dmd.sum()
        sum_S_t = S_t.sum()

        # Gross return: end / adjusted-start
        values[t] = values[t - 1] * (sum_S_t / sum_S_dmd) if sum_S_dmd > 0 else values[t - 1]

    return pd.Series(values, index=trading_dates, name="I_mu")


# ---------------------------------------------------------------------------
# Price-weighted benchmark
# ---------------------------------------------------------------------------

def build_price_weighted_benchmark(
    P: pd.DataFrame,
    N: pd.DataFrame,
    m: pd.DataFrame,
    quarterly_universes: Dict[pd.Timestamp, List[str]],
    trading_dates: pd.DatetimeIndex,
    I0: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """Build chain-linked price-weighted benchmark I_nu(t) and divisor kappa(t).

    The index gross return on day t is the ratio of total price to adjusted
    total price:

        I_nu(t) = I_nu(t-1) * [sum_{i in U} P_i(t)] / [sum_{i in U} P^diamond_i(t)]

    The divisor kappa tracks the cumulative split adjustments for the current
    constituents.  On split days it is updated to keep Σ̃ = Σ/kappa continuous:

        kappa(t) = kappa(t-1) * [sum_{i in U_t} P^diamond_i(t)] / [sum_{i in U_t} P_i(t-1)]

    Parameters
    ----------
    P, N, m : pd.DataFrame
        Raw prices, shares outstanding, split multipliers.
    quarterly_universes : dict
        {rebalance_date: [tickers]}.
    trading_dates : pd.DatetimeIndex
    I0 : float
        Initial index level.

    Returns
    -------
    (I_nu, kappa) : tuple of pd.Series, both indexed by *trading_dates*.
    """
    active_universes = _precompute_active_universes(trading_dates, quarterly_universes)

    index_vals = np.empty(len(trading_dates))
    kappa_vals = np.empty(len(trading_dates))
    index_vals[0] = I0
    kappa_vals[0] = 1.0

    for t in range(1, len(trading_dates)):
        date = trading_dates[t]
        date_prev = trading_dates[t - 1]
        universe = active_universes[t]
        cols = [c for c in universe if c in P.columns]

        P_prev = P.loc[date_prev, cols].values
        m_t = m.loc[date, cols].values
        P_t = P.loc[date, cols].values

        # Adjusted start-of-day prices
        P_dmd = P_prev / m_t   # P^diamond_i(t)

        sum_P_dmd = P_dmd.sum()
        sum_P_t = P_t.sum()

        # Chain-linked price-weighted return
        gross_ret = sum_P_t / sum_P_dmd if sum_P_dmd > 0 else 1.0
        index_vals[t] = index_vals[t - 1] * gross_ret

        # Divisor update: kappa absorbs the split-induced drop in total price.
        # On a split day, sum_P^diamond < sum_P(t-1), so kappa decreases.
        sum_P_prev = P_prev.sum()
        kappa_factor = sum_P_dmd / sum_P_prev if sum_P_prev > 0 else 1.0
        kappa_vals[t] = kappa_vals[t - 1] * kappa_factor

    I_nu = pd.Series(index_vals, index=trading_dates, name="I_nu")
    kappa = pd.Series(kappa_vals, index=trading_dates, name="kappa")
    return I_nu, kappa


# ---------------------------------------------------------------------------
# Benchmark gap decomposition
# ---------------------------------------------------------------------------

def compute_benchmark_decomposition(
    L_mu_P: pd.Series,
    L_mu_S: pd.Series,
    L_nu_P: pd.Series,
    L_nu_S: pd.Series,
) -> pd.DataFrame:
    """Decompose the gap between price-weighted and cap-weighted benchmarks.

    Definitions (all as log-level series relative to a common origin):

        D     = L_nu^P - L_mu^S   (headline gap: price-weight/price-universe
                                    vs cap-weight/cap-universe)
        W^P   = L_nu^P - L_mu^P   (weighting effect on the price universe)
        W^S   = L_nu^S - L_mu^S   (weighting effect on the cap universe)
        C^mu  = L_mu^P - L_mu^S   (selection effect under cap weights)
        C^nu  = L_nu^P - L_nu^S   (selection effect under price weights)

    Accounting identities that must hold exactly:

        D = W^P + C^mu  (via L_mu^P cancellation)
        D = W^S + C^nu  (via L_nu^S cancellation)

    Parameters
    ----------
    L_mu_P : pd.Series  Log cap-weighted index on price universe.
    L_mu_S : pd.Series  Log cap-weighted index on cap universe.
    L_nu_P : pd.Series  Log price-weighted index on price universe.
    L_nu_S : pd.Series  Log price-weighted index on cap universe.

    Returns
    -------
    pd.DataFrame
        Columns: D, W_P, W_S, C_mu, C_nu, check_WP_Cmu, check_WS_Cnu.
        The last two columns should equal D and serve as verification.
    """
    D = L_nu_P - L_mu_S
    W_P = L_nu_P - L_mu_P
    W_S = L_nu_S - L_mu_S
    C_mu = L_mu_P - L_mu_S
    C_nu = L_nu_P - L_nu_S

    return pd.DataFrame({
        "D": D,
        "W_P": W_P,
        "W_S": W_S,
        "C_mu": C_mu,
        "C_nu": C_nu,
        "check_WP_Cmu": W_P + C_mu,   # must equal D
        "check_WS_Cnu": W_S + C_nu,   # must equal D
    })


# ---------------------------------------------------------------------------
# Gibbs entropy
# ---------------------------------------------------------------------------

def compute_gibbs_entropy(
    weights: pd.DataFrame,
    eps: float = 1e-15,
) -> pd.Series:
    """Compute Gibbs (Shannon) entropy H(w) = -sum_i w_i log(w_i) for each date.

    Weights equal to zero are treated as contributing zero to the entropy
    (consistent with the convention 0 * log 0 = 0 in information theory).

    Parameters
    ----------
    weights : pd.DataFrame
        Weight matrix, shape (T, d).  Each row should sum to approximately 1.
    eps : float
        Small threshold below which a weight is treated as zero.

    Returns
    -------
    pd.Series
        Entropy values indexed by *weights.index*.
    """
    w = weights.values.astype(float)
    # Replace near-zero weights with eps before log to avoid log(0)
    safe = np.where(w > eps, w, eps)
    log_safe = np.where(w > eps, np.log(safe), 0.0)   # 0 * log(0) → 0
    H = -(w * log_safe).sum(axis=1)
    return pd.Series(H, index=weights.index, name="gibbs_entropy")
