"""
split_mechanics.py
==================
Core numerical routines for price-based Stochastic Portfolio Theory (SPT)
with stock-split awareness.

Notation used throughout this module (mirroring the paper):
    d          – number of assets
    T          – number of time steps
    P_i(t)    – price of asset i at time t
    N_i(t)    – shares outstanding of asset i at time t
    S_i(t)    – market capitalisation P_i(t) * N_i(t)
    Σ(t)      – raw total price  sum_i P_i(t)
    κ(t)      – price-index divisor
    Σ̃(t)     – normalised total price  Σ(t) / κ(t)
    ν_i(t)   – price weight  P_i(t) / Σ(t)
    μ_i(t)   – cap weight    S_i(t) / Λ(t)
    Λ(t)     – total market capitalisation  sum_i S_i(t)
    ϑ_i(t)  – number of shares of asset i held by the portfolio
    V^ϑ(t)  – portfolio wealth  sum_i ϑ_i(t) P_i(t)
    m_i(t)  – split multiplier for asset i at time t (1.0 on non-split days)

All array functions expect NumPy arrays and perform no implicit copies unless
otherwise noted.  Index conventions: time axis is axis 0, asset axis is axis 1.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Weight functions
# ---------------------------------------------------------------------------


def compute_price_weights(prices: np.ndarray) -> np.ndarray:
    """Compute price weights ν_i = P_i / Σ_j P_j.

    Parameters
    ----------
    prices : np.ndarray
        1-D array of asset prices at a single time step, shape ``(d,)``.
        All entries must be strictly positive.

    Returns
    -------
    np.ndarray
        1-D array of price weights of shape ``(d,)`` that sums to 1.

    Raises
    ------
    ValueError
        If *prices* is empty, contains non-positive values, or does not sum
        to a positive total.

    Examples
    --------
    >>> compute_price_weights(np.array([100.0, 50.0, 50.0]))
    array([0.5 , 0.25, 0.25])
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size == 0:
        raise ValueError("prices must be a non-empty 1-D array.")
    if np.any(prices <= 0):
        raise ValueError("All prices must be strictly positive.")

    total = prices.sum()
    weights = prices / total

    # Sanity check (guards against catastrophic cancellation)
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError(
            f"Price weights do not sum to 1 (sum={weights.sum():.6e})."
        )
    return weights


def compute_cap_weights(caps: np.ndarray) -> np.ndarray:
    """Compute capitalisation weights μ_i = S_i / Σ_j S_j.

    Parameters
    ----------
    caps : np.ndarray
        1-D array of market capitalisations at a single time step, shape
        ``(d,)``.  All entries must be strictly positive.

    Returns
    -------
    np.ndarray
        1-D array of cap weights of shape ``(d,)`` that sums to 1.

    Raises
    ------
    ValueError
        If *caps* is empty or contains non-positive values.

    Examples
    --------
    >>> compute_cap_weights(np.array([1e6, 2e6, 1e6]))
    array([0.25, 0.5 , 0.25])
    """
    caps = np.asarray(caps, dtype=float)
    if caps.ndim != 1 or caps.size == 0:
        raise ValueError("caps must be a non-empty 1-D array.")
    if np.any(caps <= 0):
        raise ValueError("All market caps must be strictly positive.")

    total = caps.sum()
    weights = caps / total

    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise RuntimeError(
            f"Cap weights do not sum to 1 (sum={weights.sum():.6e})."
        )
    return weights


# ---------------------------------------------------------------------------
# Divisor mechanics
# ---------------------------------------------------------------------------


def divisor_update(
    kappa_prev: float,
    sigma_prev: float,
    sigma_curr: float,
    is_split_day: bool,
) -> float:
    """Update the price-index divisor κ across a single time step.

    The divisor is only adjusted on days when at least one asset splits:

        κ(t) = κ(t−1) · Σ(t) / Σ(t−1)    [split day]
        κ(t) = κ(t−1)                       [non-split day]

    On a split day the update rule ensures continuity of the normalised total
    price Σ̃(t) = Σ(t)/κ(t):

        Σ̃(t) = Σ(t) / [κ(t−1) · Σ(t)/Σ(t−1)] = Σ(t−1) / κ(t−1) = Σ̃(t−1)

    Parameters
    ----------
    kappa_prev : float
        Divisor value at the previous time step, κ(t−1).  Must be positive.
    sigma_prev : float
        Raw total price at the previous step, Σ(t−1) = Σ_i P_i(t−1).
    sigma_curr : float
        Raw total price at the current step, Σ(t) = Σ_i P_i(t).
    is_split_day : bool
        ``True`` if at least one asset underwent a split at the current step.

    Returns
    -------
    float
        Updated divisor κ(t).

    Raises
    ------
    ValueError
        If *kappa_prev* or *sigma_prev* is non-positive.
    """
    if kappa_prev <= 0:
        raise ValueError(f"kappa_prev must be positive; got {kappa_prev}.")
    if sigma_prev <= 0:
        raise ValueError(f"sigma_prev must be positive; got {sigma_prev}.")

    if is_split_day:
        return kappa_prev * sigma_curr / sigma_prev
    return kappa_prev


def compute_normalized_total_price(
    prices: np.ndarray,
    kappa: float,
) -> float:
    """Compute the normalised total price Σ̃(t) = Σ(t) / κ(t).

    Parameters
    ----------
    prices : np.ndarray
        1-D array of asset prices at a single time step.
    kappa : float
        Divisor at the same time step.  Must be positive.

    Returns
    -------
    float
        Normalised total price Σ̃(t).

    Raises
    ------
    ValueError
        If *kappa* is non-positive or any price is non-positive.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size == 0:
        raise ValueError("prices must be a non-empty 1-D array.")
    if np.any(prices <= 0):
        raise ValueError("All prices must be strictly positive.")
    if kappa <= 0:
        raise ValueError(f"kappa must be positive; got {kappa}.")

    return float(prices.sum() / kappa)


# ---------------------------------------------------------------------------
# Split detection
# ---------------------------------------------------------------------------


def detect_split_events(
    N_series: pd.DataFrame,
    S_series: pd.DataFrame,
    tau_sh: float = 0.15,
    tau_mc: float = 0.15,
) -> pd.DataFrame:
    """Heuristically detect split multipliers from share-count and cap data.

    Implements the detection rule from the paper:

        ρ_i(t) = N_i(t) / N_i(t−1)          (share-count ratio)
        η_i(t) = S_i(t) / S_i(t−1)          (market-cap ratio)

    Classification at each (t, i):
        m_i(t) = 1         if |ρ_i(t) − 1| ≤ τ_sh          [no change]
        m_i(t) = ρ_i(t)   if |ρ_i(t) − 1| > τ_sh
                              and |η_i(t) − 1| ≤ τ_mc       [split detected]
        m_i(t) = 1         otherwise                         [non-split action]

    The first row (t=0) has no predecessor and is set to 1 by convention.

    Parameters
    ----------
    N_series : pd.DataFrame
        Share-count time series, shape ``(T+1, d)``.  Each column corresponds
        to one asset; values must be positive.
    S_series : pd.DataFrame
        Market-capitalisation time series, shape ``(T+1, d)``.  Must have the
        same shape and column names as *N_series*.
    tau_sh : float
        Detection threshold for fractional share-count change.  Default 0.15.
    tau_mc : float
        Detection threshold for fractional market-cap change.  Default 0.15.

    Returns
    -------
    pd.DataFrame
        Split-multiplier DataFrame of the same shape and column names as
        *N_series*.  Entry ``(t, i)`` equals the detected multiplier for
        asset *i* at time *t* (1.0 on non-split or undetected days).

    Raises
    ------
    ValueError
        If *N_series* and *S_series* have different shapes or if threshold
        values are negative.
    """
    if N_series.shape != S_series.shape:
        raise ValueError(
            f"N_series and S_series must have the same shape; "
            f"got {N_series.shape} vs {S_series.shape}."
        )
    if tau_sh < 0 or tau_mc < 0:
        raise ValueError("Thresholds tau_sh and tau_mc must be non-negative.")

    N = N_series.values.astype(float)
    S = S_series.values.astype(float)
    T_plus_1, d = N.shape

    M = np.ones((T_plus_1, d))  # default: no split

    if T_plus_1 < 2:
        return pd.DataFrame(M, index=N_series.index, columns=N_series.columns)

    # Compute ratios for t >= 1
    rho = N[1:] / N[:-1]   # share-count ratio, shape (T, d)
    eta = S[1:] / S[:-1]   # cap ratio, shape (T, d)

    share_changed = np.abs(rho - 1.0) > tau_sh
    cap_stable = np.abs(eta - 1.0) <= tau_mc

    # Where shares changed significantly AND cap stayed stable → split detected
    is_split = share_changed & cap_stable
    M[1:][is_split] = rho[is_split]
    # All other non-trivial share changes (non-split corporate actions) remain 1

    return pd.DataFrame(M, index=N_series.index, columns=N_series.columns)


# ---------------------------------------------------------------------------
# Adjusted start-of-day prices
# ---------------------------------------------------------------------------


def compute_adjusted_start_of_day(
    P_prev: np.ndarray,
    N_curr: np.ndarray,
    m_curr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute split-adjusted start-of-day prices and capitalisations.

    On split days for asset *i*, the "true" economic opening price seen by a
    holder of pre-split shares is P_i(t−1)/m_i(t), not P_i(t−1).  Using this
    adjusted price ensures that the overnight price change attributed to the
    *return* component excludes the mechanical split-induced drop:

        P⋄_i(t) = P_i(t−1) / m_i(t)
        S⋄_i(t) = P⋄_i(t) · N_i(t)

    On non-split days m_i(t) = 1, so P⋄_i = P_i(t−1) and S⋄_i = S_i(t−1).

    Parameters
    ----------
    P_prev : np.ndarray
        Prices at the previous time step, shape ``(d,)``.
    N_curr : np.ndarray
        Shares outstanding at the current time step, shape ``(d,)``.
    m_curr : np.ndarray
        Split multipliers at the current time step, shape ``(d,)``.
        Non-split assets have m_curr[i] = 1.0.

    Returns
    -------
    P_diamond : np.ndarray
        Adjusted start-of-day prices, shape ``(d,)``.
    S_diamond : np.ndarray
        Adjusted start-of-day capitalisations, shape ``(d,)``.

    Raises
    ------
    ValueError
        If the three arrays have different lengths or if any multiplier is
        non-positive.
    """
    P_prev = np.asarray(P_prev, dtype=float)
    N_curr = np.asarray(N_curr, dtype=float)
    m_curr = np.asarray(m_curr, dtype=float)

    if not (P_prev.shape == N_curr.shape == m_curr.shape):
        raise ValueError(
            "P_prev, N_curr, and m_curr must all have the same shape."
        )
    if np.any(m_curr <= 0):
        raise ValueError("All multipliers in m_curr must be positive.")

    P_diamond = P_prev / m_curr
    S_diamond = P_diamond * N_curr
    return P_diamond, S_diamond


# ---------------------------------------------------------------------------
# Self-financing check
# ---------------------------------------------------------------------------


def self_financing_check(
    theta: np.ndarray,
    X: np.ndarray,
    is_split: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Check the self-financing condition for a split-aware strategy.

    In the price-based SPT framework the wealth process is:

        V^ϑ(t) = Σ_i ϑ_i(t) · X_i(t)

    and satisfies the algebraic decomposition:

        ΔV^ϑ(t) = ϑ(t−1) · ΔX(t)  +  Δϑ(t) · X(t)
                  └─ return component ─┘  └─ adjustment ─┘

    A strategy is *self-financing* when the adjustment component
    ``Δϑ(t) · X(t)`` is zero on every non-split day.  On split days the
    adjustment component captures the mechanical share-count change and is
    non-zero even without active rebalancing.

    The ``defect`` key reports the discrepancy between the left-hand side
    (``wealth_change``) and the right-hand side (``return_component +
    adjustment_component``).  By the algebraic identity above, the defect is
    zero to floating-point precision; a non-zero defect indicates numerical
    issues or an inconsistent input.

    Parameters
    ----------
    theta : np.ndarray
        Share holdings ϑ(t), shape ``(T+1, d)``.
    X : np.ndarray
        Asset prices X(t) (may be raw or adjusted), shape ``(T+1, d)``.
    is_split : np.ndarray
        Boolean array indicating split days, shape ``(T+1, d)``.

    Returns
    -------
    dict with keys:
        ``'wealth'``
            V^ϑ(t), shape ``(T+1,)``.
        ``'wealth_change'``
            ΔV^ϑ(t) = V^ϑ(t) − V^ϑ(t−1), shape ``(T+1,)`` with index 0 set
            to NaN.
        ``'return_component'``
            ϑ(t−1) · ΔX(t) = Σ_i ϑ_i(t−1) · (X_i(t) − X_i(t−1)), shape
            ``(T+1,)`` with index 0 set to NaN.
        ``'adjustment_component'``
            Δϑ(t) · X(t) = Σ_i (ϑ_i(t) − ϑ_i(t−1)) · X_i(t), shape
            ``(T+1,)`` with index 0 set to NaN.
        ``'defect'``
            wealth_change − (return_component + adjustment_component), shape
            ``(T+1,)`` with index 0 set to NaN.  Should be ≈ 0 everywhere.

    Raises
    ------
    ValueError
        If *theta*, *X*, or *is_split* have incompatible shapes, or if *theta*
        has fewer than 2 rows.
    """
    theta = np.asarray(theta, dtype=float)
    X = np.asarray(X, dtype=float)
    is_split = np.asarray(is_split, dtype=bool)

    if theta.ndim != 2 or X.ndim != 2:
        raise ValueError("theta and X must be 2-D arrays of shape (T+1, d).")
    if theta.shape != X.shape:
        raise ValueError(
            f"theta and X must have the same shape; "
            f"got {theta.shape} vs {X.shape}."
        )
    if is_split.shape != theta.shape:
        raise ValueError(
            f"is_split must match theta's shape {theta.shape}; "
            f"got {is_split.shape}."
        )
    T_plus_1 = theta.shape[0]
    if T_plus_1 < 2:
        raise ValueError("theta must have at least 2 rows (t=0 and t=1).")

    wealth = (theta * X).sum(axis=1)  # shape (T+1,)

    wealth_change = np.full(T_plus_1, np.nan)
    return_component = np.full(T_plus_1, np.nan)
    adjustment_component = np.full(T_plus_1, np.nan)
    defect = np.full(T_plus_1, np.nan)

    # t = 1, ..., T
    dX = X[1:] - X[:-1]                              # shape (T, d)
    dtheta = theta[1:] - theta[:-1]                  # shape (T, d)

    wealth_change[1:] = wealth[1:] - wealth[:-1]
    return_component[1:] = (theta[:-1] * dX).sum(axis=1)
    adjustment_component[1:] = (dtheta * X[1:]).sum(axis=1)
    defect[1:] = wealth_change[1:] - (return_component[1:] + adjustment_component[1:])

    return {
        "wealth": wealth,
        "wealth_change": wealth_change,
        "return_component": return_component,
        "adjustment_component": adjustment_component,
        "defect": defect,
    }


# ---------------------------------------------------------------------------
# Defect measures Q and C
# ---------------------------------------------------------------------------


def compute_defect_Q(
    theta: np.ndarray,
    nu: np.ndarray,
    P: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    """Compute the price-weight defect Q^{ϑ,ν}(t) (paper eq. 2.9).

    Measures the discrepancy between actual portfolio wealth and the wealth
    implied by naive price weighting without split-aware divisor adjustment:

        Q^{ϑ,ν}(t) = V^ϑ(t) − κ(t) · [Σ_i ϑ_i(t) · ν_i(t)] · Σ̃(t)

    Since κ(t) · Σ̃(t) = Σ(t) and ν_i = P_i/Σ, the second term equals
    V^ϑ(t) when κ and ν are both consistently updated through splits.
    Consequently, Q = 0 when the divisor is correctly maintained.

    A non-zero defect arises when **ν** and/or **κ** are computed naively
    (e.g. the divisor is not updated on split days, so κ_naive < κ_correct
    after a split, inflating Σ̃_naive and producing Q ≠ 0).  Comparing Q
    under the naive and corrected divisors therefore quantifies the cost of
    ignoring split mechanics.

    Parameters
    ----------
    theta : np.ndarray
        Share holdings ϑ(t), shape ``(T+1, d)``.
    nu : np.ndarray
        Price weights ν_i(t), shape ``(T+1, d)``.  May be naive (no split
        correction) or corrected depending on the analysis.
    P : np.ndarray
        Raw asset prices P_i(t), shape ``(T+1, d)``.
    kappa : np.ndarray
        Divisor κ(t), shape ``(T+1,)``.  All values must be positive.

    Returns
    -------
    np.ndarray
        Defect values Q^{ϑ,ν}(t), shape ``(T+1,)``.

    Raises
    ------
    ValueError
        If array shapes are inconsistent or *kappa* contains non-positive
        values.
    """
    theta = np.asarray(theta, dtype=float)
    nu = np.asarray(nu, dtype=float)
    P = np.asarray(P, dtype=float)
    kappa = np.asarray(kappa, dtype=float)

    if theta.ndim != 2 or nu.ndim != 2 or P.ndim != 2:
        raise ValueError("theta, nu, and P must be 2-D arrays of shape (T+1, d).")
    if not (theta.shape == nu.shape == P.shape):
        raise ValueError("theta, nu, and P must all have the same shape.")
    T_plus_1 = theta.shape[0]
    if kappa.shape != (T_plus_1,):
        raise ValueError(
            f"kappa must have shape ({T_plus_1},); got {kappa.shape}."
        )
    if np.any(kappa <= 0):
        raise ValueError("All values in kappa must be positive.")

    # Actual portfolio wealth
    V = (theta * P).sum(axis=1)  # shape (T+1,)

    # Normalised total price Σ̃(t) = Σ(t) / κ(t)
    Sigma = P.sum(axis=1)        # shape (T+1,)
    Sigma_tilde = Sigma / kappa  # shape (T+1,)

    # Price-index-implied wealth: κ(t) · (ϑ·ν)(t) · Σ̃(t)
    theta_dot_nu = (theta * nu).sum(axis=1)           # shape (T+1,)
    implied = kappa * theta_dot_nu * Sigma_tilde      # shape (T+1,)

    return V - implied


def compute_defect_C(
    theta: np.ndarray,
    nu: np.ndarray,
    P: np.ndarray,
    kappa: np.ndarray,
    Sigma_tilde: np.ndarray,
) -> np.ndarray:
    """Compute the cumulative jump defect C^{ϑ,ν}(t) (paper eq. 2.10).

    Accumulates the wealth-change discontinuities attributable to stock splits:

        C^{ϑ,ν}(t) = Σ_{0<s≤t} [V^ϑ(s) − V^ϑ(s−)]
                    − Σ_{0<s≤t} ϑ(s−) · ΔX(s)

    In the discrete-time setting the two sums simplify to:

        C(t) = Σ_{s=1}^t Δϑ(s) · X(s)
             = Σ_{s=1}^t Σ_i (ϑ_i(s) − ϑ_i(s−1)) · P_i(s)

    which is the cumulative *adjustment component* of the wealth decomposition
    (see :func:`self_financing_check`).

    On non-split days a self-financing strategy satisfies Δϑ(s) = 0, so C
    does not accumulate.  On split days, a split-corrected strategy
    automatically adjusts ϑ_i → ϑ_i · m_i, contributing a non-zero term that
    captures the *free* wealth gain from the split.  A naive strategy that
    ignores splits leaves C = 0 on split days but then mis-tracks the economic
    wealth going forward.

    Parameters
    ----------
    theta : np.ndarray
        Share holdings ϑ(t), shape ``(T+1, d)``.
    nu : np.ndarray
        Price weights ν_i(t), shape ``(T+1, d)``.  Retained for API
        consistency and alternative formulations; not used in the primary
        computation.
    P : np.ndarray
        Raw asset prices P_i(t), shape ``(T+1, d)``.
    kappa : np.ndarray
        Divisor κ(t), shape ``(T+1,)``.  Retained for API consistency; not
        used in the primary computation.
    Sigma_tilde : np.ndarray
        Normalised total price Σ̃(t), shape ``(T+1,)``.  Retained for API
        consistency; not used in the primary computation.

    Returns
    -------
    np.ndarray
        Cumulative defect C^{ϑ,ν}(t), shape ``(T+1,)``.  ``C[0] = 0`` by
        convention (no history before t=0).

    Raises
    ------
    ValueError
        If *theta* and *P* have incompatible shapes or fewer than 2 rows.
    """
    theta = np.asarray(theta, dtype=float)
    P = np.asarray(P, dtype=float)

    if theta.ndim != 2 or P.ndim != 2:
        raise ValueError("theta and P must be 2-D arrays of shape (T+1, d).")
    if theta.shape != P.shape:
        raise ValueError(
            f"theta and P must have the same shape; "
            f"got {theta.shape} vs {P.shape}."
        )
    T_plus_1 = theta.shape[0]
    if T_plus_1 < 2:
        raise ValueError("theta must have at least 2 rows.")

    # Instantaneous adjustment component at each step s = 1, ..., T
    dtheta = theta[1:] - theta[:-1]                    # shape (T, d)
    instant = (dtheta * P[1:]).sum(axis=1)             # shape (T,)

    C = np.zeros(T_plus_1)
    C[1:] = np.cumsum(instant)
    return C


# ---------------------------------------------------------------------------
# Cross-basis identity
# ---------------------------------------------------------------------------


def compute_cross_basis_identity(
    nu: np.ndarray,
    mu: np.ndarray,
    N: np.ndarray,
    P: np.ndarray,
    S: np.ndarray,
    kappa: np.ndarray,
    Sigma_tilde: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Verify the cross-basis identity from price-based SPT.

    In the price-based SPT framework two equivalent expressions for portfolio
    wealth yield the following ratio identity for any holdings ϑ:

        V^{ϑ⊙N^{-1}}(μ)               Σ̃(t)
        ─────────────────  =  ─────────────────────  =  Σ̃(t) / Λ(t)
            V^{κϑ}(ν)              Λ(t)

    where the two portfolio valuations are:

        V^{ϑ⊙N^{-1}}(μ)  =  Σ_i (ϑ_i / N_i) · μ_i
                          =  V^ϑ / Λ

        V^{κϑ}(ν)         =  Σ_i (κ · ϑ_i) · ν_i
                          =  V^ϑ / Σ̃

    Here ϑ_i = 1 (unit holdings) is used as a canonical test vector,
    so V^ϑ = Σ_i P_i = Σ(t).  The identity becomes:

        LHS = [Σ_i P_i / (N_i · Λ)] · Λ  /  [κ · Σ_i P_i/Σ · Σ̃^{-1}·Σ̃]
            = Σ̃ / Λ  = RHS

    Parameters
    ----------
    nu : np.ndarray
        Price weights ν_i(t) = P_i(t)/Σ(t), shape ``(T+1, d)``.
    mu : np.ndarray
        Capitalisation weights μ_i(t) = S_i(t)/Λ(t), shape ``(T+1, d)``.
    N : np.ndarray
        Shares outstanding N_i(t), shape ``(T+1, d)``.  All values must be
        positive.
    P : np.ndarray
        Asset prices P_i(t), shape ``(T+1, d)``.
    S : np.ndarray
        Market capitalisations S_i(t) = P_i(t)·N_i(t), shape ``(T+1, d)``.
    kappa : np.ndarray
        Divisor κ(t), shape ``(T+1,)``.  All values must be positive.
    Sigma_tilde : np.ndarray
        Normalised total price Σ̃(t) = Σ(t)/κ(t), shape ``(T+1,)``.

    Returns
    -------
    dict with keys:
        ``'LHS'``
            Left-hand side V^{ϑ⊙N^{-1}}(μ) / V^{κϑ}(ν), shape ``(T+1,)``.
        ``'RHS'``
            Right-hand side Σ̃(t) / Λ(t), shape ``(T+1,)``.
        ``'max_abs_error'``
            Scalar: maximum of |LHS − RHS| over all time steps.

    Raises
    ------
    ValueError
        If array shapes are inconsistent or *kappa* or *N* contain
        non-positive values.
    """
    nu = np.asarray(nu, dtype=float)
    mu = np.asarray(mu, dtype=float)
    N = np.asarray(N, dtype=float)
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    Sigma_tilde = np.asarray(Sigma_tilde, dtype=float)

    shape_2d = nu.shape
    if not all(arr.shape == shape_2d for arr in (mu, N, P, S)):
        raise ValueError(
            "nu, mu, N, P, S must all have the same 2-D shape (T+1, d)."
        )
    T_plus_1 = shape_2d[0]
    if kappa.shape != (T_plus_1,) or Sigma_tilde.shape != (T_plus_1,):
        raise ValueError(
            f"kappa and Sigma_tilde must have shape ({T_plus_1},)."
        )
    if np.any(kappa <= 0):
        raise ValueError("All values in kappa must be positive.")
    if np.any(N <= 0):
        raise ValueError("All values in N must be positive.")

    # Canonical unit holdings: ϑ_i = 1 for all i
    # V^ϑ = Σ_i P_i = Σ(t)
    Sigma = P.sum(axis=1)                 # shape (T+1,)
    Lambda = S.sum(axis=1)                # shape (T+1,)

    # Cap-basis valuation: V^{ϑ⊙N^{-1}}(μ) = Σ_i (ϑ_i/N_i) · μ_i
    # With ϑ_i = 1: = Σ_i (1/N_i) · μ_i = Σ_i μ_i/N_i
    V_cap = (mu / N).sum(axis=1)          # shape (T+1,)

    # Price-basis valuation: V^{κϑ}(ν) = Σ_i (κ · ϑ_i) · ν_i
    # With ϑ_i = 1: = κ · Σ_i ν_i = κ  (since ν sums to 1)
    V_price = kappa * nu.sum(axis=1)      # shape (T+1,)

    # Guard against zero denominator
    with np.errstate(divide="ignore", invalid="ignore"):
        LHS = np.where(V_price != 0.0, V_cap / V_price, np.nan)

    RHS = Sigma_tilde / Lambda

    max_abs_error = float(np.nanmax(np.abs(LHS - RHS)))

    return {
        "LHS": LHS,
        "RHS": RHS,
        "max_abs_error": max_abs_error,
    }
