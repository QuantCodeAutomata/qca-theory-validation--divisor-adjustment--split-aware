"""
synthetic_paths.py
==================
Generators for synthetic price paths used in price-based SPT experiments.

Each generator returns a :class:`pandas.DataFrame` whose rows correspond to
discrete time steps ``t = 0, 1, ..., T``.  Prices follow geometric Brownian
motion (GBM) with optional pre-scheduled stock-split events.

Split mechanics
---------------
A split event for asset *k* at time *τ* with multiplier *m* applies the
following instantaneous adjustments (on top of the regular GBM step):

    P_k(τ) ←  P_k(τ) / m          (price drops by factor m)
    N_k(τ) ←  N_k(τ−1) · m        (shares outstanding rise by factor m)
    S_k(τ) =  P_k(τ) · N_k(τ)     (market cap is unchanged)

The split multiplier for each asset at each time step is recorded in the
``m_i`` column (value 1.0 on non-split days).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def generate_two_asset_paths(
    T: int = 30,
    dt: float = 1.0,
    seed: int = 42,
    p1_0: float = 100.0,
    p2_0: float = 50.0,
    n1_0: float = 1000.0,
    n2_0: float = 2000.0,
    sigma1: float = 0.01,
    sigma2: float = 0.01,
    mu1: float = 0.001,
    mu2: float = 0.001,
    split_events: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Generate a two-asset GBM price path with optional split events.

    Prices evolve according to:

        P_i(t) = P_i(t−1) · exp((μ_i − σ_i²/2)·dt + σ_i·√dt·Z_i(t))

    followed by any split adjustments scheduled at time *t*.

    Parameters
    ----------
    T : int
        Number of time steps (the returned DataFrame has T+1 rows).
    dt : float
        Length of each time step (used in GBM drift and diffusion terms).
    seed : int
        Random seed passed to :class:`numpy.random.Generator`.
    p1_0, p2_0 : float
        Initial prices of assets 1 and 2.
    n1_0, n2_0 : float
        Initial shares outstanding of assets 1 and 2.
    sigma1, sigma2 : float
        Annualised volatilities (GBM diffusion coefficients).
    mu1, mu2 : float
        Drift rates (GBM drift coefficients per unit time).
    split_events : list of dict, optional
        Sequence of split-event descriptors.  Each dict must contain:

        - ``'t'`` (*int*): time index at which the split takes effect.
        - ``'asset'`` (*int*): asset identifier, either ``1`` or ``2``.
        - ``'multiplier'`` (*float*): split ratio *m* (e.g. 2.0 for a 2-for-1
          split, 3.0 for a 3-for-1 split).

    Returns
    -------
    pd.DataFrame
        Columns: ``t``, ``P1``, ``P2``, ``N1``, ``N2``, ``S1``, ``S2``,
        ``m1``, ``m2``.

        - ``P{i}`` – price of asset *i* (post-split if applicable).
        - ``N{i}`` – shares outstanding of asset *i*.
        - ``S{i}`` – market capitalisation ``P_i * N_i``.
        - ``m{i}`` – split multiplier (1.0 on non-split days).

    Raises
    ------
    ValueError
        If any split event specifies an asset other than 1 or 2, a time index
        outside ``[1, T]``, or a non-positive multiplier.
    """
    if split_events is None:
        split_events = []

    # Validate split events
    for ev in split_events:
        if ev['asset'] not in (1, 2):
            raise ValueError(f"asset must be 1 or 2; got {ev['asset']}")
        if not (1 <= ev['t'] <= T):
            raise ValueError(f"split time {ev['t']} is outside [1, {T}]")
        if ev['multiplier'] <= 0:
            raise ValueError(f"multiplier must be positive; got {ev['multiplier']}")

    # Build lookup: {t: {asset_index: multiplier}}
    split_lookup: Dict[int, Dict[int, float]] = {}
    for ev in split_events:
        t_ev = ev['t']
        if t_ev not in split_lookup:
            split_lookup[t_ev] = {}
        # last event wins if duplicates exist for the same (t, asset)
        split_lookup[t_ev][ev['asset']] = float(ev['multiplier'])

    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal(T)
    Z2 = rng.standard_normal(T)

    P1 = np.empty(T + 1)
    P2 = np.empty(T + 1)
    N1 = np.empty(T + 1)
    N2 = np.empty(T + 1)
    S1 = np.empty(T + 1)
    S2 = np.empty(T + 1)
    m1 = np.ones(T + 1)
    m2 = np.ones(T + 1)

    P1[0], P2[0] = p1_0, p2_0
    N1[0], N2[0] = n1_0, n2_0
    S1[0] = p1_0 * n1_0
    S2[0] = p2_0 * n2_0

    drift1 = (mu1 - 0.5 * sigma1 ** 2) * dt
    drift2 = (mu2 - 0.5 * sigma2 ** 2) * dt
    diff1 = sigma1 * np.sqrt(dt)
    diff2 = sigma2 * np.sqrt(dt)

    for t in range(1, T + 1):
        # GBM step
        P1[t] = P1[t - 1] * np.exp(drift1 + diff1 * Z1[t - 1])
        P2[t] = P2[t - 1] * np.exp(drift2 + diff2 * Z2[t - 1])
        N1[t] = N1[t - 1]
        N2[t] = N2[t - 1]

        # Apply scheduled splits
        if t in split_lookup:
            if 1 in split_lookup[t]:
                mult = split_lookup[t][1]
                P1[t] /= mult
                N1[t] *= mult
                m1[t] = mult
            if 2 in split_lookup[t]:
                mult = split_lookup[t][2]
                P2[t] /= mult
                N2[t] *= mult
                m2[t] = mult

        S1[t] = P1[t] * N1[t]
        S2[t] = P2[t] * N2[t]

    return pd.DataFrame(
        {
            "t": np.arange(T + 1, dtype=int),
            "P1": P1,
            "P2": P2,
            "N1": N1,
            "N2": N2,
            "S1": S1,
            "S2": S2,
            "m1": m1,
            "m2": m2,
        }
    )


def generate_multi_asset_paths(
    d: int = 3,
    T: int = 50,
    seed: int = 42,
    p0: Optional[np.ndarray] = None,
    n0: Optional[np.ndarray] = None,
    sigma: float = 0.015,
    mu: float = 0.0005,
    split_events: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Generate a *d*-asset GBM price path with optional split events.

    All assets share the same scalar drift *μ* and volatility *σ* but are
    driven by independent Brownian increments.  Per-asset initial conditions
    can be supplied via ``p0`` and ``n0``.

    Parameters
    ----------
    d : int
        Number of assets.
    T : int
        Number of time steps (the returned DataFrame has T+1 rows).
    seed : int
        Random seed passed to :class:`numpy.random.Generator`.
    p0 : array-like of shape (d,), optional
        Initial prices.  Defaults to ``[100, 50, 100/3, ...]`` (i.e.
        ``100/(i+1)`` for asset *i*).
    n0 : array-like of shape (d,), optional
        Initial shares outstanding.  Defaults to 1 000 for every asset.
    sigma : float
        Common GBM volatility coefficient.
    mu : float
        Common GBM drift coefficient.
    split_events : list of dict, optional
        Sequence of split-event descriptors.  Each dict must contain:

        - ``'t'`` (*int*): time index at which the split takes effect.
        - ``'asset'`` (*int*): 1-based asset index (``1`` through ``d``).
        - ``'multiplier'`` (*float*): split ratio *m*.

    Returns
    -------
    pd.DataFrame
        Columns: ``t``, ``P_1``, ..., ``P_d``, ``N_1``, ..., ``N_d``,
        ``S_1``, ..., ``S_d``, ``m_1``, ..., ``m_d``.

    Raises
    ------
    ValueError
        If initial arrays have wrong length, or split events are out of range.
    """
    if p0 is None:
        p0 = np.array([100.0 / (i + 1) for i in range(d)])
    else:
        p0 = np.asarray(p0, dtype=float)

    if n0 is None:
        n0 = np.full(d, 1000.0)
    else:
        n0 = np.asarray(n0, dtype=float)

    if p0.shape != (d,):
        raise ValueError(f"p0 must have length d={d}; got shape {p0.shape}")
    if n0.shape != (d,):
        raise ValueError(f"n0 must have length d={d}; got shape {n0.shape}")
    if np.any(p0 <= 0):
        raise ValueError("All initial prices in p0 must be positive.")
    if np.any(n0 <= 0):
        raise ValueError("All initial share counts in n0 must be positive.")

    if split_events is None:
        split_events = []

    # Validate and build lookup: {t: {0-based asset index: multiplier}}
    split_lookup: Dict[int, Dict[int, float]] = {}
    for ev in split_events:
        asset_1based = ev['asset']
        if not (1 <= asset_1based <= d):
            raise ValueError(f"asset must be in [1, {d}]; got {asset_1based}")
        t_ev = ev['t']
        if not (1 <= t_ev <= T):
            raise ValueError(f"split time {t_ev} is outside [1, {T}]")
        mult = float(ev['multiplier'])
        if mult <= 0:
            raise ValueError(f"multiplier must be positive; got {mult}")
        if t_ev not in split_lookup:
            split_lookup[t_ev] = {}
        split_lookup[t_ev][asset_1based - 1] = mult  # store 0-based

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((T, d))  # shape (T, d)

    P = np.empty((T + 1, d))
    N = np.empty((T + 1, d))
    S = np.empty((T + 1, d))
    M = np.ones((T + 1, d))

    P[0] = p0
    N[0] = n0
    S[0] = p0 * n0

    drift = mu - 0.5 * sigma ** 2

    for t in range(1, T + 1):
        P[t] = P[t - 1] * np.exp(drift + sigma * Z[t - 1])
        N[t] = N[t - 1].copy()

        if t in split_lookup:
            for idx, mult in split_lookup[t].items():
                P[t, idx] /= mult
                N[t, idx] *= mult
                M[t, idx] = mult

        S[t] = P[t] * N[t]

    data: Dict[str, np.ndarray] = {"t": np.arange(T + 1, dtype=int)}
    for i in range(d):
        data[f"P_{i + 1}"] = P[:, i]
        data[f"N_{i + 1}"] = N[:, i]
        data[f"S_{i + 1}"] = S[:, i]
        data[f"m_{i + 1}"] = M[:, i]

    return pd.DataFrame(data)


def generate_repeated_split_reset_paths(
    T: int = 100,
    d: int = 2,
    split_period: int = 20,
    appreciation_rate: float = 0.02,
    split_multiplier: float = 2.0,
    p0: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic split-reset paths (Example 3.5 style).

    All *d* assets start at price *p0* and appreciate at a constant
    multiplicative rate ``appreciation_rate`` per step:

        P_i(t) = P_i(t−1) · exp(appreciation_rate)

    Every ``split_period`` steps all assets undergo a simultaneous
    ``split_multiplier``-for-1 split: price is divided by ``split_multiplier``
    and shares outstanding are multiplied by ``split_multiplier``.  This
    creates a periodic *sawtooth* pattern in raw prices while market
    capitalisation grows monotonically.

    The construction follows the spirit of Example 3.5 in the accompanying
    paper: in the long run the divisor κ(t) grows without bound, while the
    normalised total price Σ̃(t) remains bounded, illustrating the divergence
    between naive price-index tracking and the divisor-adjusted version.

    Parameters
    ----------
    T : int
        Number of time steps (the returned DataFrame has T+1 rows).
    d : int
        Number of assets (all behave identically).
    split_period : int
        Number of steps between successive splits for every asset.
    appreciation_rate : float
        Deterministic drift per step (no diffusion term; σ = 0).
    split_multiplier : float
        Split ratio applied to every asset at each split date.
    p0 : float
        Common initial price for all assets.
    seed : int
        Random seed (unused for the deterministic path; kept for API
        consistency with the other generators).

    Returns
    -------
    pd.DataFrame
        Same column layout as :func:`generate_multi_asset_paths`:
        ``t``, ``P_1``, ..., ``P_d``, ``N_1``, ..., ``N_d``,
        ``S_1``, ..., ``S_d``, ``m_1``, ..., ``m_d``.

    Raises
    ------
    ValueError
        If ``split_period`` is not a positive integer or ``split_multiplier``
        is not greater than 1.
    """
    if split_period < 1:
        raise ValueError(f"split_period must be ≥ 1; got {split_period}")
    if split_multiplier <= 1.0:
        raise ValueError(
            f"split_multiplier must be > 1.0 for a meaningful split; "
            f"got {split_multiplier}"
        )

    split_events: List[Dict] = []
    for asset in range(1, d + 1):
        for t_split in range(split_period, T + 1, split_period):
            split_events.append(
                {"t": t_split, "asset": asset, "multiplier": split_multiplier}
            )

    return generate_multi_asset_paths(
        d=d,
        T=T,
        seed=seed,
        p0=np.full(d, p0),
        n0=np.full(d, 1000.0),
        sigma=0.0,  # deterministic: no diffusion
        mu=appreciation_rate,
        split_events=split_events,
    )
