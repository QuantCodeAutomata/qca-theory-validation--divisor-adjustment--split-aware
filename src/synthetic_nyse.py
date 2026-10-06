"""
synthetic_nyse.py
=================
Generator for a large synthetic equity market calibrated to resemble the
NYSE universe over the period 2002–2021.

Since the CRSP (Center for Research in Security Prices) database is proprietary
and unavailable in this environment, all empirical experiments that would
normally use CRSP data instead use the synthetic dataset produced by
:func:`generate_synthetic_nyse`.  The synthetic market is a *structural analog*:
it reproduces the key statistical features — cross-sectional price dispersion,
market-cap heterogeneity, and realistic split frequency — but should not be
interpreted as replicating specific historical CRSP returns.

Calibration targets (NYSE 2002–2021 approximate characteristics)
-----------------------------------------------------------------
* Universe size: ~600 eligible stocks (select top 500 for experiments)
* Annual market return: ~8 % (geometric mean, equity risk premium included)
* Cross-sectional annual volatility: ~20 %  (std of per-stock annualised returns)
* Individual stock volatility: 15–50 % annualised
* Forward split frequency: ~8 % of stocks per year
* Reverse split frequency: ~2 % of stocks per year
* Nominal price range: $5 – $500 at inception
* Market-cap range: $100 M – $1 T at inception

Model
-----
Prices follow independent geometric Brownian motions with heterogeneous drift
and volatility drawn from cross-sectional distributions:

    sigma_i ~ Uniform(0.12, 0.45)   (annual vol)
    mu_i    ~ Normal(mu_annual, sigma_cross^2)  (annual drift)

    log P_i(t) = log P_i(t-1) + (mu_i - sigma_i^2/2) * dt + sigma_i * sqrt(dt) * Z_i(t)

where dt = 1/252.

Splits are applied deterministically based on price thresholds:
    * Forward 2:1 split:  price crosses the HIGH_THRESH from below while a
      per-stock split-eligible Bernoulli draw is positive (annual prob = split_prob_annual).
    * Reverse 1:2 split:  price crosses the LOW_THRESH from above with a
      lower annual probability (reverse_split_prob_annual).

After a split/reverse-split:
    * P_i(t) is divided / multiplied by the split ratio.
    * N_i(t) is multiplied / divided by the split ratio.
    * m_i(t) is set to the split ratio (1.0 on non-split days).
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


# Price thresholds that trigger split consideration
_HIGH_THRESH = 120.0   # forward split candidate if price exceeds this
_LOW_THRESH = 4.0      # reverse split candidate if price falls below this


def generate_synthetic_nyse(
    n_stocks: int = 600,
    n_trading_days: int = 5036,
    seed: int = 42,
    split_prob_annual: float = 0.08,
    reverse_split_prob_annual: float = 0.02,
    sigma_cross: float = 0.20,
    mu_annual: float = 0.08,
    start_date: str = "2002-01-02",
    price_range: Tuple[float, float] = (5.0, 500.0),
    cap_range: Tuple[float, float] = (1e8, 1e12),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Generate a synthetic NYSE-like equity dataset for 2002–2021.

    Parameters
    ----------
    n_stocks : int
        Total number of stocks in the synthetic universe.  The top 500 by
        price or cap are selected for each experiment.
    n_trading_days : int
        Number of simulated trading days (≈ 252 × 20 for a 20-year panel).
    seed : int
        NumPy random seed for reproducibility.
    split_prob_annual : float
        Annual probability that an eligible stock undergoes a forward split.
    reverse_split_prob_annual : float
        Annual probability that an eligible stock undergoes a reverse split.
    sigma_cross : float
        Standard deviation of the cross-sectional annual drift distribution.
    mu_annual : float
        Mean annual log-drift for all stocks.
    start_date : str
        First trading date (ISO format).
    price_range : tuple of float
        (min, max) initial nominal price drawn from a log-uniform distribution.
    cap_range : tuple of float
        (min, max) initial market cap drawn from a log-uniform distribution.

    Returns
    -------
    P : pd.DataFrame, shape (n_trading_days, n_stocks)
        Raw nominal closing prices.  Column names are 'S0000', 'S0001', …
    N : pd.DataFrame, shape (n_trading_days, n_stocks)
        Shares outstanding (adjusted for splits).
    m : pd.DataFrame, shape (n_trading_days, n_stocks)
        Split multipliers: 1.0 on non-split days, 2.0 (forward 2:1),
        1.5 (forward 3:2), or 0.5 (reverse 1:2).
    trading_dates : pd.DatetimeIndex
        Business-day sequence of length n_trading_days starting from start_date.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Trading date index
    # ------------------------------------------------------------------
    all_bdays = pd.bdate_range(start=start_date, periods=n_trading_days + 50)
    trading_dates = all_bdays[:n_trading_days]

    tickers = [f"S{i:04d}" for i in range(n_stocks)]
    dt = 1.0 / 252.0  # one trading day

    # ------------------------------------------------------------------
    # 2. Cross-sectional parameter draws
    # ------------------------------------------------------------------
    # Per-stock annualised volatility: Uniform[0.12, 0.45]
    sigma_vec = rng.uniform(0.12, 0.45, size=n_stocks)  # shape (n,)

    # Per-stock annualised drift: Normal(mu_annual, sigma_cross^2)
    mu_vec = rng.normal(mu_annual, sigma_cross, size=n_stocks)  # shape (n,)

    # Clamp extreme drifts to keep simulation stable over 20 years
    mu_vec = np.clip(mu_vec, -0.30, 0.60)

    # Daily GBM parameters
    drift_daily = (mu_vec - 0.5 * sigma_vec ** 2) * dt   # shape (n,)
    diffusion_daily = sigma_vec * np.sqrt(dt)              # shape (n,)

    # ------------------------------------------------------------------
    # 3. Initial prices and shares
    # ------------------------------------------------------------------
    # Initial prices: log-uniform in price_range
    log_p_min, log_p_max = np.log(price_range[0]), np.log(price_range[1])
    p0 = np.exp(rng.uniform(log_p_min, log_p_max, size=n_stocks))

    # Initial caps: log-uniform in cap_range
    log_c_min, log_c_max = np.log(cap_range[0]), np.log(cap_range[1])
    cap0 = np.exp(rng.uniform(log_c_min, log_c_max, size=n_stocks))

    # Initial shares: N_i = cap_i / P_i
    n0 = cap0 / p0

    # ------------------------------------------------------------------
    # 4. Daily split probability (convert annual to daily)
    # ------------------------------------------------------------------
    # P(split on any given day | eligible) ≈ annual_prob / 252
    split_prob_daily = split_prob_annual / 252.0
    rev_split_prob_daily = reverse_split_prob_annual / 252.0

    # Pre-draw all split Bernoulli indicators (T × n_stocks)
    fwd_split_draws = rng.random(size=(n_trading_days, n_stocks))
    rev_split_draws = rng.random(size=(n_trading_days, n_stocks))

    # Mix of 2:1 and 3:2 forward splits: 70% are 2:1, 30% are 3:2
    split_type_draws = rng.random(size=(n_trading_days, n_stocks))

    # ------------------------------------------------------------------
    # 5. Simulate price paths day-by-day, applying splits in-place
    # ------------------------------------------------------------------
    # Allocate output arrays
    P_arr = np.empty((n_trading_days, n_stocks), dtype=float)
    N_arr = np.empty((n_trading_days, n_stocks), dtype=float)
    m_arr = np.ones((n_trading_days, n_stocks), dtype=float)

    # Set initial conditions (day 0)
    P_arr[0] = p0
    N_arr[0] = n0

    # GBM noise: shape (T-1, n_stocks)
    Z = rng.standard_normal(size=(n_trading_days - 1, n_stocks))

    for t in range(1, n_trading_days):
        # GBM price update from previous day
        log_ret = drift_daily + diffusion_daily * Z[t - 1]
        P_new = P_arr[t - 1] * np.exp(log_ret)
        N_new = N_arr[t - 1].copy()
        m_new = np.ones(n_stocks)

        # --- Forward splits: triggered when price exceeds _HIGH_THRESH ---
        fwd_eligible = P_new > _HIGH_THRESH
        fwd_event = fwd_eligible & (fwd_split_draws[t] < split_prob_daily)

        for i in np.where(fwd_event)[0]:
            # Choose between 2:1 and 3:2 split
            if split_type_draws[t, i] < 0.70:
                mult = 2.0    # 2-for-1 split
            else:
                mult = 1.5    # 3-for-2 split
            P_new[i] /= mult
            N_new[i] *= mult
            m_new[i] = mult

        # --- Reverse splits: triggered when price falls below _LOW_THRESH ---
        rev_eligible = P_new < _LOW_THRESH
        rev_event = rev_eligible & (rev_split_draws[t] < rev_split_prob_daily)

        for i in np.where(rev_event)[0]:
            # Reverse 1:2 split (m = 0.5 means price doubles, shares halve)
            mult = 0.5
            P_new[i] /= mult    # price doubles
            N_new[i] *= mult    # shares halve
            m_new[i] = mult

        # Guard against prices going to zero or negative
        P_new = np.maximum(P_new, 0.01)

        P_arr[t] = P_new
        N_arr[t] = N_new
        m_arr[t] = m_new

    # ------------------------------------------------------------------
    # 6. Package into DataFrames
    # ------------------------------------------------------------------
    P = pd.DataFrame(P_arr, index=trading_dates, columns=tickers)
    N = pd.DataFrame(N_arr, index=trading_dates, columns=tickers)
    m = pd.DataFrame(m_arr, index=trading_dates, columns=tickers)

    return P, N, m, trading_dates


def compute_split_statistics(
    m: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise split events across the synthetic dataset.

    Parameters
    ----------
    m : pd.DataFrame
        Split-multiplier matrix returned by :func:`generate_synthetic_nyse`.

    Returns
    -------
    pd.DataFrame
        Columns: year, n_fwd_2to1, n_fwd_3to2, n_rev, total_splits.
    """
    rows = []
    for year, grp in m.groupby(m.index.year):
        n_fwd_2 = int((grp == 2.0).values.sum())
        n_fwd_15 = int((grp == 1.5).values.sum())
        n_rev = int((grp == 0.5).values.sum())
        rows.append({
            "year": year,
            "n_fwd_2to1": n_fwd_2,
            "n_fwd_3to2": n_fwd_15,
            "n_rev": n_rev,
            "total_splits": n_fwd_2 + n_fwd_15 + n_rev,
        })
    return pd.DataFrame(rows).set_index("year")
