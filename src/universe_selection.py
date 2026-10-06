"""
universe_selection.py
=====================
Quarterly universe reconstitution for price-based SPT experiments.

Two parallel universes are maintained throughout each experiment:

    U_m^P  –  top-n stocks by *nominal closing price* on the ranking date s_m
    U_m^S  –  top-n stocks by *market capitalisation* on the ranking date s_m

where s_m is the last trading day of quarter m-1 (the day immediately before
the new universe becomes active).  The new universe is first used on hat_t_m,
the first trading day of quarter m.

This mimics the mechanics of price-weighted indices such as the DJIA (which
effectively selects high-nominal-price stocks) versus cap-weighted indices.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Quarterly calendar
# ---------------------------------------------------------------------------

def build_quarterly_reconstitution_calendar(
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Identify quarterly rebalance dates from a trading-date index.

    For each calendar quarter that appears in *trading_dates* we find:

        hat_t_m – first trading day of the quarter (portfolio goes live)
        s_m     – immediately preceding trading day (ranking / snapshot date)

    The first quarter has no s_m (ranking would fall before the sample), so it
    is excluded from the returned calendar.

    Parameters
    ----------
    trading_dates : pd.DatetimeIndex
        Ordered sequence of all trading days in the sample.

    Returns
    -------
    pd.DataFrame
        Columns: ``quarter`` (Period[Q]), ``hat_t_m``, ``s_m``.
        Indexed from 0 by integer.
    """
    dates = pd.Series(trading_dates, index=trading_dates)
    quarters = dates.dt.to_period("Q")

    # First trading day of each distinct quarter
    first_day_of_quarter = (
        dates.groupby(quarters.values).first().rename("hat_t_m")
    )

    rows = []
    for i, (q, hat_t) in enumerate(first_day_of_quarter.items()):
        # s_m is the trading day immediately before hat_t_m
        idx = trading_dates.get_loc(hat_t)
        if idx == 0:
            # No previous day available; skip the very first quarter
            continue
        s_m = trading_dates[idx - 1]
        rows.append({"quarter": q, "hat_t_m": hat_t, "s_m": s_m})

    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stock selection
# ---------------------------------------------------------------------------

def select_top_by_price(
    P: pd.DataFrame,
    date: pd.Timestamp,
    n: int = 500,
    exclude_missing: bool = True,
) -> List[str]:
    """Select top-n stocks by nominal closing price on *date*.

    Parameters
    ----------
    P : pd.DataFrame
        Prices, shape (T, n_stocks), indexed by trading dates.
    date : pd.Timestamp
        Ranking date (s_m in the quarterly calendar).
    n : int
        Number of stocks to select.
    exclude_missing : bool
        If True, drop NaN prices before ranking.

    Returns
    -------
    list of str
        Ticker symbols of the top-n stocks, ordered descending by price.
    """
    row = P.loc[date]
    if exclude_missing:
        row = row.dropna()
    # Keep only positive, finite prices
    row = row[np.isfinite(row) & (row > 0)]
    selected = row.nlargest(min(n, len(row))).index.tolist()
    return selected


def select_top_by_cap(
    S: pd.DataFrame,
    date: pd.Timestamp,
    n: int = 500,
    exclude_missing: bool = True,
) -> List[str]:
    """Select top-n stocks by market capitalisation on *date*.

    Parameters
    ----------
    S : pd.DataFrame
        Market capitalisations (price × shares), same shape as P.
    date : pd.Timestamp
        Ranking date.
    n : int
        Number of stocks to select.
    exclude_missing : bool
        Drop NaN entries before ranking.

    Returns
    -------
    list of str
        Ticker symbols of the top-n stocks, ordered descending by cap.
    """
    row = S.loc[date]
    if exclude_missing:
        row = row.dropna()
    row = row[np.isfinite(row) & (row > 0)]
    selected = row.nlargest(min(n, len(row))).index.tolist()
    return selected


# ---------------------------------------------------------------------------
# Build full quarterly universe dictionaries
# ---------------------------------------------------------------------------

def build_quarterly_universes(
    P: pd.DataFrame,
    S: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    n: int = 500,
) -> Tuple[Dict[pd.Timestamp, List[str]], Dict[pd.Timestamp, List[str]]]:
    """Build U_m^P (top by price) and U_m^S (top by cap) for all quarters.

    Each universe is indexed by hat_t_m (first trading day of the quarter)
    and contains the n stocks ranked on s_m (the last day of the previous
    quarter).

    The first quarter in the sample is used with the stocks ranked on the
    very first trading day (since there is no prior quarter end available).

    Parameters
    ----------
    P : pd.DataFrame
        Raw prices, indexed by trading dates.
    S : pd.DataFrame
        Market caps, indexed by trading dates.
    trading_dates : pd.DatetimeIndex
        Full trading-date index.
    n : int
        Universe size.

    Returns
    -------
    (universes_P, universes_S) : tuple of dicts
        Each maps {hat_t_m: [ticker, ...]} for all quarters.
    """
    calendar = build_quarterly_reconstitution_calendar(trading_dates)

    universes_P: Dict[pd.Timestamp, List[str]] = {}
    universes_S: Dict[pd.Timestamp, List[str]] = {}

    # Seed the very first quarter from the first trading date
    first_date = trading_dates[0]
    first_hat = trading_dates[0]
    universes_P[first_hat] = select_top_by_price(P, first_date, n=n)
    universes_S[first_hat] = select_top_by_cap(S, first_date, n=n)

    for _, row in calendar.iterrows():
        hat_t_m: pd.Timestamp = row["hat_t_m"]
        s_m: pd.Timestamp = row["s_m"]
        universes_P[hat_t_m] = select_top_by_price(P, s_m, n=n)
        universes_S[hat_t_m] = select_top_by_cap(S, s_m, n=n)

    return universes_P, universes_S


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_universe_diagnostics(
    universes_P: Dict[pd.Timestamp, List[str]],
    universes_S: Dict[pd.Timestamp, List[str]],
) -> pd.DataFrame:
    """Compute overlap and turnover diagnostics between consecutive quarters.

    For each quarter m > 0 the following are computed:

        overlap_PS  – |U_m^P ∩ U_m^S|  (cross-method intersection)
        turnover_P  – |U_m^P △ U_{m-1}^P| / 2  (one-sided replacements)
        turnover_S  – |U_m^S △ U_{m-1}^S| / 2  (one-sided replacements)
        jaccard_P   – Jaccard similarity of consecutive price universes
        jaccard_S   – Jaccard similarity of consecutive cap universes

    Parameters
    ----------
    universes_P : dict
        {hat_t_m: list of tickers} for price-ranked universe.
    universes_S : dict
        {hat_t_m: list of tickers} for cap-ranked universe.

    Returns
    -------
    pd.DataFrame
        One row per quarter, indexed by hat_t_m.
    """
    dates_P = sorted(universes_P.keys())
    dates_S = sorted(universes_S.keys())
    # Use the union of all rebalance dates
    all_dates = sorted(set(dates_P) | set(dates_S))

    rows = []
    for i, date in enumerate(all_dates):
        U_P = set(universes_P.get(date, []))
        U_S = set(universes_S.get(date, []))

        # Cross-method overlap for this quarter
        overlap_PS = len(U_P & U_S)

        # Consecutive-quarter turnover
        if i > 0:
            prev_date = all_dates[i - 1]
            U_P_prev = set(universes_P.get(prev_date, []))
            U_S_prev = set(universes_S.get(prev_date, []))

            # Additions to current universe from previous
            turnover_P = len(U_P - U_P_prev)   # stocks added = stocks dropped
            turnover_S = len(U_S - U_S_prev)

            # Jaccard similarity
            union_P = U_P | U_P_prev
            union_S = U_S | U_S_prev
            jaccard_P = len(U_P & U_P_prev) / len(union_P) if union_P else 1.0
            jaccard_S = len(U_S & U_S_prev) / len(union_S) if union_S else 1.0
        else:
            turnover_P = turnover_S = np.nan
            jaccard_P = jaccard_S = np.nan

        rows.append({
            "hat_t_m": date,
            "size_P": len(U_P),
            "size_S": len(U_S),
            "overlap_PS": overlap_PS,
            "turnover_P": turnover_P,
            "turnover_S": turnover_S,
            "jaccard_P": jaccard_P,
            "jaccard_S": jaccard_S,
        })

    return pd.DataFrame(rows).set_index("hat_t_m")
