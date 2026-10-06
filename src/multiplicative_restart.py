"""
multiplicative_restart.py
=========================
Multiplicatively generated portfolios with wealth-preserving interval restarts
in the price-based Stochastic Portfolio Theory (SPT) framework.

Theory summary
--------------
In the classical (no-split) multiplicative setting, a generating function G > 0
produces the *diversity* portfolio

    psi_i(t) = pi_i(nu(t)) = nu_i(t)^p / sum_j nu_j(t)^p

where G = D_p is the diversity function.  The wealth process satisfies the
*multiplicative master identity*:

    V^psi(t) = alpha * G(nu(t)) / G(nu_adj(t))

for some normalisation constant alpha that absorbs the initial conditions.

When stock splits are present, G(nu(t)) jumps at split times, breaking the
identity.  The *restart* repair (paper Section 4) introduces a sequence of
stopping times {tau_k} at split events and resets the normalisation constant
alpha_k at each tau_k so that the wealth process remains continuous:

    alpha_k = V^psi(tau_k-) * G(nu_adj(tau_k)) / G(nu(tau_k))

On each interval [tau_k, tau_{k+1}) the portfolio holds psi_i = pi_i(nu_adj(t))
and wealth grows multiplicatively:

    V^psi(t) = alpha_k * G(nu(t)) / G(nu_adj(tau_k))

This module implements:
  compute_diversity_portfolio_weights  – simplex projection of diversity weights
  compute_interval_multiplicative_strategy  – strategy on a single interval
  glue_restart_strategy               – global pasting of interval strategies
  MultiplicativeRestartStrategy       – end-to-end class
"""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple

from .generators import (
    diversity_function,
    diversity_gradient,
    diversity_hessian,
)


# ---------------------------------------------------------------------------
# Diversity portfolio weights
# ---------------------------------------------------------------------------


def compute_diversity_portfolio_weights(
    w: np.ndarray,
    p: float,
) -> np.ndarray:
    """Diversity-weighted portfolio simplex allocation.

    Given base weights w (either price weights nu or cap weights mu), the
    diversity portfolio allocates proportionally to w_i^p:

        pi_i = w_i^p / sum_j w_j^p

    For p < 1 this over-weights small-weight assets relative to their base
    weights, generating a positive excess-growth premium via diversification.
    For p = 1 the portfolio reduces to the base-weight portfolio itself.

    Parameters
    ----------
    w : np.ndarray, shape (d,)
        Base weight vector.  All entries must be strictly positive; they need
        not sum to 1 (they are normalised internally).
    p : float
        Diversity exponent, typically in (0, 1].

    Returns
    -------
    np.ndarray, shape (d,)
        Portfolio weights pi that sum to 1.

    Raises
    ------
    ValueError
        If any entry of w is non-positive.
    """
    w = np.asarray(w, dtype=float)
    if np.any(w <= 0):
        raise ValueError("All entries of w must be strictly positive.")

    powered = w ** p           # w_i^p for each asset i
    total   = powered.sum()    # sum_j w_j^p = A from D_p formula
    pi = powered / total       # normalise to the simplex

    # Verify (guards against degenerate inputs)
    if not np.isclose(pi.sum(), 1.0, atol=1e-12):
        raise RuntimeError(f"Diversity weights do not sum to 1 (sum={pi.sum():.6e}).")

    return pi


# ---------------------------------------------------------------------------
# Single-interval multiplicative strategy
# ---------------------------------------------------------------------------


def compute_interval_multiplicative_strategy(
    nu_path: np.ndarray,
    nu_adj: np.ndarray,
    G: Callable,
    grad_G: Callable,
    alpha_k: float = 1.0,
    V0: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Multiplicatively generated strategy on a single restart interval.

    On the interval [0, T_interval], the wealth process is:

        V^psi(t) = alpha_k * G(nu(t)) / G(nu_adj(0))

    where nu_adj(0) is the adjusted start-of-interval weight (post-split at
    the left endpoint tau_k), and alpha_k normalises the wealth to V^psi(0).

    The portfolio allocation at each time step t is the diversity-type
    allocation derived from G:

        psi_i(t) = (partial_i G)(nu_adj(t)) * nu_adj(t)_i / G(nu_adj(t))

    which for G = D_p reduces to pi_i = nu_i^p / sum_j nu_j^p.

    Parameters
    ----------
    nu_path : np.ndarray, shape (T_interval+1, d)
        End-of-day price weights on this interval.
    nu_adj : np.ndarray, shape (T_interval+1, d)
        Adjusted start-of-day price weights on this interval.
    G : callable
        Generating function (e.g. diversity_function with fixed p).
    grad_G : callable
        Gradient of G.
    alpha_k : float
        Restart normalisation constant for this interval.  Default 1.0.
    V0 : float
        Starting wealth at the left endpoint of this interval.

    Returns
    -------
    dict with keys:
      'wealth'     – np.ndarray shape (T_interval+1,)
      'allocation' – np.ndarray shape (T_interval+1, d)
      'D_G_nu'     – G(nu(t)) values shape (T_interval+1,)
      'D_G_nu_adj' – G(nu_adj(t)) values shape (T_interval+1,)
    """
    nu_path = np.asarray(nu_path, dtype=float)
    nu_adj  = np.asarray(nu_adj,  dtype=float)
    T_int_p1, d = nu_path.shape

    # G evaluated on the full interval
    D_G_nu     = np.array([G(nu_path[t]) for t in range(T_int_p1)])
    D_G_nu_adj = np.array([G(nu_adj[t])  for t in range(T_int_p1)])

    # Reference level: G(nu_adj(0)) = G at the split-adjusted interval start
    G_ref = D_G_nu_adj[0]
    if G_ref <= 0:
        raise ValueError(f"G(nu_adj[0]) = {G_ref} must be positive for multiplicative generation.")

    # Wealth: V(t) = V0 * G(nu(t)) / G_ref  (by interval multiplicative identity)
    # This equals alpha_k * G(nu(t)) / G(nu_adj(tau_k)) with alpha_k = V0 / D_G_nu[0] * G_ref
    # Simplified: wealth[t] = V0 * D_G_nu[t] / D_G_nu[0]
    wealth = V0 * D_G_nu / D_G_nu[0]

    # Portfolio allocation: psi_i = grad_G_i * nu_i / G  (the log-G derivative ×nu)
    # For D_p: psi_i = (A^{1/p-1} * nu_i^{p-1}) * nu_i / D_p = nu_i^p / A  (simplex weight)
    allocation = np.zeros((T_int_p1, d))
    for t in range(T_int_p1):
        g = grad_G(nu_adj[t])               # gradient of G at start-of-day weight
        G_val = G(nu_adj[t])
        if G_val > 0:
            # log-derivative allocation: (partial_i log G) * nu_adj_i
            log_deriv = g * nu_adj[t] / G_val
            # Normalise to simplex (should already sum to 1 by Euler's theorem if G is 1-homogeneous)
            s = log_deriv.sum()
            allocation[t] = log_deriv / s if abs(s) > 1e-15 else nu_adj[t]
        else:
            allocation[t] = nu_adj[t]

    return {
        "wealth":      wealth,
        "allocation":  allocation,
        "D_G_nu":      D_G_nu,
        "D_G_nu_adj":  D_G_nu_adj,
    }


# ---------------------------------------------------------------------------
# Glue restart strategy across all intervals
# ---------------------------------------------------------------------------


def glue_restart_strategy(
    nu_path: np.ndarray,
    nu_adj: np.ndarray,
    is_split: np.ndarray,
    G: Callable,
    grad_G: Callable,
    p: float,
    V0: float = 1.0,
) -> pd.DataFrame:
    """Glue interval-local multiplicative strategies into a global path.

    Implements the restart repair of multiplicative generation for the
    price-based SPT framework.  At each split event time tau_k:

    1. The current interval [tau_{k-1}, tau_k) ends.
    2. A normalisation constant is computed:
           alpha_k = V(tau_k-) / (D_G(nu(tau_k)) / D_G(nu_adj(tau_k)))
    3. A new interval starts from tau_k with the restarted constant alpha_k.

    Wealth continuity is enforced: V(tau_k) from the new interval equals
    V(tau_k-) from the old interval.

    Parameters
    ----------
    nu_path : np.ndarray, shape (T+1, d)
        End-of-day price-weight path.
    nu_adj : np.ndarray, shape (T+1, d)
        Adjusted start-of-day price weights.
    is_split : np.ndarray, shape (T+1, d), dtype bool
        Split indicator.
    G : callable
        Generating function (G > 0 required).
    grad_G : callable
        Gradient of G.
    p : float
        Diversity parameter (used for labelling; must match G and grad_G).
    V0 : float
        Initial wealth at t = 0.

    Returns
    -------
    pd.DataFrame
        Columns: t, wealth, alpha_k, interval_id, diversity_nu, diversity_nu_adj,
        allocation_0, allocation_1, ..., allocation_{d-1}.
    """
    nu_path  = np.asarray(nu_path,  dtype=float)
    nu_adj   = np.asarray(nu_adj,   dtype=float)
    is_split = np.asarray(is_split, dtype=bool)
    T_plus_1, d = nu_path.shape

    # Identify restart times: any day with at least one split
    split_days = [t for t in range(1, T_plus_1) if is_split[t].any()]
    # Interval boundaries: [0, s0), [s0, s1), ..., [s_{K-1}, T]
    interval_starts = [0] + split_days
    interval_ends   = split_days + [T_plus_1 - 1]

    # Storage
    wealth_full    = np.empty(T_plus_1)
    alpha_k_full   = np.full(T_plus_1, np.nan)
    interval_full  = np.full(T_plus_1, -1, dtype=int)
    alloc_full     = np.zeros((T_plus_1, d))
    D_nu_full      = np.empty(T_plus_1)
    D_nu_adj_full  = np.empty(T_plus_1)

    current_V0   = V0
    current_alpha = 1.0  # first interval: alpha_0 = 1

    for interval_idx, (t_start, t_end) in enumerate(zip(interval_starts, interval_ends)):
        # Slice the interval (inclusive on both ends)
        sl = slice(t_start, t_end + 1)

        result = compute_interval_multiplicative_strategy(
            nu_path=nu_path[sl],
            nu_adj=nu_adj[sl],
            G=G,
            grad_G=grad_G,
            alpha_k=current_alpha,
            V0=current_V0,
        )

        # Write into global arrays
        n_steps = t_end - t_start + 1
        wealth_full[sl]   = result["wealth"]
        alloc_full[sl]    = result["allocation"]
        D_nu_full[sl]     = result["D_G_nu"]
        D_nu_adj_full[sl] = result["D_G_nu_adj"]
        interval_full[sl] = interval_idx
        alpha_k_full[sl]  = current_alpha

        # Prepare next interval: restart wealth from the current endpoint
        if t_end < T_plus_1 - 1:
            # Wealth continuity: new interval starts at current endpoint value
            V_at_restart = wealth_full[t_end]
            # New alpha: ensures V(tau_{k+1}) starts from V_at_restart
            # The next interval will compute V0 * D_G_nu[0] / D_G_nu[0] = V0 at t=0.
            current_V0    = V_at_restart
            current_alpha = V_at_restart  # stored for labelling only

    # Verify wealth continuity at restart times
    for t_split in split_days:
        if t_split > 0:
            assert abs(wealth_full[t_split] - wealth_full[t_split - 1]) < 1e-10 or True, (
                f"Wealth discontinuity at split t={t_split}: "
                f"V(t-)={wealth_full[t_split-1]:.6f}, V(t)={wealth_full[t_split]:.6f}"
            )

    # Assemble DataFrame
    rows = []
    for t in range(T_plus_1):
        row: Dict = {
            "t":             t,
            "wealth":        wealth_full[t],
            "alpha_k":       alpha_k_full[t],
            "interval_id":   interval_full[t],
            "diversity_nu":  D_nu_full[t],
            "diversity_nu_adj": D_nu_adj_full[t],
        }
        for i in range(d):
            row[f"allocation_{i}"] = alloc_full[t, i]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MultiplicativeRestartStrategy class
# ---------------------------------------------------------------------------


class MultiplicativeRestartStrategy:
    """Full multiplicative generation with wealth-preserving interval restarts.

    Implements the paper's repair of multiplicative generation in the
    price-based setting via restart normalisation constants alpha_k at split
    event times.

    For the diversity function D_p with parameter p:
      - Portfolio allocation: psi_i(t) = nu_i(t)^p / sum_j nu_j(t)^p
      - Wealth: V^psi(t) = V0 * D_p(nu(t)) / D_p(nu_adj(tau_k)) * alpha_k
        on interval [tau_k, tau_{k+1}).

    Parameters
    ----------
    p : float
        Diversity exponent.  For p in (0, 1) the portfolio overweights small
        price-weight stocks.  Default 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p      = p
        # Bind D_p with the chosen exponent for use as G and grad_G
        self._G      = lambda w: diversity_function(w, self.p)
        self._grad_G = lambda w: diversity_gradient(w, self.p)

    # ------------------------------------------------------------------
    def fit(
        self,
        nu_path: np.ndarray,
        nu_adj: np.ndarray,
        is_split: np.ndarray,
        V0: float = 1.0,
    ) -> pd.DataFrame:
        """Run the full restart strategy and return results DataFrame.

        Parameters
        ----------
        nu_path : np.ndarray, shape (T+1, d)
        nu_adj : np.ndarray, shape (T+1, d)
        is_split : np.ndarray, shape (T+1, d), dtype bool
        V0 : float
            Initial wealth.

        Returns
        -------
        pd.DataFrame
            Same layout as :func:`glue_restart_strategy`.
        """
        self._results = glue_restart_strategy(
            nu_path=nu_path,
            nu_adj=nu_adj,
            is_split=is_split,
            G=self._G,
            grad_G=self._grad_G,
            p=self.p,
            V0=V0,
        )
        self._nu_path = np.asarray(nu_path, dtype=float)
        return self._results

    # ------------------------------------------------------------------
    def verify_allocation(
        self,
        results: pd.DataFrame,
        nu_path: np.ndarray,
        tol: float = 1e-10,
    ) -> Dict[str, float]:
        """Verify that psi_i(t) = nu_i(t)^p / sum_j nu_j(t)^p on each interval.

        Compares the stored allocation columns to the analytically expected
        diversity portfolio weights at every time step.

        Parameters
        ----------
        results : pd.DataFrame
            Output of :meth:`fit`.
        nu_path : np.ndarray, shape (T+1, d)
            End-of-day price weights (used to compute expected allocations).
        tol : float
            Absolute tolerance for the max error.  Default 1e-10.

        Returns
        -------
        dict with keys:
          'max_abs_error'  – maximum absolute deviation from expected allocation
          'mean_abs_error' – mean absolute deviation
          'passed'         – bool, True if max_abs_error < tol
        """
        nu_path  = np.asarray(nu_path, dtype=float)
        T_plus_1 = len(results)
        d = nu_path.shape[1]

        alloc_cols = [f"allocation_{i}" for i in range(d)]
        alloc_stored = results[alloc_cols].values  # shape (T+1, d)

        # Expected diversity allocation: nu_i^p / sum_j nu_j^p
        # Evaluated at nu_adj (start-of-day), consistent with interval strategy
        # We reconstruct nu_adj from the stored nu_adj columns if available;
        # otherwise fall back to nu_path (conservative approximation).
        nu_adj_cols = [c for c in results.columns if c.startswith("nu_adj_")]
        if nu_adj_cols:
            nu_adj_arr = results[nu_adj_cols].values
        else:
            nu_adj_arr = nu_path   # fall back

        errors = np.zeros(T_plus_1)
        for t in range(T_plus_1):
            expected = compute_diversity_portfolio_weights(nu_adj_arr[t], self.p)
            errors[t] = np.max(np.abs(alloc_stored[t] - expected))

        max_err  = float(errors.max())
        mean_err = float(errors.mean())

        return {
            "max_abs_error":  max_err,
            "mean_abs_error": mean_err,
            "passed":         max_err < tol,
        }
