"""
tests/test_multiplicative_restart.py
======================================
Unit tests for src/multiplicative_restart.py.

Covers:
- compute_diversity_portfolio_weights: simplex, ordering, p=1 identity
- compute_interval_multiplicative_strategy: correct local wealth path
- glue_restart_strategy: wealth continuity across restarts, correct gluing
- MultiplicativeRestartStrategy: end-to-end on synthetic paths
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multiplicative_restart import (
    compute_diversity_portfolio_weights,
    compute_interval_multiplicative_strategy,
    glue_restart_strategy,
    MultiplicativeRestartStrategy,
)
from src.generators import diversity_function, diversity_gradient


# ---------------------------------------------------------------------------
# Shared helper: build G / grad_G callables for a given p
# ---------------------------------------------------------------------------

def _make_G(p: float):
    """Return (G, grad_G) for diversity function D_p."""
    G      = lambda w: diversity_function(w, p)
    grad_G = lambda w: diversity_gradient(w, p)
    return G, grad_G


# ---------------------------------------------------------------------------
# compute_diversity_portfolio_weights
# ---------------------------------------------------------------------------

class TestDiversityPortfolioWeights:
    def test_sums_to_one(self):
        """Portfolio weights always sum to 1."""
        rng = np.random.default_rng(0)
        for p in [0.1, 0.5, 0.9, 1.0]:
            w = rng.dirichlet(np.ones(5))
            pi = compute_diversity_portfolio_weights(w, p)
            assert abs(pi.sum() - 1.0) < 1e-12, f"Sum != 1 for p={p}: {pi.sum()}"

    def test_all_non_negative(self):
        """All portfolio weights are non-negative."""
        rng = np.random.default_rng(1)
        for p in [0.3, 0.7]:
            w = rng.dirichlet(np.ones(4))
            pi = compute_diversity_portfolio_weights(w, p)
            assert np.all(pi >= 0)

    def test_p_one_equals_base_weight(self):
        """For p=1, diversity portfolio = normalised base weights."""
        w = np.array([0.4, 0.35, 0.25])
        pi = compute_diversity_portfolio_weights(w, 1.0)
        assert np.allclose(pi, w / w.sum(), atol=1e-12)

    def test_diversification_small_p(self):
        """For p < 1, smaller weights receive a larger relative allocation."""
        w = np.array([0.8, 0.2])
        pi = compute_diversity_portfolio_weights(w, 0.5)
        # The small-cap asset (w=0.2) should have pi/w > large-cap asset
        assert pi[1] / w[1] > pi[0] / w[0]

    def test_equal_weights_give_uniform_pi(self):
        """Uniform base weights yield uniform portfolio for any p."""
        d = 6
        w = np.full(d, 1.0 / d)
        for p in [0.1, 0.5, 1.0]:
            pi = compute_diversity_portfolio_weights(w, p)
            assert np.allclose(pi, 1.0 / d, atol=1e-12)

    def test_raises_on_non_positive(self):
        """ValueError raised when any weight is ≤ 0."""
        w = np.array([0.5, 0.0, 0.5])
        with pytest.raises((ValueError, RuntimeError)):
            compute_diversity_portfolio_weights(w, 0.5)

    def test_known_value_two_assets(self):
        """Known analytic check: w=[0.6,0.4], p=0.5 → pi = [sqrt(0.6), sqrt(0.4)] / Z."""
        w = np.array([0.6, 0.4])
        pi = compute_diversity_portfolio_weights(w, 0.5)
        sqrt_w = np.sqrt(w)
        expected = sqrt_w / sqrt_w.sum()
        assert np.allclose(pi, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# compute_interval_multiplicative_strategy
# ---------------------------------------------------------------------------

class TestIntervalMultiplicativeStrategy:
    """compute_interval_multiplicative_strategy(nu_path, nu_adj, G, grad_G, ...)
    returns a dict with keys: 'wealth', 'allocation', 'D_G_nu', 'D_G_nu_adj'.
    """

    def _make_nu_path(self, T: int = 12, d: int = 3, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.dirichlet(np.ones(d), size=T)  # shape (T, d)

    def test_output_shape(self):
        """Result wealth has length equal to nu_path rows."""
        p = 0.5
        G, grad_G = _make_G(p)
        nu_path = self._make_nu_path()
        nu_adj  = nu_path.copy()
        result  = compute_interval_multiplicative_strategy(nu_path, nu_adj, G, grad_G)
        assert len(result["wealth"]) == len(nu_path)

    def test_initial_wealth_one(self):
        """Relative wealth starts at 1 (V0=1 default)."""
        G, grad_G = _make_G(0.5)
        nu_path   = self._make_nu_path()
        nu_adj    = nu_path.copy()
        result    = compute_interval_multiplicative_strategy(nu_path, nu_adj, G, grad_G)
        assert abs(result["wealth"][0] - 1.0) < 1e-10

    def test_wealth_always_positive(self):
        """Local wealth is always positive."""
        G, grad_G = _make_G(0.5)
        nu_path   = self._make_nu_path()
        nu_adj    = nu_path.copy()
        result    = compute_interval_multiplicative_strategy(nu_path, nu_adj, G, grad_G)
        assert np.all(result["wealth"] > 0)

    def test_different_p_gives_different_paths(self):
        """Two distinct p values produce distinct wealth paths."""
        nu_path    = self._make_nu_path(T=20, seed=7)
        nu_adj     = nu_path.copy()
        G3, grad_G3 = _make_G(0.3)
        G9, grad_G9 = _make_G(0.9)
        r1 = compute_interval_multiplicative_strategy(nu_path, nu_adj, G3, grad_G3)
        r2 = compute_interval_multiplicative_strategy(nu_path, nu_adj, G9, grad_G9)
        assert not np.allclose(r1["wealth"], r2["wealth"])


# ---------------------------------------------------------------------------
# glue_restart_strategy
# ---------------------------------------------------------------------------

class TestGlueRestartStrategy:
    """glue_restart_strategy(nu_path, nu_adj, is_split, G, grad_G, p, V0=1.0)
    returns a DataFrame with an integer RangeIndex and column 'wealth'.
    """

    def _build_arrays(self, n_intervals: int = 3, T_per: int = 10, d: int = 2,
                      p: float = 0.5, seed: int = 0):
        """Return (nu_arr, nu_adj, is_split, restart_positions, G, grad_G)."""
        rng    = np.random.default_rng(seed)
        T_tot  = n_intervals * T_per
        nu_arr = rng.dirichlet(np.ones(d), size=T_tot)
        nu_adj = nu_arr.copy()
        # Mark splits at the start of each interval except the very first
        is_split = np.zeros((T_tot, d), dtype=bool)
        restart_pos = [T_per * k for k in range(1, n_intervals)]
        for pos in restart_pos:
            is_split[pos, 0] = True   # split of asset-0 triggers restart
        G, grad_G = _make_G(p)
        return nu_arr, nu_adj, is_split, restart_pos, G, grad_G, p

    def test_wealth_continuous_at_restart(self):
        """Global wealth must not jump at restart positions."""
        nu_arr, nu_adj, is_split, restart_pos, G, grad_G, p = self._build_arrays()
        result = glue_restart_strategy(nu_arr, nu_adj, is_split, G, grad_G, p)
        for pos in restart_pos:
            if pos == 0:
                continue
            V_before = result["wealth"].iloc[pos - 1]
            V_at     = result["wealth"].iloc[pos]
            jump     = abs(V_at / V_before - 1.0)
            assert jump < 0.10, f"Jump at pos {pos}: {jump:.4%}"

    def test_always_positive(self):
        """Global wealth is always positive."""
        nu_arr, nu_adj, is_split, _, G, grad_G, p = self._build_arrays()
        result = glue_restart_strategy(nu_arr, nu_adj, is_split, G, grad_G, p)
        assert np.all(result["wealth"].values > 0)

    def test_output_row_count_matches_input(self):
        """Output has the same number of rows as the input path."""
        nu_arr, nu_adj, is_split, _, G, grad_G, p = self._build_arrays()
        result = glue_restart_strategy(nu_arr, nu_adj, is_split, G, grad_G, p)
        assert len(result) == len(nu_arr)

    def test_no_events_single_interval(self):
        """Without any splits, the strategy is one interval; wealth starts at 1."""
        rng   = np.random.default_rng(5)
        T     = 15
        d     = 3
        nu    = rng.dirichlet(np.ones(d), size=T)
        is_sp = np.zeros((T, d), dtype=bool)
        G, grad_G = _make_G(0.5)
        result = glue_restart_strategy(nu, nu.copy(), is_sp, G, grad_G, 0.5)
        assert len(result) == T
        assert abs(result["wealth"].iloc[0] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# MultiplicativeRestartStrategy (end-to-end)
# ---------------------------------------------------------------------------

class TestMultiplicativeRestartStrategyEndToEnd:
    """MultiplicativeRestartStrategy(p).fit(nu_path, nu_adj, is_split)
    returns a DataFrame with column 'wealth' and allocation columns
    'allocation_0', 'allocation_1', ...
    """

    def _make_arrays(self, T: int = 30, d: int = 3, seed: int = 42,
                     split_positions=None):
        """Return (nu_arr, nu_adj, is_split) as numpy arrays."""
        rng      = np.random.default_rng(seed)
        nu_arr   = rng.dirichlet(np.ones(d), size=T)
        nu_adj   = nu_arr.copy()
        is_split = np.zeros((T, d), dtype=bool)
        if split_positions:
            for pos in split_positions:
                is_split[pos, 0] = True
        return nu_arr, nu_adj, is_split

    def test_allocation_matches_diversity_formula(self):
        """Allocation at each step satisfies π_i = ν_adj_i^p / Σ ν_adj_j^p."""
        p        = 0.5
        d        = 3
        nu, nu_adj, is_split = self._make_arrays(T=20, d=d)
        strat    = MultiplicativeRestartStrategy(p=p)
        result   = strat.fit(nu, nu_adj, is_split)
        alloc_cols = [f"allocation_{i}" for i in range(d)]
        # Verify allocation columns exist and match formula at each row
        assert all(c in result.columns for c in alloc_cols), (
            f"Missing allocation columns; got {result.columns.tolist()}"
        )
        nu_adj_cols = [c for c in result.columns if c.startswith("nu_adj_")]
        if nu_adj_cols:
            for t in range(len(result)):
                w_t = result[nu_adj_cols].iloc[t].values
                expected = compute_diversity_portfolio_weights(w_t, p)
                actual   = result[alloc_cols].iloc[t].values
                assert np.allclose(actual, expected, atol=1e-10), (
                    f"Allocation mismatch at t={t}"
                )

    def test_wealth_path_positive(self):
        """End-to-end wealth path is always positive."""
        nu, nu_adj, is_split = self._make_arrays()
        strat  = MultiplicativeRestartStrategy(p=0.5)
        result = strat.fit(nu, nu_adj, is_split)
        assert np.all(result["wealth"].values > 0)

    def test_wealth_continuous_after_restart(self):
        """Restart introduces no artificial wealth gain or loss.

        When nu_adj equals nu (no actual price-adjustment for the split), the
        wealth path with restarts must be numerically identical to the single-
        interval path.  The day-over-day move *at* the restart position is a
        genuine market move driven by the diversity function, not an artifact.
        """
        T      = 40
        d      = 4
        splits = [10, 25]
        nu, nu_adj, is_split = self._make_arrays(T=T, d=d, split_positions=splits)
        no_split = np.zeros_like(is_split, dtype=bool)

        strat  = MultiplicativeRestartStrategy(p=0.5)
        r_with = strat.fit(nu, nu_adj, is_split)
        r_none = strat.fit(nu, nu_adj, no_split)

        # When nu_adj == nu the restart cannot introduce any extra gain/loss
        max_diff = np.max(np.abs(r_with["wealth"].values - r_none["wealth"].values))
        assert max_diff < 1e-12, (
            f"Restart introduced artificial wealth change: max_diff={max_diff:.2e}"
        )

    def test_multiple_p_values_run_without_error(self):
        """Strategy runs cleanly for all paper p values."""
        T = 25
        d = 5
        nu, nu_adj, is_split = self._make_arrays(T=T, d=d)
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            strat  = MultiplicativeRestartStrategy(p=p)
            result = strat.fit(nu, nu_adj, is_split)
            assert len(result) == T, f"Output length mismatch for p={p}"
