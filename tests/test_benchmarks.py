"""
tests/test_benchmarks.py
=========================
Unit tests for src/benchmarks.py.

Covers:
- build_cap_weighted_benchmark / build_price_weighted_benchmark
- compute_benchmark_decomposition: exact identities D = W^P+C^mu = W^S+C^nu
- compute_gibbs_entropy: correct values, log(n) reference
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmarks import (
    build_cap_weighted_benchmark,
    build_price_weighted_benchmark,
    compute_benchmark_decomposition,
    compute_gibbs_entropy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_universe(T: int = 22, seed: int = 0):
    """
    Return (P, N, m, quarterly_universes, trading_dates) for 3 assets.

    One 2-for-1 split on asset 'A' at index t=10.
    All assets are active in a single 'quarter' covering the full period.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=T, freq="B")
    assets = ["A", "B", "C"]

    P_vals = np.ones((T, 3)) * np.array([100.0, 50.0, 200.0])
    P_vals += rng.standard_normal((T, 3)).cumsum(axis=0) * 0.5
    P_vals = np.clip(P_vals, 5.0, None)
    N_vals = np.ones((T, 3)) * np.array([1_000.0, 2_000.0, 500.0])

    # Pure 2-for-1 split on asset A at t=10
    P_vals[10:, 0] = P_vals[9, 0] / 2.0
    N_vals[10:, 0] = N_vals[9, 0] * 2.0

    P = pd.DataFrame(P_vals, index=idx, columns=assets)
    N = pd.DataFrame(N_vals, index=idx, columns=assets)

    # Compute split multipliers via heuristic (τ = 0.15)
    m = pd.DataFrame(1.0, index=idx, columns=assets)
    for t in range(1, T):
        ratio_N = N.iloc[t] / N.iloc[t - 1]
        S = P * N
        ratio_S = S.iloc[t] / S.iloc[t - 1]
        for c in assets:
            if abs(ratio_N[c] - 1) > 0.15 and abs(ratio_S[c] - 1) <= 0.15:
                m.at[idx[t], c] = float(ratio_N[c])

    # Single quarter: reconstitution at first date, active all T days
    quarterly_universes = {idx[0]: assets}
    trading_dates = idx
    return P, N, m, quarterly_universes, trading_dates


def _log_level(I: pd.Series) -> pd.Series:
    """Compute log(I/I_0) for a positive benchmark series."""
    return np.log(I / I.iloc[0])


# ---------------------------------------------------------------------------
# build_cap_weighted_benchmark
# ---------------------------------------------------------------------------

class TestCapWeightedBenchmark:
    def test_starts_at_one(self):
        """Cap-weighted benchmark starts at 1."""
        P, N, m, qu, td = _make_minimal_universe()
        I_mu = build_cap_weighted_benchmark(P, N, m, qu, td)
        assert abs(I_mu.iloc[0] - 1.0) < 1e-12

    def test_always_positive(self):
        """Cap-weighted benchmark is always positive."""
        P, N, m, qu, td = _make_minimal_universe()
        I_mu = build_cap_weighted_benchmark(P, N, m, qu, td)
        assert np.all(I_mu.values > 0)

    def test_length_matches_trading_dates(self):
        """Output length matches number of trading dates."""
        P, N, m, qu, td = _make_minimal_universe(T=30)
        I_mu = build_cap_weighted_benchmark(P, N, m, qu, td, I0=1.0)
        assert len(I_mu) == 30

    def test_returns_series(self):
        """Output is a pandas Series."""
        P, N, m, qu, td = _make_minimal_universe()
        I_mu = build_cap_weighted_benchmark(P, N, m, qu, td)
        assert isinstance(I_mu, pd.Series)

    def test_split_continuity(self):
        """Cap-weighted benchmark should not jump mechanically at a pure split."""
        P, N, m, qu, td = _make_minimal_universe()
        I_mu = build_cap_weighted_benchmark(P, N, m, qu, td)
        # Capitalization unchanged at split → benchmark gross return near 1
        jump = abs(I_mu.iloc[10] / I_mu.iloc[9] - 1.0)
        assert jump < 0.05, f"Unexpected cap-benchmark jump at split: {jump:.3%}"


# ---------------------------------------------------------------------------
# build_price_weighted_benchmark
# ---------------------------------------------------------------------------

class TestPriceWeightedBenchmark:
    def test_returns_tuple(self):
        """build_price_weighted_benchmark returns (I_nu, kappa) tuple."""
        P, N, m, qu, td = _make_minimal_universe()
        result = build_price_weighted_benchmark(P, N, m, qu, td)
        assert isinstance(result, tuple) and len(result) == 2

    def test_starts_at_one(self):
        """Price-weighted benchmark starts at 1, kappa starts at 1."""
        P, N, m, qu, td = _make_minimal_universe()
        I_nu, kappa = build_price_weighted_benchmark(P, N, m, qu, td)
        assert abs(I_nu.iloc[0] - 1.0) < 1e-12
        assert abs(kappa.iloc[0] - 1.0) < 1e-12

    def test_always_positive(self):
        """Both I_nu and kappa are always positive."""
        P, N, m, qu, td = _make_minimal_universe()
        I_nu, kappa = build_price_weighted_benchmark(P, N, m, qu, td)
        assert np.all(I_nu.values > 0)
        assert np.all(kappa.values > 0)

    def test_split_continuity_I_nu(self):
        """Price-weighted benchmark does not jump at a pure split day."""
        P, N, m, qu, td = _make_minimal_universe()
        I_nu, _ = build_price_weighted_benchmark(P, N, m, qu, td)
        jump = abs(I_nu.iloc[10] / I_nu.iloc[9] - 1.0)
        assert jump < 0.05, f"Large jump in I_nu at split: {jump:.3%}"

    def test_kappa_decreases_at_forward_split(self):
        """Divisor κ decreases (or stays flat) at a forward split."""
        P, N, m, qu, td = _make_minimal_universe()
        _, kappa = build_price_weighted_benchmark(P, N, m, qu, td)
        # At a 2-for-1 split, total price halves for that asset → kappa drops
        assert kappa.iloc[10] <= kappa.iloc[9] + 1e-10, (
            f"kappa did not decrease at split: {kappa.iloc[9]:.4f} → {kappa.iloc[10]:.4f}"
        )

    def test_length_matches(self):
        """Output series match trading_dates length."""
        P, N, m, qu, td = _make_minimal_universe(T=25)
        I_nu, kappa = build_price_weighted_benchmark(P, N, m, qu, td)
        assert len(I_nu) == 25 and len(kappa) == 25


# ---------------------------------------------------------------------------
# compute_benchmark_decomposition
# ---------------------------------------------------------------------------

class TestBenchmarkDecomposition:
    def _get_log_levels(self, seed: int = 0):
        """Build two benchmark sets and return their log levels."""
        P, N, m, qu, td = _make_minimal_universe(T=25, seed=seed)
        I_mu_P = build_cap_weighted_benchmark(P, N, m, qu, td)
        I_nu_P, _ = build_price_weighted_benchmark(P, N, m, qu, td)
        # "S" universe: use same data for simplicity (focus is on identity algebra)
        I_mu_S = build_cap_weighted_benchmark(P, N, m, qu, td)
        I_nu_S, _ = build_price_weighted_benchmark(P, N, m, qu, td)
        L_mu_P = _log_level(I_mu_P)
        L_mu_S = _log_level(I_mu_S)
        L_nu_P = _log_level(I_nu_P)
        L_nu_S = _log_level(I_nu_S)
        return L_mu_P, L_mu_S, L_nu_P, L_nu_S

    def test_returns_dataframe(self):
        """Decomposition returns a DataFrame."""
        L_mu_P, L_mu_S, L_nu_P, L_nu_S = self._get_log_levels()
        result = compute_benchmark_decomposition(L_mu_P, L_mu_S, L_nu_P, L_nu_S)
        assert isinstance(result, pd.DataFrame)

    def test_D_definition(self):
        """D = L_nu_P - L_mu_S."""
        L_mu_P, L_mu_S, L_nu_P, L_nu_S = self._get_log_levels()
        result = compute_benchmark_decomposition(L_mu_P, L_mu_S, L_nu_P, L_nu_S)
        expected = L_nu_P.values - L_mu_S.values
        assert np.allclose(result["D"].values, expected, atol=1e-13)

    def test_identity_D_equals_WP_plus_Cmu(self):
        """D = W_P + C_mu holds pointwise (numerical identity)."""
        L_mu_P, L_mu_S, L_nu_P, L_nu_S = self._get_log_levels()
        df = compute_benchmark_decomposition(L_mu_P, L_mu_S, L_nu_P, L_nu_S)
        err = np.abs(df["D"].values - (df["W_P"].values + df["C_mu"].values))
        assert err.max() < 1e-12, f"D ≠ W_P + C_mu; max err = {err.max():.2e}"

    def test_identity_D_equals_WS_plus_Cnu(self):
        """D = W_S + C_nu holds pointwise."""
        L_mu_P, L_mu_S, L_nu_P, L_nu_S = self._get_log_levels()
        df = compute_benchmark_decomposition(L_mu_P, L_mu_S, L_nu_P, L_nu_S)
        err = np.abs(df["D"].values - (df["W_S"].values + df["C_nu"].values))
        assert err.max() < 1e-12, f"D ≠ W_S + C_nu; max err = {err.max():.2e}"

    def test_check_columns_present(self):
        """Output contains expected decomposition columns."""
        L_mu_P, L_mu_S, L_nu_P, L_nu_S = self._get_log_levels()
        df = compute_benchmark_decomposition(L_mu_P, L_mu_S, L_nu_P, L_nu_S)
        for col in ("D", "W_P", "W_S", "C_mu", "C_nu"):
            assert col in df.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# compute_gibbs_entropy
# ---------------------------------------------------------------------------

class TestGibbsEntropy:
    def test_uniform_weights_max_entropy(self):
        """Uniform weights on d assets → H = log(d)."""
        d = 5
        idx = pd.date_range("2020-01-02", periods=3, freq="B")
        w = pd.DataFrame(np.full((3, d), 1.0 / d), index=idx)
        H = compute_gibbs_entropy(w)
        assert np.allclose(H.values, np.log(d), atol=1e-11)

    def test_degenerate_weights_zero_entropy(self):
        """Near-point-mass weight → H ≈ 0."""
        idx = pd.date_range("2020-01-02", periods=3, freq="B")
        # clip at 1e-15 avoids exact zero; entropy should be tiny
        w = pd.DataFrame([[1.0 - 1e-14, 1e-14], [1.0 - 1e-14, 1e-14],
                          [1.0 - 1e-14, 1e-14]], index=idx)
        H = compute_gibbs_entropy(w)
        assert np.all(H.values <= 1e-10)

    def test_non_negative(self):
        """H ≥ 0 for any valid weight matrix."""
        rng = np.random.default_rng(17)
        idx = pd.date_range("2020-01-02", periods=20, freq="B")
        w = pd.DataFrame(rng.dirichlet(np.ones(4), size=20), index=idx)
        H = compute_gibbs_entropy(w)
        assert np.all(H.values >= -1e-12)

    def test_bounded_by_log_n(self):
        """H ≤ log(n) for any n-dimensional weight vector."""
        rng = np.random.default_rng(77)
        n = 8
        idx = pd.date_range("2020-01-02", periods=15, freq="B")
        w = pd.DataFrame(rng.dirichlet(np.ones(n), size=15), index=idx)
        H = compute_gibbs_entropy(w)
        assert np.all(H.values <= np.log(n) + 1e-11)

    def test_returns_series(self):
        """compute_gibbs_entropy returns a pandas Series."""
        idx = pd.date_range("2020-01-02", periods=5, freq="B")
        w = pd.DataFrame(np.full((5, 3), 1.0 / 3), index=idx)
        H = compute_gibbs_entropy(w)
        assert isinstance(H, pd.Series)

    def test_length_preserved(self):
        """Output length matches input."""
        idx = pd.date_range("2020-01-02", periods=50, freq="B")
        w = pd.DataFrame(np.full((50, 3), 1.0 / 3), index=idx)
        H = compute_gibbs_entropy(w)
        assert len(H) == 50
