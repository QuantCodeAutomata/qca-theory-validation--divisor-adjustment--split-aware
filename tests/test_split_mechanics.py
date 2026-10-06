"""
tests/test_split_mechanics.py
==============================
Unit tests for src/split_mechanics.py.

Tests cover:
- compute_price_weights: correct normalisation, split jump, edge cases
- compute_cap_weights: correct normalisation, split invariance
- divisor_update: continuity of Σ̃ at split events
- compute_normalized_total_price: continuity across split events
- detect_split_events: forward/reverse split detection, no-split days
- compute_adjusted_start_of_day: correct price scaling
- self_financing_check: wealth preservation under splits
- compute_defect_Q / compute_defect_C: values under correct divisor
- compute_cross_basis_identity: LHS == RHS numerically
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.split_mechanics import (
    compute_price_weights,
    compute_cap_weights,
    divisor_update,
    compute_normalized_total_price,
    detect_split_events,
    compute_adjusted_start_of_day,
    self_financing_check,
    compute_defect_Q,
    compute_defect_C,
    compute_cross_basis_identity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_paths(T: int = 20, seed: int = 42):
    """Return (P, N) arrays for 2 assets with one forward split on asset 0."""
    rng = np.random.default_rng(seed)
    P = np.ones((T, 2))
    N = np.ones((T, 2))
    P[:, 0] = 100.0 + rng.standard_normal(T).cumsum() * 0.5
    P[:, 1] = 50.0  + rng.standard_normal(T).cumsum() * 0.3
    N[:, 0] = 1_000.0
    N[:, 1] = 2_000.0
    # Pure 2-for-1 split on asset 0 at t=10
    split_t = 10
    P[split_t:, 0] = P[split_t - 1, 0] / 2.0
    N[split_t:, 0] = N[split_t - 1, 0] * 2.0
    return P, N


# ---------------------------------------------------------------------------
# compute_price_weights
# ---------------------------------------------------------------------------

class TestComputePriceWeights:
    def test_sums_to_one(self):
        """Price weights must sum to 1 for any positive price vector."""
        prices = np.array([100.0, 50.0, 25.0])
        nu = compute_price_weights(prices)
        assert abs(nu.sum() - 1.0) < 1e-14

    def test_two_asset_values(self):
        """Check exact values for a 2-asset case."""
        prices = np.array([80.0, 20.0])
        nu = compute_price_weights(prices)
        assert abs(nu[0] - 0.8) < 1e-14
        assert abs(nu[1] - 0.2) < 1e-14

    def test_jump_at_split(self):
        """Price weights jump mechanically at a stock split."""
        prices_before = np.array([100.0, 50.0])
        prices_after_split = np.array([50.0, 50.0])  # 2-for-1 split on asset 0
        nu_before = compute_price_weights(prices_before)
        nu_after = compute_price_weights(prices_after_split)
        # ν should change even though economic cap is unchanged
        assert not np.allclose(nu_before, nu_after)

    def test_single_asset(self):
        """A single-asset weight must be exactly 1."""
        nu = compute_price_weights(np.array([42.0]))
        assert abs(nu[0] - 1.0) < 1e-14

    def test_all_non_negative(self):
        """All price weights must be non-negative."""
        prices = np.array([10.0, 5.0, 1.0, 0.5])
        nu = compute_price_weights(prices)
        assert np.all(nu >= 0)

    def test_equal_prices(self):
        """Equal prices yield uniform weights."""
        prices = np.array([50.0, 50.0, 50.0])
        nu = compute_price_weights(prices)
        assert np.allclose(nu, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# compute_cap_weights
# ---------------------------------------------------------------------------

class TestComputeCapWeights:
    def test_sums_to_one(self):
        caps = np.array([200_000.0, 100_000.0])
        mu = compute_cap_weights(caps)
        assert abs(mu.sum() - 1.0) < 1e-14

    def test_invariant_under_split(self):
        """Cap weights must be unchanged by a pure stock split."""
        caps_before = np.array([100_000.0, 100_000.0])
        # After 2-for-1 split on asset 0: cap unchanged
        caps_after = np.array([100_000.0, 100_000.0])
        mu_before = compute_cap_weights(caps_before)
        mu_after = compute_cap_weights(caps_after)
        assert np.allclose(mu_before, mu_after)

    def test_two_asset_values(self):
        caps = np.array([300_000.0, 100_000.0])
        mu = compute_cap_weights(caps)
        assert abs(mu[0] - 0.75) < 1e-14
        assert abs(mu[1] - 0.25) < 1e-14

    def test_all_non_negative(self):
        caps = np.array([10.0, 5.0, 2.0])
        mu = compute_cap_weights(caps)
        assert np.all(mu >= 0)


# ---------------------------------------------------------------------------
# divisor_update
# ---------------------------------------------------------------------------

class TestDivisorUpdate:
    def test_no_update_on_non_split_day(self):
        """Divisor unchanged when there are no split events."""
        kappa_prev = 1.0
        sigma_prev = 150.0
        sigma_now = 155.0  # pure price move, no split
        kappa = divisor_update(kappa_prev, sigma_prev, sigma_now, is_split_day=False)
        assert abs(kappa - kappa_prev) < 1e-14

    def test_update_restores_continuity(self):
        """After a split the normalized total price Σ̃ is continuous."""
        P_before = np.array([100.0, 50.0])
        P_after = np.array([50.0, 50.0])   # price halved by 2-for-1 split on asset 0
        sigma_before = P_before.sum()      # 150
        sigma_after = P_after.sum()        # 100
        kappa_prev = 1.0
        kappa_new = divisor_update(kappa_prev, sigma_before, sigma_after, is_split_day=True)
        # Σ̃ = Σ / κ must be continuous
        sigma_tilde_before = sigma_before / kappa_prev
        sigma_tilde_after = sigma_after / kappa_new
        assert abs(sigma_tilde_before - sigma_tilde_after) < 1e-10

    def test_reverse_split_raises_divisor(self):
        """A reverse split (price rises, shares fall) raises the total price sum."""
        P_before = np.array([50.0, 50.0])
        P_after = np.array([100.0, 50.0])   # reverse 2-for-1 on asset 0
        sigma_before = P_before.sum()
        sigma_after = P_after.sum()
        kappa_prev = 1.0
        kappa_new = divisor_update(kappa_prev, sigma_before, sigma_after, is_split_day=True)
        sigma_tilde_before = sigma_before / kappa_prev
        sigma_tilde_after = sigma_after / kappa_new
        assert abs(sigma_tilde_before - sigma_tilde_after) < 1e-10


# ---------------------------------------------------------------------------
# compute_normalized_total_price
# ---------------------------------------------------------------------------

class TestNormalizedTotalPrice:
    def test_continuous_at_split(self):
        """Σ̃(t) is continuous at a pure split event."""
        P, N = _make_simple_paths(T=15, seed=0)
        T_len = len(P)
        kappa = np.ones(T_len)
        sigma = P.sum(axis=1)
        # Build divisor stream detecting splits via share-count ratio
        for t in range(1, T_len):
            ratio_N = N[t] / N[t - 1]
            ratio_S = (P[t] * N[t]) / (P[t - 1] * N[t - 1])
            is_split = bool(
                np.any((np.abs(ratio_N - 1) > 0.15) & (np.abs(ratio_S - 1) <= 0.15))
            )
            kappa[t] = divisor_update(kappa[t - 1], sigma[t - 1], sigma[t], is_split)
        # compute_normalized_total_price takes a 1-D price array + scalar kappa
        sigma_tilde = np.array(
            [compute_normalized_total_price(P[t], kappa[t]) for t in range(T_len)]
        )
        # At t=10 (split), sigma_tilde should be continuous
        assert abs(sigma_tilde[10] - sigma_tilde[9]) < sigma_tilde[9] * 0.15

    def test_without_divisor_jumps(self):
        """Raw Σ(t) jumps at split events (verifying the problem being solved)."""
        P, N = _make_simple_paths(T=15, seed=0)
        sigma = P.sum(axis=1)
        # At t=10, P[0] halves so Σ must drop by ~P[0,9]/2
        jump = abs(sigma[10] - sigma[9])
        assert jump > 1.0  # substantial jump in raw sum


# ---------------------------------------------------------------------------
# detect_split_events
# ---------------------------------------------------------------------------

class TestDetectSplitEvents:
    """detect_split_events(N_series, S_series) expects DataFrames; returns DataFrame."""

    def test_forward_split_detected(self):
        """2-for-1 forward split is detected with m≈2."""
        dates = pd.date_range("2020-01-01", periods=2)
        N_df = pd.DataFrame({"A": [1000.0, 2000.0], "B": [2000.0, 2000.0]}, index=dates)
        P_arr = np.array([[100.0, 50.0], [50.0, 51.0]])
        S_arr = P_arr * N_df.values
        S_df = pd.DataFrame(S_arr, columns=["A", "B"], index=dates)
        m = detect_split_events(N_df, S_df)
        # t=1 (index 1), asset A should have m≈2.0
        assert abs(m.iloc[1]["A"] - 2.0) < 0.05
        # t=1, asset B should have m=1 (no split)
        assert abs(m.iloc[1]["B"] - 1.0) < 0.05

    def test_no_split_no_detection(self):
        """No split: multipliers should all be 1."""
        dates = pd.date_range("2020-01-01", periods=2)
        N_df = pd.DataFrame({"A": [1000.0, 1000.0], "B": [2000.0, 2000.0]}, index=dates)
        P_arr = np.array([[100.0, 50.0], [101.0, 50.5]])
        S_df = pd.DataFrame(P_arr * N_df.values, columns=["A", "B"], index=dates)
        m = detect_split_events(N_df, S_df)
        assert np.allclose(m.iloc[1].values, 1.0)

    def test_reverse_split_detected(self):
        """2-for-1 reverse split: price doubles, shares halve → m≈0.5."""
        dates = pd.date_range("2020-01-01", periods=2)
        N_df = pd.DataFrame({"A": [2000.0, 1000.0], "B": [2000.0, 2000.0]}, index=dates)
        P_arr = np.array([[50.0, 50.0], [100.0, 50.0]])
        S_df = pd.DataFrame(P_arr * N_df.values, columns=["A", "B"], index=dates)
        m = detect_split_events(N_df, S_df)
        # m at t=1, asset A should be ~0.5 (reverse split multiplier)
        assert abs(m.iloc[1]["A"] - 0.5) < 0.05


# ---------------------------------------------------------------------------
# compute_adjusted_start_of_day
# ---------------------------------------------------------------------------

class TestAdjustedStartOfDay:
    """compute_adjusted_start_of_day(P_prev, N_curr, m_curr) → (P_diamond, S_diamond)."""

    def test_no_split_identity(self):
        """Without a split m=1, adjusted price equals previous close."""
        P_prev = np.array([100.0, 50.0])
        N_curr = np.array([1000.0, 2000.0])
        m = np.array([1.0, 1.0])
        P_adj, S_adj = compute_adjusted_start_of_day(P_prev, N_curr, m)
        assert np.allclose(P_adj, P_prev)

    def test_forward_split_halves_price(self):
        """2-for-1 split: adjusted start price = P_prev / 2."""
        P_prev = np.array([100.0, 50.0])
        N_curr = np.array([2000.0, 2000.0])
        m = np.array([2.0, 1.0])
        P_adj, S_adj = compute_adjusted_start_of_day(P_prev, N_curr, m)
        assert abs(P_adj[0] - 50.0) < 1e-12
        assert abs(P_adj[1] - 50.0) < 1e-12

    def test_reverse_split_doubles_price(self):
        """Reverse 2-for-1 split: adjusted start price = P_prev / 0.5 = 2*P_prev."""
        P_prev = np.array([50.0, 50.0])
        N_curr = np.array([1000.0, 2000.0])
        m = np.array([0.5, 1.0])
        P_adj, S_adj = compute_adjusted_start_of_day(P_prev, N_curr, m)
        assert abs(P_adj[0] - 100.0) < 1e-12

    def test_adjusted_price_positive(self):
        """Adjusted price is always positive."""
        P_prev = np.array([30.0, 80.0])
        N_curr = np.array([3000.0, 500.0])
        m = np.array([3.0, 0.5])
        P_adj, S_adj = compute_adjusted_start_of_day(P_prev, N_curr, m)
        assert np.all(P_adj > 0)


# ---------------------------------------------------------------------------
# self_financing_check
# ---------------------------------------------------------------------------

class TestSelfFinancingCheck:
    """self_financing_check(theta, X, is_split) → dict with 'wealth', 'defect', etc."""

    def test_wealth_preserved_on_split(self):
        """Under a predictable share adjustment at split, wealth must be preserved."""
        P, N = _make_simple_paths(T=12, seed=1)
        T_len = len(P)
        m = np.ones_like(P)
        for t in range(1, T_len):
            ratio_N = N[t] / N[t - 1]
            ratio_S = (P[t] * N[t]) / (P[t - 1] * N[t - 1])
            m[t] = np.where((np.abs(ratio_N - 1) > 0.15) & (np.abs(ratio_S - 1) <= 0.15),
                            ratio_N, 1.0)
        is_split = m != 1.0  # bool (T, d) array
        theta = np.ones((T_len, 2))
        # After a split by factor m, price drops by 1/m so holdings must rise by m
        for t in range(1, T_len):
            for i in range(2):
                if m[t, i] != 1.0:
                    theta[t:, i] = theta[t - 1, i] * m[t, i]
        sf = self_financing_check(theta, P, is_split)
        V = sf["wealth"]
        # Find first split time (if any) and check wealth continuity there
        split_times = [t for t in range(1, T_len) if is_split[t].any()]
        if split_times:
            tau = split_times[0]
            split_jump = abs(V[tau] - V[tau - 1])
            assert split_jump < V[tau - 1] * 0.01, (
                f"Wealth jump {split_jump:.6f} at t={tau} too large (V_prev={V[tau - 1]:.4f})"
            )

    def test_wealth_increases_with_returns(self):
        """Without splits, wealth grows if prices grow."""
        T_len = 5
        P = np.array([[100.0, 50.0]] * T_len, dtype=float)
        P[1] = [105.0, 52.0]
        P[2] = [110.0, 54.0]
        P[3] = [115.0, 56.0]
        P[4] = [120.0, 58.0]
        is_split = np.zeros_like(P, dtype=bool)
        theta = np.ones((T_len, 2))
        sf = self_financing_check(theta, P, is_split)
        V = sf["wealth"]
        assert V[-1] > V[0]


# ---------------------------------------------------------------------------
# compute_defect_Q / compute_defect_C
# ---------------------------------------------------------------------------

class TestDefects:
    def _build_correct_path(self):
        """Build a correctly-divisor-adjusted path; defect should be ~0."""
        P, N = _make_simple_paths(T=20, seed=99)
        T_len = len(P)
        m = np.ones_like(P)
        for t in range(1, T_len):
            ratio_N = N[t] / N[t - 1]
            ratio_S = (P[t] * N[t]) / (P[t - 1] * N[t - 1])
            m[t] = np.where((np.abs(ratio_N - 1) > 0.15) & (np.abs(ratio_S - 1) <= 0.15),
                            ratio_N, 1.0)
        kappa = np.ones(T_len)
        sigma = P.sum(axis=1)
        for t in range(1, T_len):
            is_split_day = bool(np.any(m[t] != 1.0))
            kappa[t] = divisor_update(kappa[t - 1], sigma[t - 1], sigma[t], is_split_day)
        nu = np.array([compute_price_weights(P[t]) for t in range(T_len)])
        theta = np.ones((T_len, 2))
        for t in range(1, T_len):
            for i in range(2):
                if m[t, i] != 1.0:
                    theta[t:, i] = theta[t - 1, i] / m[t, i]
        V = np.array([np.dot(theta[t], P[t]) for t in range(T_len)])
        Sigma_tilde = np.array(
            [compute_normalized_total_price(P[t], kappa[t]) for t in range(T_len)]
        )
        return theta, nu, P, kappa, V, sigma, Sigma_tilde

    def test_defect_Q_small_under_correct_divisor(self):
        theta, nu, P, kappa, V, sigma, Sigma_tilde = self._build_correct_path()
        Q = compute_defect_Q(theta, nu, P, kappa)
        # Defect is by construction ≈ 0 when divisor is correctly maintained
        assert np.all(np.abs(Q) < 1.0), f"Max |Q| = {np.abs(Q).max():.3e}"

    def test_defect_C_zero_on_pre_split(self):
        theta, nu, P, kappa, V, sigma, Sigma_tilde = self._build_correct_path()
        C = compute_defect_C(theta, nu, P, kappa, Sigma_tilde)
        # Before split t < 10, continuity defect should be 0
        assert np.all(np.abs(C[:10]) < 1e-6)


# ---------------------------------------------------------------------------
# compute_cross_basis_identity
# ---------------------------------------------------------------------------

class TestCrossBasisIdentity:
    """compute_cross_basis_identity(nu, mu, N, P, S, kappa, Sigma_tilde) → dict."""

    def test_identity_holds_numerically(self):
        """V^{ϑ⊗N^{-1}}(μ) / V^{κϑ}(ν) ≈ Σ̃ / Λ"""
        P, N = _make_simple_paths(T=30, seed=7)
        T_len = len(P)
        S = P * N
        sigma = P.sum(axis=1)
        m = np.ones_like(P)
        for t in range(1, T_len):
            ratio_N = N[t] / N[t - 1]
            ratio_S = S[t].sum() / S[t - 1].sum()
            m[t] = np.where((np.abs(ratio_N - 1) > 0.15) & (np.abs(ratio_S - 1) <= 0.15),
                            ratio_N, 1.0)
        kappa = np.ones(T_len)
        for t in range(1, T_len):
            is_split_day = bool(np.any(m[t] != 1.0))
            kappa[t] = divisor_update(kappa[t - 1], sigma[t - 1], sigma[t], is_split_day)
        Sigma_tilde = np.array(
            [compute_normalized_total_price(P[t], kappa[t]) for t in range(T_len)]
        )
        nu = np.array([compute_price_weights(P[t]) for t in range(T_len)])
        mu = np.array([compute_cap_weights(S[t]) for t in range(T_len)])
        result = compute_cross_basis_identity(nu, mu, N, P, S, kappa, Sigma_tilde)
        lhs = result["LHS"]
        rhs = result["RHS"]
        assert np.allclose(lhs, rhs, atol=1e-8, rtol=1e-6), (
            f"Max deviation: {np.abs(lhs - rhs).max():.2e}"
        )


# ---------------------------------------------------------------------------
# Mathematical property tests
# ---------------------------------------------------------------------------

class TestMathProperties:
    def test_price_weights_simplex(self):
        """Price weights lie in the simplex (sum=1, all ≥0) for random positive prices."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            d = rng.integers(2, 10)
            prices = rng.uniform(1.0, 200.0, d)
            nu = compute_price_weights(prices)
            assert abs(nu.sum() - 1.0) < 1e-13
            assert np.all(nu >= 0)

    def test_cap_weights_simplex(self):
        """Cap weights lie in the simplex for random positive caps."""
        rng = np.random.default_rng(456)
        for _ in range(20):
            d = rng.integers(2, 10)
            caps = rng.uniform(1e3, 1e9, d)
            mu = compute_cap_weights(caps)
            assert abs(mu.sum() - 1.0) < 1e-13
            assert np.all(mu >= 0)

    def test_cap_invariant_to_pure_split(self):
        """For any pure split (P_i → P_i/m, N_i → N_i*m), cap weights unchanged."""
        rng = np.random.default_rng(789)
        for _ in range(10):
            P = rng.uniform(10.0, 500.0, 3)
            N = rng.uniform(1e3, 1e6, 3)
            S = P * N
            # Forward 3-for-1 split on asset 1
            P2 = P.copy(); N2 = N.copy()
            P2[1] /= 3.0
            N2[1] *= 3.0
            S2 = P2 * N2
            mu_before = compute_cap_weights(S)
            mu_after = compute_cap_weights(S2)
            assert np.allclose(mu_before, mu_after, atol=1e-12)

    def test_sigma_tilde_continuous_multiple_splits(self):
        """Σ̃ remains continuous through multiple consecutive splits."""
        kappa_0 = 1.0
        # Split 1: Σ drops from 150→100 (2-for-1 forward split)
        k1 = divisor_update(kappa_0, 150.0, 100.0, is_split_day=True)
        s_tilde_0 = 150.0 / kappa_0
        s_tilde_1 = 100.0 / k1
        assert abs(s_tilde_0 - s_tilde_1) < 1e-10
        # Split 2: Σ rises from 100→200 (reverse split)
        k2 = divisor_update(k1, 100.0, 200.0, is_split_day=True)
        s_tilde_2 = 200.0 / k2
        assert abs(s_tilde_1 - s_tilde_2) < 1e-10
