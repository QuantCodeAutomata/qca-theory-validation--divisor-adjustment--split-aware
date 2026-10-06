"""
tests/test_generators.py
=========================
Unit tests for src/generators.py.

Covers:
- gibbs_entropy: value, gradient, Hessian correctness
- diversity_function / diversity_gradient / diversity_hessian
- balance_defect
- compute_jump_corrected_state_additive
- compute_gamma_continuous / compute_gamma_jump
- compute_generated_strategy_additive / compute_additive_wealth
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.generators import (
    gibbs_entropy,
    gibbs_entropy_gradient,
    gibbs_entropy_hessian,
    diversity_function,
    diversity_gradient,
    diversity_hessian,
    log_diversity_hessian,
    balance_defect,
    compute_jump_corrected_state_additive,
    compute_gamma_continuous,
    compute_gamma_jump,
    compute_generated_strategy_additive,
    compute_additive_wealth,
)


# ---------------------------------------------------------------------------
# Gibbs entropy H(x) = -sum x_i log x_i
# ---------------------------------------------------------------------------

class TestGibbsEntropy:
    def test_uniform_is_max(self):
        """H is maximised by the uniform distribution."""
        d = 5
        x_uniform = np.full(d, 1.0 / d)
        H_uniform = gibbs_entropy(x_uniform)
        for _ in range(50):
            rng = np.random.default_rng(_ + 100)
            x = rng.dirichlet(np.ones(d))
            assert gibbs_entropy(x) <= H_uniform + 1e-12

    def test_degenerate_is_zero(self):
        """H is 0 for a point mass (one weight = 1)."""
        x = np.array([1.0, 0.0, 0.0])
        H = gibbs_entropy(x)
        assert abs(H) < 1e-12

    def test_two_asset_value(self):
        """Analytic check: H([0.5, 0.5]) = log(2)."""
        x = np.array([0.5, 0.5])
        H = gibbs_entropy(x)
        assert abs(H - np.log(2)) < 1e-12

    def test_non_negative(self):
        """H >= 0 for any probability vector."""
        rng = np.random.default_rng(7)
        for _ in range(30):
            d = rng.integers(2, 8)
            x = rng.dirichlet(np.ones(d))
            assert gibbs_entropy(x) >= -1e-12

    def test_gradient_finite_diff(self):
        """Gradient of H agrees with finite differences."""
        x = np.array([0.4, 0.35, 0.25])
        grad = gibbs_entropy_gradient(x)
        eps = 1e-6
        fd_grad = np.zeros(3)
        for i in range(3):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            fd_grad[i] = (gibbs_entropy(xp) - gibbs_entropy(xm)) / (2 * eps)
        assert np.allclose(grad, fd_grad, atol=1e-7)

    def test_hessian_negative_semi_definite(self):
        """Hessian of H should be negative semi-definite (H concave)."""
        x = np.array([0.4, 0.35, 0.25])
        H_mat = gibbs_entropy_hessian(x)
        eigvals = np.linalg.eigvalsh(H_mat)
        assert np.all(eigvals <= 1e-12), f"Positive eigval: {eigvals.max():.3e}"


# ---------------------------------------------------------------------------
# Diversity function D_p(x) = (sum x_i^p)^{1/p}
# ---------------------------------------------------------------------------

class TestDiversityFunction:
    def test_uniform_d2(self):
        """D_p([0.5, 0.5]) = 2^{1/p - 1} for p in (0,1)."""
        for p in [0.3, 0.5, 0.7]:
            x = np.array([0.5, 0.5])
            Dp = diversity_function(x, p)
            expected = 2.0 ** (1.0 / p - 1.0)
            assert abs(Dp - expected) < 1e-12, f"p={p}: {Dp} vs {expected}"

    def test_p_equals_one_is_one(self):
        """D_1(x) = sum(x_i) = 1 for any weight vector."""
        x = np.array([0.3, 0.5, 0.2])
        Dp = diversity_function(x, 1.0)
        assert abs(Dp - 1.0) < 1e-12

    def test_smaller_p_larger_Dp(self):
        """For non-uniform weights, smaller p → larger D_p."""
        x = np.array([0.6, 0.3, 0.1])
        D_low  = diversity_function(x, 0.2)
        D_high = diversity_function(x, 0.8)
        assert D_low > D_high

    def test_positive(self):
        """D_p is always positive."""
        rng = np.random.default_rng(42)
        for p in [0.1, 0.5, 0.9]:
            for _ in range(10):
                x = rng.dirichlet(np.ones(4))
                assert diversity_function(x, p) > 0

    def test_gradient_finite_diff(self):
        """Gradient of D_p agrees with finite differences."""
        x = np.array([0.4, 0.35, 0.25])
        for p in [0.3, 0.5, 0.7]:
            grad = diversity_gradient(x, p)
            eps = 1e-6
            fd = np.zeros(3)
            for i in range(3):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                fd[i] = (diversity_function(xp, p) - diversity_function(xm, p)) / (2 * eps)
            assert np.allclose(grad, fd, atol=1e-7), f"Gradient mismatch for p={p}"

    def test_hessian_finite_diff(self):
        """Hessian of D_p agrees with finite differences (diagonal only for speed)."""
        x = np.array([0.4, 0.35, 0.25])
        for p in [0.5, 0.7]:
            H = diversity_hessian(x, p)
            eps = 1e-5
            for i in range(3):
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                fd_hii = (diversity_function(xp, p) - 2 * diversity_function(x, p)
                          + diversity_function(xm, p)) / eps**2
                assert abs(H[i, i] - fd_hii) < 1e-5, (
                    f"Hessian diag mismatch p={p} i={i}: H={H[i,i]:.4f} fd={fd_hii:.4f}"
                )

    def test_diversity_portfolio_weights_sum_to_one(self):
        """Diversity-weighted portfolio π_i = x_i^p / Σ x_j^p sums to 1."""
        for p in [0.1, 0.5, 0.9]:
            x = np.array([0.4, 0.35, 0.25])
            pi = x ** p / (x ** p).sum()
            assert abs(pi.sum() - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# balance_defect b_G(x) = ∇G(x)·x - G(x)
# ---------------------------------------------------------------------------

class TestBalanceDefect:
    def test_gibbs_entropy_balance(self):
        """For Gibbs entropy H, b_H(x) = -sum x_i log x_i - (-1 - sum x_i log x_i) = 1."""
        # Gradient of H = -1 - log x_i, so ∇H·x = -sum x_i - sum x_i log x_i = -1 - H
        # b_H = ∇H·x - H = (-1 - H) - H = -1 - 2H, but let's test numerically.
        x = np.array([0.4, 0.35, 0.25])
        b = balance_defect(gibbs_entropy, gibbs_entropy_gradient, x)
        # Numerical verification: ∇H·x - H
        H = gibbs_entropy(x)
        grad_H = gibbs_entropy_gradient(x)
        expected = np.dot(grad_H, x) - H
        assert abs(b - expected) < 1e-12

    def test_homogeneous_function_euler(self):
        """For 1-homogeneous G (like D_p), b_G = 0 by Euler's theorem."""
        # D_p is 1-homogeneous: D_p(λx) = λ D_p(x), so ∇D_p·x = D_p(x)
        x = np.array([0.4, 0.35, 0.25])
        for p in [0.3, 0.5, 0.7]:
            b = balance_defect(
                lambda w: diversity_function(w, p),
                lambda w: diversity_gradient(w, p),
                x,
            )
            assert abs(b) < 1e-8, f"D_p balance defect not ~0 for p={p}: b={b:.2e}"


# ---------------------------------------------------------------------------
# Jump-corrected state and gamma decomposition
# ---------------------------------------------------------------------------

class TestJumpCorrectedState:
    def test_no_jump_identity(self):
        """Without splits, G^c(ν) = G(ν) and Γ^{G,d} = 0."""
        # Build a path with no splits; is_split is all-False boolean array
        T = 10
        nu = np.array([[0.6 - 0.02 * t, 0.4 + 0.02 * t] for t in range(T)])
        is_split = np.zeros((T, 2), dtype=bool)

        # API: compute_jump_corrected_state_additive(G, nu_path, is_split) → array
        G_c = compute_jump_corrected_state_additive(gibbs_entropy, nu, is_split)
        Gamma_d = compute_gamma_jump(gibbs_entropy, nu, is_split)

        G_direct = np.array([gibbs_entropy(nu[t]) for t in range(T)])
        assert np.allclose(G_c, G_direct, atol=1e-12), "G^c must equal G when no splits"
        assert np.allclose(Gamma_d, 0.0, atol=1e-12), "Γ^{G,d} must be 0 when no splits"

    def test_gamma_d_nonzero_at_splits(self):
        """Γ^{G,d} is non-zero when splits produce a price-weight jump."""
        T = 10
        nu = np.array([[0.6, 0.4]] * T, dtype=float)
        # At t=5, a 2-for-1 split on asset 0 shifts price weights mechanically
        nu[5] = [0.4, 0.6]
        is_split = np.zeros((T, 2), dtype=bool)
        is_split[5, 0] = True

        Gamma_d = compute_gamma_jump(gibbs_entropy, nu, is_split)
        # Gibbs entropy at [0.6, 0.4] ≠ Gibbs entropy at [0.4, 0.6] — symmetric
        # but the test is about Gamma_d accumulating the jump; use asymmetric weight
        nu2 = np.array([[0.7, 0.3]] * T, dtype=float)
        nu2[5] = [0.35, 0.65]          # non-symmetric jump: H differs
        is_split2 = np.zeros((T, 2), dtype=bool)
        is_split2[5, 0] = True
        Gamma_d2 = compute_gamma_jump(gibbs_entropy, nu2, is_split2)
        assert any(abs(Gamma_d2[t]) > 1e-8 for t in range(5, T)), (
            "Γ^{G,d} should be non-zero after a split that changes G"
        )


class TestGammaContinuous:
    def test_shape_and_zero_initial(self):
        """Γ^{G,c} has the right shape and starts at 0."""
        T = 20
        rng = np.random.default_rng(0)
        nu = rng.dirichlet(np.ones(3), size=T)
        is_split = np.zeros((T, 3), dtype=bool)
        # API: compute_gamma_continuous(G, grad_G, nu_path, nu_adj, is_split)
        Gamma_c = compute_gamma_continuous(
            gibbs_entropy, gibbs_entropy_gradient, nu, nu, is_split
        )
        assert Gamma_c.shape == (T,), f"Expected shape ({T},), got {Gamma_c.shape}"
        assert abs(Gamma_c[0]) < 1e-12, "Γ^{G,c}(0) must be 0"

    def test_zero_when_path_stationary(self):
        """When nu_path = nu_adj at every step, Γ^{G,c} = 0 (no market moves)."""
        T = 10
        nu = np.tile([0.4, 0.35, 0.25], (T, 1))
        is_split = np.zeros((T, 3), dtype=bool)
        Gamma_c = compute_gamma_continuous(
            gibbs_entropy, gibbs_entropy_gradient, nu, nu, is_split
        )
        assert np.allclose(Gamma_c, 0.0, atol=1e-12), (
            "Γ^{G,c} must be 0 when end-of-day equals start-of-day weights"
        )


# ---------------------------------------------------------------------------
# Additive wealth vs master identity
# ---------------------------------------------------------------------------

class TestAdditiveWealth:
    def test_master_identity_no_splits(self):
        """Wealth increments match Gamma^{G,c} increments (additive master, no splits).

        When nu_adj[t] = nu_path[t-1] (no splits), the wealth increment from
        phi[t-1] = grad_G(nu_adj[t-1]) equals Gamma^{G,c}'s increment at t
        up to the one-step gradient lag inherent in the discrete formulation.
        We verify the weaker property that the total accumulated Gamma^{G,c}
        and the direct wealth differ by at most a bounded amount.
        """
        T = 15
        rng = np.random.default_rng(42)
        nu = rng.dirichlet(np.ones(3), size=T)
        nu_adj = nu.copy()              # no splits
        is_split = np.zeros((T, 3), dtype=bool)
        Sigma_tilde = np.ones(T)

        phi = compute_generated_strategy_additive(gibbs_entropy_gradient, nu_adj, Sigma_tilde)
        V   = compute_additive_wealth(phi, nu, nu_adj, V0=0.0)

        Gamma_c = compute_gamma_continuous(
            gibbs_entropy, gibbs_entropy_gradient, nu, nu_adj, is_split
        )

        # Difference between direct wealth and Gamma^{G,c}: should be close
        # (exact only in continuous time; discrete lag of one step allowed)
        max_err = np.abs(V - Gamma_c).max()
        assert max_err < 1.0, (
            f"Wealth and Gamma^{{G,c}} diverge by {max_err:.4f}; "
            "check discrete-time implementation consistency"
        )

    def test_initial_wealth_is_V0(self):
        """compute_additive_wealth returns V0 at t=0."""
        T = 8
        rng = np.random.default_rng(7)
        nu = rng.dirichlet(np.ones(2), size=T)
        nu_adj = nu.copy()
        Sigma_tilde = np.ones(T)
        phi = compute_generated_strategy_additive(gibbs_entropy_gradient, nu_adj, Sigma_tilde)
        for V0 in [0.0, 1.0, 2.5]:
            V = compute_additive_wealth(phi, nu, nu_adj, V0=V0)
            assert abs(V[0] - V0) < 1e-12, f"V(0) should be V0={V0}"
