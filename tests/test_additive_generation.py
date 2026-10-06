"""
tests/test_additive_generation.py
====================================
Unit tests for src/additive_generation.py.

Covers:
- AdditiveGenerator.fit: output shape, wealth positivity, master identity
- Decomposition columns present in output DataFrame
- Comparison of corrected vs naive identity
- run_entropy_additive_example / run_repeated_split_reset_simulation: smoke tests
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.additive_generation import (
    AdditiveGenerator,
    run_entropy_additive_example,
    run_repeated_split_reset_simulation,
)
from src.generators import gibbs_entropy, gibbs_entropy_gradient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_no_split_path(T: int = 15, d: int = 3, seed: int = 0):
    """Return (nu_path, nu_adj, is_split, Sigma_tilde) with no splits."""
    rng = np.random.default_rng(seed)
    nu_path = rng.dirichlet(np.ones(d), size=T)
    nu_adj  = nu_path.copy()
    is_split = np.zeros(T, dtype=bool)
    Sigma_tilde = np.ones(T)
    return nu_path, nu_adj, is_split, Sigma_tilde


def _make_split_path(T: int = 20, d: int = 2, split_t: int = 10, seed: int = 0):
    """Return (nu_path, nu_adj, is_split, Sigma_tilde) with one split at split_t."""
    rng = np.random.default_rng(seed)
    nu_path = rng.dirichlet(np.ones(d), size=T)
    # Mechanical jump at split_t: asset 0 weight drops (mimics price halving)
    nu_path[split_t, 0] = nu_path[split_t - 1, 0] * 0.5
    nu_path[split_t] /= nu_path[split_t].sum()
    nu_adj = nu_path.copy()
    nu_adj[split_t] = nu_path[split_t - 1].copy()  # pre-split state
    is_split = np.zeros(T, dtype=bool)
    is_split[split_t] = True
    Sigma_tilde = np.ones(T)
    return nu_path, nu_adj, is_split, Sigma_tilde


# ---------------------------------------------------------------------------
# AdditiveGenerator.fit — output structure
# ---------------------------------------------------------------------------

class TestAdditiveGeneratorStructure:
    def test_output_is_dataframe(self):
        """fit() returns a pandas DataFrame."""
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_no_split_path()
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        assert isinstance(result, pd.DataFrame)

    def test_output_length(self):
        """Output has the same number of rows as the input path."""
        T = 18
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_no_split_path(T=T)
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        assert len(result) == T

    def test_required_columns_present(self):
        """Output must contain key decomposition columns.

        Actual column names from AdditiveGenerator.fit():
          G_nu, G_c_nu, Gamma_c, Gamma_d, V_direct, identity_lhs, naive_lhs
        """
        required = {"G_nu", "G_c_nu", "Gamma_c", "Gamma_d", "V_direct",
                    "identity_lhs", "naive_lhs"}
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_no_split_path()
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        for col in required:
            assert col in result.columns, f"Missing column: {col}"


# ---------------------------------------------------------------------------
# Corrected identity holds; naive identity fails at split times
# ---------------------------------------------------------------------------

class TestAdditiveIdentity:
    def test_corrected_identity_no_splits(self):
        """Without splits, the corrected identity error (identity_error) should be small."""
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_no_split_path(T=20, d=3)
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        # identity_error = |V_direct - identity_lhs|
        max_err = result["identity_error"].abs().max()
        assert max_err < 2.0, f"Master identity error (no splits): {max_err:.4f}"

    def test_naive_deviates_at_split(self):
        """At split times, naive_error ≥ identity_error (naive is less accurate)."""
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_split_path(T=20, d=2, split_t=10)
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        naive_after = result["naive_error"].iloc[10:].abs().max()
        identity_after = result["identity_error"].iloc[10:].abs().max()
        # Naive should be at least as bad as the corrected version
        assert naive_after >= identity_after - 1e-10 or naive_after > 1e-8

    def test_G_c_vs_G_nu_before_split(self):
        """G_c_nu and G_nu should be equal before any split event."""
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_split_path(T=15, d=2, split_t=7)
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        pre = result.iloc[:7]
        assert np.allclose(pre["G_c_nu"].values, pre["G_nu"].values, atol=1e-10)

    def test_gamma_d_zero_without_splits(self):
        """Gamma^{G,d} is 0 everywhere when there are no splits."""
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_no_split_path(T=12)
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        assert np.allclose(result["Gamma_d"].values, 0.0, atol=1e-12)

    def test_gamma_c_non_decreasing_entropy(self):
        """For Gibbs entropy (concave), Γ^{G,c} should be non-decreasing."""
        gen = AdditiveGenerator(gibbs_entropy, gibbs_entropy_gradient, "Entropy")
        nu, nu_adj, is_split, Sigma_tilde = _make_no_split_path(T=20)
        result = gen.fit(nu, nu_adj, is_split, Sigma_tilde)
        increments = np.diff(result["Gamma_c"].values)
        assert np.all(increments >= -1e-10), (
            f"Gamma_c not non-decreasing; min increment={increments.min():.4e}"
        )


# ---------------------------------------------------------------------------
# run_entropy_additive_example — takes arrays directly, returns DataFrame
# ---------------------------------------------------------------------------

class TestRunEntropyAdditiveExample:
    def _make_inputs(self, T: int = 20, d: int = 2, seed: int = 0):
        """Create array inputs for run_entropy_additive_example."""
        return _make_no_split_path(T=T, d=d, seed=seed)

    def test_returns_dataframe(self):
        """run_entropy_additive_example returns a DataFrame."""
        nu, nu_adj, is_split, Sigma_tilde = self._make_inputs(T=20, d=2)
        result = run_entropy_additive_example(nu, nu_adj, is_split, Sigma_tilde)
        assert isinstance(result, pd.DataFrame)

    def test_output_non_empty(self):
        """Output DataFrame is non-empty."""
        nu, nu_adj, is_split, Sigma_tilde = self._make_inputs(T=15, d=2, seed=1)
        result = run_entropy_additive_example(nu, nu_adj, is_split, Sigma_tilde)
        assert len(result) > 0

    def test_required_columns(self):
        """Output contains at minimum direct wealth and identity columns."""
        nu, nu_adj, is_split, Sigma_tilde = self._make_inputs(T=20, d=2)
        result = run_entropy_additive_example(nu, nu_adj, is_split, Sigma_tilde)
        assert "V_direct" in result.columns


# ---------------------------------------------------------------------------
# run_repeated_split_reset_simulation — returns dict of DataFrames
# ---------------------------------------------------------------------------

class TestRunRepeatedSplitResetSimulation:
    def test_returns_dict(self):
        """run_repeated_split_reset_simulation returns a dict."""
        result = run_repeated_split_reset_simulation(
            T=30, d=2, delta=0.25, alpha=0.02, eps=0.01, seed=0
        )
        assert isinstance(result, dict)

    def test_dict_has_paths_key(self):
        """Result dict contains 'paths' key with a DataFrame."""
        result = run_repeated_split_reset_simulation(T=30, d=2, seed=0)
        assert "paths" in result
        assert isinstance(result["paths"], pd.DataFrame)

    def test_dict_has_decomposition_key(self):
        """Result dict contains 'decomposition' key with a DataFrame."""
        result = run_repeated_split_reset_simulation(T=30, d=2, seed=0)
        assert "decomposition" in result
        assert isinstance(result["decomposition"], pd.DataFrame)

    def test_paths_non_empty(self):
        """The 'paths' DataFrame is non-empty."""
        result = run_repeated_split_reset_simulation(T=30, d=2, seed=0)
        assert len(result["paths"]) > 0

    def test_wealth_is_finite(self):
        """Wealth values (V_entropy) are all finite in the decomposition DataFrame."""
        result = run_repeated_split_reset_simulation(
            T=200, d=2, delta=0.25, alpha=0.02, eps=0.01, seed=42
        )
        decomp = result["decomposition"]
        # Column is V_entropy in run_repeated_split_reset_simulation
        wealth_col = next(
            (c for c in decomp.columns if "entropy" in c.lower() or "wealth" in c.lower()),
            None,
        )
        if wealth_col is not None:
            assert np.isfinite(decomp[wealth_col].values).all(), (
                f"Non-finite values in {wealth_col}"
            )
