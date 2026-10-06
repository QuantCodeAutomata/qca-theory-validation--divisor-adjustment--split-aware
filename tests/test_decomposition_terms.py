"""
tests/test_decomposition_terms.py
===================================
Unit tests for src/decomposition_terms.py.

Covers:
- hessian_quadratic_form_Dp: sign, scale, edge cases
- compute_C_p: zero when w = w_adj; sign for outperforming diversification
- compute_E_p: always non-positive for concave p (p < 1) due to -ve Hessian term
- compute_J_p: zero on non-split dates; correct sign on split dates
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.decomposition_terms import (
    hessian_quadratic_form_Dp,
    compute_C_p,
    compute_E_p,
    compute_J_p,
)
from src.generators import diversity_function


# ---------------------------------------------------------------------------
# hessian_quadratic_form_Dp
# ---------------------------------------------------------------------------

class TestHessianQF:
    def test_zero_for_zero_delta(self):
        """No weight change → quadratic form is 0."""
        x = np.array([0.4, 0.35, 0.25])
        delta = np.zeros(3)
        qf = hessian_quadratic_form_Dp(x, delta, p=0.5)
        assert abs(qf) < 1e-14

    def test_finite_for_small_p(self):
        """QF should be finite even for p=0.1 with 10 assets."""
        d = 10
        x = np.full(d, 1.0 / d)
        delta = np.random.default_rng(0).standard_normal(d) * 0.01
        delta -= delta.mean()
        qf = hessian_quadratic_form_Dp(x, delta, p=0.1)
        assert np.isfinite(qf), f"QF not finite for p=0.1: {qf}"

    def test_sign_for_p_less_than_one(self):
        """For p < 1, (1/2) delta^T [H_{Dp}/Dp] delta should be non-positive
        (D_p is concave in the weight simplex for p < 1)."""
        rng = np.random.default_rng(123)
        x = rng.dirichlet(np.ones(5))
        for p in [0.3, 0.5, 0.7]:
            delta = rng.standard_normal(5) * 0.01
            delta -= delta.mean()
            qf = hessian_quadratic_form_Dp(x, delta, p)
            # The normalised Hessian of D_p (for p<1) is negative semidefinite
            # so qf = (1/2) delta^T H/Dp delta should be ≤ 0
            assert qf <= 1e-12, f"QF > 0 for p={p}: {qf:.4e}"

    def test_matches_finite_diff(self):
        """QF(x, delta, p) ≈ 0.5 * finite-diff second directional derivative / D_p."""
        x = np.array([0.4, 0.35, 0.25])
        delta = np.array([0.02, -0.01, -0.01])
        p = 0.5
        eps = 1e-5
        # Finite-diff second derivative of D_p along delta, divided by D_p
        Dp_centre = diversity_function(x, p)
        Dp_plus   = diversity_function(x + eps * delta / np.linalg.norm(delta), p)
        Dp_minus  = diversity_function(x - eps * delta / np.linalg.norm(delta), p)
        fd_second = (Dp_plus - 2 * Dp_centre + Dp_minus) / eps**2
        # scale by ||delta||^2
        fd_qf = 0.5 * fd_second * np.linalg.norm(delta)**2 / Dp_centre
        qf = hessian_quadratic_form_Dp(x, delta, p)
        assert abs(qf - fd_qf) < 1e-5, f"QF={qf:.6f} vs fd={fd_qf:.6f}"

    def test_empty_weights_returns_zero(self):
        """All-zero x returns 0 safely."""
        x = np.zeros(3)
        delta = np.ones(3)
        qf = hessian_quadratic_form_Dp(x, delta, p=0.5)
        assert qf == 0.0


# ---------------------------------------------------------------------------
# compute_C_p
# ---------------------------------------------------------------------------

class TestComputeCp:
    def _make_series(self, values, name="x"):
        idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
        return pd.Series(values, index=idx, name=name)

    def test_zero_when_no_change(self):
        """C_p = 0 when w_end = w_adj (no intraday change)."""
        vals = np.array([1.2, 1.3, 1.25, 1.28])
        D_end = self._make_series(vals)
        D_adj = self._make_series(vals)  # same
        C_p = compute_C_p(D_end, D_adj)
        assert np.allclose(C_p.values, 0.0)

    def test_accumulates_correctly(self):
        """C_p is a cumulative sum of log ratios."""
        D_end = self._make_series([1.0, 1.05, 1.10])
        D_adj = self._make_series([1.0, 1.00, 1.05])
        C_p = compute_C_p(D_end, D_adj)
        expected = np.cumsum(np.log([1.0 / 1.0, 1.05 / 1.00, 1.10 / 1.05]))
        assert np.allclose(C_p.values, expected, atol=1e-12)

    def test_negative_when_Dp_declines(self):
        """C_p is negative when diversity function consistently declines intraday."""
        D_end = self._make_series([1.0, 0.95, 0.90])
        D_adj = self._make_series([1.0, 1.00, 1.00])
        C_p = compute_C_p(D_end, D_adj)
        assert C_p.iloc[-1] < 0

    def test_handles_degenerate_ratio(self):
        """Degenerate (0/0) ratio is silently treated as 0 increment."""
        D_end = self._make_series([1.0, 0.0, 1.2])
        D_adj = self._make_series([1.0, 0.0, 1.0])
        C_p = compute_C_p(D_end, D_adj)  # should not raise
        assert np.isfinite(C_p.values).all()


# ---------------------------------------------------------------------------
# compute_E_p
# ---------------------------------------------------------------------------

class TestComputeEp:
    def _make_df(self, arrays, assets=None):
        idx = pd.date_range("2020-01-01", periods=len(arrays), freq="B")
        cols = assets or [f"A{i}" for i in range(arrays.shape[1])]
        return pd.DataFrame(arrays, index=idx, columns=cols)

    def test_zero_when_no_intraday_move(self):
        """E_p = 0 when w_end == w_adj at every date."""
        w = np.array([[0.4, 0.35, 0.25]] * 5)
        w_end = self._make_df(w)
        w_adj = self._make_df(w.copy())
        E_p = compute_E_p(w_end, w_adj, p=0.5)
        assert np.allclose(E_p.values, 0.0, atol=1e-12)

    def test_non_negative_for_p_less_one(self):
        """E_p ≥ 0 for p < 1.

        In classical SPT the excess-growth rate is positive.  In the discrete
        approximation used here, the normalised Hessian of D_p is negative
        semi-definite on the simplex for 0 < p < 1, so qf_vals ≤ 0 and
        E_p = -cumsum(qf_vals) ≥ 0.
        """
        rng = np.random.default_rng(5)
        T, d = 20, 4
        w_adj_arr = rng.dirichlet(np.ones(d), size=T)
        perturbation = rng.standard_normal((T, d)) * 0.01
        perturbation -= perturbation.mean(axis=1, keepdims=True)
        w_end_arr = np.clip(w_adj_arr + perturbation, 1e-6, None)
        w_end_arr /= w_end_arr.sum(axis=1, keepdims=True)
        w_end = self._make_df(w_end_arr)
        w_adj = self._make_df(w_adj_arr)
        for p in [0.3, 0.5, 0.7]:
            E_p = compute_E_p(w_end, w_adj, p=p)
            assert E_p.iloc[-1] >= -1e-10, f"E_p < 0 for p={p}: {E_p.iloc[-1]:.4e}"

    def test_finite_small_p(self):
        """E_p remains finite for small p with a 10-asset universe."""
        rng = np.random.default_rng(11)
        T, d = 30, 10
        w_adj_arr = rng.dirichlet(np.ones(d), size=T)
        w_end_arr = w_adj_arr.copy()
        w_end = self._make_df(w_end_arr)
        w_adj = self._make_df(w_adj_arr)
        E_p = compute_E_p(w_end, w_adj, p=0.1)
        assert np.isfinite(E_p.values).all()

    def test_index_preserved(self):
        """Output DatetimeIndex matches input."""
        w = np.array([[0.5, 0.5]] * 7)
        w_df = self._make_df(w)
        E_p = compute_E_p(w_df, w_df.copy(), p=0.5)
        assert (E_p.index == w_df.index).all()


# ---------------------------------------------------------------------------
# compute_J_p
# ---------------------------------------------------------------------------

class TestComputeJp:
    def _weight_df(self, arrays):
        idx = pd.date_range("2020-01-01", periods=len(arrays), freq="B")
        return pd.DataFrame(arrays, index=idx)

    def test_zero_on_non_split_dates(self):
        """J_p = 0 everywhere when there are no split dates."""
        T = 10
        nu_adj_arr  = np.tile([0.6, 0.4], (T, 1))
        nu_prev_arr = np.tile([0.6, 0.4], (T, 1))
        idx = pd.date_range("2020-01-01", periods=T, freq="B")
        nu_adj  = pd.DataFrame(nu_adj_arr,  index=idx)
        nu_prev = pd.DataFrame(nu_prev_arr, index=idx)
        split_dates = pd.DatetimeIndex([])
        J_p = compute_J_p(nu_adj, nu_prev, split_dates, 0.5, idx)
        assert np.allclose(J_p.values, 0.0)

    def test_nonzero_on_split_date(self):
        """J_p picks up a non-zero increment on a split date.

        NOTE: nu_adj and nu_prev must NOT be mere permutations of each other,
        because D_p is symmetric: D_p([a,b]) == D_p([b,a]).  Use weights with
        different D_p values so log(ratio) != 0.
        """
        T = 5
        idx = pd.date_range("2020-01-01", periods=T, freq="B")
        # uniform post-split → D_p = 2.0 for p=0.5
        nu_adj_arr  = np.tile([0.5, 0.5], (T, 1))
        # concentrated pre-split → D_p = (√0.8 + √0.2)^2 ≈ 1.80 for p=0.5
        nu_prev_arr = np.tile([0.8, 0.2], (T, 1))
        nu_adj  = pd.DataFrame(nu_adj_arr,  index=idx)
        nu_prev = pd.DataFrame(nu_prev_arr, index=idx)
        split_date = idx[2]
        split_dates = pd.DatetimeIndex([split_date])
        J_p = compute_J_p(nu_adj, nu_prev, split_dates, 0.5, idx)
        # Increment at split_date should be non-zero
        assert abs(J_p.loc[split_date]) > 1e-10

    def test_cumulative_from_split_date(self):
        """After the split, J_p stays constant (no further increments)."""
        T = 8
        idx = pd.date_range("2020-01-01", periods=T, freq="B")
        nu_adj_arr  = np.tile([0.4, 0.6], (T, 1))
        nu_prev_arr = np.tile([0.6, 0.4], (T, 1))
        nu_adj  = pd.DataFrame(nu_adj_arr,  index=idx)
        nu_prev = pd.DataFrame(nu_prev_arr, index=idx)
        split_date = idx[3]
        split_dates = pd.DatetimeIndex([split_date])
        J_p = compute_J_p(nu_adj, nu_prev, split_dates, 0.5, idx)
        # All values after split_date should equal J_p at split_date
        after = J_p[idx >= split_date]
        assert np.allclose(after.values, after.values[0])

    def test_multiple_split_dates(self):
        """J_p accumulates correctly over two split dates."""
        T = 10
        idx = pd.date_range("2020-01-01", periods=T, freq="B")
        nu_adj_arr  = np.tile([0.5, 0.5], (T, 1)).astype(float)
        nu_prev_arr = np.tile([0.6, 0.4], (T, 1)).astype(float)
        nu_adj  = pd.DataFrame(nu_adj_arr,  index=idx)
        nu_prev = pd.DataFrame(nu_prev_arr, index=idx)
        split_dates = pd.DatetimeIndex([idx[2], idx[6]])
        J_p = compute_J_p(nu_adj, nu_prev, split_dates, 0.5, idx)
        # J_p at idx[6] should be larger in absolute magnitude than at idx[2]
        assert abs(J_p.loc[idx[6]]) >= abs(J_p.loc[idx[2]])
