"""
exp/validate_theory_divisor.py  –  Experiment 1
================================================
Validates the core price-based SPT quantity relationships in a two-asset
market with controlled stock-split events.

Market setup (Example 2.5 analog)
----------------------------------
  d = 2 assets, T = 30 trading days
  Asset 1 (index 0):
    - 2:1 forward split at t = 10  (price halves, shares double)
    - 1:2 reverse split at t = 20  (price doubles, shares halve)
  Asset 2 (index 1): continuous GBM, no splits.

Notation (mirroring split_mechanics.py)
-----------------------------------------
  P_i(t)     – nominal price of asset i at end of day t
  N_i(t)     – shares outstanding of asset i
  S_i(t)     – market cap  P_i(t) * N_i(t)
  m_i(t)     – split multiplier (1.0 on non-split days)
  Σ(t)       – raw total price  sum_i P_i(t)
  κ(t)       – price-index divisor  (updated at splits to keep Σ̃ continuous)
  Σ̃(t)      – adjusted index  Σ(t) / κ(t)
  Λ(t)       – total market cap  sum_i S_i(t)
  ν_i(t)     – price weight  P_i(t) / Σ(t)
  μ_i(t)     – cap weight  S_i(t) / Λ(t)

Hypotheses verified
--------------------
  H1. S_i(t) preserved across pure splits: market cap unchanged at split moment.
  H2. ν_i jumps at split times (price weight discontinuity).
  H3. Σ(t) jumps at split times (raw price index discontinuity).
  H4. Σ̃(t) does NOT jump at split times (divisor-adjusted index continuity).
  H5. Self-financing portfolio: wealth preserved under predictable share adjustments.
  H6. Defect Q ≈ 0  (price-weight identity: V = κ · (ϑ·ν) · Σ̃).
  H7. Defect C accumulates at split times (adjustment component from free shares).
  H8. Cross-basis identity: (V_cap / V_price) ≈ Σ̃ / Λ.

Usage
-----
  python exp/validate_theory_divisor.py
"""

import sys
import os

# Ensure src is importable when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.synthetic_paths import generate_two_asset_paths
from src.split_mechanics import (
    compute_price_weights,
    compute_cap_weights,
    divisor_update,
    compute_adjusted_start_of_day,
    self_financing_check,
    compute_defect_Q,
    compute_defect_C,
    compute_cross_basis_identity,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

T     = 30
SEED  = 42
TOL   = 1e-8    # numerical tolerance for hypothesis tests

SPLIT_EVENTS = [
    {"t": 10, "asset": 1, "multiplier": 2.0},   # 2:1 forward split at t=10
    {"t": 20, "asset": 1, "multiplier": 0.5},   # 1:2 reverse split at t=20
]

PASS = "✓ PASS"
FAIL = "✗ FAIL"

# ---------------------------------------------------------------------------
# Step 1: Generate raw path
# ---------------------------------------------------------------------------

print("=" * 70)
print("Experiment 1 – Validate Theory Divisor (Example 2.5 analog)")
print("=" * 70)
print(f"\nGenerating two-asset market: T={T}, seed={SEED}")
print("  Asset 1: 2:1 forward split at t=10, 1:2 reverse split at t=20")
print("  Asset 2: continuous GBM, no splits\n")

raw_df = generate_two_asset_paths(
    T=T,
    seed=SEED,
    split_events=SPLIT_EVENTS,
)

# Extract arrays (1-based asset labeling in the DataFrame)
t_arr = raw_df["t"].values      # shape (T+1,)
P1    = raw_df["P1"].values      # nominal prices
P2    = raw_df["P2"].values
N1    = raw_df["N1"].values      # shares outstanding
N2    = raw_df["N2"].values
S1    = raw_df["S1"].values      # market caps  S_i = P_i * N_i
S2    = raw_df["S2"].values
m1    = raw_df["m1"].values      # split multipliers
m2    = raw_df["m2"].values

# Stack into 2-D arrays for convenience
P     = np.column_stack([P1, P2])    # (T+1, 2)
N     = np.column_stack([N1, N2])
S_cap = np.column_stack([S1, S2])   # market caps (S_i = P_i * N_i in notation)
m     = np.column_stack([m1, m2])   # split multipliers

is_split = np.abs(m - 1.0) > 1e-10   # True on any split day (forward or reverse)

# ---------------------------------------------------------------------------
# Step 2: Build full derived path (σ, κ, Σ̃, ν, μ, etc.)
# ---------------------------------------------------------------------------

d = 2
Sigma      = P.sum(axis=1)             # raw total price Σ(t)
Lambda_mc  = S_cap.sum(axis=1)         # total market cap Λ(t) = sum_i S_i

# Price-index divisor κ(t): starts at 1, updated at split days
kappa_div  = np.empty(T + 1)
kappa_div[0] = 1.0
for t in range(1, T + 1):
    kappa_div[t] = divisor_update(
        kappa_prev=kappa_div[t - 1],
        sigma_prev=Sigma[t - 1],
        sigma_curr=Sigma[t],
        is_split_day=bool(is_split[t].any()),
    )

Sigma_tilde = Sigma / kappa_div        # divisor-adjusted index Σ̃(t) = Σ/κ

# Price weights ν_i and cap weights μ_i
nu = np.zeros((T + 1, d))
mu = np.zeros((T + 1, d))
for t in range(T + 1):
    nu[t] = compute_price_weights(P[t])
    mu[t] = compute_cap_weights(S_cap[t])

# Adjusted start-of-day prices P^◇ and caps S^◇
P_diamond = np.zeros((T + 1, d))
S_diamond = np.zeros((T + 1, d))
P_diamond[0] = P[0]
S_diamond[0] = S_cap[0]
for t in range(1, T + 1):
    P_dmd_t, S_dmd_t = compute_adjusted_start_of_day(P[t - 1], N[t], m[t])
    P_diamond[t] = P_dmd_t
    S_diamond[t] = S_dmd_t

# ---------------------------------------------------------------------------
# Step 3: Build a buy-and-hold portfolio (self-financing with split adjustment)
# A portfolio that starts with 1 share of each asset at t=0 and receives
# free shares at each split (ϑ_i(t) = N_i(t) / N_i(0)).
# ---------------------------------------------------------------------------

theta = np.zeros((T + 1, d))
theta[0] = np.array([1.0, 1.0])       # start with 1 share each
for t in range(1, T + 1):
    # On any day, holdings = proportional to share count ratio
    theta[t, 0] = theta[0, 0] * (N1[t] / N1[0])
    theta[t, 1] = theta[0, 1] * (N2[t] / N2[0])

# Self-financing check: verify the adjustment component is zero on non-split days
sf = self_financing_check(theta, P, is_split)
wealth         = sf["wealth"]
adj_component  = sf["adjustment_component"]
defect_sf      = sf["defect"]

# ---------------------------------------------------------------------------
# Step 4: Defects Q and C
# ---------------------------------------------------------------------------

Q_defect = compute_defect_Q(theta, nu, P, kappa_div)
C_defect = compute_defect_C(theta, nu, P, kappa_div, Sigma_tilde)

# ---------------------------------------------------------------------------
# Step 5: Cross-basis identity
# ---------------------------------------------------------------------------

cross_basis = compute_cross_basis_identity(
    nu=nu,
    mu=mu,
    N=N,
    P=P,
    S=S_cap,
    kappa=kappa_div,
    Sigma_tilde=Sigma_tilde,
)

# ---------------------------------------------------------------------------
# Step 6: Run all hypothesis tests
# ---------------------------------------------------------------------------

results_tests = {}

# H1: Market cap preserved at split times
# S_i(t-1) ≈ P_diamond_i(t) * N_i(t) (cap at the opening post-split = cap before)
h1_errors = []
for t in range(1, T + 1):
    if is_split[t].any():
        for i in range(d):
            if is_split[t, i]:
                cap_before = S_cap[t - 1, i]
                cap_after  = P_diamond[t, i] * N[t, i]
                h1_errors.append(abs(cap_before - cap_after) / max(cap_before, 1e-15))
h1_max_rel_err = max(h1_errors) if h1_errors else 0.0
results_tests["H1_cap_preserved"] = (h1_max_rel_err < TOL, h1_max_rel_err)

# H2: ν_i jumps at split times (should be non-trivial change for asset 0 at t=10,20)
h2_nu_changes_at_splits = []
for t in range(1, T + 1):
    if is_split[t].any():
        delta_nu = np.max(np.abs(nu[t] - nu[t - 1]))
        h2_nu_changes_at_splits.append((t, delta_nu))
# Expect significant jumps (> some threshold say 0.05)
h2_min_jump = min(v for _, v in h2_nu_changes_at_splits) if h2_nu_changes_at_splits else 0
results_tests["H2_nu_jumps"] = (h2_min_jump > 0.01, h2_min_jump)

# H3: Σ(t) jumps at split times
h3_sigma_jumps = []
for t in range(1, T + 1):
    if is_split[t].any():
        delta_sigma = abs(Sigma[t] - Sigma[t - 1])
        h3_sigma_jumps.append((t, delta_sigma))
h3_min_jump = min(v for _, v in h3_sigma_jumps) if h3_sigma_jumps else 0
results_tests["H3_Sigma_jumps"] = (h3_min_jump > 0.01, h3_min_jump)

# H4: Σ̃(t) does NOT jump at split times
h4_tilde_jumps = []
for t in range(1, T + 1):
    if is_split[t].any():
        delta_tilde = abs(Sigma_tilde[t] - Sigma_tilde[t - 1])
        h4_tilde_jumps.append((t, delta_tilde))
h4_max_jump = max(v for _, v in h4_tilde_jumps) if h4_tilde_jumps else 0.0
results_tests["H4_Sigma_tilde_no_jump"] = (h4_max_jump < TOL, h4_max_jump)

# H5: Self-financing defect is zero (algebraic identity check)
h5_max_defect = float(np.nanmax(np.abs(defect_sf[1:])))
results_tests["H5_self_financing"] = (h5_max_defect < TOL, h5_max_defect)

# H6: Defect Q ≈ 0 (price-weight identity)
h6_max_Q = float(np.max(np.abs(Q_defect)))
results_tests["H6_defect_Q_zero"] = (h6_max_Q < TOL, h6_max_Q)

# H7: Defect C accumulates only at split times (zero on non-split days)
C_increments = np.diff(C_defect)
non_split_C = [abs(C_increments[t - 1]) for t in range(1, T + 1) if not is_split[t].any()]
h7_max_non_split_C = max(non_split_C) if non_split_C else 0.0
results_tests["H7_defect_C_split_only"] = (h7_max_non_split_C < TOL, h7_max_non_split_C)

# H8: Cross-basis identity holds
h8_max_err = cross_basis["max_abs_error"]
results_tests["H8_cross_basis"] = (h8_max_err < TOL, h8_max_err)

# ---------------------------------------------------------------------------
# Step 7: Print Example 2.5 analog table
# ---------------------------------------------------------------------------

print("─" * 70)
print("Example 2.5 Analog: Key Quantities")
print("─" * 70)
header = (f"{'t':>4}  {'P1':>8}  {'P2':>8}  {'N1':>8}  {'N2':>8}  "
          f"{'nu_1':>7}  {'nu_2':>7}  {'Sigma':>8}  {'kappa':>8}  {'Sigma~':>8}  "
          f"{'m1':>5}  {'m2':>5}")
print(header)
print("─" * len(header))

SPLIT_MARKER_TS = {e["t"] for e in SPLIT_EVENTS}
for t in range(T + 1):
    marker = " <-- SPLIT" if t in SPLIT_MARKER_TS else ""
    print(
        f"{t:>4}  {P1[t]:>8.3f}  {P2[t]:>8.3f}  {N1[t]:>8.0f}  {N2[t]:>8.0f}  "
        f"{nu[t,0]:>7.4f}  {nu[t,1]:>7.4f}  {Sigma[t]:>8.3f}  "
        f"{kappa_div[t]:>8.6f}  {Sigma_tilde[t]:>8.3f}  "
        f"{m1[t]:>5.2f}  {m2[t]:>5.2f}{marker}"
    )

# ---------------------------------------------------------------------------
# Step 8: Print hypothesis test results
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("Hypothesis Test Results")
print("=" * 70)
labels = {
    "H1_cap_preserved":       "H1: S_i preserved at splits (rel. error)",
    "H2_nu_jumps":            "H2: ν jumps at splits (min jump)",
    "H3_Sigma_jumps":         "H3: Σ jumps at splits (min jump)",
    "H4_Sigma_tilde_no_jump": "H4: Σ̃ continuous at splits (max jump)",
    "H5_self_financing":      "H5: Self-financing defect (max |defect|)",
    "H6_defect_Q_zero":       "H6: Defect Q ≈ 0 (max |Q|)",
    "H7_defect_C_split_only": "H7: Defect C non-split increment (max)",
    "H8_cross_basis":         "H8: Cross-basis identity (max |LHS-RHS|)",
}
all_passed = True
for key, label in labels.items():
    passed, value = results_tests[key]
    status = PASS if passed else FAIL
    if not passed:
        all_passed = False
    print(f"  {status}  {label}: {value:.3e}")

print("\n" + ("ALL HYPOTHESES PASSED ✓" if all_passed else "SOME HYPOTHESES FAILED ✗"))

# ---------------------------------------------------------------------------
# Step 9: Build and save results table
# ---------------------------------------------------------------------------

table_rows = []
for t in range(T + 1):
    table_rows.append({
        "t":            t,
        "P1":           P1[t],
        "P2":           P2[t],
        "N1":           N1[t],
        "N2":           N2[t],
        "S1":           S1[t],
        "S2":           S2[t],
        "m1":           m1[t],
        "m2":           m2[t],
        "Sigma":        Sigma[t],
        "kappa_div":    kappa_div[t],
        "Sigma_tilde":  Sigma_tilde[t],
        "Lambda_mc":    Lambda_mc[t],
        "nu_1":         nu[t, 0],
        "nu_2":         nu[t, 1],
        "mu_1":         mu[t, 0],
        "mu_2":         mu[t, 1],
        "theta_1":      theta[t, 0],
        "theta_2":      theta[t, 1],
        "wealth":       wealth[t],
        "defect_Q":     Q_defect[t],
        "defect_C":     C_defect[t],
        "cross_LHS":    cross_basis["LHS"][t],
        "cross_RHS":    cross_basis["RHS"][t],
    })

table_df = pd.DataFrame(table_rows)
csv_path = os.path.join(RESULTS_DIR, "exp1_table.csv")
table_df.to_csv(csv_path, index=False)
print(f"\nResults table saved → {csv_path}")

# ---------------------------------------------------------------------------
# Step 10: Generate plots
# ---------------------------------------------------------------------------

# ── Plot 1: Prices, weights, and divisor ──────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Experiment 1: Two-Asset Market with Splits", fontsize=13, fontweight="bold")

ax = axes[0, 0]
ax.plot(t_arr, P1, "b-o", ms=3, label="$P_1$ (splits at t=10,20)")
ax.plot(t_arr, P2, "r-s", ms=3, label="$P_2$ (no splits)")
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("Price")
ax.set_title("Nominal Prices"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t_arr, N1, "b-o", ms=3, label="$N_1$")
ax.plot(t_arr, N2, "r-s", ms=3, label="$N_2$")
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("Shares")
ax.set_title("Shares Outstanding"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(t_arr, Sigma,       "k-o",  ms=3, label="$\\Sigma$ (raw, jumps)")
ax.plot(t_arr, Sigma_tilde, "g-s",  ms=3, label="$\\tilde{\\Sigma}$ (adjusted, continuous)")
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("Index value")
ax.set_title("Price Index (raw vs adjusted)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(t_arr, nu[:, 0], "b-o", ms=3, label="$\\nu_1$ (jumps at splits)")
ax.plot(t_arr, nu[:, 1], "r-s", ms=3, label="$\\nu_2$")
ax.plot(t_arr, mu[:, 0], "b--", ms=3, label="$\\mu_1$", alpha=0.5)
ax.plot(t_arr, mu[:, 1], "r--", ms=3, label="$\\mu_2$", alpha=0.5)
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("Weight")
ax.set_title("Price weights $\\nu$ (solid) vs Cap weights $\\mu$ (dashed)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

ax = axes[2, 0]
ax.plot(t_arr, kappa_div, "m-o", ms=3, label="$\\kappa$ (divisor)")
ax.set_xlabel("t"); ax.set_ylabel("Divisor $\\kappa$")
ax.set_title("Price-index Divisor"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")

ax = axes[2, 1]
ax.plot(t_arr, wealth,    "b-o",  ms=3, label="Wealth $V^{\\vartheta}$")
ax.plot(t_arr, C_defect,  "r--s", ms=3, label="Cumulative defect C")
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("Value")
ax.set_title("Buy-and-Hold Wealth and Defect C")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plot1_path = os.path.join(RESULTS_DIR, "exp1_prices_weights.png")
fig.savefig(plot1_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved → {plot1_path}")

# ── Plot 2: Split-time zoom + cross-basis ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
fig.suptitle("Experiment 1: Split Events and Cross-Basis Identity", fontsize=13, fontweight="bold")

# Σ continuity zoom around t=10
ax = axes[0, 0]
t_zoom = slice(8, 13)
ax.plot(t_arr[t_zoom], Sigma[t_zoom],       "k-o",  ms=5, label="$\\Sigma$ (raw)")
ax.plot(t_arr[t_zoom], Sigma_tilde[t_zoom], "g-s",  ms=5, label="$\\tilde{\\Sigma}$")
ax.axvline(10, color="gray", lw=1.0, ls="--", label="Split t=10")
ax.set_xlabel("t"); ax.set_ylabel("Index value")
ax.set_title("Zoom: Split at t=10 (Σ jumps, Σ̃ continuous)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ν zoom around t=20
ax = axes[0, 1]
t_zoom2 = slice(18, 23)
ax.plot(t_arr[t_zoom2], nu[t_zoom2, 0], "b-o", ms=5, label="$\\nu_1$ (jumps)")
ax.plot(t_arr[t_zoom2], nu[t_zoom2, 1], "r-s", ms=5, label="$\\nu_2$")
ax.axvline(20, color="gray", lw=1.0, ls="--", label="Split t=20")
ax.set_xlabel("t"); ax.set_ylabel("Price weight")
ax.set_title("Zoom: Reverse Split at t=20 (ν jumps)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Cross-basis identity: LHS vs RHS
ax = axes[1, 0]
ax.plot(t_arr, cross_basis["LHS"], "b-o", ms=3, label="LHS")
ax.plot(t_arr, cross_basis["RHS"], "r--s", ms=3, label="RHS")
ax.set_xlabel("t"); ax.set_ylabel("Ratio")
ax.set_title(f"Cross-Basis Identity (max |LHS-RHS|={h8_max_err:.2e})")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Market cap S_i over time
ax = axes[1, 1]
ax.plot(t_arr, S1, "b-o", ms=3, label="$S_1 = P_1 N_1$ (market cap)")
ax.plot(t_arr, S2, "r-s", ms=3, label="$S_2 = P_2 N_2$")
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")
ax.set_xlabel("t"); ax.set_ylabel("Market cap")
ax.set_title("Market Caps $S_i = P_i N_i$ (preserved at splits)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plot2_path = os.path.join(RESULTS_DIR, "exp1_splits_crossbasis.png")
fig.savefig(plot2_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved → {plot2_path}")

# ── Plot 3: Defect summary ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Experiment 1: Defects Q and C", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(t_arr, Q_defect, "b-o", ms=3, label="|Q| (should be ≈ 0)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("t"); ax.set_ylabel("Q(t)")
ax.set_title(f"Defect Q (max={np.max(np.abs(Q_defect)):.1e})")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")

ax = axes[1]
ax.plot(t_arr, C_defect, "r-o", ms=3, label="Cumulative C")
ax.set_xlabel("t"); ax.set_ylabel("C(t)")
ax.set_title("Defect C (accumulates at splits)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
for t_sp in SPLIT_MARKER_TS:
    ax.axvline(t_sp, color="gray", lw=0.8, ls="--")

plt.tight_layout()
plot3_path = os.path.join(RESULTS_DIR, "exp1_defects.png")
fig.savefig(plot3_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"Plot saved → {plot3_path}")

# Final summary
print("\n" + "=" * 70)
print("Experiment 1 complete.")
print("=" * 70)
