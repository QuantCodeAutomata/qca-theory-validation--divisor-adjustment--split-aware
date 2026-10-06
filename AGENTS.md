# AGENTS.md — Price-Based SPT Repository

## Project Overview
Price-based Stochastic Portfolio Theory (SPT) with stock-split awareness.
All code is in `/workspace/project/`.

## Directory Layout
```
src/                   # Library modules
  __init__.py
  synthetic_paths.py   # GBM price-path generator with scheduled split events
  split_mechanics.py   # Core SPT mechanics (defects Q, C; cross-basis identity)
  benchmarks.py        # Price/cap weight indices, divisor update
  generators.py        # Gibbs entropy, diversity D_p; additive/mult generation
  additive_generation.py  # AdditiveGenerator class + simulation helpers
  multiplicative_restart.py  # MultiplicativeRestartStrategy + glue logic
exp/                   # Self-contained experiment scripts (runnable directly)
  validate_theory_divisor.py       # Exp 1: split mechanics & index theory
  validate_functional_generation.py # Exp 2: additive & multiplicative generation
  replicate_nyse_benchmarks.py     # Exp 3: NYSE benchmark replication
  replicate_nyse_portfolios.py     # Exp 4: diversity portfolio replication
data/                  # Input data files
results/               # All output CSVs and PNGs (auto-created by scripts)
tests/                 # pytest test files
```

## Key Mathematical Conventions

### Price-weight ν vs cap-weight μ
- ν_i(t) = P_i(t) / Σ(t)  where  Σ(t) = Σ_i P_i(t)  (price-weighted index)
- μ_i(t) = S_i(t) / Λ(t)  where  S_i = P_i·N_i,  Λ = Σ_i S_i  (cap-weighted)

### Split multiplier convention
- `m_i(t)` = share multiplier applied at day t (stored in df column `m1`, `m2`)
- Forward 2:1 split: m=2, P /= 2, N *= 2
- Reverse 1:2 split: m=0.5, P /= 0.5 (=×2), N *= 0.5
- **CRITICAL**: `is_split = np.abs(m - 1.0) > 1e-10`  (NOT `m > 1.0` — reverse splits have m<1)

### Adjusted start-of-day weight ν◇
- ν◇_i(t) = P_i(t-1)/m_i(t) / Σ◇(t)  — pre-market, post-adjustment price weight
- Used as the evaluation point for gradient in additive generation

### Divisor κ (DJIA-style)
- κ(t) = κ(t-1) · Σ(t) / Σ(t-1)  on split days  →  Σ̃(t) = Σ̃(t-1)  (exact)
- κ(t) = κ(t-1)  on non-split days

## Known Gotchas / Lessons Learned

1. **Reverse splits and `is_split`**: Always use `np.abs(m - 1.0) > 1e-10` to flag split
   events. Using `m > 1.0` misses reverse splits (m=0.5) and breaks tests H3, H4, H7 in Exp 1.

2. **Additive identity tolerance**: The corrected master identity
   V^φ(t) = H^c(ν(t)) + Γ^{H,c}(t) is a first-order Taylor approximation.
   The O(vol²·T) error is ~5e-3 for vol=0.008, T=60. Use `TOL_IDENTITY = 1e-2`.

3. **Matplotlib format strings**: `"r:.^"` is invalid (two marker symbols). Use `"r--^"` etc.

4. **naive_lhs interpretation**: naive_lhs = G(ν(t)) + Γ^{G,d}(t). Under repeated splits,
   Γ^{G,d} accumulates large negative values, driving naive_lhs strongly negative (to -1.06
   for 10 splits with delta=0.25). Actual wealth V^φ stays positive. This is the Example 3.5
   pathological behavior.

5. **`compute_diversity_portfolio_weights`** lives in `src/multiplicative_restart.py`,
   NOT in `src/generators.py`.

## Running Experiments
```bash
cd /workspace/project
python exp/validate_theory_divisor.py       # Exp 1 (all 8 hypotheses should PASS)
python exp/validate_functional_generation.py # Exp 2 (all 7 hypotheses should PASS)
```

## Dependencies
numpy, pandas, matplotlib, seaborn, scipy, sympy, pytest
Python: /usr/local/bin/python
