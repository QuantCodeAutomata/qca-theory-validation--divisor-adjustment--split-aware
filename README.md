# QCA Theory Validation: Divisor Adjustment, Split-Aware SPT and Diversity Portfolios

## Overview

This repository provides a Python implementation and empirical validation of
**price-based Stochastic Portfolio Theory (SPT)** extended to handle stock
splits and other share-count discontinuities.  Classical SPT works with
capitalisation weights and is invariant to stock splits by construction.
Price-based SPT, which expresses portfolios in terms of raw prices and a
*divisor* (analogous to the divisor used in price-weighted indices such as the
Dow Jones Industrial Average), requires explicit bookkeeping whenever a
corporate action changes the number of shares outstanding without changing
total market capitalisation.

The central object is the *normalised total price*

```
Σ̃(t) = Σ(t) / κ(t),    Σ(t) = Σ_i P_i(t)
```

where the divisor `κ(t)` is updated only on split days so that `Σ̃` is
continuous across splits.  The repository reproduces the key identities and
defect decompositions from the accompanying paper (referred to below as *QCA*),
verifies the self-financing condition for split-aware strategies, and
demonstrates that naive price-weight portfolios accumulate a non-zero defect
whenever splits occur.

### Key theoretical contributions validated

| Symbol | Description |
|--------|-------------|
| `κ(t)` | Price-index divisor, updated on split days only |
| `Σ̃(t)` | Divisor-normalised total price |
| `ν_i(t)` | Price weight `P_i / Σ` |
| `Q^{ϑ,ν}(t)` | Wealth defect from naive price weighting (eq. 2.9) |
| `C^{ϑ,ν}(t)` | Cumulative jump defect from split discontinuities (eq. 2.10) |
| Cross-basis identity | Ratio of cap-basis to price-basis wealth equals `Σ̃ / Λ` |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md          # Description of synthetic data storage
│
├── exp/                   # Experiment scripts (one file per experiment)
│   ├── exp1_divisor_adjustment.py
│   ├── exp2_self_financing.py
│   ├── exp3_split_reset.py
│   └── exp4_diversity_portfolio.py
│
├── results/               # Figures and numerical outputs (not gitignored)
│
├── src/
│   ├── __init__.py
│   ├── synthetic_paths.py # Synthetic price-path generators
│   └── split_mechanics.py # Divisor, weight, defect, and detection functions
│
└── tests/
    ├── test_synthetic_paths.py
    └── test_split_mechanics.py
```

---

## Experiments

### Experiment 1 — Divisor Adjustment and the Normalised Total Price

**Script:** `exp/exp1_divisor_adjustment.py`

Generates a two-asset price path with a single 2-for-1 stock split for asset 1
at `t = 10`.  The experiment plots the raw total price `Σ(t)` alongside the
divisor-normalised price `Σ̃(t)` to confirm that `Σ̃` is continuous across the
split date while `Σ` exhibits a jump.  It also records the divisor `κ(t)` and
verifies that the divisor update rule

```
κ(t) = κ(t−1) · Σ(t) / Σ(t−1)    [on split days only]
```

correctly absorbs the jump.  Numerical output includes the maximum absolute
deviation of `Σ̃` before versus after the split.

### Experiment 2 — Self-Financing Condition and Defect Decomposition

**Script:** `exp/exp2_self_financing.py`

Constructs an equal-price-weight (ν) portfolio rebalanced daily and verifies the
self-financing identity for split-aware strategies.  The defect `Q^{ϑ,ν}(t)`
(eq. 2.9) and the jump-defect `C^{ϑ,ν}(t)` (eq. 2.10) are computed at each
time step.  The experiment confirms that:

* A split-aware (divisor-corrected) strategy accumulates zero defect.
* A naive price-weight strategy accumulates a defect proportional to the
  magnitude of the split.

Results are displayed as time-series plots of `Q` and `C` for both strategies.

### Experiment 3 — Repeated Split-Reset Paths (Example 3.5 Style)

**Script:** `exp/exp3_split_reset.py`

Simulates multi-period dynamics where assets appreciate at a constant rate and
undergo periodic splits every `split_period` steps (the *split-reset* regime of
Example 3.5 in the paper).  The experiment tracks the long-run behaviour of:

* The raw price path vs the adjusted path.
* The cumulative log-return of a price-weight portfolio vs a
  capitalisation-weight portfolio.
* The cross-basis identity ratio `Σ̃ / Λ` over time.

This illustrates that, in the split-reset regime, the divisor grows without
bound while `Σ̃` remains bounded, and that the cap-weight portfolio dominates
the naive price-weight portfolio after a sufficient number of splits.

### Experiment 4 — Diversity-Weighted Portfolio under Splits

**Script:** `exp/exp4_diversity_portfolio.py`

Uses a `d = 5` asset universe with heterogeneous volatilities and scheduled
splits at different times for different assets.  Implements the *diversity
portfolio* `π^p_i = μ_i^p / Σ_j μ_j^p` for `p ∈ {0.5, 1.0, 2.0}` where
`μ_i` denotes capitalisation weights.  The experiment:

1. Detects split events heuristically from simulated share-count data using
   `detect_split_events`.
2. Computes adjusted start-of-day prices `P⋄_i(t)` and `S⋄_i(t)`.
3. Evaluates the cross-basis identity numerically.
4. Compares terminal wealth of diversity portfolios across the three values of
   `p`, with and without split correction, to quantify the impact of ignoring
   splits on portfolio performance.

---

## Installation

Requires Python 3.9 or later.

```bash
# Clone the repository
git clone <repo-url>
cd <repo-directory>

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running individual experiments

```bash
python exp/exp1_divisor_adjustment.py
python exp/exp2_self_financing.py
python exp/exp3_split_reset.py
python exp/exp4_diversity_portfolio.py
```

Each script saves figures to `results/` and prints a numerical summary to
standard output.

### Running the test suite

```bash
pytest tests/ -v
```

### Using the library directly

```python
from src.synthetic_paths import generate_two_asset_paths, generate_multi_asset_paths
from src.split_mechanics import (
    compute_price_weights,
    compute_cap_weights,
    divisor_update,
    detect_split_events,
    self_financing_check,
    compute_defect_Q,
    compute_defect_C,
    compute_cross_basis_identity,
)

# Generate a two-asset path with a 2-for-1 split of asset 1 at t=10
df = generate_two_asset_paths(
    T=30,
    split_events=[{'t': 10, 'asset': 1, 'multiplier': 2.0}],
)

# Inspect prices, shares, and capitalisations
print(df[['t', 'P1', 'P2', 'N1', 'N2', 'S1', 'S2', 'm1', 'm2']].head(15))
```
