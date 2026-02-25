# Shock Propagation Framework

## Overview

The shock propagation framework models asset returns as driven by underlying risk factors (shocks) with regime-dependent loadings. This enables:

1. **Factor-driven returns:** Decompose returns into systematic (shock-driven) and idiosyncratic components
2. **Stress scenarios:** Deterministically apply shock scenarios to compute stressed returns
3. **Stochastic simulation:** Monte Carlo paths with regime-switching shocks
4. **Variance decomposition:** Analyze how much variance is explained by each shock factor

## Mathematical Formulation

### Return Dynamics

Returns are modeled as:
$$r_t = \mu_{s_t} + B_{s_t} u_t + \varepsilon_t$$

where:
- $r_t \in \mathbb{R}^N$ — asset returns at time $t$
- $s_t \in \{1, \ldots, K\}$ — regime at time $t$ (stochastic, from Markov chain)
- $\mu_{s_t} \in \mathbb{R}^N$ — regime-conditional mean return
- $B_{s_t} \in \mathbb{R}^{N \times M}$ — regime-conditional loading matrix
- $u_t \in \mathbb{R}^M$ — shock vector (systematic risk factors)
- $\varepsilon_t \in \mathbb{R}^N$ — idiosyncratic noise
- $K$ — number of regimes
- $N$ — number of assets
- $M$ — number of shocks (fixed, typically 2-5)

### Component Interpretation

**Mean Component:** $\mu_{s_t}$
- The expected return given the current regime
- Regime-switching allows different average returns in calm vs. stressed regimes

**Shock Component:** $B_{s_t} u_t$
- Factor-driven returns
- Loading matrix $B_{s_t}$ determines how each asset responds to each shock
- Allows regime-dependent sensitivity (e.g., correlation with shocks can shift)

**Idiosyncratic Component:** $\varepsilon_t$
- Firm-specific risk not explained by factors
- Assumed $\varepsilon_t \sim N(0, \Sigma_{s_t})$
- Covariance $\Sigma_{s_t}$ is regime-conditional

### Shock Process

The shock vector $u_t$ can be:

**Deterministic (stress testing):**
$$u_t = \bar{u} \quad \text{(fixed vector)}$$
Used to apply predetermined scenarios, e.g., "bond yields rise by 100bp, equity volatility spikes by 20%".

**Stochastic (Monte Carlo):**
$$u_t \sim N(0, \Sigma_u)$$
For forward-looking simulation. Default: $\Sigma_u = I_M$ (standardized shocks).

### Variance Decomposition

Total variance:
$$\text{Var}(r_t | s_t) = B_{s_t} \Sigma_u B_{s_t}^T + \Sigma_{s_t}$$

Breaking into:
- **Systematic variance:** $B_{s_t} \Sigma_u B_{s_t}^T$ — driven by shocks
- **Idiosyncratic variance:** $\Sigma_{s_t}$ — firm-specific risk

Percentage of variance explained by shocks:
$$\text{systematic ratio} = \frac{\text{trace}(B_{s_t} \Sigma_u B_{s_t}^T)}{\text{trace}(B_{s_t} \Sigma_u B_{s_t}^T + \Sigma_{s_t})}$$

This ratio varies by regime, allowing shocks to be more or less important depending on market conditions.

## Implementation Details

### Class: `ShockModel`

Manages loading matrices and shock mechanics.

**Attributes:**
- `n_assets` — number of assets (N)
- `n_shocks` — number of shock factors (M)
- `n_regimes` — number of regimes (K)
- `loading_matrices` — array of shape (K, N, M)

**Key Methods:**

```python
def compute_shock_impact(shocks: NDArray, regime: int) -> NDArray:
    """
    Compute shock-driven return: B_{regime} @ shocks
    
    Shape: (M,) @ (N, M) -> (N,)
    """
```

```python
def simulate_shocks(n_steps: int, shock_std=None, random_seed=None) -> NDArray:
    """
    Generate stochastic shock process u_t ~ N(0, diag(shock_std)^2)
    
    Returns: (n_steps, M) array
    """
```

```python
def stress_test(shock_scenario: NDArray, regime: int, mean_return=None) -> NDArray:
    """
    Deterministic stress: r = mu + B_{regime} @ shock_scenario
    
    Useful for "what if yields rise 100bp?" scenarios
    """
```

### Class: `ReturnWithShocks`

Composes regime-conditional means, shock loadings, and idiosyncratic noise.

**Attributes:**
- `shock_model` — ShockModel instance
- `regime_means` — shape (K, N) array of regime-conditional means
- `regime_covs` — shape (K, N, N) array of regime-conditional idiosyncratic covariances

**Key Methods:**

```python
def generate_returns(regime_path, shocks, idiosyncratic_noise=None, random_seed=None) -> NDArray:
    """
    Generate full return path: r_t = mu_{s_t} + B_{s_t} u_t + eps_t
    
    Inputs:
    - regime_path: (T,) array of regime indices
    - shocks: (T, M) array of shock values
    - idiosyncratic_noise: (T, N) array (sampled if None)
    
    Returns: (T, N) array of asset returns
    """
```

```python
def compute_systematic_variance(regime, shock_cov=None) -> NDArray:
    """
    Systematic covariance: B_k @ Sigma_u @ B_k^T
    
    Shape: (N, N)
    """
```

```python
def variance_decomposition(regime, shock_cov=None) -> Dict:
    """
    Decompose total variance into systematic and idiosyncratic components.
    
    Returns: {
        'systematic_variance': float,      # trace(systematic cov)
        'idiosyncratic_variance': float,   # trace(idiosyncratic cov)
        'systematic_ratio': float          # fraction of variance from shocks
    }
    """
```

## Usage Examples

### Example 1: Stress Test (Bond Yields + Equity Volatility)

```python
import numpy as np
from regimes.shocks import ShockModel

# Define shock model: 2 assets, 2 shocks, 1 regime
B = np.array([
    [1.0, -2.0],    # Bond: affected by yields (↑) and vol (↓)
    [0.5,  3.0]     # Equity: affected by yields (↑) and vol (↑)
])
shock_model = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, 
                         loading_matrices=B[np.newaxis, :, :])

# Stress scenario: yields +100bp, vol spike +20%
u_stress = np.array([1.0, 0.2])

# Apply with mean returns
mu = np.array([0.05, 0.08])
stressed_returns = shock_model.stress_test(u_stress, regime=0, mean_return=mu)
# Result: Bond return stressed down, Equity up
```

### Example 2: Variance Decomposition

```python
from regimes.shocks import ReturnWithShocks

# Continue from above
mu_all = mu[np.newaxis, :]  # (1, 2)
Sigma = np.eye(2)[np.newaxis, :, :]  # Identity idiosyncratic covariance
rws = ReturnWithShocks(shock_model, mu_all, Sigma)

# Analyze variance sources
decomp = rws.variance_decomposition(regime=0, shock_cov=np.eye(2))
print(f"Systematic variance: {decomp['systematic_variance']:.4f}")
print(f"Idiosyncratic variance: {decomp['idiosyncratic_variance']:.4f}")
print(f"Ratio: {decomp['systematic_ratio']:.2%}")
# Output: Shocks explain X%, firm-specific risk explains (1-X)%
```

### Example 3: Monte Carlo Simulation

```python
from regimes.markov import MarkovChain
import numpy as np

# Set up regime-switching
P = np.array([[0.95, 0.05], [0.10, 0.90]])
mc = MarkovChain(P)

# Simulate regime path
regime_path = mc.simulate_path(n_steps=252, initial_regime=0, random_seed=42)

# Simulate shocks
shocks = shock_model.simulate_shocks(n_steps=252, random_seed=42)

# Generate full return paths
returns = rws.generate_returns(regime_path, shocks, random_seed=42)
# returns shape: (252, 2) — one year of daily returns
```

## Graphical Model (DAG)

```
Regime s_t
    ↓
    +─→ μ_{s_t} ─┐
    │            ├→ r_t (return)
    +─→ B_{s_t} ─┤
         ↓       │
         u_t ────┤
         (shock) │
                 │
    ε_t ─────────┘
    (idiosyncratic)
```

**Dependencies:**
- Returns depend on **current regime** s_t
- Regime determines both mean and loading matrix
- Shocks are **exogenous** (not regime-dependent in this model)
- Idiosyncratic noise is regime-conditional (variance depends on s_t)

## Computational Notes

### Numerical Stability

1. **Cholesky decomposition:** For sampling idiosyncratic noise, we use Cholesky decomposition of $\Sigma_{s_t}$ rather than eigendecomposition. This is numerically stable even for ill-conditioned covariances.

2. **Log-determinant:** The variance decomposition uses `np.trace()` rather than log-determinant, avoiding numerical issues with nearly-singular matrices.

3. **Type safety:** All matrices are stored as `np.float64`. Input validation ensures row-stochasticity, symmetry, and positive-definiteness.

### Time Complexity

- `compute_shock_impact`: O(NM) — matrix-vector product
- `simulate_shocks`: O(TM) — T time steps, M shocks
- `generate_returns`: O(TN²) — T time steps, N² from noise sampling (Cholesky per regime)
- `variance_decomposition`: O(N³) — matrix multiplications

For typical use (N=10-50 assets, M=2-5 shocks, T=252-1000 steps), this is well under 1 second.

## Parameters & Priors (for Bayesian Inference)

### Prior on Loading Matrices

In the context of PyMC inference (Commit 5), priors on $B_k$ could be:
- **Centered:** $B_{knm} \sim N(0, \sigma_B)$ — mild regularization
- **Sparse:** $B_{knm} \sim \text{Laplace}(0, \lambda)$ — some loadings driven to zero
- **Informed:** Use domain knowledge (e.g., bonds always have negative yield sensitivity)

### Prior on Idiosyncratic Variance

- $\text{diag}(\Sigma_k) \sim \text{HalfNormal}(\sigma)$ — half-normal for per-asset volatility
- $\Sigma_k \sim \text{LKJ}(\eta) \times \text{HalfNormal}$ — matrix with LKJ correlation prior

### Shock Covariance

If shocks themselves are inferred:
- $\Sigma_u \sim \text{LKJ}(\eta)$ — regularized correlation structure
- Or fix $\Sigma_u = I$ — standardized shocks (simpler, often sufficient)

## Future Extensions

1. **Shock-regime coupling:** $u_t$ could depend on s_t (e.g., volatility shocks larger in stressed regime)
2. **Time-varying loadings:** Smooth changes in $B_k$ over time (e.g., via GP priors)
3. **Sparse loadings:** Many loadings exactly zero via horseshoe or spike-and-slab priors
4. **Observable shocks:** Calibrate loading matrices to observable risk factors (yields, spreads, etc.)

## References

**Key papers:**
- Hamilton (1989) — Regime-switching models
- Guidolin & Timmermann (2007) — Regime-switching in multi-asset allocation
- Rossi (2013) — Shock identification and VAR models

**Implementation notes:**
- Shock model architecture inspired by multi-factor models in quantitative finance
- Variance decomposition follows Fama-French framework
