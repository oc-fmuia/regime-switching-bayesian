# Bayesian Model Builder for Regime-Switching Returns

## Overview

The `ModelBuilder` class assembles a complete Bayesian model for regime-switching multi-asset returns with shock propagation. It uses PyMC to manage:

1. **Regime dynamics:** Markov chain transition probabilities
2. **Return model:** Regime-conditional Student-t distributions
3. **Shock propagation:** Factor loadings and systematic risk
4. **Priors:** Flexible prior specifications with sensible defaults

## Mathematical Model

The full Bayesian model:

$$
\begin{align}
s_t &\sim \text{Markov}(P) \quad \text{(regime dynamics)}\\
P_{ij} &\sim \text{Dirichlet}(\alpha) \quad \text{(transition priors)}\\
\mu_k &\sim N(\mu_0, \sigma_0^2) \quad \text{(mean returns)}\\
\sigma_{k,n} &\sim \text{HalfNormal}(\sigma_v) \quad \text{(volatilities)}\\
\nu_k &\sim \text{Exponential}(1/\nu_0) \quad \text{(degrees of freedom)}\\
B_{k,nm} &\sim N(0, \sigma_B^2) \quad \text{(factor loadings)}\\
r_t &\sim \text{StudentT}(\nu_{s_t}, \mu_{s_t}, \Sigma_{s_t}) \quad \text{(returns)}
\end{align}
$$

where:
- $s_t \in \{1, \ldots, K\}$ is the regime at time $t$
- $P \in \mathbb{R}^{K \times K}$ is the transition matrix
- $\mu_k \in \mathbb{R}^N$ are regime-conditional means
- $\sigma_k \in \mathbb{R}^N$ are regime-conditional volatilities
- $B_k \in \mathbb{R}^{N \times M}$ are factor loadings per regime
- $\Sigma_{s_t}$ is the total covariance (systematic + idiosyncratic)

## Components

### PriorSpec

Specifies prior distributions for all model parameters.

```python
from inference.model_builder import PriorSpec

spec = PriorSpec(
    dirichlet_alpha=1.0,        # Transition matrix: higher = more uncertain
    mean_loc=0.0,               # Center for mean returns
    mean_scale=0.05,            # Spread for mean returns
    vol_scale=0.1,              # Scale for volatilities
    lkj_eta=2.0,                # Correlation concentration (unused for now)
    df_mean=10.0,               # Prior mean for degrees of freedom
    loading_scale=0.5,          # Scale for factor loadings
)
```

**Parameter Guidance:**
- `dirichlet_alpha`: Use 1.0 for uninformed, >1 to concentrate on certain transitions
- `mean_scale`: Should be ~0.5-1× typical expected returns
- `vol_scale`: Should be ~0.05-0.2 depending on asset class
- `df_mean`: 10-20 for moderate tail risk, 5-10 for heavy tails

### ModelBuilder

Constructs the PyMC model. Usage:

```python
from inference.model_builder import ModelBuilder

# Define problem dimensions
mb = ModelBuilder(
    n_assets=5,          # Number of assets
    n_regimes=3,         # Number of regimes
    n_shocks=2,          # Number of risk factors
    n_obs=250,           # Number of time steps
    prior_spec=spec,     # (Optional) custom priors
)

# Build model (without inference yet)
model = mb.build(returns_data=returns)  # Returns shape: (250, 5)
```

**Optional Parameters:**
- `returns_data`: Observed returns (T, N). If provided, likelihood is included.
- `regime_path`: Known regime sequence (T,). If not provided, inferred.
- `shocks`: Observed shock realizations (T, M). If not provided, generated from prior.

## Model Structure

### Prior Specification

| Parameter | Prior | Shape | Interpretation |
|-----------|-------|-------|-----------------|
| Transition matrix | Dirichlet(α) | (K, K) | Row-by-row transition probabilities |
| Stationary dist. | Dirichlet(1) | (K,) | Long-run regime probabilities |
| Regime means | Normal(0, 0.05²) | (K, N) | Average return per regime |
| Volatilities | HalfNormal(0.1) | (K, N) | Volatility per asset per regime |
| DF (tail risk) | Exponential(0.1) | (K,) | Heavy-tailedness parameter |
| Loadings | Normal(0, 0.5²) | (K, N, M) | Factor sensitivities |

### Likelihood (if data provided)

$$p(r_t | s_t) = \text{StudentT}(\nu_{s_t}, \mu_{s_t}, \Sigma_{s_t})$$

where $\Sigma_{s_t}$ combines:
- **Systematic:** $B_{s_t} I B_{s_t}^T$ (factor-driven)
- **Idiosyncratic:** $\text{diag}(\sigma_{s_t})^2$ (firm-specific)

## Usage Examples

### Example 1: Build Model Without Data

```python
mb = ModelBuilder(n_assets=3, n_regimes=2, n_shocks=2, n_obs=100)
model = mb.build()  # Prior-only model for prior predictive checks
```

### Example 2: Build Model With Observed Data

```python
import numpy as np

# Simulate returns
returns = np.random.randn(100, 3) * 0.01

mb = ModelBuilder(n_assets=3, n_regimes=2, n_shocks=2, n_obs=100)
model = mb.build(returns_data=returns)  # Model conditioned on data
```

### Example 3: Custom Priors

```python
from inference.model_builder import PriorSpec, ModelBuilder

# Heavy-tailed (financial crisis risk)
spec_crisis = PriorSpec(
    df_mean=7.0,        # Lower DF = heavier tails
    vol_scale=0.15,     # Expect higher volatility
)

mb = ModelBuilder(
    n_assets=5,
    n_regimes=3,
    n_shocks=2,
    n_obs=250,
    prior_spec=spec_crisis,
)
model = mb.build(returns_data=returns)
```

### Example 4: Known Regime Path

```python
# If regimes are known (e.g., from market classification)
regime_path = np.array([0, 0, 0, 1, 1, 1, 0, 0, ...])

mb = ModelBuilder(n_assets=3, n_regimes=2, n_shocks=2, n_obs=len(regime_path))
model = mb.build(regime_path=regime_path, returns_data=returns)
```

## MCMC Inference (PyMC Sampling)

Once the model is built, use PyMC's `sample()` method:

```python
import pymc as pm

with model:
    # NUTS sampling (default)
    idata = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        cores=2,
        random_seed=42,
        progressbar=True,
    )
    
    # Posterior predictive samples
    pm.sample_posterior_predictive(idata, extend_inferencedata=True)

# Access results
print(idata.posterior)  # Posterior samples
print(idata.posterior_predictive)  # Predictive samples
```

**Sampling Parameters:**
- `draws`: Number of post-burn-in samples per chain (typically 500-2000)
- `tune`: Number of burn-in steps (typically =draws)
- `chains`: Parallel chains (2-4 recommended)
- `target_accept`: NUTS acceptance rate (0.8-0.95)

## Testing & Validation

The module includes comprehensive tests at multiple scales:

```bash
python3 src/inference/test_model_builder.py
```

**Test Coverage:**
- Small (n=10): Model instantiation, basic validation
- Medium (n=50): Multi-regime/asset/shock configurations
- Large (n=100): Realistic scenarios and scaling
- Optional: Full NUTS inference (uncomment in test file)

**Timing Expectations:**
- Model construction: <100ms
- Prior predictive sampling: ~1-5s per 1000 samples
- NUTS inference: 5-60s per 1000 draws (depends on dimension, data size, chains)

## Prior Sensitivity

The model's behavior depends strongly on priors. Here are typical configurations:

### Conservative (Low Volatility Periods)
```python
PriorSpec(
    mean_scale=0.03,    # Expect small moves
    vol_scale=0.05,     # Lower volatility
    df_mean=15.0,       # Thinner tails
)
```

### Crisis (High Volatility, Heavy Tails)
```python
PriorSpec(
    mean_scale=0.1,     # Larger expected moves
    vol_scale=0.2,      # Higher volatility
    df_mean=5.0,        # Heavier tails
)
```

### Default (Uninformed)
```python
PriorSpec()  # Neutral across conditions
```

## Computational Notes

### Scalability

Computation time scales with:
1. **Number of observations (T):** Linear in likelihood evaluation
2. **Number of assets (N):** Quadratic in covariance operations
3. **Number of regimes (K):** Linear (mixture model)
4. **Number of shocks (M):** Linear in factor loading size

Typical rule of thumb:
- Small (N=3-5, T=100-250): <1 min per 1000 draws
- Medium (N=10, T=250-500): 1-5 min per 1000 draws
- Large (N=20+, T=1000+): 10+ min per 1000 draws

### Memory Usage

Memory scales as O(T×N + K×N²) for:
- Likelihood computation
- Posterior storage
- Covariance matrices

Typical: <500MB for T=1000, N=10, K=3

## Future Extensions

1. **Time-varying priors:** Learn prior parameters from rolling windows
2. **LKJ correlations:** Proper correlation priors per regime
3. **Hierarchical structure:** Shared hyper-priors across assets
4. **Sequential inference:** Online learning as new data arrives
5. **Model comparison:** Marginal likelihood for regime count selection

## References

**Key Papers:**
- Carlin & Polson (1992) — Markov chain Monte Carlo and regime-switching
- Tran et al. (2015) — Variational inference for regime-switching models
- Gelfand & Smith (1990) — Sampling methods in inference

**PyMC:**
- https://docs.pymc.io
- NUTS sampler: Hoffman & Gelman (2014)
- LKJ priors: Lewandowski et al. (2009)
