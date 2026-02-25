# Inference and Diagnostics for Regime-Switching Models

## Overview

The inference module provides end-to-end MCMC sampling and convergence diagnostics:

1. **NUTSSampler:** Orchestrates PyMC NUTS sampling with configurable parameters
2. **DiagnosticsComputer:** Computes Rhat, ESS, divergence rates
3. **PosteriorPredictiveCheck:** Model validation via posterior predictive checks
4. **InferenceSummary:** Encapsulates results, diagnostics, and timing

## NUTS Sampler

### Basic Usage

```python
from inference.sampler import NUTSSampler
from inference.model_builder import ModelBuilder

# Build model (Commit 5)
mb = ModelBuilder(n_assets=5, n_regimes=3, n_shocks=2, n_obs=250)
model = mb.build(returns_data=returns)

# Sample with NUTS
sampler = NUTSSampler()
summary = sampler.sample(
    model,
    draws=1000,
    tune=1000,
    chains=2,
    cores=2,
    random_seed=42,
)

print(summary)
# InferenceSummary(draws=1000, tune=1000, chains=2, time=45.2s)
```

### Configuration

```python
sampler = NUTSSampler(
    target_accept=0.85,    # NUTS acceptance rate (0.7-0.95)
    max_treedepth=10,      # Max tree depth (5-15)
)
```

**Parameters:**
- `target_accept`: Higher = more careful sampling, slower. Default 0.85.
- `max_treedepth`: Higher = more evaluations per step. Default 10 (=2^10=1024 max).

### Return Value

`NUTSSampler.sample()` returns `InferenceSummary`:

```python
summary.idata              # arviz.InferenceData (posterior, chains, diagnostics)
summary.n_draws            # Number of post-burn-in draws per chain
summary.n_tune             # Number of burn-in steps per chain
summary.n_chains           # Number of parallel chains
summary.total_samples      # Total posterior samples (n_draws * n_chains)
summary.sampling_time      # Total sampling time (seconds)
```

## Convergence Diagnostics

### Rhat (Potential Scale Reduction Factor)

Measures whether multiple chains have converged to same posterior distribution.

$$\hat{R} = \sqrt{\frac{\text{Var}_{\text{between}} + \text{Var}_{\text{within}}}{\text{Var}_{\text{within}}}}$$

**Interpretation:**
- $\hat{R} < 1.01$: Excellent convergence ✓
- $1.01 \le \hat{R} < 1.05$: Good convergence ✓
- $1.05 \le \hat{R} < 1.10$: Marginal (re-run with more samples) ⚠️
- $\hat{R} \ge 1.10$: Poor convergence (model/sampler issue) ❌

**Automatic Computation:**
PyMC's `sample()` computes Rhat for all variables automatically. Access via:

```python
summary.idata.posterior  # View posterior samples
az.summary(summary.idata)  # View Rhat + other diagnostics in table
```

### ESS (Effective Sample Size)

Accounts for autocorrelation in MCMC samples. Effective samples are much fewer than nominal samples.

$$\text{ESS} = \frac{N}{2 \tau_{\text{int}}}$$

where $\tau_{\text{int}}$ is the integrated autocorrelation time.

**Interpretation:**
- $\text{ESS} > 400$ per chain: Excellent ✓
- $\text{ESS} > 200$ per chain: Adequate ✓
- $\text{ESS} < 100$ per chain: Problematic ❌

**Compute Manually:**

```python
from inference.sampler import DiagnosticsComputer

posterior_samples = summary.idata.posterior["regime_means"].values
# Shape: (chains, draws, regimes, assets)

# For each variable
ess = DiagnosticsComputer.ess(posterior_samples[0, :])  # Chain 0

print(f"ESS: {ess:.0f} effective samples from {len(posterior_samples[0, :])} draws")
```

### Divergences

NUTS sampler detects divergences (high-curvature areas where integrator fails).

**Interpretation:**
- $< 0.5\%$ divergences: Excellent ✓
- $< 2\%$ divergences: Acceptable ✓
- $\ge 2\%$ divergences: Problem (increase `tune`, reparameterize) ❌

**Automatic Check:**

```python
divergence_rate = DiagnosticsComputer.divergence_rate(summary.idata)
print(f"Divergence rate: {divergence_rate:.2%}")

if divergence_rate > 0.05:
    print("Warning: High divergence rate!")
```

## Posterior Predictive Checks

Validate whether the model generates plausible data by comparing observed data to posterior predictive samples.

### Basic PPC

```python
from inference.sampler import PosteriorPredictiveCheck

# Generate posterior predictive samples (from model)
import pymc as pm
with model:
    pm.sample_posterior_predictive(summary.idata)

# Compute PPC statistics
ppc_stats = PosteriorPredictiveCheck.compute_ppcheck(
    summary.idata,
    observed_data=returns,
    dim_name="returns",
)

print(ppc_stats)
# {
#   'mean_pvalue': 0.48,    # ✓ Close to 0.5 is good
#   'std_pvalue': 0.52,     # ✓ Close to 0.5 is good
#   'max_pvalue': 0.45,     # ✓ Close to 0.5 is good
# }
```

**Interpretation:**
- p-value $\approx 0.5$: Model generates data similar to observations ✓
- p-value $\ll 0.5$: Observations more extreme than predictions (under-fit)
- p-value $\gg 0.5$: Observations less extreme than predictions (over-fit)

### Summary Statistics

```python
from inference.sampler import PosteriorPredictiveCheck

stats = PosteriorPredictiveCheck.summary_stats(
    summary.idata,
    var_names=["regime_means", "volatilities", "degrees_of_freedom"],
)

for var_name, var_stats in stats.items():
    print(f"\n{var_name}:")
    print(f"  Mean: {var_stats['mean']:.4f}")
    print(f"  Std:  {var_stats['std']:.4f}")
    print(f"  95% HDI: [{var_stats['hdi_low']:.4f}, {var_stats['hdi_high']:.4f}]")
    print(f"  Rhat: {var_stats['rhat']:.4f}")
    print(f"  ESS:  {var_stats['ess_bulk']:.0f}")
```

## Workflow Example

Complete workflow from model building to diagnosis:

```python
import numpy as np
import arviz as az
from inference.model_builder import ModelBuilder, PriorSpec
from inference.sampler import NUTSSampler, DiagnosticsComputer, PosteriorPredictiveCheck

# 1. Generate synthetic data
np.random.seed(42)
n_obs, n_assets, n_regimes, n_shocks = 250, 5, 3, 2
returns = np.random.randn(n_obs, n_assets) * 0.01

# 2. Build model
spec = PriorSpec(
    mean_scale=0.05,
    vol_scale=0.1,
    df_mean=10.0,
)
mb = ModelBuilder(
    n_assets=n_assets,
    n_regimes=n_regimes,
    n_shocks=n_shocks,
    n_obs=n_obs,
    prior_spec=spec,
)
model = mb.build(returns_data=returns)

# 3. Sample with NUTS
sampler = NUTSSampler(target_accept=0.85)
summary = sampler.sample(
    model,
    draws=1000,
    tune=1000,
    chains=2,
    cores=2,
    random_seed=42,
)

print(f"Sampling completed: {summary.sampling_time:.1f}s")

# 4. Check diagnostics
div_rate = DiagnosticsComputer.divergence_rate(summary.idata)
print(f"Divergence rate: {div_rate:.2%}")

if div_rate > 0.05:
    print("⚠️ Warning: High divergence rate!")
else:
    print("✓ Divergence rate acceptable")

# 5. Posterior predictive checks
import pymc as pm
with model:
    pm.sample_posterior_predictive(summary.idata, random_seed=42)

ppc_stats = PosteriorPredictiveCheck.compute_ppcheck(
    summary.idata,
    observed_data=returns,
)

print(f"\nPPC p-values: {ppc_stats}")

# 6. Summary statistics
stats = PosteriorPredictiveCheck.summary_stats(summary.idata)

for var_name, var_stats in stats.items():
    print(f"\n{var_name}:")
    print(f"  Rhat: {var_stats['rhat']:.4f}")
    if var_stats['rhat'] > 1.01:
        print("  ⚠️ Potential convergence issue")
    print(f"  ESS: {var_stats['ess_bulk']:.0f}")
    if var_stats['ess_bulk'] < 400:
        print("  ⚠️ Low effective sample size")

# 7. Visualize posterior
az.plot_trace(summary.idata, var_names=["regime_means"])
```

## Troubleshooting

### High Divergence Rate (>2%)

**Causes:**
- Model has problematic geometry (e.g., high curvature)
- Step size too large

**Solutions:**
1. Increase `tune` (burn-in) steps
2. Increase `target_accept` (e.g., 0.90-0.95)
3. Reparameterize model (reduce correlations, normalize data)
4. Check prior specification (overly informative can cause issues)

```python
sampler = NUTSSampler(target_accept=0.90)  # More careful
summary = sampler.sample(model, draws=1000, tune=2000)  # More burn-in
```

### Low ESS

**Causes:**
- High autocorrelation in posterior
- Insufficient sampling
- Model/data mismatch

**Solutions:**
1. Increase `draws` per chain
2. Increase `chains` (more parallel sampling)
3. Check model specification
4. Verify data preprocessing

```python
summary = sampler.sample(model, draws=2000, tune=2000, chains=4, cores=4)
```

### Rhat > 1.01

**Causes:**
- Chains haven't converged
- Bimodal/multimodal posterior
- Numerical issues

**Solutions:**
1. Run longer (`draws`, `tune`)
2. Use multiple chains (ensures coverage)
3. Check prior specification
4. Visualize posterior (`az.plot_trace`)

```python
summary = sampler.sample(model, draws=2000, tune=2000, chains=4)
az.plot_trace(summary.idata)  # Visualize all chains
```

## Performance Notes

### Timing Estimates

Sampling time scales with:
- **Number of observations (T):** Linear in likelihood computation
- **Number of assets (N):** Quadratic in covariance operations
- **Number of regimes (K):** Linear (mixture model)
- **Tuning steps & draws:** Linear

Rough estimates for 2 chains on 2 cores:
- Small (N=3, T=100): 5-15 seconds
- Medium (N=5, T=250): 30-60 seconds
- Large (N=10, T=500): 2-5 minutes
- Very large (N=20, T=1000): 10-30 minutes

### Memory Usage

MCMC memory usage is dominated by storing posterior samples:
- Per variable: ~8 bytes × draws × chains × variable_size
- Example (1000 draws, 2 chains, 5 assets): ~80 KB per variable

Total: typically <500 MB for realistic problems.

## References

**Key Papers:**
- Gelman & Rubin (1992) — Rhat convergence diagnostic
- Vehtari et al. (2021) — ESS and effective sample size
- Hoffman & Gelman (2014) — NUTS sampler
- Gelfand & Smith (1990) — Posterior predictive checks

**Software:**
- PyMC: https://docs.pymc.io
- ArviZ: https://arviz-devs.github.io (diagnostics and plotting)
- Stan manual: https://mc-stan.org/users/documentation/case-studies
