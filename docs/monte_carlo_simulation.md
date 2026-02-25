# Monte Carlo Simulation for Scenario Analysis

## Overview

The Monte Carlo simulator generates forward-looking scenarios from posterior samples, enabling:

1. **Path generation:** Regime-switching returns with shocks
2. **Path analytics:** Mean, volatility, quantiles across scenarios
3. **Portfolio metrics:** VaR, CVaR, Sharpe ratio, maximum drawdown
4. **Scenario decomposition:** Per-regime performance analysis

## Basic Usage

```python
from simulation.simulator import MonteCarloSimulator

# Initialize simulator
sim = MonteCarloSimulator(
    n_assets=5,        # Number of assets
    n_regimes=3,       # Number of regimes
    n_shocks=2,        # Number of shock factors
    n_scenarios=1000,  # Monte Carlo paths to generate
)

# Generate scenarios
paths, regime_paths, shocks = sim.generate_paths(
    n_steps=252,                    # Trading days (1 year)
    transition_matrix=P,            # From posterior
    regime_means=mu,                # From posterior
    regime_covs=Sigma,              # From posterior
    loading_matrices=B,             # From posterior (optional)
    initial_regime=0,               # Starting regime
    random_seed=42,
)

# Paths shape: (1000, 252, 5) — 1000 scenarios, 252 steps, 5 assets
```

## Path Generation

Generate Monte Carlo paths combining regime-switching, returns, and shocks.

### Return Dynamics

$$r_{t} = \mu_{s_t} + B_{s_t} u_t + \varepsilon_t$$

- **$\mu_{s_t}$:** Regime-conditional mean
- **$B_{s_t} u_t$:** Shock-driven component (optional)
- **$\varepsilon_t$:** Idiosyncratic noise

### Regime Switching

Regimes evolve via Markov chain:

$$s_t \sim P(s_{t-1}, \cdot)$$

### Shock Scenarios

Shocks are standard normal and regime-conditional:

$$u_t \sim N(0, I_M)$$

## Path Analytics

### Compute Path Statistics

```python
stats = sim.compute_path_statistics(paths)

stats['mean']           # (252, 5) mean return per step
stats['std']            # (252, 5) volatility per step
stats['median']         # (252, 5) median return
stats['quantile_5']     # (252, 5) 5th percentile (downside)
stats['quantile_95']    # (252, 5) 95th percentile (upside)
stats['cumulative']     # (1000, 252, 5) cumulative returns per path
```

### Portfolio Metrics

```python
metrics = sim.compute_portfolio_metrics(
    paths,
    weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Equal weight
    risk_free_rate=0.02,  # 2% annual
)

print(f"Expected return:    {metrics['mean_return']:.4f}")
print(f"Volatility:         {metrics['volatility']:.4f}")
print(f"Sharpe ratio:       {metrics['sharpe_ratio']:.4f}")
print(f"VaR (95%):          {metrics['var_95']:.4f}")
print(f"CVaR (95%):         {metrics['cvar_95']:.4f}")
print(f"Max drawdown:       {metrics['max_drawdown']:.4f}")
```

**Metrics Interpretation:**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Mean Return** | Average path return | Long-term expected growth |
| **Volatility** | Std dev of returns | Risk magnitude |
| **Sharpe Ratio** | (μ - rf) / σ | Return per unit risk |
| **VaR (95%)** | 5th percentile | Loss with 95% confidence |
| **CVaR (95%)** | Mean of tail | Expected loss if VaR exceeded |
| **Max Drawdown** | Peak-to-trough decline | Worst drawdown scenario |

## Scenario Analysis

Decompose performance by regime.

### Per-Regime Breakdown

```python
analysis = sim.scenario_analysis(
    paths,
    regime_paths,
    regime_labels=["Normal Market", "Stressed Market", "Recovery"],
)

for regime_name, stats in analysis.items():
    print(f"\n{regime_name}:")
    print(f"  Frequency:     {stats['frequency']:.1%}")
    print(f"  Mean return:   {stats['mean_return']:.4f}")
    print(f"  Volatility:    {stats['volatility']:.4f}")
```

**Output:**
```
Normal Market:
  Frequency:     65.5%
  Mean return:   0.0012
  Volatility:    0.0089

Stressed Market:
  Frequency:     25.3%
  Mean return:   -0.0015
  Volatility:    0.0245

Recovery:
  Frequency:     9.2%
  Mean return:   0.0028
  Volatility:    0.0145
```

## Advanced: Custom Stress Scenarios

Combine simulator with shock inputs for deterministic stress testing.

```python
# Scenario: Bond yields +100bp, equity vol spike +20%
shock_scenario = np.array([1.0, 0.2])  # Standardized shocks

# Apply via ModelBuilder (for posterior predictive)
# Or directly compute return impact
mu_stressed = regime_means[regime_idx] + loading_matrices[regime_idx] @ shock_scenario

print(f"Stressed returns: {mu_stressed}")
```

## Full Workflow: Posterior to Scenarios

```python
import pymc as pm
from inference.model_builder import ModelBuilder
from inference.sampler import NUTSSampler
from simulation.simulator import MonteCarloSimulator

# 1. Build and sample model (Commits 5-6)
mb = ModelBuilder(n_assets=5, n_regimes=3, n_shocks=2, n_obs=250)
model = mb.build(returns_data=returns)

sampler = NUTSSampler()
summary = sampler.sample(model, draws=1000, tune=1000, chains=2)

# 2. Extract posterior samples
posterior = summary.idata.posterior

# 3. Run Monte Carlo scenarios
sim = MonteCarloSimulator(n_assets=5, n_regimes=3, n_shocks=2, n_scenarios=1000)

# Sample a random draw from posterior
draw_idx = np.random.randint(0, posterior['regime_means'].shape[0] * posterior['regime_means'].shape[1])
chain_idx = draw_idx // posterior['regime_means'].shape[1]
sample_idx = draw_idx % posterior['regime_means'].shape[1]

P = summary.idata.posterior['transition_matrix'].values[chain_idx, sample_idx]
mu = summary.idata.posterior['regime_means'].values[chain_idx, sample_idx]
Sigma = np.eye(5) * 0.01  # Approximation
B = summary.idata.posterior['loading_matrices'].values[chain_idx, sample_idx]

# Generate paths
paths, regime_paths, shocks = sim.generate_paths(
    n_steps=252,
    transition_matrix=P,
    regime_means=mu,
    regime_covs=Sigma,
    loading_matrices=B,
)

# 4. Analyze scenarios
metrics = sim.compute_portfolio_metrics(paths)
analysis = sim.scenario_analysis(paths, regime_paths)

print(f"Expected annual return: {metrics['mean_return'] * 252:.2%}")
print(f"Annual volatility:      {metrics['volatility'] * np.sqrt(252):.2%}")
print(f"Sharpe ratio:           {metrics['sharpe_ratio']:.2f}")
print(f"Value-at-Risk (95%):    {metrics['var_95']:.4f}")
```

## Performance Characteristics

### Computational Complexity

- **Path generation:** O(n_scenarios × n_steps × n_assets²)
- **Portfolio metrics:** O(n_scenarios × n_steps)
- **Scenario analysis:** O(n_scenarios × n_steps × n_regimes)

### Typical Timings

| Config | Time | Notes |
|--------|------|-------|
| (5 assets, 3 regimes, 100 scenarios, 100 steps) | <100ms | Instant |
| (10 assets, 3 regimes, 500 scenarios, 250 steps) | 0.5-1s | Quick analysis |
| (20 assets, 5 regimes, 1000 scenarios, 500 steps) | 5-10s | Full-scale |

### Memory Usage

Dominated by path storage: ~8 bytes × n_scenarios × n_steps × n_assets

- (1000, 252, 5): ~10 MB
- (1000, 252, 20): ~40 MB
- (10000, 252, 20): ~400 MB

## Use Cases

### 1. Risk Assessment

```python
# What's the worst 5% outcome?
print(f"VaR: {metrics['var_95']:.2%}")
print(f"Expected tail loss (CVaR): {metrics['cvar_95']:.2%}")
```

### 2. Portfolio Optimization

```python
# Evaluate different allocations
for weights in [equal_weights, min_var_weights, equal_risk_weights]:
    metrics = sim.compute_portfolio_metrics(paths, weights=weights)
    print(f"Weights: {weights}, Sharpe: {metrics['sharpe_ratio']:.2f}")
```

### 3. Scenario Comparison

```python
# How does performance change across regimes?
analysis = sim.scenario_analysis(paths, regime_paths)
for regime, stats in analysis.items():
    print(f"{regime}: {stats['mean_return']:.2%} (prob: {stats['frequency']:.0%})")
```

### 4. Drawdown Analysis

```python
# What's the worst drawdown?
print(f"Maximum drawdown: {metrics['max_drawdown']:.2%}")

# Can we tolerate it?
if abs(metrics['max_drawdown']) > 0.20:
    print("⚠️ Drawdown exceeds 20% risk limit")
```

## References

**Monte Carlo Methods:**
- Glasserman (2004) — Monte Carlo Methods in Financial Engineering
- Kloeden & Platen (1992) — Numerical Solution of SDEs

**Scenario Analysis:**
- Broadie & Glasserman (1997) — Pricing American-style securities
- Boyle et al. (1997) — Monte Carlo methods for securities pricing

**Regime-Switching:**
- Hamilton (1989) — A new approach to economic analysis with regime switching
- Guidolin & Timmermann (2007) — Asset allocation under multivariate regime switching
