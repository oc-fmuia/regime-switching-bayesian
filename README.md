# Regime-Switching Bayesian Multi-Asset Framework

A Bayesian Hidden Markov Model for equity returns, fitted via NUTS
(PyMC / NumPyro), with forward-filter backward-sampler regime recovery
and interactive [marimo](https://marimo.io/) notebooks. The current
release (v0) demonstrates the full pipeline on synthetic data with two
regimes and three equity indices.

## What's implemented (v0)

- **K = 2 Hidden Markov Model** with multivariate Normal emissions
  (diagonal covariance; identity correlation within regimes).
- **Priors calibrated to monthly returns:** sticky Dirichlet on the
  transition matrix, `LKJCholeskyCov` on regime covariances, Normal on
  regime means.
- **Analytically marginalised likelihood** via a normalised forward
  algorithm (`pytensor.scan`, JIT-compiled to `jax.lax.scan` under
  NumPyro).
- **NUTS inference** with NumPyro/JAX backend by default (PyMC fallback
  available).
- **Label-switching detection** and post-hoc permutation alignment
  across chains.
- **Forward-filter backward-sampler (FFBS)** for full posterior regime
  recovery.
- **Regime-aware allocation demo:** proof-of-concept strategy that
  reduces equity exposure when filtered bear probability exceeds a
  threshold.

## Repository layout

```
src/regime_switching_bayesian/
    data_gen.py      Synthetic K-regime HMM data generation
    model.py         PyMC model (forward algorithm + pm.Potential)
    inference.py     NUTS fitting, diagnostics, FFBS, label alignment
    plotting.py      Regime probability, return, and posterior plots
notebooks/
    01_bayesian_hmm_fundamentals.py   Blog-style narrative with allocation demo
tests/                       Pytest suite (unit + slow integration)
docs/                        Math spec, finance spec, PyMC spec, impl plans
research/                    Exploratory marginalization probes
```

## Getting started

This project uses [pixi](https://pixi.sh) for environment management.

```bash
# Install pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash
```

### Activate an environment

**macOS note:** `pixi shell` spawns a subshell using the system
`/bin/bash` (3.2, frozen by Apple due to GPLv3 licensing). To stay in
your current shell (zsh, bash 5, etc.), use `shell-hook` instead:

```bash
eval "$(pixi shell-hook -e dev)"
```

On Linux, `pixi shell -e dev` works fine. The `shell-hook` form works
everywhere.

### Run the notebooks

```bash
eval "$(pixi shell-hook -e notebook)"
marimo edit notebooks/01_bayesian_hmm_fundamentals.py
```

### Run the tests

```bash
eval "$(pixi shell-hook -e dev)"
pytest
```

## Notebook series

This repository is organised as a series of notebooks, each building on
the previous one.

1. **Bayesian HMM fundamentals** (`01_bayesian_hmm_fundamentals.py`)
   *Status: complete.*
   Two-regime (bull/bear) HMM on synthetic data with multivariate Normal
   emissions and diagonal covariance. Covers the full pipeline: data
   generation, marginalised likelihood via the forward algorithm, NUTS
   inference, FFBS regime recovery, smoothed vs filtered probabilities,
   and a regime-aware 80/20 equity/risk-free allocation demo. Includes
   appendices on the forward algorithm derivation and label switching.

2. **Scenario forward simulation and performance evaluation.**
   *Status: planned.*
   Simulate forward return paths under different regime assumptions
   (e.g. pin the bear regime for 12 months, shock covariances) and
   compute regime-conditional portfolio metrics (VaR, CVaR, maximum
   drawdown distributions, Sharpe/Calmar ratios) with full parameter
   uncertainty propagated.

3. **Regime-dependent correlations, fat-tailed emissions, and
   autoregression.**
   *Status: planned.*
   Replace the identity correlation assumption with regime-dependent
   correlation matrices via LKJCholeskyCov. Replace multivariate Normal
   emissions with multivariate Student-t to capture intra-regime fat
   tails. Add optional AR(p) dynamics within each regime to model
   momentum and mean-reversion effects.

4. **Time-varying transition probabilities with exogenous covariates.**
   *Status: planned.*
   Condition transition probabilities on observable macro variables
   (e.g. VIX, yield curve slope) via a softmax link, enabling scenario
   analysis questions such as "what happens to regime probabilities if
   the VIX doubles?".

5. **Real data applications.**
   *Status: planned.*
   Apply the full pipeline to historical market data (e.g. S&P 500,
   long-duration Treasuries, gold, 2000-2025) with walk-forward
   re-estimation for proper out-of-sample evaluation.

## License

MIT
