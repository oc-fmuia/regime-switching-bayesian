"""
Bayesian inference module for regime-switching models.

This module provides complete PyMC-based inference pipeline:
1. ModelBuilder: Assemble full PyMC model with priors
2. NUTSSampler: NUTS sampling with convergence diagnostics
3. DiagnosticsComputer: Rhat, ESS, divergence rates
4. PosteriorPredictiveCheck: Model validation

**Usage:**
```python
from inference.model_builder import ModelBuilder, PriorSpec
from inference.sampler import NUTSSampler

# 1. Define and build model
mb = ModelBuilder(n_assets=5, n_regimes=3, n_shocks=2, n_obs=250)
model = mb.build(returns_data=returns)

# 2. Sample with NUTS
sampler = NUTSSampler()
summary = sampler.sample(model, draws=1000, tune=1000, chains=2)

# 3. Check convergence diagnostics (auto-computed)
print(summary.idata)  # Full posterior, diagnostics included
```

**Key Classes:**
- PriorSpec: Prior specification (7 customizable parameters)
- ModelBuilder: PyMC model assembly
- NUTSSampler: NUTS sampling orchestration
- DiagnosticsComputer: Rhat, ESS, divergence analysis
- PosteriorPredictiveCheck: PPCs and summary statistics
- InferenceSummary: Sampling results and diagnostics
"""

from inference.model_builder import ModelBuilder, PriorSpec
from inference.sampler import (
    NUTSSampler,
    DiagnosticsComputer,
    PosteriorPredictiveCheck,
    InferenceSummary,
)

__all__ = [
    "ModelBuilder",
    "PriorSpec",
    "NUTSSampler",
    "DiagnosticsComputer",
    "PosteriorPredictiveCheck",
    "InferenceSummary",
]
