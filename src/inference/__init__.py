"""
Bayesian inference module for regime-switching models.

This module provides PyMC-based Bayesian inference for regime-switching
multi-asset return models with shock propagation.

**Components:**
- ModelBuilder: Assemble full PyMC model with priors
- PriorSpec: Prior specification and customization
- Inference utilities: NUTS sampling, diagnostics, predictions

**Usage:**
```python
from inference.model_builder import ModelBuilder, PriorSpec

# Define problem
mb = ModelBuilder(
    n_assets=5,
    n_regimes=3,
    n_shocks=2,
    n_obs=250
)

# Build model
model = mb.build(returns_data=returns)

# Run inference
import pymc as pm
with model:
    idata = pm.sample(draws=1000, tune=1000)
```
"""

from inference.model_builder import ModelBuilder, PriorSpec

__all__ = [
    "ModelBuilder",
    "PriorSpec",
]
