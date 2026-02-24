# Regime-Switching Bayesian Multi-Asset Framework

A production-quality Python framework for Bayesian regime-switching modeling of multi-asset returns with shock propagation, parameter uncertainty quantification, and Monte Carlo simulation.

## Overview

This framework implements a hierarchical Bayesian model for capturing **structural breaks, fat tails, and shock dynamics** in financial returns:

- **Regime-switching Markov chain:** K discrete regimes with Dirichlet-prior transition probabilities
- **Multivariate Student-t returns:** Fat-tailed, regime-conditional asset distributions (vs. Gaussian)
- **Shock propagation:** Explicit modeling of exogenous shocks (e.g., market stress, volatility spikes) and their asset-specific loadings
- **Bayesian parameter uncertainty:** All parameters (means, covariances, tail indices, shock loadings) treated as random variables
- **NUTS inference:** Rigorous Markov chain Monte Carlo via PyMC
- **Monte Carlo simulation:** Forward-looking scenario analysis conditional on posterior beliefs

## Key Features

✓ **Mathematically rigorous** — Full derivations, graphical models, prior justification  
✓ **Production code** — Type hints, docstrings, modular architecture, unit tests  
✓ **Educationally clear** — Two notebooks: one teaches theory, one lets you play with parameters  
✓ **Statistically sound** — Convergence diagnostics, posterior predictive checks, residual analysis  
✓ **Practically useful** — Tunable assets (2-20), regimes (2-5), adjustable priors  

## Installation

```bash
# Clone repo
git clone https://github.com/oc-fmuia/regime-switching-bayesian.git
cd regime-switching-bayesian

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

**Generate synthetic data, build model, infer, simulate:**

```python
from src.bayesian import ModelBuilder
from src.simulation import MonteCarloSimulator
import numpy as np

# Initialize
builder = ModelBuilder(n_assets=5, n_regimes=2, n_shocks=3)

# Build PyMC model
model = builder.build_model(data=returns_data)

# Inference (NUTS sampling)
idata = builder.infer(model, draws=2000, tune=1000)

# Simulation
simulator = MonteCarloSimulator(idata, n_assets=5, n_regimes=2)
paths = simulator.simulate(n_paths=10000, horizon=252)
```

See `notebooks/` for detailed examples.

## Documentation

- **`docs/math_foundations.md`** — Bayesian theory (why Student-t, why LKJ, Dirichlet priors)
- **`docs/regime_switching.md`** — Full mathematical specification (generative model, priors, likelihood)
- **`docs/shock_propagation.md`** — Shock mechanics and return dynamics
- **Code docstrings** — All functions thoroughly documented with type hints

## Notebooks

1. **`notebooks/01_model_building.ipynb`** — Educational walkthrough
   - Regime dynamics and Markov chains
   - Student-t returns and why fat tails matter
   - Shock loadings and return dynamics
   - PyMC model construction
   - NUTS inference and diagnostics
   - Posterior predictive checks

2. **`notebooks/02_interactive_exploration.ipynb`** — Interactive sandbox
   - Tune number of regimes (2-5)
   - Tune number of assets (2-20)
   - Adjust Student-t ν prior
   - Compare prior specifications
   - Define shock scenarios
   - Run simulations and visualize

## Project Structure

```
regime-switching-bayesian/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
│
├── src/
│   ├── regimes/
│   │   ├── __init__.py
│   │   └── markov.py              # Markov chain + transition priors
│   ├── returns/
│   │   ├── __init__.py
│   │   ├── student_t.py           # Student-t model specification
│   │   └── covariance.py          # LKJ prior + correlation handling
│   ├── shocks/
│   │   ├── __init__.py
│   │   ├── shock_model.py         # Shock process + loading matrix
│   │   └── propagation.py         # Return + shock integration
│   ├── bayesian/
│   │   ├── __init__.py
│   │   ├── model_builder.py       # PyMC model construction
│   │   ├── inference.py           # NUTS sampling + diagnostics
│   │   └── priors.py              # Prior specifications
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── monte_carlo.py         # Path simulation engine
│   └── utils/
│       ├── __init__.py
│       └── visualization.py       # Plotting + diagnostic plots
│
├── notebooks/
│   ├── 01_model_building.ipynb
│   └── 02_interactive_exploration.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_regimes.py
│   ├── test_returns.py
│   ├── test_shocks.py
│   ├── test_inference.py
│   └── test_simulation.py
│
└── docs/
    ├── math_foundations.md
    ├── regime_switching.md
    └── shock_propagation.md
```

## Model Summary

### Generative Process

```
s_t ∈ {1, …, K}                          # Regime (Markov latent state)
P_{ij} = P(s_t = j | s_{t-1} = i)       # Transition probabilities

r_t | s_t = k ~ ST(μ_k, Σ_k, ν_k)      # Student-t returns (regime-conditional)
r_t = μ_{s_t} + B_{s_t} u_t + ε_t       # With shock loading

u_t ~ N(0, I)                            # Shocks (standardized)
ε_t ~ ST(0, Σ_{s_t}, ν_{s_t})           # Idiosyncratic noise
```

### Priors

- **Regimes:** Dirichlet(1) on each row of transition matrix
- **Means:** Normal(0, 1) 
- **Covariances:** LKJ(2) for correlation, HalfNormal for scales
- **Tail index:** Exponential(0.1) on ν_k (default; tunable)
- **Shock loadings:** Normal(0, 0.5) on B entries

## Why Bayesian?

1. **Parameter uncertainty propagates** → Decision-makers see full distribution of outcomes
2. **Principled priors** → Avoid overfitting to noise, incorporate domain knowledge
3. **Posterior predictive checks** → Validate whether model captures data structure
4. **Sequential learning** → Update beliefs as new data arrives

## Performance

- **Inference time:** ~5-10 minutes for 2000 draws + 1000 tune (NUTS, 4 cores, 5-10 assets, 2-3 regimes)
- **Simulation speed:** ~1000 paths/second (NumPy, vectorized)
- **Memory:** ~2-3 GB for full posterior (2000 draws, 10 assets, 3 regimes)

## Citation

If you use this framework, cite:

```bibtex
@software{regime_switching_bayesian_2026,
  author = {Fernando, PyMC Labs},
  title = {Regime-Switching Bayesian Multi-Asset Framework},
  year = {2026},
  url = {https://github.com/oc-fmuia/regime-switching-bayesian}
}
```

## License

MIT License. See LICENSE file.

## Contributing

Contributions welcome. Please:
1. Add unit tests for new features
2. Follow PEP 8 (enforced via Black)
3. Add docstrings and type hints
4. Update documentation

## Contact

Questions? Reach out or open an issue on GitHub.

---

**Built with:**
- [PyMC](https://www.pymc.io) — Bayesian inference
- [ArviZ](https://arviz-devs.github.io/arviz/) — Diagnostics & visualization
- [NumPy](https://numpy.org) — Numerical computing
