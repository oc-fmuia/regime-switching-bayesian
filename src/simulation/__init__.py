"""
Monte Carlo simulation module for regime-switching models.

This module provides forward-looking scenario generation and analysis:
- MonteCarloSimulator: Generate paths from posterior draws
- Path statistics: Mean, volatility, quantiles across scenarios
- Portfolio analytics: VaR, CVaR, Sharpe ratio, max drawdown
- Scenario analysis: Per-regime performance decomposition

**Usage:**
```python
from simulation.simulator import MonteCarloSimulator

sim = MonteCarloSimulator(n_assets=5, n_regimes=3, n_shocks=2, n_scenarios=1000)

paths, regime_paths, shocks = sim.generate_paths(
    n_steps=252,
    transition_matrix=P,
    regime_means=mu,
    regime_covs=Sigma,
    loading_matrices=B,
)

metrics = sim.compute_portfolio_metrics(paths)
analysis = sim.scenario_analysis(paths, regime_paths)
```
"""

from simulation.simulator import MonteCarloSimulator

__all__ = [
    "MonteCarloSimulator",
]
