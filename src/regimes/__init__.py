"""
Regime-switching models: Markov chains and shock propagation.

This module implements regime dynamics and shock-driven returns:

**Markov Chains (markov.py):**
- Discrete-time Markov chains for regime switching
- Transition probability matrices
- Stationary distribution computation
- Regime path simulation

**Shock Propagation (shocks.py):**
- Factor-driven returns with regime-dependent loadings
- Deterministic stress testing
- Stochastic shock simulation
- Variance decomposition (systematic vs. idiosyncratic)
"""

from src.regimes.markov import MarkovChain, create_symmetric_markov_chain
from src.regimes.shocks import ShockModel, ReturnWithShocks

__all__ = [
    # Markov chain
    "MarkovChain",
    "create_symmetric_markov_chain",
    # Shock propagation
    "ShockModel",
    "ReturnWithShocks",
]
