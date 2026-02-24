"""
Regime-switching model: Markov chain dynamics.

This module implements the regime component of the framework:
- Discrete-time Markov chains for regime switching
- Transition probability matrices
- Stationary distribution computation
- Regime path simulation
"""

from src.regimes.markov import MarkovChain, create_symmetric_markov_chain

__all__ = [
    "MarkovChain",
    "create_symmetric_markov_chain",
]
