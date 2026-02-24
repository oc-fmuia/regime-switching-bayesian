"""
Regime-switching Markov chain model.

This module implements a discrete-time Markov chain for modeling regime dynamics.
States represent different market regimes (e.g., calm, stressed, volatile) and
transition probabilities are estimated from data using Bayesian inference.

Mathematical formulation:
    s_t ∈ {1, …, K}                       # Regime state at time t
    P(s_t = j | s_{t-1} = i) = P_{ij}    # Transition probability
    P_{ij} ~ Dirichlet(α)                 # Prior on transition probabilities

The Dirichlet prior is symmetric with α_k = 1 for all k, representing
a maximally uninformed prior that treats all transitions equally likely.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class MarkovChain:
    """
    Discrete-time Markov chain for regime switching.

    A Markov chain with K states where transitions depend only on the current
    state (memoryless property). This is the natural model for regime dynamics.

    Attributes
    ----------
    n_regimes : int
        Number of regimes (K)
    transition_matrix : NDArray[np.float64]
        Transition probability matrix P of shape (K, K) where P[i, j] is
        the probability of transitioning from regime i to regime j.
        Each row sums to 1.
    stationary_dist : NDArray[np.float64]
        Stationary distribution of the Markov chain (long-run regime probabilities).
        Computed as the left eigenvector corresponding to eigenvalue 1.
    """

    def __init__(
        self,
        transition_matrix: NDArray[np.float64],
        validate: bool = True
    ) -> None:
        """
        Initialize Markov chain.

        Parameters
        ----------
        transition_matrix : NDArray[np.float64]
            Transition probability matrix of shape (K, K).
            Must be row-stochastic (rows sum to 1, all entries in [0, 1]).
        validate : bool, optional
            If True, validate that transition_matrix is row-stochastic.
            Default is True.

        Raises
        ------
        ValueError
            If transition_matrix is not row-stochastic.
        """
        self.transition_matrix = transition_matrix.astype(np.float64)
        self.n_regimes: int = transition_matrix.shape[0]

        if validate:
            self._validate_transition_matrix()

        self.stationary_dist = self._compute_stationary_distribution()

    def _validate_transition_matrix(self) -> None:
        """
        Validate that transition matrix is row-stochastic.

        A transition matrix is row-stochastic if:
        - Shape is (K, K)
        - All entries are in [0, 1]
        - Each row sums to 1 (within numerical tolerance)

        Raises
        ------
        ValueError
            If any validation check fails.
        """
        if self.transition_matrix.shape[0] != self.transition_matrix.shape[1]:
            raise ValueError(
                f"Transition matrix must be square. Got shape {self.transition_matrix.shape}"
            )

        if not np.all((self.transition_matrix >= 0) & (self.transition_matrix <= 1)):
            raise ValueError("All transition probabilities must be in [0, 1]")

        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-10):
            raise ValueError(
                f"Each row must sum to 1. Got row sums: {row_sums}"
            )

    def _compute_stationary_distribution(self) -> NDArray[np.float64]:
        """
        Compute stationary distribution of Markov chain.

        The stationary distribution π satisfies:
            π P = π   (left eigenvector, eigenvalue 1)
            Σ π_i = 1

        For irreducible, aperiodic chains (typical in regime-switching),
        this is unique and represents the long-run probability of each regime.

        Returns
        -------
        NDArray[np.float64]
            Stationary distribution of shape (K,), sums to 1.
        """
        # Compute left eigenvector (equivalently, right eigenvector of P^T)
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])

        # Normalize to sum to 1
        stationary = stationary / stationary.sum()

        return stationary.astype(np.float64)

    def simulate_path(
        self,
        n_steps: int,
        initial_regime: Optional[int] = None,
        random_seed: Optional[int] = None
    ) -> NDArray[np.int64]:
        """
        Simulate a path of regime states.

        Uses the transition matrix to generate a sequence of regimes.
        At each step t, the next regime s_{t+1} is sampled from the
        distribution P(· | s_t).

        Parameters
        ----------
        n_steps : int
            Length of path (number of transitions)
        initial_regime : int, optional
            Starting regime (0-indexed). If None, sample from stationary distribution.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NDArray[np.int64]
            Regime path of shape (n_steps,) with values in {0, …, K-1}.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize
        if initial_regime is None:
            current_regime = np.random.choice(
                self.n_regimes,
                p=self.stationary_dist
            )
        else:
            if not (0 <= initial_regime < self.n_regimes):
                raise ValueError(
                    f"initial_regime must be in {{0, …, {self.n_regimes - 1}}}"
                )
            current_regime = initial_regime

        path = np.zeros(n_steps, dtype=np.int64)
        path[0] = current_regime

        # Simulate transitions
        for t in range(1, n_steps):
            # Transition probabilities from current regime
            transition_probs = self.transition_matrix[current_regime, :]

            # Sample next regime
            current_regime = np.random.choice(
                self.n_regimes,
                p=transition_probs
            )
            path[t] = current_regime

        return path

    def expected_duration(self, regime: int) -> float:
        """
        Expected duration (holding time) in a regime.

        For regime i, the expected number of steps until exiting is:
            E[duration | regime i] = 1 / (1 - P_{ii})

        This is the expected value of a geometric distribution with parameter
        (1 - P_{ii}).

        Parameters
        ----------
        regime : int
            Regime index (0-indexed).

        Returns
        -------
        float
            Expected duration in that regime.

        Raises
        ------
        ValueError
            If regime index is out of bounds or if P_{ii} = 1 (absorbing state).
        """
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}"
            )

        self_prob = self.transition_matrix[regime, regime]

        if self_prob >= 1.0:
            raise ValueError(
                f"Regime {regime} is absorbing (P_{{{regime},{regime}}} = 1). "
                "Expected duration is infinite."
            )

        return 1.0 / (1.0 - self_prob)

    def expected_durations(self) -> NDArray[np.float64]:
        """
        Expected durations for all regimes.

        Returns
        -------
        NDArray[np.float64]
            Array of shape (K,) with expected duration in each regime.
        """
        return np.array([
            self.expected_duration(i)
            for i in range(self.n_regimes)
        ])

    def ergodic_probabilities(
        self,
        n_steps: int,
        initial_dist: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Probability distribution over regimes after n steps.

        Given an initial distribution π_0, computes π_n via:
            π_n = π_0 P^n

        As n → ∞, π_n → stationary_dist (for ergodic chains).

        Parameters
        ----------
        n_steps : int
            Number of transitions.
        initial_dist : NDArray[np.float64], optional
            Initial distribution over regimes of shape (K,).
            If None, use stationary distribution.

        Returns
        -------
        NDArray[np.float64]
            Distribution over regimes after n steps, shape (K,), sums to 1.
        """
        if initial_dist is None:
            initial_dist = self.stationary_dist
        else:
            if len(initial_dist) != self.n_regimes:
                raise ValueError(
                    f"initial_dist must have length {self.n_regimes}"
                )
            if not np.isclose(initial_dist.sum(), 1.0):
                raise ValueError("initial_dist must sum to 1")

        # Compute P^n via matrix exponentiation
        P_n = np.linalg.matrix_power(self.transition_matrix, n_steps)

        # Apply initial distribution
        return initial_dist @ P_n

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MarkovChain(n_regimes={self.n_regimes}, "
            f"stationary_dist={self.stationary_dist})"
        )


def create_symmetric_markov_chain(
    n_regimes: int,
    self_transition_prob: float
) -> MarkovChain:
    """
    Create a symmetric Markov chain with equal transition probabilities.

    Useful for generating test data or as a baseline model.
    All off-diagonal transitions have equal probability.

    Parameters
    ----------
    n_regimes : int
        Number of regimes (K).
    self_transition_prob : float
        Probability of staying in the same regime (P_{ii}).
        Must be in (0, 1).

    Returns
    -------
    MarkovChain
        Symmetric Markov chain with specified self-transition probability.

    Raises
    ------
    ValueError
        If self_transition_prob is not in (0, 1).
    """
    if not (0 < self_transition_prob < 1):
        raise ValueError(
            f"self_transition_prob must be in (0, 1). Got {self_transition_prob}"
        )

    # Off-diagonal probability (equal for all transitions)
    off_diag_prob = (1 - self_transition_prob) / (n_regimes - 1)

    # Build transition matrix
    P = np.full((n_regimes, n_regimes), off_diag_prob)
    np.fill_diagonal(P, self_transition_prob)

    return MarkovChain(P, validate=False)
