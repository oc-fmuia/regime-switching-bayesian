"""
Unit tests for regime-switching Markov chain model.

Tests cover:
- Initialization and validation
- Transition matrix properties
- Stationary distribution computation
- Simulation
- Expected durations
- Edge cases and error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from src.regimes.markov import MarkovChain, create_symmetric_markov_chain


class TestMarkovChainInitialization:
    """Tests for MarkovChain initialization and validation."""

    def test_valid_2x2_matrix(self) -> None:
        """Test initialization with valid 2x2 transition matrix."""
        P = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        mc = MarkovChain(P)
        assert mc.n_regimes == 2
        assert_array_almost_equal(mc.transition_matrix, P)

    def test_valid_3x3_matrix(self) -> None:
        """Test initialization with valid 3x3 transition matrix."""
        P = np.array([
            [0.8, 0.15, 0.05],
            [0.1, 0.7, 0.2],
            [0.05, 0.15, 0.8]
        ])
        mc = MarkovChain(P)
        assert mc.n_regimes == 3

    def test_non_square_matrix_raises_error(self) -> None:
        """Test that non-square matrices raise ValueError."""
        P = np.array([
            [0.5, 0.5, 0.0],
            [0.3, 0.7, 0.0]
        ])
        with pytest.raises(ValueError, match="must be square"):
            MarkovChain(P)

    def test_invalid_probabilities_raise_error(self) -> None:
        """Test that probabilities outside [0, 1] raise ValueError."""
        P = np.array([
            [0.9, 0.2],  # Row sum > 1
            [0.1, 0.8]
        ])
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MarkovChain(P)

    def test_non_stochastic_rows_raise_error(self) -> None:
        """Test that rows not summing to 1 raise ValueError."""
        P = np.array([
            [0.9, 0.05],  # Row sum < 1
            [0.2, 0.8]
        ])
        with pytest.raises(ValueError, match="sum to 1"):
            MarkovChain(P)

    def test_validation_disabled(self) -> None:
        """Test that validation can be disabled."""
        P = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        # Should not raise even if we pass invalid matrix (if validate=False)
        mc = MarkovChain(P, validate=False)
        assert mc.n_regimes == 2


class TestStationaryDistribution:
    """Tests for stationary distribution computation."""

    def test_stationary_dist_2_regime(self) -> None:
        """Test stationary distribution for 2-regime chain."""
        # Known solution: P = [[0.9, 0.1], [0.2, 0.8]]
        # Stationary: π = [2/3, 1/3]
        P = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        mc = MarkovChain(P)
        expected = np.array([2/3, 1/3])
        assert_allclose(mc.stationary_dist, expected, atol=1e-6)

    def test_stationary_dist_sums_to_one(self) -> None:
        """Test that stationary distribution sums to 1."""
        P = np.random.dirichlet([1, 1, 1], size=3)
        mc = MarkovChain(P)
        assert_allclose(mc.stationary_dist.sum(), 1.0, atol=1e-10)

    def test_stationary_dist_satisfies_invariance(self) -> None:
        """Test that π P = π (stationary property)."""
        P = np.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        mc = MarkovChain(P)
        # π P should equal π
        result = mc.stationary_dist @ P
        assert_allclose(result, mc.stationary_dist, atol=1e-10)

    def test_symmetric_chain_uniform_stationary(self) -> None:
        """Test that symmetric chain has uniform stationary distribution."""
        # Symmetric chain: all transitions equal
        P = np.ones((3, 3)) / 3
        mc = MarkovChain(P)
        expected = np.ones(3) / 3
        assert_allclose(mc.stationary_dist, expected, atol=1e-10)


class TestSimulation:
    """Tests for regime path simulation."""

    def test_simulate_path_length(self) -> None:
        """Test that simulated path has correct length."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        path = mc.simulate_path(n_steps=100, random_seed=42)
        assert len(path) == 100

    def test_simulate_path_values_in_range(self) -> None:
        """Test that simulated path values are valid regime indices."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        path = mc.simulate_path(n_steps=1000, random_seed=42)
        assert np.all(path >= 0) and np.all(path < 2)

    def test_simulate_with_initial_regime(self) -> None:
        """Test simulation with specified initial regime."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        path = mc.simulate_path(n_steps=10, initial_regime=1, random_seed=42)
        assert path[0] == 1

    def test_simulate_with_invalid_initial_regime_raises(self) -> None:
        """Test that invalid initial regime raises ValueError."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        with pytest.raises(ValueError, match="initial_regime must be in"):
            mc.simulate_path(n_steps=10, initial_regime=5)

    def test_simulate_reproducibility(self) -> None:
        """Test that same seed produces same path."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        path1 = mc.simulate_path(n_steps=100, random_seed=42)
        path2 = mc.simulate_path(n_steps=100, random_seed=42)
        assert_array_almost_equal(path1, path2)

    def test_simulate_different_seeds_different_paths(self) -> None:
        """Test that different seeds produce different paths."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        path1 = mc.simulate_path(n_steps=100, random_seed=42)
        path2 = mc.simulate_path(n_steps=100, random_seed=43)
        assert not np.array_equal(path1, path2)


class TestExpectedDurations:
    """Tests for expected duration (holding time) in regimes."""

    def test_expected_duration_calculation(self) -> None:
        """Test expected duration formula: E[duration] = 1 / (1 - P_ii)."""
        P = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        mc = MarkovChain(P)
        # E[duration | regime 0] = 1 / (1 - 0.9) = 10
        # E[duration | regime 1] = 1 / (1 - 0.8) = 5
        assert_allclose(mc.expected_duration(0), 10.0, atol=1e-6)
        assert_allclose(mc.expected_duration(1), 5.0, atol=1e-6)

    def test_expected_durations_array(self) -> None:
        """Test expected_durations() returns array of durations."""
        P = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        mc = MarkovChain(P)
        durations = mc.expected_durations()
        assert len(durations) == 2
        assert_allclose(durations[0], 10.0, atol=1e-6)
        assert_allclose(durations[1], 5.0, atol=1e-6)

    def test_absorbing_state_raises_error(self) -> None:
        """Test that absorbing state (P_ii = 1) raises ValueError."""
        P = np.array([
            [1.0, 0.0],  # Absorbing state 0
            [0.5, 0.5]
        ])
        mc = MarkovChain(P)
        with pytest.raises(ValueError, match="absorbing"):
            mc.expected_duration(0)


class TestErgodicProbabilities:
    """Tests for multi-step transition probabilities."""

    def test_one_step_equals_transition_matrix(self) -> None:
        """Test that n=1 gives the transition matrix itself."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        initial_dist = np.array([1.0, 0.0])  # Start in regime 0
        result = mc.ergodic_probabilities(n_steps=1, initial_dist=initial_dist)
        assert_allclose(result, P[0, :], atol=1e-10)

    def test_convergence_to_stationary(self) -> None:
        """Test that probabilities converge to stationary distribution."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        initial_dist = np.array([1.0, 0.0])
        # After many steps, should approach stationary
        result = mc.ergodic_probabilities(n_steps=1000, initial_dist=initial_dist)
        assert_allclose(result, mc.stationary_dist, atol=1e-4)

    def test_invalid_initial_dist_raises(self) -> None:
        """Test that invalid initial distribution raises ValueError."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])
        mc = MarkovChain(P)
        with pytest.raises(ValueError, match="must have length"):
            mc.ergodic_probabilities(n_steps=1, initial_dist=np.array([0.5]))


class TestCreateSymmetricMarkovChain:
    """Tests for symmetric Markov chain creation helper."""

    def test_create_2_regime_chain(self) -> None:
        """Test creation of 2-regime symmetric chain."""
        mc = create_symmetric_markov_chain(n_regimes=2, self_transition_prob=0.9)
        assert mc.n_regimes == 2
        assert_allclose(mc.transition_matrix[0, 0], 0.9, atol=1e-10)
        assert_allclose(mc.transition_matrix[0, 1], 0.1, atol=1e-10)

    def test_create_3_regime_chain(self) -> None:
        """Test creation of 3-regime symmetric chain."""
        mc = create_symmetric_markov_chain(n_regimes=3, self_transition_prob=0.7)
        assert mc.n_regimes == 3
        # Diagonal should be 0.7
        assert_allclose(np.diag(mc.transition_matrix), [0.7, 0.7, 0.7], atol=1e-10)
        # Off-diagonal should be (1-0.7)/(3-1) = 0.15
        assert_allclose(mc.transition_matrix[0, 1], 0.15, atol=1e-10)

    def test_invalid_self_prob_raises(self) -> None:
        """Test that invalid self-transition probability raises ValueError."""
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            create_symmetric_markov_chain(n_regimes=2, self_transition_prob=0.0)

        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            create_symmetric_markov_chain(n_regimes=2, self_transition_prob=1.0)


class TestMarkovChainIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: create, compute, simulate."""
        # Create chain
        P = np.array([
            [0.85, 0.15],
            [0.25, 0.75]
        ])
        mc = MarkovChain(P)

        # Check properties
        assert mc.n_regimes == 2
        stationary = mc.stationary_dist
        assert_allclose(stationary.sum(), 1.0)

        # Simulate
        path = mc.simulate_path(n_steps=500, random_seed=42)
        assert len(path) == 500
        assert np.all((path >= 0) & (path < 2))

        # Compute durations
        durations = mc.expected_durations()
        assert all(d > 1.0 for d in durations)  # Duration >= 1

    def test_reproducible_analysis(self) -> None:
        """Test that analysis is reproducible across runs."""
        P = np.array([
            [0.8, 0.2],
            [0.3, 0.7]
        ])

        # First run
        mc1 = MarkovChain(P)
        path1 = mc1.simulate_path(n_steps=100, random_seed=123)
        stat1 = mc1.stationary_dist

        # Second run (independent instance)
        mc2 = MarkovChain(P)
        path2 = mc2.simulate_path(n_steps=100, random_seed=123)
        stat2 = mc2.stationary_dist

        assert_array_almost_equal(path1, path2)
        assert_array_almost_equal(stat1, stat2)
