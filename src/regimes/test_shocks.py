"""
Comprehensive unit tests for shock propagation framework.

Tests ShockModel and ReturnWithShocks classes including:
- Initialization and validation
- Loading matrix operations
- Shock impact computation
- Deterministic stress testing
- Stochastic simulation
- Variance decomposition
- Edge cases and error handling
"""

import sys
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, '/home/fmuia/.openclaw/workspace-fernando/regime-switching-bayesian/src')

from regimes.shocks import ShockModel, ReturnWithShocks


class TestShockModelInitialization:
    """Test ShockModel initialization and validation."""

    def test_init_with_default_matrices(self):
        """Test initialization with default (zero) loading matrices."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        
        assert sm.n_assets == 3
        assert sm.n_shocks == 2
        assert sm.n_regimes == 2
        assert sm.loading_matrices.shape == (2, 3, 2)
        assert np.allclose(sm.loading_matrices, 0.0)

    def test_init_with_provided_matrices(self):
        """Test initialization with pre-specified loading matrices."""
        B = np.random.randn(2, 3, 2)  # (K, N, M)
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2, loading_matrices=B)
        
        assert np.allclose(sm.loading_matrices, B)

    def test_init_invalid_dimensions(self):
        """Test that non-positive dimensions are rejected."""
        with pytest.raises(ValueError):
            ShockModel(n_assets=0, n_shocks=2, n_regimes=2)
        
        with pytest.raises(ValueError):
            ShockModel(n_assets=3, n_shocks=-1, n_regimes=2)
        
        with pytest.raises(ValueError):
            ShockModel(n_assets=3, n_shocks=2, n_regimes=0)

    def test_init_mismatched_matrix_shape(self):
        """Test that mismatched loading matrix shape is rejected."""
        B_wrong = np.random.randn(2, 4, 2)  # Wrong n_assets
        
        with pytest.raises(ValueError):
            ShockModel(n_assets=3, n_shocks=2, n_regimes=2, loading_matrices=B_wrong)

    def test_init_type_conversion(self):
        """Test that loading matrices are converted to float64."""
        B = np.random.randint(0, 5, size=(2, 3, 2))  # Integer array
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2, loading_matrices=B)
        
        assert sm.loading_matrices.dtype == np.float64


class TestLoadingMatrixOperations:
    """Test getting and setting loading matrices."""

    def test_get_loading_matrix(self):
        """Test retrieving loading matrix for a specific regime."""
        B = np.random.randn(3, 4, 2)
        sm = ShockModel(n_assets=4, n_shocks=2, n_regimes=3, loading_matrices=B)
        
        # Get regime 1
        B_1 = sm.get_loading_matrix(1)
        assert B_1.shape == (4, 2)
        assert np.allclose(B_1, B[1, :, :])

    def test_get_loading_matrix_invalid_regime(self):
        """Test that invalid regime index raises error."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        
        with pytest.raises(ValueError):
            sm.get_loading_matrix(-1)
        
        with pytest.raises(ValueError):
            sm.get_loading_matrix(2)

    def test_set_loading_matrices(self):
        """Test updating loading matrices."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        B_new = np.random.randn(2, 3, 2)
        
        sm.set_loading_matrices(B_new)
        assert np.allclose(sm.loading_matrices, B_new)

    def test_set_loading_matrices_invalid_shape(self):
        """Test that wrong shape is rejected when setting."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        B_wrong = np.random.randn(2, 4, 2)  # Wrong n_assets
        
        with pytest.raises(ValueError):
            sm.set_loading_matrices(B_wrong)


class TestShockImpact:
    """Test shock impact computation."""

    def test_compute_shock_impact_simple(self):
        """Test basic shock impact computation B @ u."""
        B = np.array([
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0]]
        ])
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2, loading_matrices=B)
        
        u = np.array([1.0, 2.0])
        
        # Regime 0: [[1, 0], [0, 1], [0.5, 0.5]] @ [1, 2]^T = [1, 2, 1.5]
        impact_0 = sm.compute_shock_impact(u, regime=0)
        expected_0 = np.array([1.0, 2.0, 1.5])
        assert np.allclose(impact_0, expected_0)
        
        # Regime 1: [[0.5, 0.5], [0, 0], [1, 0]] @ [1, 2]^T = [1.5, 0, 1]
        impact_1 = sm.compute_shock_impact(u, regime=1)
        expected_1 = np.array([1.5, 0.0, 1.0])
        assert np.allclose(impact_1, expected_1)

    def test_compute_shock_impact_invalid_shock_dim(self):
        """Test that wrong shock dimension is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        u_wrong = np.array([1.0, 2.0, 3.0])  # Wrong length
        
        with pytest.raises(ValueError):
            sm.compute_shock_impact(u_wrong, regime=0)

    def test_compute_shock_impact_invalid_regime(self):
        """Test that invalid regime raises error."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        u = np.array([1.0, 2.0])
        
        with pytest.raises(ValueError):
            sm.compute_shock_impact(u, regime=5)


class TestShockSimulation:
    """Test stochastic shock process generation."""

    def test_simulate_shocks_shape(self):
        """Test that simulated shocks have correct shape."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        
        shocks = sm.simulate_shocks(n_steps=100)
        assert shocks.shape == (100, 2)

    def test_simulate_shocks_reproducibility(self):
        """Test that same seed gives identical simulations."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        
        shocks1 = sm.simulate_shocks(n_steps=100, random_seed=42)
        shocks2 = sm.simulate_shocks(n_steps=100, random_seed=42)
        
        assert np.array_equal(shocks1, shocks2)

    def test_simulate_shocks_statistics(self):
        """Test that simulated shocks have correct mean and std."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        
        shocks = sm.simulate_shocks(n_steps=10000, random_seed=123)
        
        # Mean should be close to 0
        assert np.allclose(np.mean(shocks, axis=0), 0.0, atol=0.05)
        
        # Std should be close to 1
        assert np.allclose(np.std(shocks, axis=0), 1.0, atol=0.05)

    def test_simulate_shocks_with_custom_std(self):
        """Test shock simulation with custom standard deviations."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        shock_std = np.array([2.0, 0.5])
        
        shocks = sm.simulate_shocks(n_steps=10000, shock_std=shock_std, random_seed=123)
        
        # Check empirical std matches specified
        empirical_std = np.std(shocks, axis=0)
        assert np.allclose(empirical_std, shock_std, atol=0.1)

    def test_simulate_shocks_invalid_std_shape(self):
        """Test that wrong shock_std shape is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        shock_std_wrong = np.array([1.0, 2.0, 3.0])  # Wrong length
        
        with pytest.raises(ValueError):
            sm.simulate_shocks(n_steps=100, shock_std=shock_std_wrong)


class TestStressTesting:
    """Test deterministic stress scenarios."""

    def test_stress_test_simple(self):
        """Test basic stress test."""
        B = np.array([
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]]
        ])
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=2, loading_matrices=B)
        
        u_stress = np.array([1.0, 0.0])
        mu = np.array([0.01, 0.02])
        
        # Regime 0: [0.01, 0.02] + [[1, 0], [0, 1]] @ [1, 0] = [0.01 + 1, 0.02 + 0]
        result = sm.stress_test(u_stress, regime=0, mean_return=mu)
        expected = np.array([1.01, 0.02])
        assert np.allclose(result, expected)

    def test_stress_test_default_mean(self):
        """Test stress test with default zero mean."""
        B = np.eye(2)[np.newaxis, :, :]  # Identity matrix
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        
        u_stress = np.array([1.0, 2.0])
        result = sm.stress_test(u_stress, regime=0)
        
        assert np.allclose(result, u_stress)

    def test_stress_test_invalid_mean_shape(self):
        """Test that wrong mean shape is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=1)
        u_stress = np.array([1.0, 2.0])
        mu_wrong = np.array([0.01, 0.02])  # Wrong length
        
        with pytest.raises(ValueError):
            sm.stress_test(u_stress, regime=0, mean_return=mu_wrong)


class TestReturnWithShocksInitialization:
    """Test ReturnWithShocks initialization and validation."""

    def test_init_valid(self):
        """Test initialization with valid inputs."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu = np.zeros((2, 3))
        Sigma = np.array([np.eye(3), np.eye(3)])
        
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        assert rws.shock_model is sm
        assert np.allclose(rws.regime_means, mu)
        assert np.allclose(rws.regime_covs, Sigma)

    def test_init_invalid_mean_shape(self):
        """Test that wrong mean shape is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu_wrong = np.zeros((2, 4))  # Wrong n_assets
        Sigma = np.array([np.eye(3), np.eye(3)])
        
        with pytest.raises(ValueError):
            ReturnWithShocks(sm, mu_wrong, Sigma)

    def test_init_invalid_cov_shape(self):
        """Test that wrong covariance shape is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu = np.zeros((2, 3))
        Sigma_wrong = np.array([np.eye(3), np.eye(4)])  # Wrong n_assets in second
        
        with pytest.raises(ValueError):
            ReturnWithShocks(sm, mu, Sigma_wrong)

    def test_init_non_symmetric_cov(self):
        """Test that non-symmetric covariance is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu = np.zeros((2, 3))
        Sigma_bad = np.eye(3)
        Sigma_bad[0, 1] = 1.0  # Non-symmetric
        Sigma = np.array([Sigma_bad, np.eye(3)])
        
        with pytest.raises(ValueError):
            ReturnWithShocks(sm, mu, Sigma)

    def test_init_non_pd_cov(self):
        """Test that non-positive-definite covariance is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu = np.zeros((2, 3))
        Sigma_bad = -np.eye(3)  # Negative definite
        Sigma = np.array([Sigma_bad, np.eye(3)])
        
        with pytest.raises(ValueError):
            ReturnWithShocks(sm, mu, Sigma)


class TestReturnGeneration:
    """Test return generation with shocks."""

    def test_generate_returns_shape(self):
        """Test that generated returns have correct shape."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu = np.zeros((2, 3))
        Sigma = np.array([np.eye(3), np.eye(3)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.array([0, 1, 0, 1])
        shocks = np.random.randn(4, 2)
        
        returns = rws.generate_returns(regime_path, shocks, random_seed=42)
        assert returns.shape == (4, 3)

    def test_generate_returns_deterministic(self):
        """Test return generation with zero shocks and noise."""
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1)
        mu = np.array([[0.01, 0.02]])
        Sigma = np.array([np.zeros((2, 2))])  # No idiosyncratic noise
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.array([0, 0])
        shocks = np.zeros((2, 2))
        noise = np.zeros((2, 2))
        
        returns = rws.generate_returns(regime_path, shocks, idiosyncratic_noise=noise)
        
        # Without shocks or noise, returns = mu
        assert np.allclose(returns, mu[[0, 0]])

    def test_generate_returns_with_shocks(self):
        """Test that shocks affect returns."""
        B = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        mu = np.array([[0.0, 0.0]])
        Sigma = np.array([np.zeros((2, 2))])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.array([0])
        shocks = np.array([[1.0, 2.0]])
        noise = np.zeros((1, 2))
        
        returns = rws.generate_returns(regime_path, shocks, idiosyncratic_noise=noise)
        
        # r = mu + B @ u = [0, 0] + [1, 2] = [1, 2]
        assert np.allclose(returns[0], [1.0, 2.0])

    def test_generate_returns_stochastic_noise(self):
        """Test stochastic noise sampling."""
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1)
        mu = np.zeros((1, 2))
        Sigma = np.array([np.eye(2)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.array([0, 0, 0])
        shocks = np.zeros((3, 2))
        
        # Generate with and without noise
        returns1 = rws.generate_returns(regime_path, shocks, idiosyncratic_noise=np.zeros((3, 2)))
        returns2 = rws.generate_returns(regime_path, shocks, random_seed=42)
        
        # With noise should differ from zero
        assert not np.allclose(returns2, returns1)

    def test_generate_returns_invalid_shock_dim(self):
        """Test that wrong shock dimension is rejected."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        mu = np.zeros((2, 3))
        Sigma = np.array([np.eye(3), np.eye(3)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.array([0, 1])
        shocks_wrong = np.random.randn(2, 3)  # Wrong n_shocks
        
        with pytest.raises(ValueError):
            rws.generate_returns(regime_path, shocks_wrong)


class TestVarianceDecomposition:
    """Test variance decomposition analysis."""

    def test_compute_systematic_variance(self):
        """Test systematic variance computation."""
        B = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        mu = np.zeros((1, 2))
        Sigma = np.zeros((1, 2, 2))  # No idiosyncratic
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        # Shock cov = I, so systematic = B @ I @ B^T = B B^T = I
        sys_var = rws.compute_systematic_variance(regime=0, shock_cov=np.eye(2))
        assert np.allclose(sys_var, np.eye(2))

    def test_compute_total_variance(self):
        """Test total variance computation."""
        B = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        mu = np.zeros((1, 2))
        Sigma = np.array([np.eye(2)])  # Idiosyncratic variance = I
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        # Total = B I B^T + I = I + I = 2I
        total_var = rws.compute_total_variance(regime=0, shock_cov=np.eye(2))
        assert np.allclose(total_var, 2 * np.eye(2))

    def test_variance_decomposition(self):
        """Test variance decomposition percentages."""
        B = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        mu = np.zeros((1, 2))
        Sigma = np.array([np.eye(2)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        decomp = rws.variance_decomposition(regime=0, shock_cov=np.eye(2))
        
        # Systematic variance = trace(I) = 2
        # Idiosyncratic = trace(I) = 2
        # Ratio = 2 / 4 = 0.5
        assert np.isclose(decomp['systematic_variance'], 2.0)
        assert np.isclose(decomp['idiosyncratic_variance'], 2.0)
        assert np.isclose(decomp['systematic_ratio'], 0.5)

    def test_variance_decomposition_all_systematic(self):
        """Test decomposition when variance is purely systematic."""
        B = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        mu = np.zeros((1, 2))
        Sigma = np.zeros((1, 2, 2))  # No idiosyncratic
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        decomp = rws.variance_decomposition(regime=0, shock_cov=np.eye(2))
        
        assert np.isclose(decomp['idiosyncratic_variance'], 0.0)
        assert np.isclose(decomp['systematic_ratio'], 1.0)

    def test_variance_decomposition_no_shocks(self):
        """Test decomposition when loadings are zero."""
        B = np.zeros((1, 2, 2))
        sm = ShockModel(n_assets=2, n_shocks=2, n_regimes=1, loading_matrices=B)
        mu = np.zeros((1, 2))
        Sigma = np.array([np.eye(2)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        decomp = rws.variance_decomposition(regime=0, shock_cov=np.eye(2))
        
        assert np.isclose(decomp['systematic_variance'], 0.0)
        assert np.isclose(decomp['systematic_ratio'], 0.0)


class TestFactorExposure:
    """Test factor exposure extraction."""

    def test_compute_factor_exposure(self):
        """Test getting factor exposure for a regime."""
        B = np.random.randn(3, 4, 2)
        sm = ShockModel(n_assets=4, n_shocks=2, n_regimes=3, loading_matrices=B)
        mu = np.zeros((3, 4))
        Sigma = np.array([np.eye(4), np.eye(4), np.eye(4)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        B_1 = rws.compute_factor_exposure(regime=1)
        assert np.allclose(B_1, B[1, :, :])


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_asset_single_shock(self):
        """Test with minimal dimensions."""
        sm = ShockModel(n_assets=1, n_shocks=1, n_regimes=1)
        mu = np.array([[0.01]])
        Sigma = np.array([[[0.01]]])
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.array([0])
        shocks = np.array([[1.0]])
        returns = rws.generate_returns(regime_path, shocks, random_seed=42)
        
        assert returns.shape == (1, 1)

    def test_many_assets_many_regimes(self):
        """Test with larger dimensions."""
        sm = ShockModel(n_assets=20, n_shocks=5, n_regimes=5)
        mu = np.random.randn(5, 20)
        Sigma = np.array([np.random.randn(20, 20) for _ in range(5)])
        Sigma = np.array([S @ S.T for S in Sigma])  # Make positive-definite
        
        rws = ReturnWithShocks(sm, mu, Sigma)
        
        regime_path = np.random.randint(0, 5, size=100)
        shocks = np.random.randn(100, 5)
        returns = rws.generate_returns(regime_path, shocks, random_seed=42)
        
        assert returns.shape == (100, 20)

    def test_repr_methods(self):
        """Test string representations."""
        sm = ShockModel(n_assets=3, n_shocks=2, n_regimes=2)
        repr_sm = repr(sm)
        assert "ShockModel" in repr_sm
        assert "3" in repr_sm
        assert "2" in repr_sm
        
        mu = np.zeros((2, 3))
        Sigma = np.array([np.eye(3), np.eye(3)])
        rws = ReturnWithShocks(sm, mu, Sigma)
        repr_rws = repr(rws)
        assert "ReturnWithShocks" in repr_rws


# Run tests if executed as script
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
