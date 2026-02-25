"""
Unit tests for Student-t return model and covariance handling.

Tests cover:
- Student-t likelihood and sampling
- Covariance matrix properties
- Conditional distributions
- Mahalanobis distance
- Edge cases and error handling
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose, assert_equal

from src.returns.student_t import StudentTReturnModel, create_symmetric_return_model
from src.returns.covariance import (
    CovarianceModel,
    create_identity_covariance,
    create_compound_symmetric_covariance
)


class TestStudentTReturnModel:
    """Tests for Student-t return model."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        means = np.array([[0.001, 0.002], [0.0005, 0.0015]])
        cov1 = np.array([[0.0004, 0.00008], [0.00008, 0.0009]])
        cov2 = np.array([[0.0005, 0.00010], [0.00010, 0.001]])
        covariances = np.array([cov1, cov2])
        dfs = np.array([10.0, 8.0])

        model = StudentTReturnModel(means, covariances, dfs)
        assert model.n_regimes == 2
        assert model.n_assets == 2

    def test_invalid_shapes_raise_error(self) -> None:
        """Test that mismatched shapes raise ValueError."""
        means = np.array([[0.001, 0.002]])
        cov = np.eye(2) * 0.0004
        covariances = np.array([cov])
        dfs = np.array([10.0])

        # Wrong covariance shape
        bad_covariances = np.array([cov, cov])  # 2 instead of 1
        with pytest.raises(ValueError, match="shape"):
            StudentTReturnModel(means, bad_covariances, dfs)

    def test_non_positive_definite_raises_error(self) -> None:
        """Test that non-positive-definite covariances raise error."""
        means = np.array([[0.001, 0.002]])
        bad_cov = np.array([[0.0004, 0.001], [0.001, 0.0001]])  # Non-PD
        covariances = np.array([bad_cov])
        dfs = np.array([10.0])

        with pytest.raises(ValueError, match="positive definite"):
            StudentTReturnModel(means, covariances, dfs)

    def test_invalid_dfs_raise_error(self) -> None:
        """Test that degrees of freedom <= 2 raise error."""
        means = np.array([[0.001, 0.002]])
        cov = np.eye(2) * 0.0004
        covariances = np.array([cov])
        dfs = np.array([2.0])  # Should be > 2

        with pytest.raises(ValueError, match="> 2"):
            StudentTReturnModel(means, covariances, dfs)

    def test_log_likelihood_shape(self) -> None:
        """Test that log-likelihood has correct shape."""
        means = np.array([[0.001, 0.002], [0.0005, 0.0015]])
        cov = np.array([[0.0004, 0.00008], [0.00008, 0.0009]])
        covariances = np.array([cov, cov])
        dfs = np.array([10.0, 8.0])

        model = StudentTReturnModel(means, covariances, dfs)

        returns = np.random.normal(0, 0.02, size=(100, 2))
        ll = model.log_likelihood(returns, regime=0)

        assert ll.shape == (100,)
        assert np.all(np.isfinite(ll))

    def test_likelihood_all_regimes_shape(self) -> None:
        """Test likelihood across all regimes."""
        means = np.array([[0.001, 0.002], [0.0005, 0.0015]])
        cov = np.array([[0.0004, 0.00008], [0.00008, 0.0009]])
        covariances = np.array([cov, cov])
        dfs = np.array([10.0, 8.0])

        model = StudentTReturnModel(means, covariances, dfs)
        returns = np.random.normal(0, 0.02, size=(50, 2))
        lls = model.log_likelihood_all_regimes(returns)

        assert lls.shape == (50, 2)
        assert np.all(np.isfinite(lls))

    def test_sampling_shape_and_reproducibility(self) -> None:
        """Test that sampling works and is reproducible."""
        means = np.array([[0.001, 0.002], [0.0005, 0.0015]])
        cov = np.array([[0.0004, 0.00008], [0.00008, 0.0009]])
        covariances = np.array([cov, cov])
        dfs = np.array([10.0, 8.0])

        model = StudentTReturnModel(means, covariances, dfs)

        # Sample and check shape
        samples1 = model.sample(regime=0, n_samples=1000, random_seed=42)
        assert samples1.shape == (1000, 2)

        # Reproducibility
        samples2 = model.sample(regime=0, n_samples=1000, random_seed=42)
        assert_array_almost_equal(samples1, samples2)

    def test_correlation_matrix_extraction(self) -> None:
        """Test correlation matrix extraction from covariance."""
        means = np.array([[0.001, 0.002]])
        # Create specific covariance
        cov = np.array([[0.0004, 0.00012], [0.00012, 0.0009]])
        covariances = np.array([cov])
        dfs = np.array([10.0])

        model = StudentTReturnModel(means, covariances, dfs)
        corr = model.correlation_matrix(regime=0)

        # Check properties
        assert corr.shape == (2, 2)
        assert_allclose(np.diag(corr), [1.0, 1.0], atol=1e-10)
        assert_allclose(corr, corr.T, atol=1e-10)  # Symmetric
        assert_allclose(corr[0, 1], 0.12 / np.sqrt(0.0004 * 0.0009), atol=1e-10)

    def test_marginal_volatilities(self) -> None:
        """Test volatility extraction."""
        means = np.array([[0.001, 0.002]])
        cov = np.array([[0.0004, 0.00012], [0.00012, 0.0009]])
        covariances = np.array([cov])
        dfs = np.array([10.0])

        model = StudentTReturnModel(means, covariances, dfs)
        vols = model.marginal_volatilities(regime=0)

        assert_allclose(vols, [np.sqrt(0.0004), np.sqrt(0.0009)], atol=1e-10)

    def test_tail_index(self) -> None:
        """Test tail index (inverse degrees of freedom)."""
        means = np.array([[0.001, 0.002]])
        cov = np.eye(2) * 0.0004
        covariances = np.array([cov])
        dfs = np.array([10.0])

        model = StudentTReturnModel(means, covariances, dfs)
        tail_idx = model.tail_index(regime=0)

        assert_allclose(tail_idx, 0.1, atol=1e-10)


class TestCovarianceModel:
    """Tests for covariance matrix handling."""

    def test_identity_covariance_initialization(self) -> None:
        """Test identity correlation covariance."""
        corr = np.eye(3)
        vols = np.array([0.01, 0.02, 0.015])

        model = CovarianceModel(corr, vols)
        assert model.n_assets == 3

    def test_covariance_property(self) -> None:
        """Test covariance matrix construction."""
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        vols = np.array([0.01, 0.02])

        model = CovarianceModel(corr, vols)
        cov = model.covariance

        # Check: Σ = diag(σ) Ω diag(σ)
        expected = corr * np.outer(vols, vols)
        assert_allclose(cov, expected, atol=1e-15)

    def test_cholesky_decomposition(self) -> None:
        """Test Cholesky decomposition."""
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        vols = np.array([0.01, 0.02])

        model = CovarianceModel(corr, vols)
        L = model.cholesky
        cov = model.covariance

        # Verify: Σ = LL^T
        reconstructed = L @ L.T
        assert_allclose(reconstructed, cov, atol=1e-15)

    def test_validation_fails_on_invalid_correlation(self) -> None:
        """Test validation catches invalid correlation matrices."""
        # Non-unit diagonal
        bad_corr = np.array([[0.9, 0.3], [0.3, 1.0]])
        vols = np.array([0.01, 0.02])

        with pytest.raises(ValueError, match="diagonal must be 1"):
            CovarianceModel(bad_corr, vols)

        # Non-symmetric
        bad_corr = np.array([[1.0, 0.3], [0.4, 1.0]])
        with pytest.raises(ValueError, match="symmetric"):
            CovarianceModel(bad_corr, vols)

    def test_mvn_sampling(self) -> None:
        """Test multivariate normal sampling."""
        corr = np.eye(2)
        vols = np.array([0.01, 0.02])
        model = CovarianceModel(corr, vols)

        mean = np.array([0.001, 0.002])
        samples = model.sample_mvn(mean, n_samples=1000, random_seed=42)

        assert samples.shape == (1000, 2)
        assert_allclose(samples.mean(axis=0), mean, atol=0.002)  # Approx

    def test_conditional_distribution(self) -> None:
        """Test conditional distribution."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        vols = np.array([0.01, 0.02])
        model = CovarianceModel(corr, vols)

        mean = np.array([0.001, 0.002])

        # Condition on X_0 = 0.001
        cond_mean, cond_cov = model.condition_on(
            indices_obs=np.array([0]),
            indices_unobs=np.array([1]),
            values_obs=np.array([0.001]),
            mean=mean
        )

        assert cond_mean.shape == (1,)
        assert cond_cov.shape == (1, 1)
        assert cond_cov[0, 0] > 0  # Positive variance

    def test_mahalanobis_distance(self) -> None:
        """Test Mahalanobis distance."""
        corr = np.eye(2)
        vols = np.array([0.01, 0.02])
        model = CovarianceModel(corr, vols)

        mean = np.array([0.001, 0.002])
        point = mean + np.array([0.01, 0.02])

        dist = model.mahalanobis_distance(point, mean)

        # With identity correlation: dist^2 = (0.01/0.01)^2 + (0.02/0.02)^2 = 2
        expected = np.sqrt(2)
        assert_allclose(dist, expected, atol=1e-10)

    def test_log_det(self) -> None:
        """Test log determinant."""
        corr = np.eye(2)
        vols = np.array([0.01, 0.02])
        model = CovarianceModel(corr, vols)

        log_det = model.log_det()

        # log det(Σ) = log det(diag(σ) diag(σ)) = 2 log(0.01) + 2 log(0.02)
        expected = 2 * np.log(0.01) + 2 * np.log(0.02)
        assert_allclose(log_det, expected, atol=1e-10)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_symmetric_return_model(self) -> None:
        """Test creation of symmetric model."""
        model = create_symmetric_return_model(
            n_regimes=3,
            n_assets=5,
            mean_return=0.0005,
            volatility=0.02,
            correlation=0.3,
            degrees_of_freedom=10.0
        )

        assert model.n_regimes == 3
        assert model.n_assets == 5
        assert_allclose(model.degrees_of_freedom, [10.0, 10.0, 10.0])

    def test_create_identity_covariance(self) -> None:
        """Test identity covariance creation."""
        model = create_identity_covariance(n_assets=4, volatility=0.02)

        assert_allclose(model.covariance, np.eye(4) * 0.02**2, atol=1e-15)

    def test_create_compound_symmetric_covariance(self) -> None:
        """Test compound symmetric covariance."""
        model = create_compound_symmetric_covariance(
            n_assets=3,
            volatility=0.01,
            correlation=0.3
        )

        corr_extracted = model.correlation_matrix
        expected_corr = np.ones((3, 3)) * 0.3
        np.fill_diagonal(expected_corr, 1.0)

        assert_allclose(corr_extracted, expected_corr, atol=1e-15)

    def test_compound_symmetric_invalid_correlation_raises(self) -> None:
        """Test that invalid correlation for compound symmetric raises."""
        with pytest.raises(ValueError, match="not valid"):
            create_compound_symmetric_covariance(
                n_assets=3,
                volatility=0.01,
                correlation=0.9  # Too high for 3 assets
            )


class TestIntegration:
    """Integration tests combining models."""

    def test_full_return_model_workflow(self) -> None:
        """Test complete workflow with return model."""
        # Create model
        model = create_symmetric_return_model(
            n_regimes=2,
            n_assets=3,
            mean_return=0.0005,
            volatility=0.02,
            correlation=0.3,
            degrees_of_freedom=10.0
        )

        # Generate returns
        returns = np.random.normal(0, 0.02, size=(100, 3))

        # Compute likelihoods
        lls = model.log_likelihood_all_regimes(returns)
        assert lls.shape == (100, 2)

        # Sample
        samples = model.sample(regime=0, n_samples=50, random_seed=42)
        assert samples.shape == (50, 3)

    def test_covariance_integration(self) -> None:
        """Test covariance model with sampling."""
        cov_model = create_compound_symmetric_covariance(
            n_assets=4,
            volatility=0.02,
            correlation=0.2
        )

        mean = np.zeros(4)
        samples = cov_model.sample_mvn(mean, n_samples=500, random_seed=42)

        # Check empirical mean and cov
        assert_allclose(samples.mean(axis=0), mean, atol=0.01)
        emp_cov = np.cov(samples.T)
        assert_allclose(
            np.diag(emp_cov),
            np.diag(cov_model.covariance),
            rtol=0.15
        )
