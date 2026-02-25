"""
Covariance matrix handling with LKJ priors.

This module provides utilities for working with covariance matrices in the
context of Bayesian inference. Uses the LKJ (Lewandowski-Kurowicka-Joe) prior
on correlation matrices, which is the standard in Bayesian modeling.

Mathematical background:
    Covariance matrix Σ can be decomposed as:
        Σ = diag(σ) Ω diag(σ)

    where:
        - σ ∈ R^B: marginal standard deviations (volatilities)
        - Ω ∈ R^{B×B}: correlation matrix (unit diagonal, symmetric)

    LKJ prior on Ω:
        Ω ~ LKJ(η)

    where η is a concentration parameter:
        - η = 1: uniform (uninformed)
        - η > 1: concentrates near identity (weak correlations)
        - η < 1: concentrates at extremes (strong correlations)

Benefits:
    - Ensures Ω is valid (positive definite, unit diagonal)
    - More efficient than placing priors directly on Σ
    - Separates correlation uncertainty from volatility uncertainty
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky, solve_triangular


class CovarianceModel:
    """
    Covariance matrix management for multivariate models.

    Handles conversion between:
    - Full covariance matrices Σ
    - Correlation + volatility decomposition (Ω, σ)
    - Cholesky decomposition (L where Σ = LL^T)
    """

    def __init__(
        self,
        correlation_matrix: NDArray[np.float64],
        volatilities: NDArray[np.float64],
        validate: bool = True
    ) -> None:
        """
        Initialize covariance model.

        Parameters
        ----------
        correlation_matrix : NDArray[np.float64]
            Correlation matrix Ω, shape (B, B)
        volatilities : NDArray[np.float64]
            Marginal volatilities σ, shape (B,)
        validate : bool, optional
            If True, validate that correlation matrix is valid.

        Raises
        ------
        ValueError
            If correlation matrix or volatilities invalid.
        """
        self.correlation_matrix = correlation_matrix.astype(np.float64)
        self.volatilities = volatilities.astype(np.float64)

        self.n_assets = len(self.volatilities)

        if validate:
            self._validate_parameters()

        # Compute covariance and Cholesky decomposition
        self._covariance: Optional[NDArray[np.float64]] = None
        self._cholesky: Optional[NDArray[np.float64]] = None

    def _validate_parameters(self) -> None:
        """
        Validate correlation matrix and volatilities.

        Checks:
        - Correlation matrix is square (B, B)
        - Diagonal is 1
        - Symmetric
        - Positive definite
        - Volatilities are positive
        """
        if self.correlation_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Correlation matrix shape {self.correlation_matrix.shape} "
                f"doesn't match (n_assets={self.n_assets}, n_assets={self.n_assets})"
            )

        # Check diagonal is 1
        diag = np.diag(self.correlation_matrix)
        if not np.allclose(diag, 1.0, atol=1e-10):
            raise ValueError(
                f"Correlation matrix diagonal must be 1. Got {diag}"
            )

        # Check symmetric
        if not np.allclose(
            self.correlation_matrix,
            self.correlation_matrix.T,
            atol=1e-10
        ):
            raise ValueError("Correlation matrix must be symmetric")

        # Check positive definite
        try:
            np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix must be positive definite")

        # Check volatilities positive
        if np.any(self.volatilities <= 0):
            raise ValueError(
                f"All volatilities must be positive. Got {self.volatilities}"
            )

    @property
    def covariance(self) -> NDArray[np.float64]:
        """
        Get covariance matrix (computed lazily).

        Returns
        -------
        NDArray[np.float64]
            Covariance matrix Σ, shape (B, B)
        """
        if self._covariance is None:
            # Σ = diag(σ) Ω diag(σ)
            self._covariance = (
                self.correlation_matrix
                * np.outer(self.volatilities, self.volatilities)
            )
        return self._covariance

    @property
    def cholesky(self) -> NDArray[np.float64]:
        """
        Get Cholesky decomposition (computed lazily).

        Returns Σ = LL^T, where L is lower triangular.

        Returns
        -------
        NDArray[np.float64]
            Cholesky factor L, shape (B, B)
        """
        if self._cholesky is None:
            self._cholesky = np.linalg.cholesky(self.covariance)
        return self._cholesky

    def sample_mvn(
        self,
        mean: NDArray[np.float64],
        n_samples: int,
        random_seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """
        Sample from multivariate normal with this covariance.

        Parameters
        ----------
        mean : NDArray[np.float64]
            Mean vector, shape (B,)
        n_samples : int
            Number of samples
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        NDArray[np.float64]
            Samples, shape (n_samples, B)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Sample standard normal
        z = np.random.normal(0, 1, size=(n_samples, self.n_assets))

        # Transform via Cholesky: x = μ + L z
        L = self.cholesky
        return mean + z @ L.T

    def condition_on(
        self,
        indices_obs: NDArray[np.int64],
        indices_unobs: NDArray[np.int64],
        values_obs: NDArray[np.float64],
        mean: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Conditional distribution of unobserved given observed variables.

        For a partition:
            [X_obs]     [μ_o]       [Σ_oo  Σ_ou]
            [X_u]  ~ N( [μ_u],     [Σ_uo  Σ_uu] )

        Conditional:
            X_u | X_obs ~ N(μ_u|o, Σ_u|o)

        where:
            μ_u|o = μ_u + Σ_uo Σ_oo^{-1} (X_obs - μ_o)
            Σ_u|o = Σ_uu - Σ_uo Σ_oo^{-1} Σ_ou

        Parameters
        ----------
        indices_obs : NDArray[np.int64]
            Indices of observed variables
        indices_unobs : NDArray[np.int64]
            Indices of unobserved variables
        values_obs : NDArray[np.float64]
            Observed values, shape (len(indices_obs),)
        mean : NDArray[np.float64]
            Prior mean, shape (B,)

        Returns
        -------
        conditional_mean : NDArray[np.float64]
            Conditional mean, shape (len(indices_unobs),)
        conditional_cov : NDArray[np.float64]
            Conditional covariance, shape (len(indices_unobs), len(indices_unobs))
        """
        Sigma = self.covariance

        Sigma_oo = Sigma[np.ix_(indices_obs, indices_obs)]
        Sigma_ou = Sigma[np.ix_(indices_obs, indices_unobs)]
        Sigma_uo = Sigma[np.ix_(indices_unobs, indices_obs)]
        Sigma_uu = Sigma[np.ix_(indices_unobs, indices_unobs)]

        # Compute conditional mean
        # μ_u|o = μ_u + Σ_uo Σ_oo^{-1} (X_obs - μ_o)
        mu_o = mean[indices_obs]
        mu_u = mean[indices_unobs]
        innovation = values_obs - mu_o

        # Use Cholesky of Sigma_oo for numerical stability
        L_oo = np.linalg.cholesky(Sigma_oo)
        alpha = solve_triangular(L_oo, innovation, lower=True)
        beta = solve_triangular(L_oo.T, alpha, lower=False)

        conditional_mean = mu_u + Sigma_uo @ beta

        # Compute conditional covariance
        # Σ_u|o = Σ_uu - Σ_uo Σ_oo^{-1} Σ_ou
        gamma = solve_triangular(L_oo, Sigma_ou.T, lower=True)
        conditional_cov = Sigma_uu - gamma.T @ gamma

        return conditional_mean, conditional_cov

    def mahalanobis_distance(
        self,
        x: NDArray[np.float64],
        mean: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Mahalanobis distance from x to mean under this covariance.

        d(x, μ) = sqrt((x - μ)^T Σ^{-1} (x - μ))

        Parameters
        ----------
        x : NDArray[np.float64]
            Point(s), shape (..., B) or (B,)
        mean : NDArray[np.float64]
            Center, shape (B,)

        Returns
        -------
        NDArray[np.float64]
            Mahalanobis distances
        """
        diff = x - mean
        Sigma = self.covariance
        Sigma_inv = np.linalg.inv(Sigma)

        if diff.ndim == 1:
            return np.sqrt(diff @ Sigma_inv @ diff)
        else:
            # Handle batch
            return np.sqrt(np.sum(diff @ Sigma_inv * diff, axis=-1))

    def log_det(self) -> float:
        """
        Log determinant of covariance matrix.

        Uses Cholesky decomposition for numerical stability:
            log det(Σ) = 2 * sum(log(diag(L)))

        Returns
        -------
        float
            Log determinant
        """
        return 2.0 * np.sum(np.log(np.diag(self.cholesky)))

    def __repr__(self) -> str:
        """String representation."""
        return f"CovarianceModel(n_assets={self.n_assets})"


def create_identity_covariance(
    n_assets: int,
    volatility: float = 1.0
) -> CovarianceModel:
    """
    Create covariance model with identity correlation and fixed volatility.

    Useful for testing and baseline models.

    Parameters
    ----------
    n_assets : int
        Number of assets
    volatility : float
        Marginal volatility for all assets

    Returns
    -------
    CovarianceModel
        Identity correlation, fixed volatility
    """
    corr = np.eye(n_assets)
    vols = np.full(n_assets, volatility)
    return CovarianceModel(corr, vols, validate=False)


def create_compound_symmetric_covariance(
    n_assets: int,
    volatility: float = 1.0,
    correlation: float = 0.5
) -> CovarianceModel:
    """
    Create covariance model with compound symmetric correlation.

    All pairwise correlations are equal (equicorrelated).

    Parameters
    ----------
    n_assets : int
        Number of assets
    volatility : float
        Marginal volatility for all assets
    correlation : float
        Common pairwise correlation (in (-1/(B-1), 1))

    Returns
    -------
    CovarianceModel
        Compound symmetric correlation structure
    """
    # Build equicorrelated matrix
    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)

    # Validate positive definiteness
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Correlation {correlation} not valid for {n_assets} assets. "
            f"Must be in (-{1/(n_assets-1):.3f}, 1)"
        )

    vols = np.full(n_assets, volatility)
    return CovarianceModel(corr, vols, validate=False)
