"""
Multivariate Student-t return model.

This module implements regime-conditional multivariate Student-t distributions
for asset returns. Student-t is used instead of Gaussian to capture fat tails
(extreme events more frequent than normal distribution predicts).

Mathematical formulation:
    r_t | s_t = k ~ StudentT_B(ν_k, μ_k, Σ_k)

Where:
    - ν_k > 2: degrees of freedom (tail heaviness)
    - μ_k ∈ R^B: regime-specific mean return
    - Σ_k ∈ R^{B×B}: regime-specific covariance matrix

As ν_k → ∞, StudentT → Gaussian (thin tails).
As ν_k → 2, StudentT → very heavy tails.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import multivariate_t


class StudentTReturnModel:
    """
    Multivariate Student-t return distribution.

    Captures regime-conditional returns with Student-t distribution,
    allowing fat tails and regime-specific risk profiles.

    Attributes
    ----------
    n_assets : int
        Number of assets (B)
    n_regimes : int
        Number of regimes (K)
    means : NDArray[np.float64]
        Regime-specific mean returns, shape (K, B)
    covariances : NDArray[np.float64]
        Regime-specific covariance matrices, shape (K, B, B)
    degrees_of_freedom : NDArray[np.float64]
        Regime-specific degrees of freedom, shape (K,)
    """

    def __init__(
        self,
        means: NDArray[np.float64],
        covariances: NDArray[np.float64],
        degrees_of_freedom: NDArray[np.float64],
        validate: bool = True
    ) -> None:
        """
        Initialize Student-t return model.

        Parameters
        ----------
        means : NDArray[np.float64]
            Regime-specific means, shape (K, B)
        covariances : NDArray[np.float64]
            Regime-specific covariances, shape (K, B, B)
        degrees_of_freedom : NDArray[np.float64]
            Degrees of freedom per regime, shape (K,)
        validate : bool, optional
            If True, validate that covariances are positive definite.

        Raises
        ------
        ValueError
            If shapes don't match or covariances invalid.
        """
        self.means = means.astype(np.float64)
        self.covariances = covariances.astype(np.float64)
        self.degrees_of_freedom = degrees_of_freedom.astype(np.float64)

        self.n_regimes, self.n_assets = self.means.shape

        if validate:
            self._validate_parameters()

    def _validate_parameters(self) -> None:
        """
        Validate model parameters.

        Checks:
        - Shape consistency
        - Positive definite covariances
        - Valid degrees of freedom (> 2)
        """
        if self.means.shape != (self.n_regimes, self.n_assets):
            raise ValueError(
                f"means shape {self.means.shape} doesn't match "
                f"(n_regimes={self.n_regimes}, n_assets={self.n_assets})"
            )

        if self.covariances.shape != (self.n_regimes, self.n_assets, self.n_assets):
            raise ValueError(
                f"covariances shape {self.covariances.shape} doesn't match "
                f"(n_regimes={self.n_regimes}, n_assets={self.n_assets}, n_assets={self.n_assets})"
            )

        if self.degrees_of_freedom.shape != (self.n_regimes,):
            raise ValueError(
                f"degrees_of_freedom shape {self.degrees_of_freedom.shape} "
                f"doesn't match (n_regimes={self.n_regimes},)"
            )

        # Check positive definiteness
        for k in range(self.n_regimes):
            try:
                np.linalg.cholesky(self.covariances[k])
            except np.linalg.LinAlgError:
                raise ValueError(
                    f"Covariance matrix for regime {k} is not positive definite"
                )

        # Check degrees of freedom
        if np.any(self.degrees_of_freedom <= 2):
            raise ValueError(
                f"All degrees of freedom must be > 2. Got {self.degrees_of_freedom}"
            )

    def log_likelihood(
        self,
        returns: NDArray[np.float64],
        regime: int
    ) -> NDArray[np.float64]:
        """
        Compute log-likelihood of returns under Student-t for a regime.

        For returns r_t and regime k:
            log p(r_t | k) = log StudentT(r_t; ν_k, μ_k, Σ_k)

        Parameters
        ----------
        returns : NDArray[np.float64]
            Return observations, shape (T, B)
        regime : int
            Regime index (0 ≤ regime < K)

        Returns
        -------
        NDArray[np.float64]
            Log-likelihood values, shape (T,)
        """
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}"
            )

        # Extract parameters for this regime
        mu = self.means[regime, :]
        Sigma = self.covariances[regime, :, :]
        nu = self.degrees_of_freedom[regime]

        # Compute log-likelihood using scipy.stats
        # Note: multivariate_t expects precision matrix or covariance
        dist = multivariate_t(loc=mu, shape=Sigma, df=nu)
        return dist.logpdf(returns)

    def likelihood(
        self,
        returns: NDArray[np.float64],
        regime: int
    ) -> NDArray[np.float64]:
        """
        Compute likelihood (not log) of returns.

        Parameters
        ----------
        returns : NDArray[np.float64]
            Return observations, shape (T, B)
        regime : int
            Regime index

        Returns
        -------
        NDArray[np.float64]
            Likelihood values, shape (T,)
        """
        return np.exp(self.log_likelihood(returns, regime))

    def log_likelihood_all_regimes(
        self,
        returns: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute log-likelihood under all regimes.

        Parameters
        ----------
        returns : NDArray[np.float64]
            Return observations, shape (T, B)

        Returns
        -------
        NDArray[np.float64]
            Log-likelihoods, shape (T, K)
        """
        T = returns.shape[0]
        log_likelihoods = np.zeros((T, self.n_regimes))

        for k in range(self.n_regimes):
            log_likelihoods[:, k] = self.log_likelihood(returns, k)

        return log_likelihoods

    def sample(
        self,
        regime: int,
        n_samples: int,
        random_seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """
        Sample returns from Student-t for a given regime.

        Parameters
        ----------
        regime : int
            Regime index
        n_samples : int
            Number of samples
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        NDArray[np.float64]
            Sampled returns, shape (n_samples, B)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}"
            )

        mu = self.means[regime, :]
        Sigma = self.covariances[regime, :, :]
        nu = self.degrees_of_freedom[regime]

        # Sample via scale-mixture representation:
        # r ~ N(μ, w*Σ) where w ~ InvGamma(ν/2, ν/2)
        w = np.random.gamma(nu / 2, 2 / nu, size=n_samples)
        samples = np.random.multivariate_normal(
            mean=np.zeros(self.n_assets),
            cov=np.eye(self.n_assets),
            size=n_samples
        )
        return mu + samples * np.sqrt(w)[:, np.newaxis]

    def correlation_matrix(self, regime: int) -> NDArray[np.float64]:
        """
        Extract correlation matrix from covariance for a regime.

        Parameters
        ----------
        regime : int
            Regime index

        Returns
        -------
        NDArray[np.float64]
            Correlation matrix, shape (B, B)
        """
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}"
            )

        Sigma = self.covariances[regime, :, :]
        std = np.sqrt(np.diag(Sigma))
        return Sigma / np.outer(std, std)

    def marginal_volatilities(self, regime: int) -> NDArray[np.float64]:
        """
        Extract marginal volatilities (standard deviations) for a regime.

        Parameters
        ----------
        regime : int
            Regime index

        Returns
        -------
        NDArray[np.float64]
            Marginal volatilities, shape (B,)
        """
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}"
            )

        return np.sqrt(np.diag(self.covariances[regime, :, :]))

    def tail_index(self, regime: int) -> float:
        """
        Get tail index (inverse of ν) for a regime.

        Lower tail index = heavier tails (more extreme events).

        Parameters
        ----------
        regime : int
            Regime index

        Returns
        -------
        float
            Tail index (1/ν)
        """
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}"
            )

        return 1.0 / self.degrees_of_freedom[regime]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StudentTReturnModel("
            f"n_regimes={self.n_regimes}, "
            f"n_assets={self.n_assets})"
        )


def create_symmetric_return_model(
    n_regimes: int,
    n_assets: int,
    mean_return: float = 0.0,
    volatility: float = 0.02,
    correlation: float = 0.3,
    degrees_of_freedom: float = 10.0
) -> StudentTReturnModel:
    """
    Create a symmetric Student-t return model for testing.

    All regimes have identical parameters.

    Parameters
    ----------
    n_regimes : int
        Number of regimes
    n_assets : int
        Number of assets
    mean_return : float
        Mean daily return (e.g., 0.0005 = 0.05%)
    volatility : float
        Daily volatility (e.g., 0.02 = 2%)
    correlation : float
        Correlation between assets (constant across pairs)
    degrees_of_freedom : float
        Degrees of freedom (tail index)

    Returns
    -------
    StudentTReturnModel
        Symmetric return model
    """
    # Create means (same for all regimes)
    means = np.full((n_regimes, n_assets), mean_return)

    # Create correlation matrix (constant correlation)
    corr = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr, 1.0)

    # Create covariance matrices (same for all regimes)
    std = np.full(n_assets, volatility)
    Sigma = corr * np.outer(std, std)

    covariances = np.tile(Sigma, (n_regimes, 1, 1))

    # Degrees of freedom (same for all regimes)
    dfs = np.full(n_regimes, degrees_of_freedom)

    return StudentTReturnModel(means, covariances, dfs, validate=False)
