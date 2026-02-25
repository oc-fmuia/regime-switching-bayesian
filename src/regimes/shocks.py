"""
Shock propagation framework for regime-switching returns model.

This module implements shock dynamics in a regime-switching context, allowing
asset returns to be driven by underlying risk factors (shocks) with regime-dependent
loadings. This enables modeling of factor-driven returns and stress scenarios.

Mathematical formulation:
    u_t ∈ ℝ^M                           # M independent shock factors
    B_k ∈ ℝ^{N × M}                    # Loading matrix for regime k
    r_t = μ_{s_t} + B_{s_t} u_t + ε_t  # Return dynamics with shocks

The shock process u_t can be:
- Deterministic (for stress testing): u_t = [u_1, u_2, ..., u_M]
- Stochastic (for Monte Carlo): u_t ~ N(0, I_M)

The residual ε_t ~ N(0, Σ_{s_t}) captures idiosyncratic risk not explained
by the factors.
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
from numpy.typing import NDArray


class ShockModel:
    """
    Regime-conditional shock propagation model.

    Manages shock loading matrices B_k for each regime, shock scenarios,
    and the composition of returns with shocks and idiosyncratic noise.

    Attributes
    ----------
    n_assets : int
        Number of assets (N)
    n_shocks : int
        Number of shock factors (M)
    n_regimes : int
        Number of regimes (K)
    loading_matrices : NDArray[np.float64]
        Stack of loading matrices of shape (K, N, M).
        loading_matrices[k] = B_k is the loading matrix for regime k.
    """

    def __init__(
        self,
        n_assets: int,
        n_shocks: int,
        n_regimes: int,
        loading_matrices: Optional[NDArray[np.float64]] = None
    ) -> None:
        """
        Initialize shock model.

        Parameters
        ----------
        n_assets : int
            Number of assets (N).
        n_shocks : int
            Number of shock factors (M).
        n_regimes : int
            Number of regimes (K).
        loading_matrices : NDArray[np.float64], optional
            Pre-specified loading matrices of shape (K, N, M).
            If None, initialized to zero (can be set later via set_loading_matrices).

        Raises
        ------
        ValueError
            If dimensions are invalid (non-positive or mismatched).
        """
        if n_assets <= 0 or n_shocks <= 0 or n_regimes <= 0:
            raise ValueError(
                f"All dimensions must be positive. Got n_assets={n_assets}, "
                f"n_shocks={n_shocks}, n_regimes={n_regimes}"
            )

        self.n_assets = n_assets
        self.n_shocks = n_shocks
        self.n_regimes = n_regimes

        if loading_matrices is None:
            # Initialize to zero
            self.loading_matrices = np.zeros(
                (n_regimes, n_assets, n_shocks), dtype=np.float64
            )
        else:
            # Validate and store
            if loading_matrices.shape != (n_regimes, n_assets, n_shocks):
                raise ValueError(
                    f"loading_matrices must have shape ({n_regimes}, {n_assets}, {n_shocks}). "
                    f"Got {loading_matrices.shape}"
                )
            self.loading_matrices = loading_matrices.astype(np.float64)

    def set_loading_matrices(
        self,
        loading_matrices: NDArray[np.float64]
    ) -> None:
        """
        Set or update loading matrices.

        Parameters
        ----------
        loading_matrices : NDArray[np.float64]
            Loading matrices of shape (K, N, M).

        Raises
        ------
        ValueError
            If shape does not match (n_regimes, n_assets, n_shocks).
        """
        if loading_matrices.shape != (self.n_regimes, self.n_assets, self.n_shocks):
            raise ValueError(
                f"Expected shape ({self.n_regimes}, {self.n_assets}, {self.n_shocks}). "
                f"Got {loading_matrices.shape}"
            )
        self.loading_matrices = loading_matrices.astype(np.float64)

    def get_loading_matrix(self, regime: int) -> NDArray[np.float64]:
        """
        Get loading matrix for a specific regime.

        Parameters
        ----------
        regime : int
            Regime index (0-indexed).

        Returns
        -------
        NDArray[np.float64]
            Loading matrix B_k of shape (N, M).

        Raises
        ------
        ValueError
            If regime index is out of bounds.
        """
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}. Got {regime}"
            )
        return self.loading_matrices[regime, :, :].copy()

    def compute_shock_impact(
        self,
        shocks: NDArray[np.float64],
        regime: int
    ) -> NDArray[np.float64]:
        """
        Compute shock impact on asset returns for a given regime.

        Parameters
        ----------
        shocks : NDArray[np.float64]
            Shock vector of shape (M,).
        regime : int
            Regime index (0-indexed).

        Returns
        -------
        NDArray[np.float64]
            Shock-driven return contribution B_k u of shape (N,).

        Raises
        ------
        ValueError
            If shock dimension or regime is invalid.
        """
        if shocks.shape != (self.n_shocks,):
            raise ValueError(
                f"shocks must have shape ({self.n_shocks},). Got {shocks.shape}"
            )
        if not (0 <= regime < self.n_regimes):
            raise ValueError(
                f"regime must be in {{0, …, {self.n_regimes - 1}}}. Got {regime}"
            )

        B_k = self.loading_matrices[regime, :, :]
        return B_k @ shocks

    def simulate_shocks(
        self,
        n_steps: int,
        shock_std: Optional[NDArray[np.float64]] = None,
        random_seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """
        Simulate stochastic shock process.

        Each shock u_{m,t} is simulated as independent Normal(0, σ_m^2).

        Parameters
        ----------
        n_steps : int
            Number of time steps.
        shock_std : NDArray[np.float64], optional
            Standard deviation of each shock factor, shape (M,).
            If None, assume σ_m = 1 for all m (standardized).
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        NDArray[np.float64]
            Shock process of shape (n_steps, M).
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if shock_std is None:
            shock_std = np.ones(self.n_shocks)
        else:
            if shock_std.shape != (self.n_shocks,):
                raise ValueError(
                    f"shock_std must have shape ({self.n_shocks},). Got {shock_std.shape}"
                )

        # Sample independent standard normals
        shocks = np.random.standard_normal((n_steps, self.n_shocks))

        # Scale by shock_std
        shocks *= shock_std[np.newaxis, :]

        return shocks

    def stress_test(
        self,
        shock_scenario: NDArray[np.float64],
        regime: int,
        mean_return: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Apply deterministic shock scenario (stress test).

        Computes return impact: r = mean_return + B_k @ shock_scenario

        Parameters
        ----------
        shock_scenario : NDArray[np.float64]
            Shock vector of shape (M,).
        regime : int
            Regime index.
        mean_return : NDArray[np.float64], optional
            Mean return μ_k of shape (N,). If None, assume zero.

        Returns
        -------
        NDArray[np.float64]
            Total return (mean + shock impact) of shape (N,).
        """
        if mean_return is None:
            mean_return = np.zeros(self.n_assets)
        else:
            if mean_return.shape != (self.n_assets,):
                raise ValueError(
                    f"mean_return must have shape ({self.n_assets},). Got {mean_return.shape}"
                )

        shock_impact = self.compute_shock_impact(shock_scenario, regime)
        return mean_return + shock_impact

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ShockModel(n_assets={self.n_assets}, n_shocks={self.n_shocks}, "
            f"n_regimes={self.n_regimes})"
        )


class ReturnWithShocks:
    """
    Regime-switching return model with shock propagation.

    Composes regime-conditional means, shock loadings, and idiosyncratic noise
    to generate asset returns:

        r_t = μ_{s_t} + B_{s_t} u_t + ε_t

    where:
    - μ_{s_t} is regime-conditional mean
    - B_{s_t} u_t is shock-driven component
    - ε_t ~ N(0, Σ_{s_t}) is idiosyncratic risk

    Attributes
    ----------
    shock_model : ShockModel
        Regime-conditional shock loading matrices.
    regime_means : NDArray[np.float64]
        Mean returns μ_k for each regime, shape (K, N).
    regime_covs : NDArray[np.float64]
        Covariance matrices Σ_k for each regime, shape (K, N, N).
    """

    def __init__(
        self,
        shock_model: ShockModel,
        regime_means: NDArray[np.float64],
        regime_covs: NDArray[np.float64]
    ) -> None:
        """
        Initialize return model with shocks.

        Parameters
        ----------
        shock_model : ShockModel
            Shock propagation model.
        regime_means : NDArray[np.float64]
            Regime-conditional means of shape (K, N).
        regime_covs : NDArray[np.float64]
            Regime-conditional covariances of shape (K, N, N).

        Raises
        ------
        ValueError
            If dimensions are inconsistent or covariances are invalid.
        """
        self.shock_model = shock_model
        n_regimes = shock_model.n_regimes
        n_assets = shock_model.n_assets

        # Validate regime_means
        if regime_means.shape != (n_regimes, n_assets):
            raise ValueError(
                f"regime_means must have shape ({n_regimes}, {n_assets}). "
                f"Got {regime_means.shape}"
            )

        # Validate regime_covs
        if regime_covs.shape != (n_regimes, n_assets, n_assets):
            raise ValueError(
                f"regime_covs must have shape ({n_regimes}, {n_assets}, {n_assets}). "
                f"Got {regime_covs.shape}"
            )

        # Check symmetry and positive-definiteness of covariances
        for k in range(n_regimes):
            cov_k = regime_covs[k, :, :]
            if not np.allclose(cov_k, cov_k.T, atol=1e-10):
                raise ValueError(f"Covariance for regime {k} is not symmetric")

            # Check positive-definite via Cholesky
            try:
                np.linalg.cholesky(cov_k)
            except np.linalg.LinAlgError:
                raise ValueError(f"Covariance for regime {k} is not positive-definite")

        self.regime_means = regime_means.astype(np.float64)
        self.regime_covs = regime_covs.astype(np.float64)

    def generate_returns(
        self,
        regime_path: NDArray[np.int64],
        shocks: NDArray[np.float64],
        idiosyncratic_noise: Optional[NDArray[np.float64]] = None,
        random_seed: Optional[int] = None
    ) -> NDArray[np.float64]:
        """
        Generate asset returns given regime path, shocks, and noise.

        Parameters
        ----------
        regime_path : NDArray[np.int64]
            Regime sequence of shape (T,) with values in {0, …, K-1}.
        shocks : NDArray[np.float64]
            Shock process of shape (T, M).
        idiosyncratic_noise : NDArray[np.float64], optional
            Pre-sampled noise ε_t of shape (T, N).
            If None, sampled internally using random_seed.
        random_seed : int, optional
            Random seed for sampling idiosyncratic noise.

        Returns
        -------
        NDArray[np.float64]
            Returns of shape (T, N).

        Raises
        ------
        ValueError
            If dimensions are inconsistent.
        """
        n_steps = len(regime_path)
        n_assets = self.shock_model.n_assets

        # Validate inputs
        if shocks.shape != (n_steps, self.shock_model.n_shocks):
            raise ValueError(
                f"shocks must have shape ({n_steps}, {self.shock_model.n_shocks}). "
                f"Got {shocks.shape}"
            )

        # Sample idiosyncratic noise if not provided
        if idiosyncratic_noise is None:
            if random_seed is not None:
                np.random.seed(random_seed)

            idiosyncratic_noise = np.zeros((n_steps, n_assets))
            for k in range(self.shock_model.n_regimes):
                cov_k = self.regime_covs[k, :, :]
                # Cholesky decomposition
                L_k = np.linalg.cholesky(cov_k)
                # Indices where regime = k
                idx_k = np.where(regime_path == k)[0]
                # Sample for this regime
                z = np.random.standard_normal((len(idx_k), n_assets))
                idiosyncratic_noise[idx_k, :] = z @ L_k.T
        else:
            if idiosyncratic_noise.shape != (n_steps, n_assets):
                raise ValueError(
                    f"idiosyncratic_noise must have shape ({n_steps}, {n_assets}). "
                    f"Got {idiosyncratic_noise.shape}"
                )

        # Construct returns
        returns = np.zeros((n_steps, n_assets))

        for t in range(n_steps):
            regime_t = regime_path[t]

            # Mean return
            mu_t = self.regime_means[regime_t, :]

            # Shock impact
            u_t = shocks[t, :]
            shock_impact = self.shock_model.compute_shock_impact(u_t, regime_t)

            # Idiosyncratic noise
            eps_t = idiosyncratic_noise[t, :]

            # Total return
            returns[t, :] = mu_t + shock_impact + eps_t

        return returns

    def compute_factor_exposure(
        self,
        regime: int
    ) -> NDArray[np.float64]:
        """
        Get factor exposure (loading matrix) for a regime.

        Parameters
        ----------
        regime : int
            Regime index.

        Returns
        -------
        NDArray[np.float64]
            Loading matrix B_k of shape (N, M).
        """
        return self.shock_model.get_loading_matrix(regime)

    def compute_systematic_variance(
        self,
        regime: int,
        shock_cov: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Compute systematic (shock-driven) variance for a regime.

        Systematic variance is: Σ_systematic = B_k Cov(u) B_k^T

        Parameters
        ----------
        regime : int
            Regime index.
        shock_cov : NDArray[np.float64], optional
            Shock covariance matrix of shape (M, M).
            If None, assume identity (standardized shocks).

        Returns
        -------
        NDArray[np.float64]
            Systematic covariance of shape (N, N).
        """
        if shock_cov is None:
            shock_cov = np.eye(self.shock_model.n_shocks)
        else:
            if shock_cov.shape != (self.shock_model.n_shocks, self.shock_model.n_shocks):
                raise ValueError(
                    f"shock_cov must be ({self.shock_model.n_shocks}, "
                    f"{self.shock_model.n_shocks}). Got {shock_cov.shape}"
                )

        B_k = self.shock_model.get_loading_matrix(regime)
        return B_k @ shock_cov @ B_k.T

    def compute_total_variance(
        self,
        regime: int,
        shock_cov: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        """
        Compute total variance (systematic + idiosyncratic) for a regime.

        Total variance: Σ_total = B_k Cov(u) B_k^T + Σ_idiosyncratic

        Parameters
        ----------
        regime : int
            Regime index.
        shock_cov : NDArray[np.float64], optional
            Shock covariance matrix. If None, assume identity.

        Returns
        -------
        NDArray[np.float64]
            Total covariance of shape (N, N).
        """
        systematic = self.compute_systematic_variance(regime, shock_cov)
        idiosyncratic = self.regime_covs[regime, :, :]
        return systematic + idiosyncratic

    def variance_decomposition(
        self,
        regime: int,
        shock_cov: Optional[NDArray[np.float64]] = None
    ) -> Dict[str, float]:
        """
        Decompose variance into systematic and idiosyncratic components.

        Parameters
        ----------
        regime : int
            Regime index.
        shock_cov : NDArray[np.float64], optional
            Shock covariance. If None, assume identity.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys:
            - 'systematic_variance': Total systematic variance (trace of systematic cov)
            - 'idiosyncratic_variance': Total idiosyncratic variance (trace of residual cov)
            - 'systematic_ratio': Fraction of variance explained by shocks
        """
        systematic = self.compute_systematic_variance(regime, shock_cov)
        idiosyncratic = self.regime_covs[regime, :, :]
        total = systematic + idiosyncratic

        sys_var = np.trace(systematic)
        idio_var = np.trace(idiosyncratic)
        total_var = np.trace(total)

        systematic_ratio = sys_var / (total_var + 1e-12)

        return {
            'systematic_variance': float(sys_var),
            'idiosyncratic_variance': float(idio_var),
            'systematic_ratio': float(systematic_ratio)
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ReturnWithShocks(n_assets={self.shock_model.n_assets}, "
            f"n_shocks={self.shock_model.n_shocks}, "
            f"n_regimes={self.shock_model.n_regimes})"
        )
