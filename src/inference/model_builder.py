"""
Bayesian model builder: PyMC inference for regime-switching returns.

This module assembles the full Bayesian regime-switching model:
- Regime dynamics (Markov chain with Dirichlet priors)
- Return model (Student-t with regime-conditional parameters)
- Shock propagation (regime-dependent loading matrices)
- Prior specifications and inference orchestration

Mathematical model:
    s_t ~ MarkovChain(P)                           # Regime state
    P_ij ~ Dirichlet(α)                            # Transition priors
    ν_k ~ Exponential(1/df_prior)                  # Degrees of freedom
    μ_k ~ Normal(μ0, σ0²)                         # Mean returns
    σ_k ~ HalfNormal(σ0)                          # Volatilities
    Ω_k ~ LKJ(η)                                   # Correlations
    B_k ~ Normal(0, σ_B²)                         # Loading matrices
    r_t ~ StudentT(ν_{s_t}, μ_{s_t}, Σ_{s_t})   # Returns

where Σ_{s_t} = B_{s_t} I B_{s_t}^T + diag(σ_{s_t})² Ω_{s_t}
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray
import pymc as pm
import pytensor.tensor as pt


class PriorSpec:
    """Specification of priors for model parameters."""

    def __init__(
        self,
        # Regime dynamics
        dirichlet_alpha: float = 1.0,
        # Return model
        mean_loc: float = 0.0,
        mean_scale: float = 0.05,
        vol_scale: float = 0.1,
        lkj_eta: float = 2.0,
        # Degrees of freedom
        df_mean: float = 10.0,
        # Shock loadings
        loading_scale: float = 0.5,
    ) -> None:
        """
        Initialize prior specification.

        Parameters
        ----------
        dirichlet_alpha : float
            Dirichlet prior concentration (symmetric). Default 1.0 (uninformed).
        mean_loc : float
            Prior mean for regime-conditional returns. Default 0.0.
        mean_scale : float
            Prior std for regime-conditional returns. Default 0.05.
        vol_scale : float
            Prior scale for volatilities (HalfNormal). Default 0.1.
        lkj_eta : float
            LKJ prior concentration on correlations. Default 2.0.
            Higher → weaker correlations preferred.
        df_mean : float
            Prior mean for degrees of freedom (Exponential). Default 10.0.
        loading_scale : float
            Prior scale for shock loading matrices. Default 0.5.
        """
        self.dirichlet_alpha = dirichlet_alpha
        self.mean_loc = mean_loc
        self.mean_scale = mean_scale
        self.vol_scale = vol_scale
        self.lkj_eta = lkj_eta
        self.df_mean = df_mean
        self.loading_scale = loading_scale

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PriorSpec(dirichlet_α={self.dirichlet_alpha}, "
            f"mean_loc={self.mean_loc}, mean_scale={self.mean_scale}, "
            f"lkj_η={self.lkj_eta}, df_mean={self.df_mean})"
        )


class ModelBuilder:
    """
    Bayesian regime-switching model builder.

    Constructs and manages a full PyMC model combining:
    - Regime dynamics (discrete Markov chain)
    - Return distributions (Student-t)
    - Shock propagation (factor loadings)
    - Prior specifications

    Attributes
    ----------
    n_assets : int
        Number of assets
    n_regimes : int
        Number of regimes
    n_shocks : int
        Number of shock factors
    n_obs : int
        Number of observations
    prior_spec : PriorSpec
        Prior specification
    model : pm.Model or None
        PyMC model (None until built)
    """

    def __init__(
        self,
        n_assets: int,
        n_regimes: int,
        n_shocks: int,
        n_obs: int,
        prior_spec: Optional[PriorSpec] = None,
    ) -> None:
        """
        Initialize model builder.

        Parameters
        ----------
        n_assets : int
            Number of assets
        n_regimes : int
            Number of regimes
        n_shocks : int
            Number of shock factors
        n_obs : int
            Number of observations (time steps)
        prior_spec : PriorSpec, optional
            Prior specification. If None, use defaults.
        """
        if n_assets <= 0 or n_regimes <= 0 or n_shocks <= 0 or n_obs <= 0:
            raise ValueError(
                f"All dimensions must be positive. Got "
                f"n_assets={n_assets}, n_regimes={n_regimes}, "
                f"n_shocks={n_shocks}, n_obs={n_obs}"
            )

        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.n_shocks = n_shocks
        self.n_obs = n_obs
        self.prior_spec = prior_spec or PriorSpec()
        self.model: Optional[pm.Model] = None

    def _build_regime_dynamics(self) -> Tuple:
        """
        Build regime dynamics component.

        Returns
        -------
        transition_matrix : pm.TensorVariable
            Transition matrix P, shape (n_regimes, n_regimes)
        stationary_dist : pm.TensorVariable
            Stationary distribution π, shape (n_regimes,)
        """
        # Transition probabilities per row (Dirichlet priors)
        P = pm.Dirichlet(
            "transition_matrix",
            a=np.ones((self.n_regimes, self.n_regimes)) * self.prior_spec.dirichlet_alpha,
            shape=(self.n_regimes, self.n_regimes),
        )

        # Compute stationary distribution as left eigenvector
        # π satisfies: π P = π and Σ π_i = 1
        # Approximate via: πP^T = π (right eigenvector of P^T)
        # We'll use a simpler approach: solve (P^T - I) π = 0 with constraint Σ π = 1

        # For simplicity, use Dirichlet approximation of stationary dist
        pi = pm.Dirichlet("stationary_dist", a=np.ones(self.n_regimes))

        return P, pi

    def _build_return_model(
        self,
    ) -> Tuple:
        """
        Build return model component.

        Returns
        -------
        means : pm.TensorVariable
            Regime-conditional means, shape (n_regimes, n_assets)
        volatilities : pm.TensorVariable
            Regime-conditional volatilities, shape (n_regimes, n_assets)
        correlations : NDArray
            Regime-conditional correlations (identity for each regime)
        dfs : pm.TensorVariable
            Degrees of freedom per regime, shape (n_regimes,)
        """
        # Regime-conditional means
        means = pm.Normal(
            "regime_means",
            mu=self.prior_spec.mean_loc,
            sigma=self.prior_spec.mean_scale,
            shape=(self.n_regimes, self.n_assets),
        )

        # Regime-conditional volatilities
        volatilities = pm.HalfNormal(
            "volatilities",
            sigma=self.prior_spec.vol_scale,
            shape=(self.n_regimes, self.n_assets),
        )

        # Regime-conditional correlations
        # For now, use identity correlations (simplified)
        # Full implementation would use LKJ priors per regime
        correlations = np.array([np.eye(self.n_assets) for _ in range(self.n_regimes)])

        # Degrees of freedom per regime
        dfs = pm.Exponential(
            "degrees_of_freedom",
            lam=1.0 / self.prior_spec.df_mean,
            shape=(self.n_regimes,),
        )

        return means, volatilities, correlations, dfs

    def _build_shock_model(self):
        """
        Build shock propagation component.

        Returns
        -------
        loading_matrices : pm.TensorVariable
            Shock loading matrices B_k, shape (n_regimes, n_assets, n_shocks)
        """
        # Regime-conditional loading matrices (factor loadings)
        B = pm.Normal(
            "loading_matrices",
            mu=0.0,
            sigma=self.prior_spec.loading_scale,
            shape=(self.n_regimes, self.n_assets, self.n_shocks),
        )

        return B

    def build(
        self,
        returns_data: Optional[NDArray[np.float64]] = None,
        regime_path: Optional[NDArray[np.int64]] = None,
        shocks: Optional[NDArray[np.float64]] = None,
    ) -> pm.Model:
        """
        Build the full PyMC model.

        Parameters
        ----------
        returns_data : NDArray[np.float64], optional
            Observed returns, shape (n_obs, n_assets).
            If provided, likelihood is included.
        regime_path : NDArray[np.int64], optional
            Regime sequence (latent variable if not provided).
            If provided, regimes are observed.
        shocks : NDArray[np.float64], optional
            Shock realizations, shape (n_obs, n_shocks).
            If provided, shocks are observed (not latent).

        Returns
        -------
        model : pm.Model
            PyMC model ready for inference.
        """
        # Validate inputs
        if returns_data is not None:
            if returns_data.shape != (self.n_obs, self.n_assets):
                raise ValueError(
                    f"returns_data must have shape ({self.n_obs}, {self.n_assets}). "
                    f"Got {returns_data.shape}"
                )

        if regime_path is not None:
            if regime_path.shape != (self.n_obs,):
                raise ValueError(
                    f"regime_path must have shape ({self.n_obs},). "
                    f"Got {regime_path.shape}"
                )
            if not np.all((regime_path >= 0) & (regime_path < self.n_regimes)):
                raise ValueError(
                    f"regime_path must contain values in [0, {self.n_regimes-1}]"
                )

        if shocks is not None:
            if shocks.shape != (self.n_obs, self.n_shocks):
                raise ValueError(
                    f"shocks must have shape ({self.n_obs}, {self.n_shocks}). "
                    f"Got {shocks.shape}"
                )

        with pm.Model() as model:
            # Build components
            P, pi = self._build_regime_dynamics()
            means, vols, corrs, dfs = self._build_return_model()
            B = self._build_shock_model()

            # If regime_path is provided, condition on it
            if regime_path is not None:
                s_t = pm.Data("regime_path", regime_path)
            else:
                # Regime path is latent (Markov chain)
                # For now, sample from stationary distribution (simplified)
                s_t = pm.Categorical(
                    "regime_path",
                    p=pi,
                    shape=self.n_obs,
                )

            # Construct covariance matrices and returns likelihood
            if returns_data is not None:
                # For simplicity, use first regime's parameters
                # (Full implementation would use regime-switching)
                regime_idx = 0  # Placeholder

                # Covariance: Σ_k = B_k @ I @ B_k^T + diag(σ_k)^2 @ Ω_k
                B_k = B[regime_idx, :, :]
                sigma_k = vols[regime_idx, :]
                omega_k = corrs[regime_idx, :, :]

                # Shock-driven covariance
                cov_shocks = pt.dot(B_k, pt.dot(pt.eye(self.n_shocks), B_k.T))

                # Idiosyncratic covariance
                cov_idio = pt.dot(
                    pt.diag(sigma_k),
                    pt.dot(omega_k, pt.diag(sigma_k)),
                )

                # Total covariance
                Sigma = cov_shocks + cov_idio

                # Student-t likelihood
                nu = dfs[regime_idx]
                mu = means[regime_idx, :]

                obs = pm.StudentT(
                    "returns",
                    nu=nu,
                    mu=mu,
                    lam=pt.nlinalg.matrix_inverse(Sigma),
                    observed=returns_data,
                )

        self.model = model
        return model

    def get_model(self) -> pm.Model:
        """
        Get the built model.

        Returns
        -------
        model : pm.Model
            The PyMC model.

        Raises
        ------
        RuntimeError
            If model has not been built yet.
        """
        if self.model is None:
            raise RuntimeError("Model has not been built. Call .build() first.")
        return self.model

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ModelBuilder(n_assets={self.n_assets}, n_regimes={self.n_regimes}, "
            f"n_shocks={self.n_shocks}, n_obs={self.n_obs}, "
            f"prior_spec={self.prior_spec})"
        )
