"""
Monte Carlo simulator for regime-switching models.

Generates forward-looking scenarios from posterior samples of regime-switching
models, enabling risk analysis, portfolio optimization, and scenario analysis.

Key components:
- Path generation (regime switches + returns + shocks)
- Portfolio analytics (returns, volatility, VaR, CVaR, Sharpe ratio)
- Scenario analysis (shock combinations, stress testing)
- Path statistics and aggregation
"""

from typing import Dict, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class MonteCarloSimulator:
    """
    Monte Carlo simulator for regime-switching multi-asset returns.

    Generates scenario paths given:
    - Posterior samples (from NUTS inference)
    - Markov regime dynamics
    - Return model (means, covariances)
    - Shock specifications

    Attributes
    ----------
    n_assets : int
        Number of assets
    n_regimes : int
        Number of regimes
    n_shocks : int
        Number of shock factors
    n_scenarios : int
        Number of Monte Carlo scenarios to generate
    """

    def __init__(
        self,
        n_assets: int,
        n_regimes: int,
        n_shocks: int,
        n_scenarios: int = 1000,
    ) -> None:
        """
        Initialize Monte Carlo simulator.

        Parameters
        ----------
        n_assets : int
            Number of assets
        n_regimes : int
            Number of regimes
        n_shocks : int
            Number of shock factors
        n_scenarios : int
            Number of Monte Carlo paths to generate. Default 1000.
        """
        if n_assets <= 0 or n_regimes <= 0 or n_shocks <= 0 or n_scenarios <= 0:
            raise ValueError(
                f"All dimensions must be positive. Got "
                f"n_assets={n_assets}, n_regimes={n_regimes}, "
                f"n_shocks={n_shocks}, n_scenarios={n_scenarios}"
            )

        self.n_assets = n_assets
        self.n_regimes = n_regimes
        self.n_shocks = n_shocks
        self.n_scenarios = n_scenarios

    def generate_paths(
        self,
        n_steps: int,
        transition_matrix: NDArray[np.float64],
        regime_means: NDArray[np.float64],
        regime_covs: NDArray[np.float64],
        loading_matrices: Optional[NDArray[np.float64]] = None,
        initial_regime: int = 0,
        random_seed: Optional[int] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
        """
        Generate Monte Carlo paths of regime-switching returns.

        Parameters
        ----------
        n_steps : int
            Number of time steps per path
        transition_matrix : NDArray[np.float64]
            Regime transition matrix, shape (n_regimes, n_regimes)
        regime_means : NDArray[np.float64]
            Regime-conditional means, shape (n_regimes, n_assets)
        regime_covs : NDArray[np.float64]
            Regime-conditional covariances, shape (n_regimes, n_assets, n_assets)
        loading_matrices : NDArray[np.float64], optional
            Shock loadings, shape (n_regimes, n_assets, n_shocks).
            If None, shocks are ignored (deterministic means + idiosyncratic noise only).
        initial_regime : int
            Starting regime. Default 0.
        random_seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        paths : NDArray[np.float64]
            Return paths, shape (n_scenarios, n_steps, n_assets)
        regime_paths : NDArray[np.int64]
            Regime sequences, shape (n_scenarios, n_steps)
        shock_paths : NDArray[np.float64]
            Shock realizations, shape (n_scenarios, n_steps, n_shocks)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Validate inputs
        if transition_matrix.shape != (self.n_regimes, self.n_regimes):
            raise ValueError(
                f"transition_matrix must have shape ({self.n_regimes}, {self.n_regimes}). "
                f"Got {transition_matrix.shape}"
            )

        if regime_means.shape != (self.n_regimes, self.n_assets):
            raise ValueError(
                f"regime_means must have shape ({self.n_regimes}, {self.n_assets}). "
                f"Got {regime_means.shape}"
            )

        if regime_covs.shape != (self.n_regimes, self.n_assets, self.n_assets):
            raise ValueError(
                f"regime_covs must have shape ({self.n_regimes}, {self.n_assets}, {self.n_assets}). "
                f"Got {regime_covs.shape}"
            )

        if not (0 <= initial_regime < self.n_regimes):
            raise ValueError(
                f"initial_regime must be in [0, {self.n_regimes-1}]. Got {initial_regime}"
            )

        # Initialize output arrays
        paths = np.zeros((self.n_scenarios, n_steps, self.n_assets))
        regime_paths = np.zeros((self.n_scenarios, n_steps), dtype=np.int64)
        shock_paths = np.zeros((self.n_scenarios, n_steps, self.n_shocks))

        # Generate paths
        for scenario in range(self.n_scenarios):
            # Initialize regime
            regime = initial_regime

            for t in range(n_steps):
                # Store regime
                regime_paths[scenario, t] = regime

                # Generate shocks
                shocks_t = np.random.randn(self.n_shocks)
                shock_paths[scenario, t, :] = shocks_t

                # Compute return: r_t = μ_{s_t} + B_{s_t} u_t + ε_t
                mu_t = regime_means[regime, :]

                # Shock component
                if loading_matrices is not None:
                    B_t = loading_matrices[regime, :, :]
                    shock_impact = B_t @ shocks_t
                else:
                    shock_impact = np.zeros(self.n_assets)

                # Idiosyncratic noise
                cov_t = regime_covs[regime, :, :]
                L_t = np.linalg.cholesky(cov_t)
                noise_t = L_t @ np.random.randn(self.n_assets)

                # Total return
                paths[scenario, t, :] = mu_t + shock_impact + noise_t

                # Transition to next regime
                regime_probs = transition_matrix[regime, :]
                regime = np.random.choice(self.n_regimes, p=regime_probs)

        return paths, regime_paths, shock_paths

    def compute_path_statistics(
        self,
        paths: NDArray[np.float64],
    ) -> Dict[str, NDArray[np.float64]]:
        """
        Compute statistics across Monte Carlo paths.

        Parameters
        ----------
        paths : NDArray[np.float64]
            Return paths, shape (n_scenarios, n_steps, n_assets)

        Returns
        -------
        stats : Dict[str, NDArray]
            Statistics per asset:
            - 'mean': Mean return per step, shape (n_steps, n_assets)
            - 'std': Std deviation per step, shape (n_steps, n_assets)
            - 'median': Median return per step, shape (n_steps, n_assets)
            - 'quantile_5': 5th percentile, shape (n_steps, n_assets)
            - 'quantile_95': 95th percentile, shape (n_steps, n_assets)
            - 'cumulative': Cumulative returns, shape (n_scenarios, n_steps, n_assets)
        """
        n_scenarios, n_steps, n_assets = paths.shape

        # Compute statistics across scenarios (axis 0)
        stats = {
            "mean": np.mean(paths, axis=0),  # (n_steps, n_assets)
            "std": np.std(paths, axis=0),  # (n_steps, n_assets)
            "median": np.median(paths, axis=0),  # (n_steps, n_assets)
            "quantile_5": np.percentile(paths, 5, axis=0),  # (n_steps, n_assets)
            "quantile_95": np.percentile(paths, 95, axis=0),  # (n_steps, n_assets)
            "cumulative": np.cumsum(paths, axis=1),  # (n_scenarios, n_steps, n_assets)
        }

        return stats

    def compute_portfolio_metrics(
        self,
        paths: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None,
        risk_free_rate: float = 0.0,
    ) -> Dict[str, float]:
        """
        Compute portfolio-level metrics.

        Parameters
        ----------
        paths : NDArray[np.float64]
            Return paths, shape (n_scenarios, n_steps, n_assets)
        weights : NDArray[np.float64], optional
            Portfolio weights, shape (n_assets,).
            If None, use equal weights (1/n_assets).
        risk_free_rate : float
            Risk-free rate for Sharpe ratio. Default 0.0.

        Returns
        -------
        metrics : Dict[str, float]
            Portfolio metrics:
            - 'mean_return': Mean annualized return
            - 'volatility': Annualized volatility
            - 'sharpe_ratio': Return / volatility - risk_free_rate
            - 'var_95': Value-at-Risk (95% confidence)
            - 'cvar_95': Conditional Value-at-Risk (95%)
            - 'max_drawdown': Maximum drawdown
        """
        n_scenarios, n_steps, n_assets = paths.shape

        # Default: equal weights
        if weights is None:
            weights = np.ones(n_assets) / n_assets
        else:
            if weights.shape != (n_assets,):
                raise ValueError(
                    f"weights must have shape ({n_assets},). Got {weights.shape}"
                )
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError(f"weights must sum to 1. Got {weights.sum()}")

        # Compute portfolio returns
        port_returns = paths @ weights  # (n_scenarios, n_steps)

        # Path-wise statistics
        path_returns = np.sum(port_returns, axis=1)  # Total return per path (n_scenarios,)
        mean_return = np.mean(path_returns)
        volatility = np.std(port_returns.flatten())

        # Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / (volatility + 1e-10)

        # VaR and CVaR
        sorted_returns = np.sort(path_returns)
        var_95_idx = int(0.05 * n_scenarios)
        var_95 = sorted_returns[var_95_idx]
        cvar_95 = np.mean(sorted_returns[:var_95_idx])

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + port_returns, axis=1)
        running_max = np.maximum.accumulate(cumulative_returns, axis=1)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)

        metrics = {
            "mean_return": float(mean_return),
            "volatility": float(volatility),
            "sharpe_ratio": float(sharpe),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95),
            "max_drawdown": float(max_drawdown),
        }

        return metrics

    def scenario_analysis(
        self,
        paths: NDArray[np.float64],
        regime_paths: NDArray[np.int64],
        regime_labels: Optional[list] = None,
    ) -> Dict:
        """
        Analyze paths by regime.

        Parameters
        ----------
        paths : NDArray[np.float64]
            Return paths, shape (n_scenarios, n_steps, n_assets)
        regime_paths : NDArray[np.int64]
            Regime sequences, shape (n_scenarios, n_steps)
        regime_labels : list, optional
            Regime names (e.g., ["Normal", "Stressed"])

        Returns
        -------
        analysis : Dict
            Per-regime statistics:
            - '{regime_label}': {
                'frequency': % of time in this regime,
                'mean_return': average return when in this regime,
                'volatility': volatility when in this regime
              }
        """
        n_scenarios, n_steps, n_assets = paths.shape

        if regime_labels is None:
            regime_labels = [f"Regime_{k}" for k in range(self.n_regimes)]

        analysis = {}

        for regime_idx in range(self.n_regimes):
            label = regime_labels[regime_idx]

            # Find times when in this regime
            in_regime = regime_paths == regime_idx
            regime_returns = paths[in_regime]

            if regime_returns.shape[0] > 0:
                frequency = np.mean(in_regime)
                mean_ret = np.mean(regime_returns)
                volatility = np.std(regime_returns)

                analysis[label] = {
                    "frequency": float(frequency),
                    "mean_return": float(mean_ret),
                    "volatility": float(volatility),
                    "n_observations": int(np.sum(in_regime)),
                }
            else:
                analysis[label] = {
                    "frequency": 0.0,
                    "mean_return": 0.0,
                    "volatility": 0.0,
                    "n_observations": 0,
                }

        return analysis

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MonteCarloSimulator(n_assets={self.n_assets}, n_regimes={self.n_regimes}, "
            f"n_shocks={self.n_shocks}, n_scenarios={self.n_scenarios})"
        )
