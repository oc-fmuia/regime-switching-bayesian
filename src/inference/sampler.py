"""
NUTS sampler and convergence diagnostics for regime-switching models.

Orchestrates PyMC sampling, computes convergence diagnostics (Rhat, ESS),
performs posterior predictive checks, and generates summary statistics.

Key diagnostics:
- Rhat (potential scale reduction): <1.01 indicates convergence
- ESS (effective sample size): >400 per chain recommended
- Divergences: <2% of draws acceptable
- Posterior predictive p-value: should be ~0.5 for well-specified model
"""

from typing import Optional, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import pymc as pm
import arviz as az


class InferenceSummary:
    """Summary statistics from MCMC inference."""

    def __init__(
        self,
        idata,  # arviz.InferenceData
        n_draws: int,
        n_tune: int,
        n_chains: int,
        sampling_time: float,
    ) -> None:
        """
        Initialize inference summary.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior inference data from PyMC
        n_draws : int
            Number of post-burn-in draws per chain
        n_tune : int
            Number of burn-in steps per chain
        n_chains : int
            Number of parallel chains
        sampling_time : float
            Total sampling time (seconds)
        """
        self.idata = idata
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.sampling_time = sampling_time
        self.total_samples = n_draws * n_chains

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"InferenceSummary(draws={self.n_draws}, tune={self.n_tune}, "
            f"chains={self.n_chains}, time={self.sampling_time:.1f}s)"
        )


class NUTSSampler:
    """
    NUTS sampler for regime-switching models.

    Orchestrates PyMC MCMC sampling with configurable parameters,
    computes convergence diagnostics, and provides inference summaries.
    """

    def __init__(
        self,
        target_accept: float = 0.85,
        max_treedepth: int = 10,
    ) -> None:
        """
        Initialize sampler.

        Parameters
        ----------
        target_accept : float
            NUTS acceptance rate target (0.7-0.95). Default 0.85.
        max_treedepth : int
            Maximum tree depth for NUTS. Default 10 (2^10 = 1024 steps max).
        """
        if not (0.5 < target_accept < 0.99):
            raise ValueError(f"target_accept must be in (0.5, 0.99). Got {target_accept}")
        if max_treedepth < 5:
            raise ValueError(f"max_treedepth must be >= 5. Got {max_treedepth}")

        self.target_accept = target_accept
        self.max_treedepth = max_treedepth

    def sample(
        self,
        model: pm.Model,
        draws: int = 1000,
        tune: int = 1000,
        chains: int = 2,
        cores: int = 2,
        random_seed: Optional[int] = None,
        progressbar: bool = True,
        return_inferencedata: bool = True,
    ) -> InferenceSummary:
        """
        Run NUTS sampling on PyMC model.

        Parameters
        ----------
        model : pm.Model
            PyMC model (from ModelBuilder.build())
        draws : int
            Number of post-burn-in samples per chain. Default 1000.
        tune : int
            Number of burn-in steps per chain. Default 1000.
        chains : int
            Number of parallel chains. Default 2.
        cores : int
            Number of cores to use. Default 2.
        random_seed : int, optional
            Random seed for reproducibility.
        progressbar : bool
            Show progress bar. Default True.
        return_inferencedata : bool
            Return arviz InferenceData. Default True.

        Returns
        -------
        summary : InferenceSummary
            Summary with posterior, diagnostics, timing.

        Raises
        ------
        RuntimeError
            If sampling divergences exceed 5% of total samples.
        """
        import time

        start_time = time.time()

        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=random_seed,
                progressbar=progressbar,
                target_accept=self.target_accept,
                return_inferencedata=return_inferencedata,
                discard_tuned_samples=True,
            )

        sampling_time = time.time() - start_time

        # Check divergences
        n_divergences = idata.sample_stats.diverging.sum().item()
        n_total = draws * chains
        div_rate = n_divergences / n_total

        if div_rate > 0.05:
            raise RuntimeError(
                f"Divergence rate too high: {div_rate:.1%} ({n_divergences}/{n_total}). "
                f"Consider increasing tune, reducing learning rate, or reparameterizing."
            )

        summary = InferenceSummary(
            idata=idata,
            n_draws=draws,
            n_tune=tune,
            n_chains=chains,
            sampling_time=sampling_time,
        )

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"NUTSSampler(target_accept={self.target_accept}, "
            f"max_treedepth={self.max_treedepth})"
        )


class DiagnosticsComputer:
    """
    Compute convergence diagnostics from posterior samples.

    Includes: Rhat, ESS, divergence rates, tail behavior.
    """

    @staticmethod
    def rhat(posterior_samples: NDArray[np.float64]) -> float:
        """
        Compute Rhat (potential scale reduction factor).

        Rhat measures whether multiple chains have converged to the same
        posterior distribution. Rhat < 1.01 indicates convergence.

        Parameters
        ----------
        posterior_samples : NDArray[np.float64]
            Posterior samples from multiple chains, shape (chains, draws).

        Returns
        -------
        rhat : float
            Potential scale reduction factor. <1.01 is good.
        """
        n_chains, n_draws = posterior_samples.shape

        if n_chains < 2:
            raise ValueError("Need at least 2 chains for Rhat")

        # Between-chain variance
        chain_means = np.mean(posterior_samples, axis=1)  # (chains,)
        grand_mean = np.mean(chain_means)
        B = n_draws * np.var(chain_means, ddof=1)

        # Within-chain variance
        chain_vars = np.var(posterior_samples, axis=1, ddof=1)  # (chains,)
        W = np.mean(chain_vars)

        # Estimated posterior variance
        var_hat = ((n_draws - 1) / n_draws) * W + (1 / n_draws) * B

        # Rhat
        rhat = np.sqrt(var_hat / W) if W > 0 else 1.0

        return float(rhat)

    @staticmethod
    def ess(posterior_samples: NDArray[np.float64]) -> float:
        """
        Compute effective sample size (ESS).

        ESS accounts for autocorrelation in MCMC samples.
        ESS > 400 per chain is recommended.

        Parameters
        ----------
        posterior_samples : NDArray[np.float64]
            Posterior samples from single chain, shape (draws,).

        Returns
        -------
        ess : float
            Effective sample size.
        """
        n = len(posterior_samples)

        # Autocorrelation via spectral density at 0
        # (simple estimate: integrate autocorrelation function)
        mean = np.mean(posterior_samples)
        c0 = np.var(posterior_samples, ddof=1)

        if c0 < 1e-10:
            return float(n)  # No variation → ESS = n

        # Estimate integrated autocorrelation
        tau_int = 0.5  # Minimum bound
        max_lag = min(n // 2, 100)

        for lag in range(1, max_lag):
            acov = np.mean(
                (posterior_samples[:-lag] - mean) * (posterior_samples[lag:] - mean)
            )
            rho = acov / c0

            if rho < 0.05:  # Stop when autocorr negligible
                break

            tau_int += rho

        ess = n / (2 * tau_int)
        return float(max(1, ess))

    @staticmethod
    def divergence_rate(idata) -> float:
        """
        Compute divergence rate from InferenceData.

        Divergences indicate areas of high curvature in parameter space
        where NUTS sampler struggles. <2% is acceptable, <0.5% is good.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior inference data from PyMC

        Returns
        -------
        div_rate : float
            Fraction of samples that diverged [0, 1].
        """
        n_divergences = idata.sample_stats.diverging.sum().item()
        n_total = idata.posterior.sizes["draw"] * idata.posterior.sizes["chain"]
        return float(n_divergences / n_total)


class PosteriorPredictiveCheck:
    """
    Posterior predictive checks for model validation.

    Compares observed data to draws from posterior predictive distribution
    to assess whether the model generates plausible data.
    """

    @staticmethod
    def compute_ppcheck(
        idata,
        observed_data: NDArray[np.float64],
        dim_name: str = "returns",
    ) -> Dict[str, float]:
        """
        Compute posterior predictive check statistics.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior inference data from PyMC
        observed_data : NDArray[np.float64]
            Observed data, shape (T, N) for returns
        dim_name : str
            Variable name in posterior_predictive

        Returns
        -------
        ppc_stats : Dict[str, float]
            Posterior predictive p-values:
            - mean_pvalue: p-value for mean
            - std_pvalue: p-value for std
            - max_pvalue: p-value for max value
        """
        if dim_name not in idata.posterior_predictive:
            return {}

        pp_samples = idata.posterior_predictive[dim_name].values
        # Shape: (chain, draw, T, N) or similar

        # Flatten to (chain*draw, ...)
        pp_flat = pp_samples.reshape(-1, *pp_samples.shape[2:])

        obs_mean = np.mean(observed_data)
        pp_means = np.mean(pp_flat, axis=tuple(range(1, pp_flat.ndim)))
        mean_pvalue = float(np.mean(pp_means >= obs_mean))

        obs_std = np.std(observed_data)
        pp_stds = np.std(pp_flat, axis=tuple(range(1, pp_flat.ndim)))
        std_pvalue = float(np.mean(pp_stds >= obs_std))

        obs_max = np.max(np.abs(observed_data))
        pp_maxs = np.max(np.abs(pp_flat), axis=tuple(range(1, pp_flat.ndim)))
        max_pvalue = float(np.mean(pp_maxs >= obs_max))

        return {
            "mean_pvalue": mean_pvalue,
            "std_pvalue": std_pvalue,
            "max_pvalue": max_pvalue,
        }

    @staticmethod
    def summary_stats(
        idata,
        var_names: Optional[list] = None,
    ) -> Dict:
        """
        Compute posterior summary statistics.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior inference data
        var_names : list, optional
            Variables to summarize. If None, use all.

        Returns
        -------
        stats : Dict
            Summary with mean, std, credible intervals, Rhat, ESS
        """
        summary_df = az.summary(
            idata,
            var_names=var_names,
            kind="stats",
        )

        # Convert to dict
        stats = {}
        for var_name in summary_df.index:
            stats[var_name] = {
                "mean": float(summary_df.loc[var_name, "mean"]),
                "std": float(summary_df.loc[var_name, "std"]),
                "hdi_low": float(summary_df.loc[var_name, "hdi_2.5%"]),
                "hdi_high": float(summary_df.loc[var_name, "hdi_97.5%"]),
                "rhat": float(summary_df.loc[var_name, "r_hat"]),
                "ess_bulk": float(summary_df.loc[var_name, "ess_bulk"]),
            }

        return stats
