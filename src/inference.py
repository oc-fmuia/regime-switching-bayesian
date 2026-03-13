"""NUTS sampling, diagnostics, and FFBS regime recovery."""

import arviz as az
import numpy as np
import pymc as pm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


def fit(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
    nuts_sampler: str = "numpyro",
    initvals: dict | None = None,
) -> az.InferenceData:
    """Run NUTS sampling on the (marginalized) model.

    Parameters
    ----------
    nuts_sampler: str
        "numpyro" (default, JAX backend) or "pymc" (C backend).
    initvals: dict | None
        Optional {var_name: value} passed to pm.sample to fix the starting
        point of the sampler. Useful for mitigating label switching by
        pinning mu to a known ordering.
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            nuts_sampler=nuts_sampler,
            initvals=initvals,
        )
    return idata


def check_diagnostics(idata: az.InferenceData) -> dict[str, bool | int | float]:
    """Return a dict of pass/fail diagnostic checks."""
    n_divergences = int(idata.sample_stats.diverging.sum())
    rhat = az.rhat(idata)
    ess = az.ess(idata)

    max_rhat = float(rhat.to_array().max())
    min_ess_bulk = float(ess.to_array().min())

    return {
        "no_divergences": n_divergences == 0,
        "rhat_ok": max_rhat < 1.01,
        "ess_ok": min_ess_bulk > 400,
        "n_divergences": n_divergences,
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
    }


def _permute_chain(
    posterior: "xr.Dataset",
    chain_idx: int,
    perm: tuple[int, ...],
    K: int,
) -> "xr.Dataset":
    """Return a copy of *posterior* with regime labels in *chain_idx* permuted.

    Handles:
      - mu (chain, draw, K, d)           -> permute axis K
      - P  (chain, draw, K, K)           -> permute rows and columns
      - chol_cov_{k}*  (all suffixes)    -> swap the per-regime variables
    """
    ds = posterior.copy(deep=True)
    perm_arr = np.array(perm)

    for name in list(ds.data_vars):
        arr = ds[name].values
        shape_tail = arr.shape[2:]

        if name == "mu" and len(shape_tail) == 2 and shape_tail[0] == K:
            arr[chain_idx] = arr[chain_idx][:, perm_arr, :]

        elif name == "P" and shape_tail == (K, K):
            arr[chain_idx] = arr[chain_idx][:, perm_arr, :][:, :, perm_arr]

    # Permute ALL per-regime variables: chol_cov_0, chol_cov_0_corr,
    # chol_cov_0_stds, chol_cov_0_cholesky-cov-packed__, etc.
    # We collect every distinct suffix (e.g. "", "_corr", "_stds",
    # "_cholesky-cov-packed__") and permute each group via a temp buffer
    # so that 3-cycles and longer cycles are handled correctly.
    chol_suffixes = set()
    for name in ds.data_vars:
        for k in range(K):
            prefix = f"chol_cov_{k}"
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                if not suffix or suffix[0] in ("_", "-"):
                    chol_suffixes.add(suffix)

    for suffix in chol_suffixes:
        tmp = {}
        for k in range(K):
            var_name = f"chol_cov_{k}{suffix}"
            if var_name in ds:
                tmp[k] = ds[var_name].values[chain_idx].copy()
        for k_new, k_old in enumerate(perm):
            if k_new == k_old:
                continue
            dst_name = f"chol_cov_{k_new}{suffix}"
            if dst_name in ds and k_old in tmp:
                ds[dst_name].values[chain_idx] = tmp[k_old]

    return ds


def _find_best_permutation(
    ref_mu_mean: np.ndarray,
    chain_mu_mean: np.ndarray,
    K: int,
) -> tuple[int, ...]:
    """Find the label permutation for a chain that best aligns its mu to the reference.

    Compares the draw-averaged mu (K, d) of a chain against the reference
    chain and returns the permutation minimising sum-of-squared differences.
    """
    from itertools import permutations

    best_perm = tuple(range(K))
    best_sse = np.inf
    for perm in permutations(range(K)):
        sse = float(np.sum((chain_mu_mean[list(perm)] - ref_mu_mean) ** 2))
        if sse < best_sse:
            best_sse = sse
            best_perm = perm
    return best_perm


def check_diagnostics_label_aware(
    idata: az.InferenceData,
    K: int = 2,
) -> dict:
    """Label-switching-aware diagnostics.

    Aligns each chain's regime labels to chain 0 by comparing posterior
    mu means (same logic as align_regime_samples but on the parameter
    space).  Then computes Rhat on the aligned posterior.

    Returns the same keys as check_diagnostics plus:
      - label_switching_detected : bool
      - best_permutations : list[tuple] per chain
      - naive_max_rhat : float (before relabeling)
    """
    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_divergences = int(idata.sample_stats.diverging.sum())

    naive_rhat = az.rhat(idata)
    _rhat_vals = naive_rhat.to_array().values
    naive_max_rhat = float(np.nanmax(_rhat_vals)) if np.any(np.isfinite(_rhat_vals)) else np.nan

    identity = tuple(range(K))
    best_assignment = [identity] * n_chains

    if n_chains >= 2 and "mu" in posterior:
        mu_all = posterior["mu"].values  # (chains, draws, K, d)
        ref_mu_mean = mu_all[0].mean(axis=0)  # (K, d)

        for c in range(1, n_chains):
            chain_mu_mean = mu_all[c].mean(axis=0)
            best_assignment[c] = _find_best_permutation(ref_mu_mean, chain_mu_mean, K)

    any_switched = any(p != identity for p in best_assignment)

    if any_switched and n_chains >= 2:
        ds = posterior.copy(deep=True)
        for c, perm in enumerate(best_assignment):
            if perm != identity:
                ds = _permute_chain(ds, c, perm, K)

        aligned_idata = idata.copy()
        aligned_idata.posterior = ds
        aligned_rhat = az.rhat(aligned_idata)
        best_max_rhat = float(np.nanmax(aligned_rhat.to_array().values))
    else:
        best_max_rhat = naive_max_rhat

    label_switching = (
        n_chains >= 2
        and np.isfinite(naive_max_rhat)
        and any_switched
    )

    ess = az.ess(idata)
    min_ess_bulk = float(np.nanmin(ess.to_array().values))

    rhat_ok = best_max_rhat < 1.01 if np.isfinite(best_max_rhat) else True

    return {
        "no_divergences": n_divergences == 0,
        "rhat_ok": rhat_ok,
        "ess_ok": min_ess_bulk > 400,
        "n_divergences": n_divergences,
        "max_rhat": best_max_rhat,
        "naive_max_rhat": naive_max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "label_switching_detected": label_switching,
        "best_permutations": best_assignment,
    }


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Compute the stationary distribution of transition matrix P."""
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    return pi / pi.sum()


def _compute_emission_loglik(
    data: np.ndarray, mu: np.ndarray, chol_covs: np.ndarray
) -> np.ndarray:
    """Vectorised emission log-likelihoods.

    Returns
    -------
    log_lik : (T, K) where log_lik[t, k] = log N(y_t | mu_k, Sigma_k)
    """
    K = mu.shape[0]
    log_lik = np.empty((data.shape[0], K))
    for k in range(K):
        cov_k = chol_covs[k] @ chol_covs[k].T
        log_lik[:, k] = multivariate_normal.logpdf(data, mu[k], cov_k)
    return log_lik


def ffbs_single(
    data: np.ndarray,
    P: np.ndarray,
    mu: np.ndarray,
    chol_covs: np.ndarray,
    pi0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    *,
    log_lik: np.ndarray | None = None,
) -> np.ndarray:
    """
    Forward-filter backward-sample for one set of parameters.

    Parameters
    ----------
    data : (T, d) observed returns
    P : (K, K) transition matrix
    mu : (K, d) regime means
    chol_covs : (K, d, d) Cholesky factors of covariance matrices
    pi0 : (K,) initial distribution; defaults to stationary distribution of P
    rng : random generator
    log_lik : (T, K) pre-computed emission log-likelihoods (skip recomputation)

    Returns
    -------
    regimes : (T,) sampled regime sequence
    """
    if rng is None:
        rng = np.random.default_rng()

    T = data.shape[0]
    K = P.shape[0]

    if pi0 is None:
        pi0 = stationary_distribution(P)

    if log_lik is None:
        log_lik = _compute_emission_loglik(data, mu, chol_covs)

    log_P = np.log(P + 1e-300)

    # Forward pass: log_alpha[t, k] = log p(y_{1:t}, s_t=k)
    log_alpha = np.empty((T, K))
    log_alpha[0] = np.log(pi0 + 1e-300) + log_lik[0]

    for t in range(1, T):
        log_alpha[t] = logsumexp(log_alpha[t - 1, :, None] + log_P, axis=0) + log_lik[t]

    # Backward sampling
    regimes = np.empty(T, dtype=int)
    log_gamma = log_alpha[T - 1] - logsumexp(log_alpha[T - 1])
    regimes[T - 1] = rng.choice(K, p=np.exp(log_gamma))

    for t in range(T - 2, -1, -1):
        log_gamma = log_alpha[t] + log_P[:, regimes[t + 1]]
        log_gamma -= logsumexp(log_gamma)
        regimes[t] = rng.choice(K, p=np.exp(log_gamma))

    return regimes


def _extract_chol_cov(
    idata: az.InferenceData, chain_idx: int, draw_idx: int, k: int, d: int
) -> np.ndarray:
    """Extract the (d, d) Cholesky factor for regime k from a posterior draw."""
    n_packed = d * (d + 1) // 2

    for name in [f"chol_cov_{k}", f"chol_cov_{k}_cholesky-cov-packed__"]:
        if name not in idata.posterior:
            continue
        vals = idata.posterior[name].values[chain_idx, draw_idx]
        if vals.shape == (d, d):
            return vals
        if vals.shape == (n_packed,):
            chol = np.zeros((d, d))
            chol[np.tril_indices(d)] = vals
            return chol

    raise KeyError(
        f"Cannot find Cholesky variable for regime {k}. "
        f"Available: {list(idata.posterior.data_vars)}"
    )


def _batch_extract_chol_covs(
    idata: az.InferenceData, K: int, d: int
) -> np.ndarray:
    """Extract all Cholesky factors at once.

    Returns
    -------
    chol_all : (n_chains, n_draws, K, d, d)
    """
    n_chains = idata.posterior.sizes["chain"]
    n_draws = idata.posterior.sizes["draw"]
    n_packed = d * (d + 1) // 2
    chol_all = np.zeros((n_chains, n_draws, K, d, d))

    for k in range(K):
        for name in [f"chol_cov_{k}", f"chol_cov_{k}_cholesky-cov-packed__"]:
            if name not in idata.posterior:
                continue
            vals = idata.posterior[name].values  # (chains, draws, ...)
            if vals.shape[-2:] == (d, d):
                chol_all[:, :, k] = vals
            elif vals.shape[-1] == n_packed:
                tril_idx = np.tril_indices(d)
                chol_all[:, :, k, tril_idx[0], tril_idx[1]] = vals
            break
    return chol_all


_ALIGN_MAX_BYTES = 512 * 1024 * 1024  # 512 MB


def align_regime_samples(regime_samples: np.ndarray, K: int = 2) -> np.ndarray:
    """Align regime labels across chains to fix label switching in FFBS output.

    Uses chain 0 as the reference.  For each subsequent chain, tests all K!
    permutations and picks the one with highest element-wise agreement.
    For K=2 this simply checks whether flipping 0<->1 improves agreement.

    Parameters
    ----------
    regime_samples : (n_chains, n_draws, T)
    K : number of regimes

    Returns
    -------
    aligned : same shape, with per-chain labels remapped

    Notes
    ------
    Internally broadcasts all K! permutations at once via a (K!, n_draws, T)
    array. Raises ValueError if the estimated allocation would exceed
    _ALIGN_MAX_BYTES (default 512 MB) to prevent OOM on large K or T.
    """
    import math
    from itertools import permutations

    n_chains, n_draws, T = regime_samples.shape

    estimated_bytes = math.factorial(K) * n_draws * T * regime_samples.dtype.itemsize
    if estimated_bytes > _ALIGN_MAX_BYTES:
        raise ValueError(
            f"align_regime_samples: vectorized permutation search would allocate "
            f"{estimated_bytes / 1e9:.2f} GB (K={K}, K!={math.factorial(K)}, "
            f"n_draws={n_draws}, T={T}, itemsize={regime_samples.dtype.itemsize}B). "
            f"Limit is {_ALIGN_MAX_BYTES / 1e9:.2f} GB. "
            f"Reduce K, increase the thinning factor in run_ffbs, or raise "
            f"_ALIGN_MAX_BYTES if you are sure the allocation is safe."
        )

    aligned = regime_samples.copy()
    if n_chains <= 1:
        return aligned

    all_perms = np.array(list(permutations(range(K))))  # (K!, K)
    ref = aligned[0]

    for c in range(1, n_chains):
        all_remapped = all_perms[:, aligned[c]]                    # (K!, n_draws, T)
        agreements = (all_remapped == ref).mean(axis=(-2, -1))     # (K!,)
        best_idx = int(np.argmax(agreements))
        best_perm = tuple(all_perms[best_idx].tolist())
        if best_perm != tuple(range(K)):
            aligned[c] = all_perms[best_idx][aligned[c]]

    return aligned


def run_ffbs(
    idata: az.InferenceData,
    data: np.ndarray,
    seed: int = 42,
    thin: int = 10,
    verbose: bool = True,
) -> np.ndarray:
    """
    Run FFBS across posterior draws.

    Parameters
    ----------
    thin : int
        Keep every `thin`-th draw (1 = all draws, 10 = every 10th).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    regime_samples : (n_chains, n_kept_draws, T) integer array
    """
    import time

    rng = np.random.default_rng(seed)
    T, d = data.shape

    posterior = idata.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    K = posterior["mu"].shape[2]

    draw_indices = np.arange(0, n_draws, thin)
    n_kept = len(draw_indices)
    total_runs = n_chains * n_kept

    if verbose:
        print(
            f"FFBS: {n_chains} chain(s) × {n_kept} draws "
            f"(thin={thin}, {n_draws} total) × T={T}, K={K}"
        )

    P_all = posterior["P"].values          # (chains, draws, K, K)
    mu_all = posterior["mu"].values        # (chains, draws, K, d)
    chol_all = _batch_extract_chol_covs(idata, K, d)

    regime_samples = np.zeros((n_chains, n_kept, T), dtype=int)

    t_start = time.perf_counter()
    done = 0

    for c in range(n_chains):
        for i, s in enumerate(draw_indices):
            log_lik = _compute_emission_loglik(data, mu_all[c, s], chol_all[c, s])
            regime_samples[c, i] = ffbs_single(
                data, P_all[c, s], mu_all[c, s], chol_all[c, s],
                rng=rng, log_lik=log_lik,
            )
            done += 1

            if verbose and (done % max(1, total_runs // 10) == 0 or done == total_runs):
                elapsed = time.perf_counter() - t_start
                rate = done / elapsed
                eta = (total_runs - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{total_runs}] "
                    f"{elapsed:.1f}s elapsed, ~{eta:.1f}s remaining "
                    f"({rate:.0f} draws/s)"
                )

    if verbose:
        print(f"FFBS complete in {time.perf_counter() - t_start:.1f}s")

    return regime_samples
