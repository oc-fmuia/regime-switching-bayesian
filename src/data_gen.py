"""Synthetic data generation for regime-switching HMM."""

from typing import Any

import numpy as np


def generate_hmm_data(
    T: int = 120,
    K: int = 2,
    d: int = 3,
    mus: np.ndarray | None = None,
    sigmas: np.ndarray | None = None,
    corr_chols: np.ndarray | None = None,
    P: np.ndarray | None = None,
    pi0: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Generate synthetic HMM regime-switching multivariate return data.

    Parameters
    ----------
    T : number of time steps
    K : number of regimes
    d : number of assets
    mus : (K, d) regime means. Defaults to Bull/Bear values.
    sigmas : (K, d) regime standard deviations.
    corr_chols : (K, d, d) Cholesky factors of correlation matrices.
    P : (K, K) transition matrix.
    pi0 : (K,) initial regime distribution.
    seed : random seed for reproducibility.

    Returns
    -------
    dict with keys "returns", "regimes", "params", "config".
    """
    rng = np.random.default_rng(seed)

    if mus is None:
        bull_base = np.array([0.01, 0.008, 0.012])
        bear_base = np.array([-0.005, -0.008, -0.003])
        mus = np.vstack([
            np.resize(bull_base, d),
            np.resize(bear_base, d),
        ])[:K]
    if sigmas is None:
        bull_vol = np.array([0.04, 0.035, 0.045])
        bear_vol = np.array([0.08, 0.09, 0.10])
        sigmas = np.vstack([
            np.resize(bull_vol, d),
            np.resize(bear_vol, d),
        ])[:K]
    if corr_chols is None:
        corr_chols = np.stack([np.eye(d)] * K)
    if P is None:
        P = np.array([[0.95, 0.05], [0.10, 0.90]])
    if pi0 is None:
        pi0 = np.array([0.8, 0.2])

    # cov_k = diag(sigma_k) @ L_k @ L_k^T @ diag(sigma_k)
    covs = np.zeros((K, d, d))
    for k in range(K):
        D = np.diag(sigmas[k])
        L = corr_chols[k]
        covs[k] = D @ L @ L.T @ D

    regimes = np.zeros(T, dtype=int)
    regimes[0] = rng.choice(K, p=pi0)
    for t in range(1, T):
        regimes[t] = rng.choice(K, p=P[regimes[t - 1]])

    returns = np.zeros((T, d))
    for t in range(T):
        returns[t] = rng.multivariate_normal(mus[regimes[t]], covs[regimes[t]])

    return {
        "returns": returns,
        "regimes": regimes,
        "params": {
            "mus": mus,
            "sigmas": sigmas,
            "corr_chols": corr_chols,
            "covs": covs,
            "P": P,
            "pi0": pi0,
        },
        "config": {"T": T, "K": K, "d": d, "seed": seed},
    }
