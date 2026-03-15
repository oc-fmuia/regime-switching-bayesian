from typing import Any

import numpy as np


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a transition matrix.

    Parameters
    ----------
    P: np.ndarray[K, K]
        Row-stochastic transition matrix.

    Returns
    -------
    np.ndarray[K,]
        Stationary probability vector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    return pi / pi.sum()


def bull_bear_params() -> dict[str, Any]:
    """
    Return parameters for a K=2 Bull/Bear regime-switching scenario.

    Returns
    -------
    dict[str, Any]
        Keys: mus float[2, 3], sigmas float[2, 3], corr_chols float[2, 3, 3],
        P float[2, 2], pi0 float[2,].

    Notes
    -----
    Asset universe: US equities (E), long-duration Treasuries (B), gold (G).
    Annualised figures: Bull +12%/+5%/+2.5% at 14%/5%/10% vol;
    Bear -22%/+7%/+13% at 28%/9%/14% vol (flight-to-quality).
    Equity-bond correlation: -0.10 (bull) vs -0.60 (bear).
    Stationary distribution: approximately 0.83 bull / 0.17 bear.
    """
    mus = np.array([
        [+0.010, +0.004, +0.002],
        [-0.020, +0.006, +0.010],
    ])
    sigmas = np.array([
        [0.040, 0.015, 0.030],
        [0.080, 0.025, 0.040],
    ])

    C0 = np.array([
        [1.00, -0.10,  0.05],
        [-0.10,  1.00,  0.10],
        [0.05,  0.10,  1.00],
    ])
    C1 = np.array([
        [ 1.00, -0.60, -0.30],
        [-0.60,  1.00,  0.50],
        [-0.30,  0.50,  1.00],
    ])
    corr_chols = np.stack([np.linalg.cholesky(C0), np.linalg.cholesky(C1)])

    P = np.array([[0.95, 0.05], [0.25, 0.75]])
    pi0 = _stationary_distribution(P)

    return {"mus": mus, "sigmas": sigmas, "corr_chols": corr_chols, "P": P, "pi0": pi0}


def growth_stagnation_crisis_params() -> dict[str, Any]:
    """
    Return parameters for a K=3 Growth/Stagnation/Crisis regime-switching scenario.

    Returns
    -------
    dict[str, Any]
        Keys: mus float[3, 3], sigmas float[3, 3], corr_chols float[3, 3, 3],
        P float[3, 3], pi0 float[3,].

    Notes
    -----
    Asset universe: US equities (E), long-duration Treasuries (B), gold (G).
    Stationary distribution: approximately (0.56, 0.29, 0.15).
    The crisis regime (~15% of time) yields ~18 expected observations over T=120,
    producing intentionally wide posteriors for its parameters.
    Equity-bond correlation: -0.10 (growth), +0.15 (stagnation), -0.55 (crisis).
    """
    mus = np.array([
        [+0.007, +0.003, +0.001],
        [ 0.000, +0.002, +0.003],
        [-0.030, +0.008, +0.015],
    ])
    sigmas = np.array([
        [0.035, 0.013, 0.028],
        [0.055, 0.018, 0.038],
        [0.100, 0.030, 0.050],
    ])

    C_growth = np.array([
        [ 1.00, -0.10,  0.05],
        [-0.10,  1.00,  0.10],
        [ 0.05,  0.10,  1.00],
    ])
    C_stagnation = np.array([
        [1.00,  0.15,  0.00],
        [0.15,  1.00,  0.05],
        [0.00,  0.05,  1.00],
    ])
    C_crisis = np.array([
        [ 1.00, -0.55, -0.25],
        [-0.55,  1.00,  0.45],
        [-0.25,  0.45,  1.00],
    ])
    corr_chols = np.stack([
        np.linalg.cholesky(C_growth),
        np.linalg.cholesky(C_stagnation),
        np.linalg.cholesky(C_crisis),
    ])

    # FM-NOTE: designed so stationary dist ≈ (0.56, 0.29, 0.15); see Notes.
    P = np.array([
        [0.92, 0.06, 0.02],
        [0.10, 0.77, 0.13],
        [0.10, 0.23, 0.67],
    ])
    pi0 = _stationary_distribution(P)

    return {"mus": mus, "sigmas": sigmas, "corr_chols": corr_chols, "P": P, "pi0": pi0}
