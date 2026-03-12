"""Tests for NUTS sampling, diagnostics, and FFBS (Steps 3+4)."""

import numpy as np
import pytest

from src.data_gen import generate_hmm_data
from src.inference import (
    _permute_chain,
    check_diagnostics,
    check_diagnostics_label_aware,
    ffbs_single,
    fit,
    run_ffbs,
    stationary_distribution,
)
from src.model import build_model


@pytest.fixture(scope="module")
def fitted_result():
    """Fit a small model once, reuse across tests in this module."""
    data = generate_hmm_data(T=30, K=2, d=2, seed=42)
    _, model_marg = build_model(data["returns"], K=2)
    idata = fit(model_marg, draws=50, tune=50, chains=1, seed=42)
    return data, model_marg, idata


def test_fit_runs(fitted_result):
    _, _, idata = fitted_result
    assert "posterior" in idata.groups()
    assert "P" in idata.posterior
    assert "mu" in idata.posterior


def test_diagnostics_keys(fitted_result):
    _, _, idata = fitted_result
    diag = check_diagnostics(idata)
    expected_keys = {
        "no_divergences", "rhat_ok", "ess_ok",
        "n_divergences", "max_rhat", "min_ess_bulk",
    }
    assert set(diag.keys()) == expected_keys


def test_stationary_distribution_basic():
    P = np.array([[0.95, 0.05], [0.10, 0.90]])
    pi = stationary_distribution(P)
    assert pi.shape == (2,)
    assert np.allclose(pi.sum(), 1.0)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-10)


def test_ffbs_single_known_params():
    data = generate_hmm_data(T=100, K=2, d=2, seed=42)
    params = data["params"]

    K, d = params["mus"].shape
    chol_covs = np.zeros((K, d, d))
    for k in range(K):
        chol_covs[k] = np.linalg.cholesky(params["covs"][k])

    regimes = ffbs_single(
        data["returns"], params["P"], params["mus"], chol_covs,
        pi0=params["pi0"], rng=np.random.default_rng(42),
    )
    assert regimes.shape == (100,)

    acc_direct = np.mean(regimes == data["regimes"])
    acc_flipped = np.mean((1 - regimes) == data["regimes"])
    accuracy = max(acc_direct, acc_flipped)
    assert accuracy > 0.70, f"FFBS accuracy too low: {accuracy:.2f}"


def test_run_ffbs_shapes(fitted_result):
    data, _, idata = fitted_result
    regime_samples = run_ffbs(idata, data["returns"], seed=42, thin=1, verbose=False)
    n_chains = idata.posterior.sizes["chain"]
    n_draws = idata.posterior.sizes["draw"]
    T = data["returns"].shape[0]
    assert regime_samples.shape == (n_chains, n_draws, T)


def test_label_aware_diagnostics_keys(fitted_result):
    _, _, idata = fitted_result
    diag = check_diagnostics_label_aware(idata, K=2)
    expected_keys = {
        "no_divergences", "rhat_ok", "ess_ok",
        "n_divergences", "max_rhat", "naive_max_rhat",
        "min_ess_bulk", "label_switching_detected", "best_permutations",
    }
    assert set(diag.keys()) == expected_keys
    assert isinstance(diag["label_switching_detected"], bool)
    assert isinstance(diag["best_permutations"], list)
    if np.isfinite(diag["max_rhat"]):
        assert diag["max_rhat"] <= diag["naive_max_rhat"]


def _make_mock_posterior(K, d, n_chains=2, n_draws=5):
    """Build a minimal xarray Dataset that looks like a posterior for _permute_chain tests."""
    import xarray as xr

    rng = np.random.default_rng(0)
    mu = rng.standard_normal((n_chains, n_draws, K, d))
    P = np.full((n_chains, n_draws, K, K), 1.0 / K)

    ds = xr.Dataset(
        {
            "mu": (["chain", "draw", "mu_dim_0", "mu_dim_1"], mu),
            "P": (["chain", "draw", "P_dim_0", "P_dim_1"], P),
        },
        coords={"chain": range(n_chains), "draw": range(n_draws)},
    )

    n_packed = d * (d + 1) // 2
    for k in range(K):
        packed = rng.standard_normal((n_chains, n_draws, n_packed))
        ds[f"chol_cov_{k}"] = (["chain", "draw", "packed"], packed)
        ds[f"chol_cov_{k}_corr"] = (["chain", "draw", "packed"], packed * 2)
    return ds


@pytest.mark.parametrize(
    "K, perm",
    [
        (2, (1, 0)),
        (3, (2, 0, 1)),
        (3, (1, 2, 0)),
        (3, (0, 2, 1)),
        (4, (1, 2, 3, 0)),
    ],
)
def test_permute_chain_chol_cov(K, perm):
    """Verify _permute_chain correctly remaps chol_cov variables for any permutation."""
    d = 2
    ds = _make_mock_posterior(K, d)
    chain_idx = 0

    originals = {}
    for k in range(K):
        for suffix in ("", "_corr"):
            name = f"chol_cov_{k}{suffix}"
            originals[name] = ds[name].values[chain_idx].copy()

    result = _permute_chain(ds, chain_idx, perm, K)

    for k_new, k_old in enumerate(perm):
        for suffix in ("", "_corr"):
            dst = f"chol_cov_{k_new}{suffix}"
            src = f"chol_cov_{k_old}{suffix}"
            np.testing.assert_array_equal(
                result[dst].values[chain_idx],
                originals[src],
                err_msg=f"perm={perm}: {dst} should contain original {src}",
            )


@pytest.mark.parametrize(
    "K, perm",
    [
        (2, (1, 0)),
        (3, (2, 0, 1)),
        (4, (1, 2, 3, 0)),
    ],
)
def test_permute_chain_mu_and_P(K, perm):
    """Verify _permute_chain correctly remaps mu and P."""
    d = 2
    ds = _make_mock_posterior(K, d)
    chain_idx = 0
    perm_arr = np.array(perm)

    orig_mu = ds["mu"].values[chain_idx].copy()
    orig_P = ds["P"].values[chain_idx].copy()

    result = _permute_chain(ds, chain_idx, perm, K)

    np.testing.assert_array_equal(
        result["mu"].values[chain_idx],
        orig_mu[:, perm_arr, :],
    )
    np.testing.assert_array_equal(
        result["P"].values[chain_idx],
        orig_P[:, perm_arr, :][:, :, perm_arr],
    )
