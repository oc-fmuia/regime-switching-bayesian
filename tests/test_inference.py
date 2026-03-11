"""Tests for NUTS sampling, diagnostics, and FFBS (Steps 3+4)."""

import numpy as np
import pytest

from src.data_gen import generate_hmm_data
from src.inference import (
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
    model = build_model(data["returns"], K=2)
    idata = fit(model, draws=50, tune=50, chains=1, seed=42)
    return data, model, idata


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
