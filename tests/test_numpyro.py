"""Validation tests for the NumPyro/JAX NUTS sampler backend.

These tests are skipped when numpyro is not installed (e.g. in the dev env).
Run with:  pixi run -e jax python -m pytest tests/test_numpyro.py -v
"""

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro")

from src.data_gen import generate_hmm_data
from src.inference import fit, run_ffbs
from src.model import build_model


@pytest.fixture(scope="module")
def small_manual_model():
    """Build a small manual-forward-algorithm model for numpyro tests."""
    data = generate_hmm_data(T=30, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2)
    return data, model


def test_manual_model_logp_finite(small_manual_model):
    """The manual forward-algorithm model has a finite logp."""
    _, model = small_manual_model
    logp_fn = model.compile_logp()
    logp_val = logp_fn(model.initial_point())
    assert np.isfinite(logp_val), f"logp is not finite: {logp_val}"


@pytest.fixture(scope="module")
def numpyro_idata(small_manual_model):
    """Sample with numpyro once, reuse across tests."""
    data, model = small_manual_model
    idata = fit(
        model, draws=50, tune=50, chains=1,
        seed=42, nuts_sampler="numpyro",
    )
    return data, idata


def test_numpyro_sampling_runs(numpyro_idata):
    """numpyro sampler produces an InferenceData with expected variables."""
    _, idata = numpyro_idata
    assert "posterior" in idata.groups()
    assert "P" in idata.posterior
    assert "mu" in idata.posterior
    assert any("chol_cov_0" in v for v in idata.posterior.data_vars)


def test_numpyro_ffbs_on_result(numpyro_idata):
    """FFBS runs on numpyro idata and produces valid regime sequences."""
    data, idata = numpyro_idata
    regime_samples = run_ffbs(idata, data["returns"], seed=42, thin=1, verbose=False)

    n_chains = idata.posterior.sizes["chain"]
    n_draws = idata.posterior.sizes["draw"]
    T = data["returns"].shape[0]

    assert regime_samples.shape == (n_chains, n_draws, T)
    assert set(np.unique(regime_samples)).issubset({0, 1})
