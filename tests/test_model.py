"""Tests for PyMC model definition (Step 2)."""

import numpy as np
import pymc as pm

from regime_switching_bayesian.data_gen import generate_hmm_data
from regime_switching_bayesian.inference import fit
from regime_switching_bayesian.model import build_model


def test_model_builds():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2)
    assert isinstance(model, pm.Model)


def test_logp_finite():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2)
    logp_fn = model.compile_logp()
    logp_val = logp_fn(model.initial_point())
    assert np.isfinite(logp_val), f"Log-probability is not finite: {logp_val}"


def test_parameter_names():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2)
    rv_names = {rv.name for rv in model.free_RVs}

    assert "P" in rv_names
    assert "mu" in rv_names
    assert any("chol_cov_0" in name for name in rv_names)
    assert any("chol_cov_1" in name for name in rv_names)


def test_order_means_adds_potentials():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2, order_means=True)
    potential_names = [p.name for p in model.potentials]
    assert "mu_order_0" in potential_names


def test_order_means_logp_finite():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2, order_means=True)
    logp_fn = model.compile_logp()
    logp_val = logp_fn(model.initial_point())
    assert np.isfinite(logp_val), f"Log-probability is not finite: {logp_val}"


def test_order_means_constrains_posterior():
    data = generate_hmm_data(T=30, K=2, d=2, seed=42)
    model = build_model(data["returns"], K=2, order_means=True)
    idata = fit(model, draws=50, tune=50, chains=1, seed=42)
    mu = idata.posterior["mu"].values  # (1, 50, 2, 2)
    assert np.all(mu[:, :, 1, 0] > mu[:, :, 0, 0]), (
        "order_means constraint violated: mu[1,0] not > mu[0,0] for all draws"
    )
