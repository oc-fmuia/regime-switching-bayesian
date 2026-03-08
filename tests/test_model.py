"""Tests for PyMC model definition (Step 2)."""

import numpy as np
import pymc as pm

from src.data_gen import generate_hmm_data
from src.model import build_model


def test_model_builds():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    model, model_marg = build_model(data["returns"], K=2)
    assert isinstance(model, pm.Model)
    assert isinstance(model_marg, pm.Model)


def test_marginalized_logp_finite():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    _, model_marg = build_model(data["returns"], K=2)
    logp_fn = model_marg.compile_logp()
    logp_val = logp_fn(model_marg.initial_point())
    assert np.isfinite(logp_val), f"Log-probability is not finite: {logp_val}"


def test_parameter_names():
    data = generate_hmm_data(T=20, K=2, d=2, seed=42)
    _, model_marg = build_model(data["returns"], K=2)
    rv_names = {rv.name for rv in model_marg.free_RVs}

    assert "P" in rv_names
    assert "mu" in rv_names
    assert any("chol_cov_0" in name for name in rv_names)
    assert any("chol_cov_1" in name for name in rv_names)
    assert "chain" not in rv_names
