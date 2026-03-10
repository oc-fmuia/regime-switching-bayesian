"""End-to-end integration test (Step 6)."""

import numpy as np
import pytest
from scipy import stats

from src.data_gen import generate_hmm_data
from src.inference import check_diagnostics, fit, run_ffbs
from src.model import build_model


@pytest.mark.slow
def test_synthetic_recovery():
    """End-to-end: generate data, fit, recover regimes, check recovery."""
    data = generate_hmm_data(T=60, K=2, d=2, seed=42)
    _, model_marg = build_model(data["returns"], K=2)

    # Single chain avoids cross-chain Rhat inflation from label switching.
    # For a thorough convergence analysis, run multiple chains with an
    # ordering constraint or post-hoc relabeling (future improvement).
    idata = fit(model_marg, draws=500, tune=500, chains=1, seed=42)
    diagnostics = check_diagnostics(idata)

    assert diagnostics["no_divergences"], (
        f"Got {diagnostics['n_divergences']} divergences"
    )

    regime_samples = run_ffbs(idata, data["returns"], seed=42, thin=1, verbose=False)
    assert regime_samples.shape == (1, 500, 60)

    modal_regimes = stats.mode(regime_samples.reshape(-1, 60), axis=0).mode
    if modal_regimes.ndim > 1:
        modal_regimes = modal_regimes[0]

    acc_direct = np.mean(modal_regimes == data["regimes"])
    acc_flipped = np.mean((1 - modal_regimes) == data["regimes"])
    accuracy = max(acc_direct, acc_flipped)

    assert accuracy > 0.75, f"Regime recovery accuracy too low: {accuracy:.2f}"
