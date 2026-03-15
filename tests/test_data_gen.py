import numpy as np
from scipy import stats

from regime_switching_bayesian.data_gen import generate_hmm_data


def test_shapes():
    data = generate_hmm_data(T=100, K=2, d=3, seed=42)
    assert data["returns"].shape == (100, 3)
    assert data["regimes"].shape == (100,)
    assert data["params"]["covs"].shape == (2, 3, 3)


def test_regimes_valid():
    data = generate_hmm_data(T=100, K=2, d=3, seed=42)
    assert set(np.unique(data["regimes"])).issubset({0, 1})


def test_deterministic_seed():
    d1 = generate_hmm_data(T=50, K=2, d=3, seed=123)
    d2 = generate_hmm_data(T=50, K=2, d=3, seed=123)
    np.testing.assert_array_equal(d1["returns"], d2["returns"])
    np.testing.assert_array_equal(d1["regimes"], d2["regimes"])


def test_different_seeds_differ():
    d1 = generate_hmm_data(T=50, K=2, d=3, seed=1)
    d2 = generate_hmm_data(T=50, K=2, d=3, seed=2)
    assert not np.array_equal(d1["returns"], d2["returns"])


def test_regime_means_differ():
    data = generate_hmm_data(T=200, K=2, d=3, seed=42)
    r = data["returns"]
    regimes = data["regimes"]

    mask_0 = regimes == 0
    mask_1 = regimes == 1

    assert mask_0.sum() > 10 and mask_1.sum() > 10, "Not enough samples per regime"

    _, pval = stats.ttest_ind(r[mask_0, 0], r[mask_1, 0])
    assert pval < 0.10, f"Regime means not significantly different (p={pval:.4f})"


def test_config_stored():
    data = generate_hmm_data(T=60, K=2, d=4, seed=99)
    assert data["config"] == {"T": 60, "K": 2, "d": 4, "seed": 99, "hard_switch_at": None}


def test_covariance_positive_definite():
    data = generate_hmm_data(T=50, K=2, d=3, seed=42)
    for k in range(2):
        eigvals = np.linalg.eigvalsh(data["params"]["covs"][k])
        assert np.all(eigvals > 0), f"Covariance for regime {k} is not positive definite"


def test_hard_switch_regimes():
    data = generate_hmm_data(T=120, K=2, d=3, seed=42, hard_switch_at=60)
    regimes = data["regimes"]
    assert np.all(regimes[:60] == 0), "First half should be regime 0"
    assert np.all(regimes[60:] == 1), "Second half should be regime 1"


def test_hard_switch_returns_shape():
    data = generate_hmm_data(T=120, K=2, d=3, seed=42, hard_switch_at=60)
    assert data["returns"].shape == (120, 3)
