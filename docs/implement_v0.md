# Implementation Plan v0

**Date:** March 2026
**Prerequisites:** `finance_spec_v0.md`, `math_spec_v0.md`, `pymc_spec_v0.md`, marginalization test results

---

## Resolved Design Decisions

These questions were open in `pymc_spec_v0.md` and are now settled by our empirical tests in `src/marginalization_engine/test_marginalization.py` (13/13 passing).

| Question | Answer | Evidence |
|----------|--------|----------|
| Can `pmx.marginalize` handle `MvNormal` emissions? | **Yes** | `TestMvNormalEmission` passed |
| Can it handle `MvNormal` + `LKJCholeskyCov`? | **Yes** | `TestMvNormalWithLKJ` passed |
| Can `pmx.recover_marginals` recover `DiscreteMarkovChain`? | **No** | Raises `NotImplementedError`; only supports `Bernoulli`, `Categorical`, `DiscreteUniform` |
| Does the manual forward algorithm (`pytensor.scan` + `pm.Potential`) work? | **Yes** | `TestManualForwardAlgorithm` passed, logp matches NumPy reference |
| Is `pymc_extras.statespace` applicable? | **No** | Linear Gaussian only; incompatible with discrete regime-switching |

**Chosen approach:**
- Model definition: **Approach A** (`pmx.DiscreteMarkovChain` + `pmx.marginalize` + `pm.MvNormal`)
- Regime recovery: **Custom FFBS in NumPy** (since `recover_marginals` does not support `DiscreteMarkovChain`)

---

## File Layout

```
src/
    __init__.py               (exists, empty)
    data_gen.py               Step 1
    model.py                  Step 2
    inference.py              Step 3 + Step 4
    plotting.py               Step 5
tests/
    conftest.py               (exists, update fixtures)
    test_data_gen.py          Step 1
    test_model.py             Step 2
    test_inference.py         Step 3 + Step 4
    test_integration.py       Step 6
notebooks/
    regime_switching_v0.py    Step 7 (marimo)
```

---

## Step 1: Synthetic Data Generation

**File:** `src/data_gen.py`
**Test:** `tests/test_data_gen.py`

### Function signature

```python
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
```

### Returns

```python
{
    "returns": np.ndarray,    # (T, d) observed returns
    "regimes": np.ndarray,    # (T,) true regime labels (int)
    "params": {
        "mus": np.ndarray,        # (K, d)
        "sigmas": np.ndarray,     # (K, d)
        "corr_chols": np.ndarray, # (K, d, d) Cholesky of correlation matrices
        "covs": np.ndarray,       # (K, d, d) full covariance matrices
        "P": np.ndarray,          # (K, K)
        "pi0": np.ndarray,        # (K,)
    },
    "config": {"T": int, "K": int, "d": int, "seed": int},
}
```

### Default parameters

When `None` is passed, use these defaults (from `finance_spec_v0.md` Section 3):

```python
mus = np.array([
    [0.01, 0.008, 0.012],      # Bull: ~1% monthly
    [-0.005, -0.008, -0.003],   # Bear: ~-0.5% monthly
])
sigmas = np.array([
    [0.04, 0.035, 0.045],       # Bull: ~4% monthly vol
    [0.08, 0.09, 0.10],         # Bear: ~9% monthly vol
])
corr_chols = np.stack([np.eye(d), np.eye(d)])  # uncorrelated
P = np.array([[0.95, 0.05],     # Bull persists
              [0.10, 0.90]])     # Bear persists
pi0 = np.array([0.8, 0.2])
```

### Logic

1. `covs[k] = diag(sigmas[k]) @ corr_chols[k] @ corr_chols[k].T @ diag(sigmas[k])`
2. `regimes[0] = rng.choice(K, p=pi0)`
3. For `t = 1..T-1`: `regimes[t] = rng.choice(K, p=P[regimes[t-1]])`
4. For `t = 0..T-1`: `returns[t] = rng.multivariate_normal(mus[regimes[t]], covs[regimes[t]])`

### Tests

- `test_shapes`: verify `returns.shape == (T, d)` and `regimes.shape == (T,)`
- `test_regimes_valid`: all regime labels in `{0, ..., K-1}`
- `test_deterministic_seed`: calling twice with same seed produces identical output
- `test_regime_means_differ`: group returns by regime, verify mean difference is statistically significant (t-test, p < 0.05 for T >= 100)

---

## Step 2: PyMC Model Definition

**File:** `src/model.py`
**Test:** `tests/test_model.py`

### Function signature

```python
def build_model(
    data: np.ndarray,
    K: int = 2,
    sticky_alpha_diag: float = 20.0,
    sticky_alpha_offdiag: float = 2.0,
    mu_prior_sigma: float = 0.05,
    lkj_eta: float = 2.0,
    sigma_prior_sigma: float = 0.10,
) -> tuple[pm.Model, pm.Model]:
    """
    Returns (unmarginalized_model, marginalized_model).
    The unmarginalized model is kept for reference; the marginalized one is used for sampling.
    """
```

### Model graph (exact PyMC code)

```python
import pymc as pm
import pymc_extras as pmx
import pytensor.tensor as pt
import numpy as np

def build_model(data, K=2, sticky_alpha_diag=20.0, sticky_alpha_offdiag=2.0,
                mu_prior_sigma=0.05, lkj_eta=2.0, sigma_prior_sigma=0.10):
    T, d = data.shape

    sticky_alpha = np.full((K, K), sticky_alpha_offdiag)
    np.fill_diagonal(sticky_alpha, sticky_alpha_diag)

    with pm.Model() as model:
        # Regime dynamics
        P = pm.Dirichlet("P", a=sticky_alpha, shape=(K, K))
        init_dist = pm.Categorical.dist(p=np.ones(K) / K)
        chain = pmx.DiscreteMarkovChain("chain", P=P, init_dist=init_dist, shape=(T,))

        # Regime means
        mu = pm.Normal("mu", mu=0.0, sigma=mu_prior_sigma, shape=(K, d))

        # Per-regime covariance via LKJCholeskyCov
        chols = []
        for k in range(K):
            chol_k, _, _ = pm.LKJCholeskyCov(
                f"chol_cov_{k}", n=d, eta=lkj_eta,
                sd_dist=pm.HalfNormal.dist(sigma=sigma_prior_sigma),
                compute_corr=True,
            )
            chols.append(chol_k)
        chol_stack = pt.stack(chols)  # (K, d, d)

        # Observation
        pm.MvNormal("obs", mu=mu[chain], chol=chol_stack[chain], observed=data)

    model_marg = pmx.marginalize(model, ["chain"])
    return model, model_marg
```

### Label switching

Not addressed in v0. The sticky Dirichlet prior provides partial mitigation. If label switching is observed during integration testing, add an ordered constraint on `mu[:, 0]` (the first asset's mean across regimes). Defer to Step 6 if needed.

### Tests

- `test_model_builds`: `build_model(data)` returns two models without error.
- `test_marginalized_logp_finite`: `model_marg.compile_logp()(model_marg.initial_point())` is finite.
- `test_parameter_names`: verify `model_marg.free_RVs` contains `P`, `mu`, `chol_cov_0`, `chol_cov_1` (and not `chain`).

---

## Step 3: NUTS Sampling

**File:** `src/inference.py`
**Test:** `tests/test_inference.py`

### Function signature

```python
def fit(
    model: pm.Model,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 42,
) -> az.InferenceData:
```

### Implementation

```python
import arviz as az
import pymc as pm

def fit(model, draws=2000, tune=2000, chains=4, target_accept=0.9, seed=42):
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=seed,
            init="jitter+adapt_diag",
        )
    return idata
```

### Diagnostics function

```python
def check_diagnostics(idata: az.InferenceData) -> dict[str, bool]:
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
```

### Tests

- `test_fit_runs`: call `fit(model_marg, draws=50, tune=50, chains=1)` on small synthetic data; verify `idata.posterior` contains expected variables.
- `test_diagnostics_on_good_run`: check that `check_diagnostics` returns a dict with expected keys.

---

## Step 4: FFBS Regime Recovery

**File:** `src/inference.py` (same file as Step 3)
**Test:** `tests/test_inference.py`

### Why custom FFBS

`pmx.recover_marginals` does not support `DiscreteMarkovChain` (raises `NotImplementedError`). We implement FFBS ourselves in NumPy. This runs post-hoc on each posterior sample, not inside NUTS.

### Function signatures

```python
def ffbs_single(
    data: np.ndarray,
    P: np.ndarray,
    mu: np.ndarray,
    chol_covs: np.ndarray,
    pi0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Forward-filter backward-sample for one posterior draw.

    Parameters
    ----------
    data : (T, d)
    P : (K, K) transition matrix
    mu : (K, d) regime means
    chol_covs : (K, d, d) Cholesky factors of covariance matrices
    pi0 : (K,) initial distribution. If None, use stationary distribution of P.
    rng : random generator

    Returns
    -------
    regimes : (T,) integer array
    """
```

```python
def run_ffbs(
    idata: az.InferenceData,
    data: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """
    Run FFBS across all posterior draws.

    Returns
    -------
    regime_samples : (n_chains, n_draws, T) integer array
    """
```

### FFBS algorithm (NumPy)

**Forward pass** -- compute `log_alpha[t, k]` for all t, k:

```python
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

log_alpha = np.zeros((T, K))
for k in range(K):
    cov_k = chol_covs[k] @ chol_covs[k].T
    log_alpha[0, k] = np.log(pi0[k]) + multivariate_normal.logpdf(data[0], mu[k], cov_k)

for t in range(1, T):
    for k in range(K):
        log_alpha[t, k] = (
            logsumexp(log_alpha[t-1] + np.log(P[:, k]))
            + multivariate_normal.logpdf(data[t], mu[k], cov_k)
        )
```

**Backward sampling:**

```python
regimes = np.zeros(T, dtype=int)
log_gamma = log_alpha[T-1] - logsumexp(log_alpha[T-1])
regimes[T-1] = rng.choice(K, p=np.exp(log_gamma))

for t in range(T-2, -1, -1):
    log_gamma = log_alpha[t] + np.log(P[:, regimes[t+1]])
    log_gamma -= logsumexp(log_gamma)
    regimes[t] = rng.choice(K, p=np.exp(log_gamma))
```

### Extracting Cholesky factors from idata

`pm.LKJCholeskyCov` stores the packed Cholesky factor. We need to unpack per regime:

```python
def _extract_chol_cov(idata, chain_idx, draw_idx, k, d):
    """Extract the (d, d) Cholesky factor for regime k from a posterior draw."""
    chol_packed = idata.posterior[f"chol_cov_{k}_cholesky-cov-packed__"].values[chain_idx, draw_idx]
    chol = np.zeros((d, d))
    chol[np.tril_indices(d)] = chol_packed
    return chol
```

Alternatively, check what variable names PyMC stores and adapt. The exact name depends on `pm.LKJCholeskyCov` internals -- verify during implementation by inspecting `idata.posterior.data_vars`.

### Stationary distribution as default pi0

If `pi0` is not an explicit model parameter (Approach A doesn't expose it), compute the stationary distribution from `P`:

```python
def stationary_distribution(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    return pi / pi.sum()
```

### Tests

- `test_ffbs_single_known_params`: run `ffbs_single` with the true generating parameters on synthetic data. Compare recovered regimes to true regimes; require accuracy > 80% (allowing label permutation).
- `test_ffbs_correctness_small`: for T=5, K=2, enumerate all 32 regime paths, compute exact posterior, and verify FFBS samples are consistent with the exact distribution (chi-squared test on marginal regime probabilities at each t, over many FFBS draws).
- `test_run_ffbs_shapes`: verify output shape is `(n_chains, n_draws, T)`.

---

## Step 5: Plotting

**File:** `src/plotting.py`
**Test:** none (visual inspection only)

### Functions

```python
def plot_regime_probabilities(
    regime_samples: np.ndarray,
    true_regimes: np.ndarray | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot P(s_t = k) over time as filled bands.
    Overlay true regime labels if provided.
    """

def plot_posterior_summary(
    idata: az.InferenceData,
    true_params: dict | None = None,
) -> matplotlib.figure.Figure:
    """
    Forest plot or trace plot of key parameters.
    Mark true values if provided.
    """
```

---

## Step 6: Integration Test

**File:** `tests/test_integration.py`

### Test

```python
def test_synthetic_recovery():
    """End-to-end: generate data, fit, recover regimes, check recovery."""
    from src.data_gen import generate_hmm_data
    from src.model import build_model
    from src.inference import fit, run_ffbs, check_diagnostics

    # Small config for speed
    data = generate_hmm_data(T=60, K=2, d=2, seed=42)
    _, model_marg = build_model(data["returns"], K=2)

    idata = fit(model_marg, draws=500, tune=500, chains=2, seed=42)
    diagnostics = check_diagnostics(idata)

    assert diagnostics["no_divergences"]
    assert diagnostics["rhat_ok"]

    regime_samples = run_ffbs(idata, data["returns"], seed=42)
    assert regime_samples.shape == (2, 500, 60)

    # Check regime recovery accuracy (allowing label permutation)
    modal_regimes = scipy.stats.mode(regime_samples.reshape(-1, 60), axis=0).mode[0]
    acc_direct = np.mean(modal_regimes == data["regimes"])
    acc_flipped = np.mean((1 - modal_regimes) == data["regimes"])
    accuracy = max(acc_direct, acc_flipped)

    assert accuracy > 0.75, f"Regime recovery accuracy too low: {accuracy:.2f}"
```

---

## Step 7: Illustrative Marimo Notebook

**File:** `notebooks/regime_switching_v0.py`
**Plan:** `docs/implement_tutorial_notebook_marimo_v0.md`

A full-pipeline marimo notebook that walks a reader through data generation, model specification, NUTS fitting, FFBS regime recovery, and posterior interpretation. Written in a "blog post" voice with narrative prose between every code cell. Imports all logic from `src.*` -- no duplication. Designed to be converted into an educative blog post later.

See `implement_tutorial_notebook_marimo_v0.md` for the detailed section outline and conventions.

---

## Implementation Order

| Step | What | File(s) | Depends on | Estimated effort |
|------|------|---------|------------|------------------|
| 1 | Data generation + tests | `data_gen.py`, `test_data_gen.py` | -- | Small |
| 2 | PyMC model + tests | `model.py`, `test_model.py` | Step 1 | Medium |
| 3 | NUTS sampling + diagnostics | `inference.py`, `test_inference.py` | Step 2 | Small |
| 4 | FFBS + tests | `inference.py`, `test_inference.py` | Step 3 | Medium |
| 5 | Plotting | `plotting.py` | Step 3 | Small |
| 6 | Integration test | `test_integration.py` | All | Medium |
| 7 | Tutorial notebook | `notebooks/regime_switching_v0.py` | Step 6 | Medium |

Steps 1-4 are the critical path. Step 5 is nice-to-have. Step 6 is the acceptance gate. Step 7 is built last, once everything works end-to-end.

---

**End of Implementation Plan v0**
