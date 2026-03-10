# PyMC Implementation Specification v0

**Version:** 0.1
**Date:** March 2026
**Scope:** Implementation plan for the v0 regime-switching model in PyMC

---

## 1. Architecture Decision: How to Handle the Discrete Latent States

The core challenge is that our model has discrete latent variables (regime states s_t) that NUTS cannot sample directly. There are three viable approaches in the PyMC ecosystem. We evaluate each and recommend a primary + fallback strategy.

### Approach A: `DiscreteMarkovChain` + `pmx.marginalize` (recommended primary)

`pymc_extras` provides `DiscreteMarkovChain` for Markov chain sequences and a `marginalize()` function that can automatically integrate out the discrete chain using a built-in forward algorithm (`MarginalDiscreteMarkovChainRV`). After sampling, `pmx.recover_marginals()` recovers the discrete states.

```python
import pymc as pm
import pymc_extras as pmx

with pm.Model() as model:
    # ... define priors ...
    chain = pmx.DiscreteMarkovChain("regime", P=P, init_dist=init_dist, shape=(T,))
    obs = pm.MvNormal("obs", mu=mu[chain], chol=L[chain], observed=data)

model_marginalized = pmx.marginalize(model, ["regime"])

with model_marginalized:
    idata = pm.sample()

idata = pmx.recover_marginals(model_marginalized, idata)
```

**Advantages:**
- Minimal custom code; the forward algorithm is handled internally.
- `recover_marginals` provides both sampled states and log-probabilities per regime.
- Maintained by the PyMC team; tested and optimized.

**Risks:**
- The marginalization machinery requires that the computational graph connecting the marginalized variable to observed variables is composed of element-wise operations. Multivariate emissions indexed by regime state may or may not satisfy this constraint.
- `DiscreteMarkovChain` marginalization is restricted to `n_lags=1` and a 2D (non-batched) transition matrix, which is fine for our model but limits future extensions.

**Mitigation:** If the multivariate emission indexing fails marginalization, fall back to Approach B.

### Approach B: Manual forward algorithm with `pytensor.scan` + `pm.Potential` (recommended fallback)

Write the log-space forward algorithm explicitly in PyTensor. Add the marginal log-likelihood to the model via `pm.Potential`. PyTensor computes gradients automatically via autodiff.

```python
import pytensor.tensor as pt
from pytensor.tensor.extra_ops import logsumexp

def forward_algorithm(data, log_pi0, log_P, mus, chols, K):
    """Compute marginal log-likelihood via forward algorithm."""

    def log_emission(z_t, k):
        return pm.logp(pm.MvNormal.dist(mu=mus[k], chol=chols[k]), z_t)

    log_lik_matrix = pt.stack(
        [pt.stack([log_emission(data[t], k) for k in range(K)]) for t in range(T)]
    )
    # ... but T is dynamic, so use scan instead (see Section 5) ...
```

This approach is modeled on the official PyMC example ["How to wrap a JAX function for use in PyMC"](https://www.pymc.io/projects/examples/en/stable/howto/wrapping_jax_function.html), which implements exactly this pattern for a univariate HMM. We adapt it to multivariate Normal emissions.

**Advantages:**
- Full control over the forward recursion.
- No dependency on `pymc_extras` marginalization internals.
- Gradients computed automatically by PyTensor.

**Disadvantages:**
- More code to write and test.
- `pytensor.scan` can be slow to compile for long sequences.

### Approach C: JAX-wrapped forward algorithm (future optimization)

Write the forward algorithm in JAX using `jax.lax.scan`, wrap it in a custom PyTensor `Op`, and provide gradients via `jax.grad`. This enables the NumPyro NUTS sampler for potentially faster sampling. Deferred to a later optimization pass.

### Why NOT `pymc_extras.statespace`

The `PyMCStateSpace` module in `pymc_extras` is built exclusively for **Linear Gaussian** state space models. It uses Kalman filters under the hood, which assume:
- Continuous latent states (not discrete regimes).
- Linear dynamics (not regime-switching).

Our model has discrete latent states that switch between emission distributions. This is fundamentally nonlinear and incompatible with the Kalman filter framework. The statespace module is **not applicable** for v0 or any regime-switching variant.

It could potentially be used for the full model's measurement model (Kalman filter for latent private returns conditioned on a fixed regime), but that is far beyond v0 scope.

---

## 2. Implementation Strategy

We attempt Approach A first. If it works, we use it. If the marginalization fails due to multivariate emission constraints, we implement Approach B. The rest of this spec covers both approaches.

---

## 3. Module Structure

```
src/
    __init__.py
    data_gen.py       # Synthetic data generation from the generative model
    model.py          # PyMC model definition (priors + forward algorithm + likelihood)
    inference.py      # NUTS sampling, FFBS regime recovery, diagnostics
    plotting.py       # Regime probability plots, posterior summaries
```

---

## 4. Data Generation (`data_gen.py`)

Synthetic data is essential for validating that the model can recover known parameters. The data generation function implements the generative model from `math_spec_v0.md` Section 8.

### Interface

```python
def generate_hmm_data(
    T: int,
    K: int,
    d: int,
    mus: np.ndarray,       # (K, d) regime means
    sigmas: np.ndarray,    # (K, d) per-asset standard deviations
    corr_chols: np.ndarray,# (K, d, d) Cholesky factors of correlation matrices
    P: np.ndarray,         # (K, K) transition matrix
    pi0: np.ndarray,       # (K,) initial distribution
    seed: int = 42,
) -> dict:
    """
    Returns dict with keys:
        "returns": np.ndarray (T, d) - observed returns
        "regimes": np.ndarray (T,)   - true regime labels
        "params": dict               - all true parameters for validation
    """
```

### Logic

1. Build full covariance matrices: `Sigma_k = diag(sigmas[k]) @ corr_chols[k] @ corr_chols[k].T @ diag(sigmas[k])`.
2. Sample `s_1 ~ Categorical(pi0)`.
3. For `t = 2, ..., T`: sample `s_t ~ Categorical(P[s_{t-1}])`.
4. For `t = 1, ..., T`: sample `z_t ~ MVN(mus[s_t], Sigma_{s_t})`.
5. Return observations and ground truth.

### Default Test Configuration

```python
DEFAULT_CONFIG = dict(
    T=120,
    K=2,
    d=3,
    mus=np.array([[0.01, 0.008, 0.012],     # Bull: ~1% monthly
                  [-0.005, -0.008, -0.003]]), # Bear: ~-0.5% monthly
    sigmas=np.array([[0.04, 0.035, 0.045],   # Bull: ~4% monthly vol
                     [0.08, 0.09, 0.10]]),    # Bear: ~9% monthly vol
    P=np.array([[0.95, 0.05],                # Bull persists
                [0.10, 0.90]]),              # Bear persists
    pi0=np.array([0.8, 0.2]),
    # corr_chols: identity (uncorrelated) as default, override for tests
)
```

---

## 5. PyMC Model Definition (`model.py`)

### 5A. Approach A: `DiscreteMarkovChain` + `marginalize`

```python
import pymc as pm
import pymc_extras as pmx
import numpy as np
import pytensor.tensor as pt

def build_model_marginalized(data: np.ndarray, K: int = 2) -> pm.Model:
    T, d = data.shape

    with pm.Model(coords={"regime": range(K), "asset": range(d), "time": range(T)}) as model:

        # --- Regime dynamics ---
        P = pm.Dirichlet("P", a=_sticky_alpha(K), dims=("regime", "regime_to"))
        init_dist = pm.Categorical.dist(p=np.ones(K) / K)
        regime = pmx.DiscreteMarkovChain("regime_seq", P=P, init_dist=init_dist, shape=(T,))

        # --- Emission parameters ---
        mu = pm.Normal("mu", mu=0.0, sigma=0.05, dims=("regime", "asset"))
        sigma = pm.HalfNormal("sigma", sigma=0.10, dims=("regime", "asset"))
        chol_corr, corr, _ = pm.LKJCholeskyCov(
            "chol_cov",
            n=d,
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(sigma=0.10),
            compute_corr=True,
        )
        # NOTE: LKJCholeskyCov returns a single (d, d) Cholesky factor.
        # For K regimes, we need K separate covariance structures.
        # This requires K calls to LKJCholeskyCov or manual construction.
        # See Section 5C for the per-regime covariance pattern.

        # --- Observation likelihood ---
        # Index emission params by regime state at each time step
        obs = pm.MvNormal("obs", mu=mu[regime], chol=chol[regime], observed=data)

    model_marg = pmx.marginalize(model, ["regime_seq"])
    return model_marg
```

**Key concern:** The `mu[regime]` and `chol[regime]` indexing must pass through `pmx.marginalize` successfully. If the marginalization engine cannot handle the multivariate indexing pattern, we fall back to Approach B.

### 5B. Approach B: Manual Forward Algorithm

```python
import pymc as pm
import pytensor.tensor as pt
import numpy as np

def build_model_forward(data: np.ndarray, K: int = 2) -> pm.Model:
    T, d = data.shape
    data_pt = pt.as_tensor_variable(data)

    with pm.Model() as model:

        # --- Regime dynamics ---
        P = pm.Dirichlet("P", a=_sticky_alpha(K), shape=(K, K))
        pi0 = pm.Dirichlet("pi0", a=np.ones(K))
        log_P = pt.log(P)
        log_pi0 = pt.log(pi0)

        # --- Emission parameters (per regime) ---
        mu = pm.Normal("mu", mu=0.0, sigma=0.05, shape=(K, d))

        # Per-regime covariance via LKJCholeskyCov (see Section 5C)
        chol_covs = []
        for k in range(K):
            chol_cov_k, _, _ = pm.LKJCholeskyCov(
                f"chol_cov_{k}", n=d, eta=2.0,
                sd_dist=pm.HalfNormal.dist(sigma=0.10),
                compute_corr=True,
            )
            chol_covs.append(chol_cov_k)
        chol_stack = pt.stack(chol_covs)  # (K, d, d)

        # --- Log-emission matrix: (T, K) ---
        log_lik = _compute_log_emissions(data_pt, mu, chol_stack, K, T, d)

        # --- Forward algorithm ---
        log_alpha_init = log_pi0 + log_lik[0]

        def forward_step(log_lik_t, log_alpha_prev):
            log_alpha_t = pt.logsumexp(
                log_alpha_prev[:, None] + log_P, axis=0
            ) + log_lik_t
            return log_alpha_t

        log_alphas, _ = pytensor.scan(
            fn=forward_step,
            sequences=[log_lik[1:]],
            outputs_info=[log_alpha_init],
        )

        log_marginal_lik = pt.logsumexp(log_alphas[-1])
        pm.Potential("hmm_loglik", log_marginal_lik)

    return model
```

### 5C. Per-Regime Covariance Pattern

`pm.LKJCholeskyCov` does not natively support a "regime" batch dimension. For K=2 and small d, the simplest approach is K separate calls:

```python
chol_covs = []
for k in range(K):
    chol_k, _, _ = pm.LKJCholeskyCov(
        f"chol_cov_{k}", n=d, eta=2.0,
        sd_dist=pm.HalfNormal.dist(sigma=0.10),
        compute_corr=True,
    )
    chol_covs.append(chol_k)

chol_stack = pt.stack(chol_covs)  # shape (K, d, d)
```

This yields K * [d + d*(d-1)/2] parameters for the covariance structure. For K=2, d=3 this is 2 * [3 + 3] = 12 parameters.

An alternative is to manually construct the Cholesky factors using `pm.LKJCorr` + `pm.HalfNormal` with explicit regime dimensions, but the loop is clearer for v0.

### 5D. Computing Log-Emissions

The log-emission matrix `log_lik[t, k]` = log N(z_t | mu_k, Sigma_k) is needed for the forward algorithm. Two options:

**Option 1: Vectorized (preferred for small K)**

```python
def _compute_log_emissions(data, mu, chol_stack, K, T, d):
    log_liks = []
    for k in range(K):
        dist_k = pm.MvNormal.dist(mu=mu[k], chol=chol_stack[k])
        log_lik_k = pm.logp(dist_k, data)  # (T,)
        log_liks.append(log_lik_k)
    return pt.stack(log_liks, axis=1)  # (T, K)
```

**Option 2: Direct MVN log-pdf computation**

```python
def _mvn_logpdf(z, mu, chol):
    diff = z - mu
    solve = pt.slinalg.solve_triangular(chol, diff, lower=True)
    log_det = pt.sum(pt.log(pt.diag(chol)))
    d = z.shape[-1]
    return -0.5 * (d * pt.log(2 * np.pi) + 2 * log_det + pt.dot(solve, solve))
```

Option 1 is cleaner and delegates the density computation to PyMC's tested implementation.

---

## 6. Inference (`inference.py`)

### 6A. NUTS Sampling

```python
def fit(model: pm.Model, draws: int = 2000, tune: int = 2000,
        chains: int = 4, target_accept: float = 0.9, seed: int = 42) -> az.InferenceData:
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            init="jitter+adapt_diag",
        )
    return idata
```

**Sampler settings rationale:**
- `target_accept=0.9`: Higher than the default 0.8 to reduce divergences in this correlated posterior.
- `init="jitter+adapt_diag"`: Standard NUTS initialization; jittering avoids symmetric start points (helps with label switching).
- `chains=4`: Minimum for reliable R-hat diagnostics.

### 6B. Forward-Filter Backward-Sampler (FFBS)

FFBS is a post-hoc step. Given posterior samples of theta = {P, pi0, mu_k, Sigma_k}, it draws regime sequences from p(s_{1:T} | z_{1:T}, theta).

If using Approach A, `pmx.recover_marginals` handles this automatically.

If using Approach B, implement FFBS in NumPy (not PyTensor -- this runs once per posterior sample, not during NUTS):

```python
def ffbs(data: np.ndarray, P: np.ndarray, pi0: np.ndarray,
         mus: np.ndarray, chol_covs: np.ndarray) -> np.ndarray:
    """
    Forward-filter backward-sample for a single posterior draw.

    Parameters
    ----------
    data : (T, d)
    P : (K, K)
    pi0 : (K,)
    mus : (K, d)
    chol_covs : (K, d, d)

    Returns
    -------
    regimes : (T,) integer array of sampled regime labels
    """
    T, d = data.shape
    K = len(pi0)

    # Forward pass: compute log alpha_t(k)
    log_alpha = np.zeros((T, K))
    for k in range(K):
        log_alpha[0, k] = np.log(pi0[k]) + _mvn_logpdf_np(data[0], mus[k], chol_covs[k])

    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = (
                logsumexp(log_alpha[t-1] + np.log(P[:, k]))
                + _mvn_logpdf_np(data[t], mus[k], chol_covs[k])
            )

    # Backward sampling
    regimes = np.zeros(T, dtype=int)
    log_gamma_T = log_alpha[T-1] - logsumexp(log_alpha[T-1])
    regimes[T-1] = np.random.choice(K, p=np.exp(log_gamma_T))

    for t in range(T-2, -1, -1):
        log_gamma = log_alpha[t] + np.log(P[:, regimes[t+1]])
        log_gamma -= logsumexp(log_gamma)
        regimes[t] = np.random.choice(K, p=np.exp(log_gamma))

    return regimes


def run_ffbs(idata: az.InferenceData, data: np.ndarray) -> np.ndarray:
    """Run FFBS for every posterior draw. Returns (chains, draws, T) array."""
    posterior = idata.posterior
    n_chains, n_draws = posterior.dims["chain"], posterior.dims["draw"]
    T = data.shape[0]
    regime_samples = np.zeros((n_chains, n_draws, T), dtype=int)

    for c in range(n_chains):
        for i in range(n_draws):
            regime_samples[c, i] = ffbs(
                data,
                posterior["P"].values[c, i],
                posterior["pi0"].values[c, i],
                posterior["mu"].values[c, i],
                # Extract chol_cov per regime ...
            )
    return regime_samples
```

### 6C. Diagnostics Checklist

After sampling, verify:

1. **R-hat < 1.01** for all parameters. Use `az.rhat(idata)`.
2. **ESS > 400** (bulk and tail) for all parameters. Use `az.ess(idata)`.
3. **No divergences.** Check `idata.sample_stats.diverging.sum()`.
4. **Trace plots** show mixing across chains. Use `az.plot_trace(idata)`.
5. **Posterior recovery** (on synthetic data): true parameter values fall within 90% HDI. Use `az.plot_posterior(idata, ref_val=true_values)`.
6. **Regime recovery**: compare FFBS regime probabilities against true regime sequence. Compute accuracy (allowing for label permutation).

---

## 7. Label Switching

With K=2 symmetric regimes, NUTS may explore solutions where regime labels are permuted (what one chain calls "regime 0" another calls "regime 1"). This is a known identifiability issue in mixture and HMM models.

### Mitigation Strategies

**Strategy 1: Ordering constraint on means (recommended for v0)**

Impose an ordering on one component of the regime means:

```python
mu_ordered = pm.Normal("mu_ordered", mu=0, sigma=0.05, shape=(K,),
                        transform=pm.distributions.transforms.ordered)
```

This breaks the symmetry by requiring mu_0 < mu_1 (or vice versa). The prior is placed on the ordered space; PyMC handles the Jacobian.

Alternatively, use `pm.math.sort` on the first asset's mean and derive the regime ordering from that.

**Strategy 2: Post-hoc relabeling**

If ordering constraints are not used during sampling, relabel posterior samples post-hoc by sorting regimes by their mean return (e.g., regime 0 = lower mean = Bear).

**Strategy 3: Informative priors**

The sticky Dirichlet prior (alpha_diag = 20) already helps by encouraging persistent regimes, which reduces (but does not eliminate) label switching.

**Recommendation:** Use Strategy 1 (ordered means on the first asset) as the primary approach. It is simple, effective, and does not distort the posterior.

---

## 8. Practical Concerns

### Compilation Time

`pytensor.scan` (Approach B) can be slow to compile for large T. For T=120, expect 30-60 seconds of compilation before sampling begins. This is a one-time cost per model build.

### Sampling Time

For K=2, d=3, T=120 (21 parameters), expect:
- **Tune:** 1000-2000 iterations.
- **Draw:** 2000 iterations.
- **Time per chain:** 2-10 minutes depending on hardware and approach.
- **Total for 4 chains (parallel):** 2-10 minutes.

### Numerical Stability

The forward algorithm must use log-space arithmetic throughout. Never exponentiate log-alpha values during the recursion. The `logsumexp` function handles the numerically stable summation.

### Prior Sensitivity

Run a prior predictive check before sampling:

```python
with model:
    prior_pred = pm.sample_prior_predictive(samples=500)
```

Verify that prior predictive samples produce returns in a realistic range (roughly -20% to +20% monthly) and that regime durations are reasonable (not switching every month, not stuck forever).

---

## 9. Testing Plan

### Unit Tests

- `test_data_gen`: Generate synthetic data, verify shapes, verify regime counts, verify return statistics are regime-dependent.
- `test_forward_algorithm`: Compare manual forward algorithm output against brute-force enumeration for T=5, K=2 (small enough to enumerate all 2^5 = 32 paths).
- `test_ffbs`: Run FFBS on synthetic data with known parameters, verify regime recovery accuracy > 80%.
- `test_model_builds`: Verify the PyMC model compiles and `model.point_logps()` returns finite values at the initial point.

### Integration Test

- Generate synthetic data with known parameters.
- Fit the model.
- Verify posterior 90% HDI contains all true parameter values.
- Verify FFBS regime sequence matches true regimes with > 80% accuracy.
- Run with `small_config` (T=20, K=2, d=2) for speed.

---

## 10. Implementation Order

| Step | What | File | Depends on |
|------|------|------|------------|
| 1 | Data generation + tests | `data_gen.py` | -- |
| 2 | Forward algorithm (Approach B) | `model.py` | Step 1 |
| 3 | Full PyMC model | `model.py` | Step 2 |
| 4 | Try Approach A (`marginalize`) | `model.py` | Step 3 |
| 5 | NUTS sampling wrapper | `inference.py` | Step 3 |
| 6 | FFBS implementation | `inference.py` | Step 5 |
| 7 | Diagnostics + plotting | `plotting.py` | Step 5 |
| 8 | Integration test (synthetic recovery) | `tests/` | All |

Steps 2-3 vs 4 are alternatives: implement the manual forward algorithm first (guaranteed to work), then attempt the cleaner `marginalize` approach. If Approach A works, it replaces Steps 2-3.

---

## 11. Dependencies

All required packages are already in `pyproject.toml`:

- `pymc >= 5.0.0` -- model definition, NUTS sampling
- `pytensor >= 2.18.0` -- tensor operations, scan, autodiff
- `arviz >= 0.14.0` -- diagnostics, plotting, InferenceData
- `numpy`, `scipy` -- data generation, FFBS, utilities
- `matplotlib`, `seaborn` -- visualization

Additionally needed (add to `pyproject.toml` dependencies):

- `pymc_extras` (formerly `pymc-experimental`) -- for Approach A (`DiscreteMarkovChain`, `marginalize`, `recover_marginals`)

---

**End of PyMC Implementation Specification v0**
