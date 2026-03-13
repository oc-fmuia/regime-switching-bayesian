# Regime-Switching Bayesian Model: Implementation Plan

## Strategy & Parameter Tuning

**Chosen Strategy:** Strategy 1 (Marginalized HMM with forward algorithm)

**Parameter Scaling for Reasonable Runtime (~10 min per chain):**
```
T = 120 months
K = 3 regimes
G = 2 geographies (reduced from 2)
H = 2 sectors (reduced from 3)
d_pub = 4 (reduced from 6)
d_priv = 3 (reduced from 4)
d = 7 total
r = 2 factors
Q = floor(T/3) = 40 quarters
```

**Expected runtime:**
- Each forward pass: ~5k operations (K² × T × d²)
- NUTS iterations: ~500-1000 per chain
- Total per chain: ~8 minutes
- 4 chains: ~20 minutes wall-clock (parallel)

---

## Project Structure

```
regime-switching-bayesian/
├── src/
│   ├── __init__.py
│   ├── data_gen.py          # Synthetic data generation
│   ├── model.py             # PyMC model definitions
│   ├── infer.py             # NUTS sampling + FFBS
│   ├── scenarios.py         # Scenario simulation
│   ├── filter_update.py     # Real-time filtering
│   ├── reporting.py         # PPC, plots, summaries
│   ├── utils.py             # Helper functions
│   └── test_*.py            # Unit tests per module
├── tests/
│   └── conftest.py          # pytest fixtures
├── main.py                  # Orchestration script
├── docs/
│   ├── math_spec.md         # Complete mathematical spec
│   ├── finance_spec.md      # Finance interpretation
│   └── implementation_notes.md  # Design decisions
├── results/                 # Output folder (generated)
├── .gitignore
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Commit Plan: 18 Small, Testable Commits

### Phase 1: Data Generation & Synthetic Truth (Commits 1–4)

**Commit 1: Project scaffold + metadata generation**
- Create directory structure
- Implement `src/utils.py` with metadata mapping functions
- Function: `generate_metadata(T, K, G, H, d_pub, d_priv, seed)`
  - Returns: geography g(ℓ), sector h(ℓ), type(ℓ) for each series
- Test: Assert shapes, no duplicates, all series mapped
- Files: `src/utils.py`, `tests/test_utils.py`, `pyproject.toml`, `.gitignore`

**Commit 2: True parameter sampling**
- `src/data_gen.py` → `sample_true_parameters(metadata, seed)`
- Sample: π0, P, ν_k, m_k, σ_a,k, σ_b,k, σ_u,k, β_k, σ_β,k, σ_βg,k, σ_βgh,k, σ_η,k, α_k, τ_k, φ, μ_r, σ_r
- Return: `TrueParams` dataclass
- Test: Assert all params within priors, P rows sum to 1, ν > 2, φ ∈ [0,1)
- Files: `src/data_gen.py`, `tests/test_data_gen_params.py`

**Commit 3: Regime path generation**
- Add to `src/data_gen.py` → `generate_regime_path(pi0, P, T, seed)`
- Sample s_1 ~ Categorical(π0), then s_t | s_{t-1} ~ Categorical(P[s_{t-1}])
- Return: s[1:T]
- Test: Assert s ∈ {1..K}, compute empirical transition frequency matches P
- Files: `src/data_gen.py`, `tests/test_data_gen_regimes.py`

**Commit 4: Monthly return generation**
- Add to `src/data_gen.py` → `generate_monthly_returns(metadata, theta_true, s_path, seed)`
- For each month t: sample z_t ~ MVT(μ_{s_t}, Σ_{s_t}, ν_{s_t})
  - Implement hierarchical μ computation
  - Implement factor covariance Σ = B B^T + D
- Split z into r_pub, x_priv
- Return: r_pub[1:T, d_pub], x_priv[1:T, d_priv]
- Test: Assert shapes, check mean/cov of z match regime parameters, assert StudentT tail behavior
- Files: `src/data_gen.py`, `tests/test_data_gen_returns.py`

### Phase 2: Measurement Model (Commits 5–6)

**Commit 5: Quarterly aggregation + smoothing**
- Add to `src/data_gen.py` → `aggregate_quarterly(x_priv, phi, R, seed)`
- Compute \tilde{y}_q = sum of 3 months of x_priv
- Apply smoothing: y_q = φ y_{q-1} + (1-φ) \tilde{y}_q + ε_q
- Handle initialization (y_0 = 0 for simplicity, document)
- Optionally drop quarters (Q_obs < Q) and apply lag L
- Return: y_priv[1:Q_obs, d_priv], obs_metadata (which q observed, at which month)
- Test: Assert Q_obs ≤ Q, check y shape, verify smoothing equation holds
- Files: `src/data_gen.py`, `tests/test_data_gen_quarterly.py`

**Commit 6: End-to-end synthetic data generation**
- Add to `src/data_gen.py` → `generate_synthetic_data(config, seed)` (master function)
- Orchestrate: metadata → params → regimes → monthly → quarterly
- Save to disk (CSV format) for reproducibility
- Return: `SyntheticData` namedtuple with all components
- Test: Full round-trip with known seed, compare saved files
- Files: `src/data_gen.py`, `tests/test_data_gen_full.py`

### Phase 3: Model Definition (Commits 7–12)

**Commit 7: Hierarchical mean model**
- `src/model.py` → `build_hierarchical_mean(pm_model, metadata, d, K, priors_dict)`
- Build: m_k, a_{k,g}, b_{k,g,h}, u_{k,ℓ}
- Compute: μ_{k,ℓ} = m_k + a_{k,g(ℓ)} + b_{k,g(ℓ),h(ℓ)} + u_{k,ℓ}
- All priors explicit (use dict for easy configuration)
- Return: mu_k tensor [K, d]
- Test: Assert shape [K, d], check priors are applied, test specific asset formula
- Files: `src/model.py`, `tests/test_model_mean.py`

**Commit 8: Factor covariance model**
- `src/model.py` → `build_hierarchical_covariance(pm_model, metadata, d, K, r, priors_dict)`
- Build: β_k, β_{k,g}, β_{k,g,h}, η_{k,ℓ} → B_k ∈ R^{d×r}
- Build: log σ_{k,ℓ} → D_k (diagonal)
- Compute: Σ_k = B_k B_k^T + D_k
- Return: B_k [K, d, r], D_k [K, d]
- Test: Assert shapes, Σ is symmetric PSD, tail params ν > 2
- Files: `src/model.py`, `tests/test_model_covariance.py`

**Commit 9: Regime dynamics (HMM structure)**
- `src/model.py` → `build_regime_dynamics(pm_model, K, alpha_diag, alpha_offdiag)`
- Build: π0 ~ Dirichlet(α_init), P_i ~ Dirichlet(α_diag, α_offdiag)
- Return: pi0, P
- Test: Assert P rows sum to 1, priors match spec
- Files: `src/model.py`, `tests/test_model_regimes.py`

**Commit 10: Measurement model (quarterly)**
- `src/model.py` → `build_measurement_model(pm_model, d_priv, priors_dict)`
- Build: φ = logistic(ψ), ψ ~ Normal(μ_ψ, σ_ψ²)
- Build: R diagonal, log r_j ~ Normal(μ_r, σ_r²)
- Return: phi, R
- Test: Assert φ ∈ [0,1), R is PSD, priors correct
- Files: `src/model.py`, `tests/test_model_measurement.py`

**Commit 11: Forward algorithm (HMM likelihood)**
- `src/model.py` → `forward_algorithm(r_data, pi0, P, mu_k, Sigma_k, nu_k)`
- Implement marginal HMM likelihood using PyTensor.scan
  - Forward recursion α_t(k) = [α_{t-1} P^T] ⊙ L(r_t | k)
  - LogSumExp stabilization
  - Return log P(r_1:T | θ)
- Test: Small synthetic example (K=2, T=10), compare with numerical differentiation
- Files: `src/model.py`, `tests/test_model_forward_algorithm.py`

**Commit 12: Full PyMC model assembly**
- `src/model.py` → `build_full_model(data, metadata, config)`
- Assemble: regime dynamics + mean + covariance + measurement + forward likelihood
- Add: `pm.Potential('obs', loglik_forward_algorithm(...))`
- Return: PyMC model object
- Test: Model shape inference, test sampling 10 draws on tiny data (T=10)
- Files: `src/model.py`, `tests/test_model_full.py`

### Phase 4: Inference Engine (Commits 13–16)

**Commit 13: NUTS sampling**
- `src/infer.py` → `fit_model(model, draws=500, tune=500, chains=2, cores=4)`
- Wrap `pm.sample()` with sensible defaults
- Return: InferenceData
- Test: Sample on synthetic tiny data, check convergence metrics (Rhat < 1.1)
- Files: `src/infer.py`, `tests/test_infer_sampling.py`

**Commit 14: Forward-filter-backward-sampler (FFBS)**
- `src/infer.py` → `forward_filter_backward_sampler(idata, r_data, metadata, config)`
- Recover regime sequence s_1:T from posterior parameters and data
- Implement: forward filtering (α_t) → backward sampling (s_t | α_t, s_{t+1})
- Return: s_samples [M, T] (M posterior draws)
- Test: Compare with ground truth regimes on synthetic data
- Files: `src/infer.py`, `tests/test_infer_ffbs.py`

**Commit 15: Posterior summaries**
- `src/infer.py` → `summarize_posterior(idata, s_samples, metadata)`
- Compute: posterior mean/SD for P, ν_k, m_k, a_{k,g}, b, u, β, σ, φ, R
- Format as DataFrames (for CSV export)
- Return: dict of summaries
- Test: Assert no NaN, shapes match, values in prior support
- Files: `src/infer.py`, `tests/test_infer_summaries.py`

**Commit 16: Posterior predictive checks**
- `src/reporting.py` → `posterior_predictive_check(idata, s_samples, r_data, metadata, config)`
- Generate: PPC samples for r_pub from posterior
- Compute: moment comparisons (mean, std, kurtosis per regime)
- Generate: tail quantiles, cross-asset correlation per regime
- Return: dict of metrics + plots
- Test: PPC samples have right shape, statistics within reasonable range
- Files: `src/reporting.py`, `tests/test_reporting_ppc.py`

### Phase 5: Latent Variable Recovery & Validation (Commits 17–18)

**Commit 17: Recover latent private returns**
- `src/infer.py` → `recover_latent_private_returns(idata, y_obs, metadata, config)`
- For each posterior draw θ, run Kalman filter on measurement model
  - State: x_t^priv (latent monthly private returns)
  - Observation: y_q^priv (quarterly, smoothed)
  - Emission: y_q = φ y_{q-1} + (1-φ) \tilde{y}_q + ε_q
- Return: x_priv_posterior [M, T, d_priv]
- Test: Shape check, compare filtered with true x_priv on synthetic data
- Files: `src/infer.py`, `tests/test_infer_latent_recovery.py`

**Commit 18: Comparison with ground truth**
- `src/reporting.py` → `compare_with_truth(x_priv_posterior, x_priv_true, s_samples, s_true, metadata)`
- Compute: MSE, correlation, regime classification accuracy
- Generate: plots of inferred vs true (time series + scatter)
- Return: evaluation dict
- Test: On synthetic data, MSE should be small, regime classification high
- Files: `src/reporting.py`, `tests/test_reporting_comparison.py`

### Phase 6: Scenarios (Commits 19–21)

**Commit 19: Scenario simulator - baseline**
- `src/scenarios.py` → `simulate_baseline(idata, s_samples, metadata, T_current, H, seed)`
- Forward simulate: s_{T+1:T+H}, z_{T+1:T+H}, y_{q'}^priv (future quarters)
- Return: paths [M, H, d]
- Test: Shape check, regimes follow transition matrix, returns have correct stats
- Files: `src/scenarios.py`, `tests/test_scenarios_baseline.py`

**Commit 20: Scenario interventions**
- `src/scenarios.py` → `apply_scenario_intervention(baseline_sim, scenario_spec)`
- Implement 5 interventions:
  - A) Regime pinning
  - B) Mean shock (Δμ)
  - C) Covariance shock (scale B or D)
  - D) Tail shock (reduce ν)
  - E) Measurement shock (change φ or R)
- Return: modified simulation
- Test: Each intervention type modifies output correctly
- Files: `src/scenarios.py`, `tests/test_scenarios_interventions.py`

**Commit 21: Portfolio analytics**
- `src/scenarios.py` → `portfolio_analytics(sim_paths, weights, config)`
- Compute: cumulative return, VaR/CVaR, drawdown, return distribution
- Generate: comparison plots (baseline vs scenarios)
- Return: metrics dict + plots
- Test: VaR monotonicity, CVaR ≥ VaR, drawdown ≤ 0
- Files: `src/scenarios.py`, `tests/test_scenarios_analytics.py`

### Phase 7: Real-Time Filtering (Commits 22–23)

**Commit 22: Monthly state update**
- `src/filter_update.py` → `update_filter(idata, s_samples_prev, r_new, y_new_if_available, metadata, config)`
- Input: posterior θ samples from fitting
- Update regime: s_T | r_1:T (likelihood update for new r_T)
- Update latent private: x_T^priv | r_1:T, y_q_if_available
- Return: s_T_filtered [M], x_T_filtered [M, d_priv]
- Test: Filtering on extended synthetic data, compare with new full fit (should be similar)
- Files: `src/filter_update.py`, `tests/test_filter_update.py`

**Commit 23: Demonstration of filtering without refitting**
- `src/filter_update.py` → `demo_online_updates()`
- Start with fit at T=100
- Sequentially add months T+1..120 using filter_update
- Compare filtered estimates vs true values
- Test: Verify filter doesn't refit θ
- Files: `src/filter_update.py`, `tests/test_filter_demo.py`

### Phase 8: Integration & Documentation (Commits 24–26)

**Commit 24: Main orchestration script**
- `main.py` → `main()`
  - Call: generate synthetic data
  - Call: fit model
  - Call: PPC
  - Call: scenario simulations
  - Call: filtering demo
  - Save all outputs to results/
- Test: Full pipeline runs without error
- Files: `main.py`, `tests/test_main.py`

**Commit 25: Documentation - Mathematical Spec**
- Write `docs/math_spec.md`
  - All symbols defined
  - All distributions stated
  - Graphical model in text
  - Forward algorithm detailed
  - Measurement model detailed
- Test: Verify against code (spot-check equations)
- Files: `docs/math_spec.md`

**Commit 26: Documentation - Finance & README**
- Write `docs/finance_spec.md`
  - Regime interpretation (growth, crisis, recovery)
  - Private return smoothing (why quarterly reported ≠ monthly latent)
  - Quarterly reporting lag effects
  - Scenario interpretation (how each intervention maps to risk questions)
- Write `README.md`
  - Installation
  - How to run: `python main.py`
  - Output structure
  - Reproducibility (seed, parameter defaults)
- Files: `docs/finance_spec.md`, `README.md`

---

## Testing Strategy

**Each commit has:**
- Unit tests in `tests/test_*.py` (pytest format)
- Asserts on shapes, types, ranges
- Small synthetic examples where possible
- No dependency on other commits (can test independently)

**Test running:**
```bash
pytest tests/test_*.py -v
```

**Test data sizes (for speed):**
- Unit tests: T=10–20, K=2, d=3
- Integration tests: T=120, K=3, d=7 (full config)

---

## Commit Metadata Template

Each commit message should follow:
```
<type>: <short description>

<longer explanation if needed>

Files: <list of files touched>
Tests: <list of test files>
```

Types: `feat` (feature), `test` (test), `docs` (documentation), `refactor` (refactoring), `fix` (bug fix)

---

## Checkpoints for Review

- **After Commit 4:** Synthetic data generation complete + reviewable
- **After Commit 12:** Full PyMC model builds (no inference yet)
- **After Commit 16:** Inference + PPC working (core done)
- **After Commit 23:** Filtering working (online capability proven)
- **After Commit 26:** Everything documented + main.py works

---

## Runtime Expectations

| Phase | Commits | Expected Time |
|-------|---------|----------------|
| Data Gen | 1–6 | 30 min (mostly setup) |
| Model | 7–12 | 1 hour (careful PyTensor code) |
| Inference | 13–18 | 2 hours (test sampling, FFBS, recovery) |
| Scenarios | 19–21 | 45 min |
| Filtering | 22–23 | 1 hour |
| Integration | 24–26 | 45 min |
| **Total** | **26** | **~6 hours** |

Per-commit average: ~15 min implementation + testing.

---

This plan ensures:
✅ Small, reviewable commits  
✅ Each commit has tests  
✅ Dependencies are clear  
✅ Can pause/resume at any checkpoint  
✅ No silent changes (explicit plan)
