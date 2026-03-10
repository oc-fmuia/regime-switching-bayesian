# Features to Be Implemented (Post-v0)

Features described in the math and finance specs that are deferred beyond the v0 blogpost release. Organized by theme, roughly ordered by priority within each section.

---

## Emission Model Extensions

### Student-t Emissions

Replace multivariate Normal emissions with multivariate Student-t to capture intra-regime fat tails. Financial returns exhibit excess kurtosis even within a single market regime; the current model can only produce portfolio-level fat tails through the regime mixture.

**Implementation sketch:**
- Add K parameters `nu_k ~ LogNormal(log(12), 0.4)` (degrees of freedom per regime)
- Replace MvNormal log-density with MvStudentT in the emission computation
- Forward algorithm is unchanged; only `_compute_emission_loglik` and `build_model_manual` emission block need modification
- Update `data_gen.py` to optionally generate from multivariate Student-t

**Model comparison angle:** enables a "Normal vs Student-t" comparison via one-step-ahead predictive density — *"Does allowing fat tails within regimes improve out-of-sample fit?"*

**Spec reference:** `math_spec.md` Section 5.

### Autoregressive Dynamics Within Regimes

Hamilton (1989) used AR(4) within each regime. The current model assumes i.i.d. emissions conditional on regime. Adding AR(p) structure would capture momentum/mean-reversion effects that vary by regime.

---

## Mean Structure

### Hierarchical Mean Decomposition

The full spec defines a 4-level hierarchy for regime-conditional means:

```
mu_{k,l} = m_k + a_{k,g(l)} + b_{k,g(l),h(l)} + u_{k,l}
```

where `g(l)` maps asset l to its geography and `h(l)` to its sector within that geography. This enables partial pooling across assets with shared characteristics.

**When needed:** becomes valuable when the asset universe grows beyond ~10 assets, where flat independent priors on each `mu_{k,l}` would be overparameterized.

**Spec reference:** `math_spec.md` Section 6.

---

## Covariance Structure

### Factor Model for Covariance

The full spec defines:

```
Sigma_k = B_k B_k^T + D_k
```

where `B_k` is a `(d, r)` factor loading matrix and `D_k` is a diagonal idiosyncratic variance matrix. This reduces covariance parameters from O(d^2) to O(d*r) when `r << d`.

**When needed:** for asset universes with d > ~10, full LKJ Cholesky becomes expensive and poorly identified. A factor model with r=2-3 factors provides a structured, lower-dimensional alternative.

**Spec reference:** `math_spec.md` Section 7.

### Shock Propagation Model

The original spec includes a deterministic shock factor structure:

```
r_t = mu_{s_t} + B_{s_t} u_t + epsilon_t
```

where `u_t` are common shock factors and `B_{s_t}` are regime-dependent loadings. This enables stress testing via counterfactual shock scenarios.

**Spec reference:** `model_comparison.md` Section 1.4.

---

## Transition Dynamics

### Time-Varying Transition Probabilities (TVTP)

Transition matrix depends on observable covariates (e.g., VIX, yield curve slope):

```
P_{ij}(t) = softmax(X_t * beta)
```

This would allow economic indicators to influence regime-switching probability, rather than assuming constant transition rates.

**Complexity:** significant model extension; requires replacing the Dirichlet prior on P rows with a regression structure and changes to the forward algorithm.

---

## Mixed-Frequency and Private Assets

### Private Asset Measurement Model

The full spec defines a latent monthly return process for private assets, observed only at quarterly frequency with smoothing and noise:

- Latent monthly: `x_t^(priv)` follows the same regime-switching dynamics
- Quarterly observation: `y_q^(priv) = phi * y_{q-1}^(priv) + (1-phi) * sum of latent monthly returns + noise`
- Smoothing parameter `phi` captures appraisal-based return smoothing
- Reporting lag `L` quarters

**Spec reference:** `math_spec.md` Section 8; `finance_spec.md` Sections 3-4.

### Mixed-Frequency Inference

Conditioning on quarterly private observations while inferring monthly latent states. Requires extending the forward algorithm with a measurement-model likelihood term at quarter boundaries.

---

## Portfolio Analytics and Scenario Framework

### Scenario Simulation Framework

Five intervention types from the finance spec:

1. **Regime pinning** — force a specific regime for T_scenario periods
2. **Mean shift** — shift mu_k by a delta
3. **Covariance shock** — scale Sigma_k by a factor
4. **Tail shock** — reduce degrees of freedom nu_k (requires Student-t emissions)
5. **Measurement shock** — perturb private asset observations

**Spec reference:** `finance_spec.md` Section 7.

### Advanced Portfolio Metrics

- Regime-conditional optimal allocation (mean-variance or risk-parity within each regime)
- Regime-aware rebalancing strategy backtest
- Maximum drawdown posterior distribution
- Regime-conditional Sharpe ratio with uncertainty

---

## Online/Real-Time Features

### Real-Time Filtering

Online regime estimation without refitting the full model. Given new observation `y_{T+1}`, update the regime probability using the forward algorithm with fixed (posterior mean) parameters.

**Spec reference:** `finance_spec.md` Section 8.

### Streaming Dashboard

Live regime probability monitor that updates as new data arrives, with alerts when P(stress regime) exceeds a threshold.
