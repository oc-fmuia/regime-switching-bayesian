# Mathematical Specification v0: Regime-Switching Bayesian Model

**Version:** 0.1
**Date:** March 2026
**Scope:** Public assets only, K = 2 regimes, multivariate Normal emissions

---

## 1. Notation and Dimensions

| Symbol | Meaning |
|--------|---------|
| T | Number of observed months |
| K | Number of regimes (K = 2 for v0) |
| d | Number of public assets |
| t | Month index, t = 1, ..., T |
| k | Regime index, k = 1, ..., K |
| z_t | Observed return vector at month t, z_t in R^d |

---

## 2. Regime Dynamics

The regime at each month is a latent discrete state governed by a first-order Markov chain.

**Transition matrix:**

$$P \in [0,1]^{K \times K}, \quad \sum_{j=1}^{K} P_{ij} = 1 \;\; \forall \, i$$

where $P_{ij} = \Pr(s_t = j \mid s_{t-1} = i)$.

**Initial distribution:**

$$\pi_0 \in \Delta^{K-1}, \quad \pi_{0,k} = \Pr(s_1 = k)$$

**Regime process:**

$$s_1 \sim \text{Categorical}(\pi_0)$$

$$s_t \mid s_{t-1} \sim \text{Categorical}(P_{s_{t-1}, \,\cdot\,}), \quad t = 2, \ldots, T$$

---

## 3. Emission Model

Conditional on the regime, returns are multivariate Normal:

$$z_t \mid (s_t = k) \sim \mathcal{N}(\mu_k, \ \Sigma_k)$$

**Parameters per regime:**

- $\mu_k \in \mathbb{R}^d$: mean return vector.
- $\Sigma_k \in \mathbb{R}^{d \times d}$: covariance matrix (symmetric positive definite).

No hierarchical decomposition of $\mu_k$ and no factor structure for $\Sigma_k$ in v0. These are free vectors and matrices with appropriate priors.

---

## 4. Covariance Parameterization

Each $\Sigma_k$ is decomposed as:

$$\Sigma_k = \text{diag}(\sigma_k) \ L_k \ L_k^T \ \text{diag}(\sigma_k)$$

where:

- $\sigma_k \in \mathbb{R}_{>0}^{d}$: per-asset standard deviations in regime k.
- $L_k$: lower-triangular Cholesky factor of the correlation matrix, drawn from an LKJ distribution.

This is the standard PyMC parameterization (`pm.LKJCholeskyCov`) that separates scale (volatility) from shape (correlation).

---

## 5. Forward Algorithm

The forward algorithm computes the marginal log-likelihood $\log p(z_{1:T} \mid \theta)$ by summing over all regime sequences without enumerating them.

**Initialization (t = 1):**

$$\log \alpha_1(k) = \log \pi_{0,k} + \log \mathcal{N}(z_1 \mid \mu_k, \Sigma_k)$$

**Recursion (t = 2, ..., T):**

$$\log \alpha_t(k) = \text{LogSumExp}_{\tilde{k}}\ \big(\log \alpha_{t-1}(\tilde{k}) + \log P_{\tilde{k},k}\big) + \log \mathcal{N}(z_t \mid \mu_k, \Sigma_k)$$

**Marginal log-likelihood:**

$$\log p(z_{1:T} \mid \theta) = \text{LogSumExp}_{k}\ \big(\log \alpha_T(k)\big)$$

where:

$$\text{LogSumExp}(a_1, \ldots, a_n) = \max_j a_j + \log \sum_{i} \exp(a_i - \max_j a_j)$$

The recursion is implemented via `pytensor.scan` over t = 2, ..., T.

---

## 6. Forward-Filter Backward-Sampler (FFBS)

After NUTS estimates the continuous parameters $\theta$, FFBS recovers posterior samples of the regime sequence $s_{1:T}$.

**Forward pass:** Run the forward algorithm to obtain $\alpha_t(k)$ for all t, k.

**Backward sampling:**

1. Normalize: $\gamma_T(k) = \alpha_T(k) \/\ \sum_{\tilde{k}} \alpha_T(\tilde{k})$.
2. Sample $s_T \sim \text{Categorical}(\gamma_T)$.
3. For $t = T-1, \ldots, 1$:
   - $\gamma_t(k) \propto \alpha_t(k) \cdot P_{k, \, s_{t+1}}$
   - Sample $s_t \sim \text{Categorical}(\gamma_t)$.

This produces one draw of $s_{1:T}$ from $p(s_{1:T} \mid z_{1:T}, \theta)$. Repeating for each posterior sample of $\theta$ gives the full posterior over regime paths.

---

## 7. Priors

### Transition matrix (sticky)

$$P_{i, \cdot} \sim \text{Dirichlet}(\alpha_i), \quad i = 1, \ldots, K$$

with sticky concentration: $\alpha_{i,i} = 20$, $\alpha_{i, j \neq i} = 2$. This encourages regimes to persist (realistic for economic regimes that last months, not days).

### Initial distribution

$$\pi_0 \sim \text{Dirichlet}(1, \ldots, 1)$$

Uninformative (symmetric).

### Regime means

$$\mu_k \sim \mathcal{N}(0, \ \sigma_\mu^2 \ I_d), \quad \sigma_\mu = 0.05$$

Centered at zero; scale of 5% reflects the order of magnitude of monthly returns.

### Per-asset standard deviations

$$\sigma_{k,\ell} \sim \text{HalfNormal}(\tau_\sigma), \quad \tau_\sigma = 0.10$$

Weakly informative; allows monthly volatilities in the range 0–30%.

### Correlation Cholesky factor

$$L_k \sim \text{LKJCholesky}(\eta)$$

with $\eta = 2$, which mildly favors correlations near zero and penalizes extreme correlations.

---

## 8. Complete Generative Model

```
Sample pi_0 ~ Dirichlet(1, ..., 1)
Sample P_i ~ Dirichlet(alpha_i)         for i = 1, ..., K
Sample mu_k ~ N(0, sigma_mu^2 I)        for k = 1, ..., K
Sample sigma_k ~ HalfNormal(tau_sigma)   for k = 1, ..., K  (each is a d-vector)
Sample L_k ~ LKJCholesky(eta)           for k = 1, ..., K
Compute Sigma_k = diag(sigma_k) L_k L_k^T diag(sigma_k)

For t = 1, ..., T:
    Sample s_t ~ Categorical(pi_0)              if t = 1
                 Categorical(P_{s_{t-1}, :})     if t >= 2
    Sample z_t ~ N(mu_{s_t}, Sigma_{s_t})
```

**Inference:** The regime sequence $s_{1:T}$ is marginalized via the forward algorithm (Section 5). NUTS samples all continuous parameters. FFBS (Section 6) recovers $s_{1:T}$ post-hoc.

---

## 9. Parameter Count

For K = 2 regimes and d assets:

| Parameter | Count | Example (d = 3) |
|-----------|-------|------------------|
| pi_0 | K - 1 | 1 |
| P (off-diagonal per row) | K(K - 1) | 2 |
| mu_k | K * d | 6 |
| sigma_k | K * d | 6 |
| L_k (correlation) | K * d(d-1)/2 | 6 |
| **Total** | | **21** |

This is a small enough model that NUTS should mix well.

---

## 10. Relationship to Full Specification

This v0 is a strict subset of the full model in `math_spec.md`. The following table maps v0 choices to their full-model generalizations:

| v0 choice | Full model generalization |
|-----------|--------------------------|
| K = 2 regimes | K >= 2 (typically 3) |
| Public assets only | Public + private, with mixed-frequency measurement model |
| $\mu_k$ as free vectors | Hierarchical: $mu_{k,l} = m_k + a_{k,g(l)} + b_{k,g(l),h(l)} + u_{k,l}$ |
| Full $\Sigma_k$ via LKJ | Factor structure: $\Sigma_k = B_k B_k^T + D_k$ |
| Multivariate Normal | Multivariate Student-t ($\nu_k$ degrees of freedom) |
| Batch inference only | Real-time filtering and scenario analysis |

Each extension is independent and can be added incrementally.

---

**End of Mathematical Specification v0**
