# Mathematical Specification: Regime-Switching Bayesian Model

**Version:** 1.0  
**Date:** March 4, 2026  
**Status:** Complete specification for implementation

---

## Table of Contents

1. [Symbol Definitions & Index Conventions](#1-symbol-definitions--index-conventions)
2. [Data & Observed Variables](#2-data--observed-variables)
3. [Latent Variables](#3-latent-variables)
4. [Regime Dynamics (HMM)](#4-regime-dynamics-hmm)
5. [Emission Model (Monthly Returns)](#5-emission-model-monthly-returns)
6. [Hierarchical Mean Specification](#6-hierarchical-mean-specification)
7. [Hierarchical Covariance Specification](#7-hierarchical-covariance-specification)
8. [Mixed-Frequency Measurement Model](#8-mixed-frequency-measurement-model)
9. [Forward Algorithm (Marginalized HMM Likelihood)](#9-forward-algorithm-marginalized-hmm-likelihood)
10. [Forward-Filter-Backward-Sampler (FFBS)](#10-forward-filter-backward-sampler-ffbs)
11. [Priors & Hyperpriors](#11-priors--hyperpriors)
12. [Graphical Model](#12-graphical-model)

---

## 1. Symbol Definitions & Index Conventions

### Time Indices

| Symbol | Meaning | Range | Notes |
|--------|---------|-------|-------|
| t | Month index | 1, 2, ..., T | Observed monthly data |
| q | Quarter index | 1, 2, ..., Q | Quarterly reporting |
| Q | Total quarters | ⌊T/3⌋ | Q = floor(T/3) |
| t(q) | Quarter-end month | 3q | Quarter q ends at month t(q) = 3q |

**Quarter-to-month mapping:**
- Quarter 1: months {1, 2, 3}, ends at t(1) = 3
- Quarter 2: months {4, 5, 6}, ends at t(2) = 6
- Quarter q: months {3q-2, 3q-1, 3q}, ends at t(q) = 3q

### Asset Indices

| Symbol | Meaning | Range | Notes |
|--------|---------|-------|-------|
| ℓ | Asset/series index | 1, 2, ..., d | All assets combined |
| g | Geography index | 1, 2, ..., G | Geographic regions |
| h | Sector index | 1, 2, ..., H | Industry sectors |
| g(ℓ) | Geography of asset ℓ | ∈ {1,...,G} | Mapping: asset → geography |
| h(ℓ) | Sector of asset ℓ | ∈ {1,...,H} | Mapping: asset → sector |
| type(ℓ) | Asset type | ∈ {pub, priv} | Public or private |

**Asset partitioning:**
- Public assets: ℓ ∈ {1, ..., d_pub}, type(ℓ) = "pub"
- Private assets: ℓ ∈ {d_pub + 1, ..., d}, type(ℓ) = "priv"
- Total assets: d = d_pub + d_priv

### Model Dimensions

| Symbol | Meaning | Constraint |
|--------|---------|-----------|
| T | Number of months | T ≥ 3 (at least one quarter) |
| K | Number of regimes | K ≥ 2 |
| G | Number of geographies | G ≥ 1 |
| H | Number of sectors | H ≥ 1 |
| d_pub | Public asset dimension | d_pub ≥ 1 |
| d_priv | Private asset dimension | d_priv ≥ 1 |
| d | Total asset dimension | d = d_pub + d_priv |
| r | Number of latent factors | 1 ≤ r ≪ d |
| Q_obs | Observed quarters | Q_obs ≤ Q |
| L | Reporting lag (months) | L ≥ 0 |

---

## 2. Data & Observed Variables

### Monthly Public Returns

$$r_t^{(\text{pub})} \in \mathbb{R}^{d_{\text{pub}}}, \quad t = 1, 2, \ldots, T$$

**Definition:** Log-returns of public assets observed monthly. Fully observed (no missing data).

**Example:** If d_pub = 4, then $r_t^{(\text{pub})} = [r_{t,1}^{(\text{pub})}, r_{t,2}^{(\text{pub})}, r_{t,3}^{(\text{pub})}, r_{t,4}^{(\text{pub})}]^T$.

### Quarterly Reported Private Returns

$$y_q^{(\text{priv})} \in \mathbb{R}^{d_{\text{priv}}}, \quad q = 1, 2, \ldots, Q_{\text{obs}}$$

**Definition:** Log-returns of private assets reported at quarterly frequency. These are *reported* values, potentially smoothed/stale relative to true monthly returns.

**Key constraints:**
- Q_obs ≤ Q (may not observe all quarters due to missing data)
- Reported at month t(q) + L, where L is reporting lag
- Contains smoothing and measurement noise (see Section 8)

---

## 3. Latent Variables

### Monthly Latent Private Returns

$$x_t^{(\text{priv})} \in \mathbb{R}^{d_{\text{priv}}}, \quad t = 1, 2, \ldots, T$$

**Definition:** True monthly log-returns of private assets. NOT observed directly; only available through quarterly reported values y_q.

**Interpretation:** What the private assets *actually returned* in month t (before any reporting smoothing or lag).

### Monthly Regime State

$$s_t \in \{1, 2, \ldots, K\}, \quad t = 1, 2, \ldots, T$$

**Definition:** Hidden discrete state indicating which regime (e.g., "growth", "crisis", "recovery") is active in month t.

**Dynamics:** Markov chain (depends only on s_{t-1}, not full history).

### Combined Monthly Latent Return Vector

$$z_t = \begin{bmatrix} r_t^{(\text{pub})} \\ x_t^{(\text{priv})} \end{bmatrix} \in \mathbb{R}^d, \quad t = 1, 2, \ldots, T$$

**Definition:** Stacking of public (observed) and private (latent) monthly returns.

**Key property:** z_t is distributed conditionally on s_t as multivariate Student-t (Section 5).

---

## 4. Regime Dynamics (HMM)

### Transition Matrix

$$P \in [0,1]^{K \times K}, \quad \sum_{j=1}^K P_{ij} = 1 \text{ for all } i$$

**Definition:** P[i,j] = P(s_t = j | s_{t-1} = i) = probability of transitioning from regime i to regime j.

**Row stochasticity:** Each row sums to 1.

### Initial Distribution

$$\pi_0 \in [0,1]^K, \quad \sum_{i=1}^K \pi_{0,i} = 1$$

**Definition:** $\pi_{0,i}$ = P(s_1 = i) = probability of starting in regime i.

### Regime Process

**Time 1:**
$$s_1 \sim \text{Categorical}(\pi_0)$$

**Time t ≥ 2:**
$$s_t | s_{t-1} \sim \text{Categorical}(P_{s_{t-1}, \cdot})$$

where P[s_{t-1}, ·] is the row of P corresponding to the previous regime.

**Shorthand:** $s_{1:T}$ denotes the entire regime sequence $(s_1, s_2, \ldots, s_T)$.

---

## 5. Emission Model (Monthly Returns)

### Conditional Distribution

Given regime s_t = k, the combined monthly return vector z_t is:

$$z_t | (s_t = k) \sim \text{MultivariateSudentT}_{\nu_k}(\mu_k, \Sigma_k)$$

**Parameters:**
- $\mu_k \in \mathbb{R}^d$: regime-k mean vector
- $\Sigma_k \in \mathbb{R}^{d \times d}$: regime-k covariance/scale matrix (symmetric PSD)
- $\nu_k > 2$: regime-k degrees of freedom (tail parameter)

### Multivariate Student-t Distribution

The PDF of $\text{MultivariateSudentT}_\nu(\mu, \Sigma)$ is:

$$f(z | \mu, \Sigma, \nu) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) (\pi \nu)^{d/2} |\Sigma|^{1/2}} \left[ 1 + \frac{1}{\nu}(z - \mu)^T \Sigma^{-1} (z - \mu) \right]^{-\frac{\nu + d}{2}}$$

**Key properties:**
- $\nu \to \infty$: recovers multivariate Gaussian with covariance Σ
- $\nu < 5$: heavy tails (higher kurtosis)
- $\nu = 1$: Cauchy distribution (infinite variance)

### Tail Index Interpretation

The excess kurtosis of univariate Student-t is:

$$\text{Kurtosis} = 3 + \frac{6}{\nu - 4} \quad (\nu > 4)$$

**Regimes:**
- Growth regime: ν_growth ≈ 10–20 (slight fat tails, kurtosis ≈ 3.5–3.2)
- Crisis regime: ν_crisis ≈ 3–5 (heavy fat tails, kurtosis ≈ 6–3)

---

## 6. Hierarchical Mean Specification

### Asset-Level Mean Decomposition

For regime k and asset ℓ:

$$\mu_{k,\ell} = m_k + a_{k, g(\ell)} + b_{k, g(\ell), h(\ell)} + u_{k,\ell}$$

where:
- $m_k \in \mathbb{R}$: Global mean shift in regime k
- $a_{k, g} \in \mathbb{R}$: Geography g effect in regime k (centered at 0)
- $b_{k, g, h} \in \mathbb{R}$: Sector h effect within geography g, regime k (centered at 0)
- $u_{k,\ell} \in \mathbb{R}$: Asset ℓ residual in regime k (centered at 0)

### Stacking into Vector Form

The full regime-k mean vector:

$$\mu_k = [m_k \mathbf{1}_d + A_k + B_k + U_k] \in \mathbb{R}^d$$

where:
- $\mathbf{1}_d$: d-dimensional vector of ones
- $A_k$: vector with $(A_k)_\ell = a_{k, g(\ell)}$
- $B_k$: vector with $(B_k)_\ell = b_{k, g(\ell), h(\ell)}$
- $U_k$: vector with $(U_k)_\ell = u_{k,\ell}$

### Hierarchical Structure Visualized

```
Global level: m_k (same for all assets in regime k)
  ↓
Geography level: a_{k,g} (depends on asset's geography)
  ↓
Sector-within-geography level: b_{k,g,h} (depends on asset's geography AND sector)
  ↓
Asset residual: u_{k,ℓ} (specific to each asset)

Final mean for asset ℓ in regime k:
  μ_{k,ℓ} = m_k + a_{k,g(ℓ)} + b_{k,g(ℓ),h(ℓ)} + u_{k,ℓ}
```

### Priors for Means

**Global mean:**
$$m_k \sim \mathcal{N}(m_0, \sigma_m^2)$$

**Hyperparameter:** $m_0$ and $\sigma_m$ are fixed (not estimated). Default: $m_0 = 0$, $\sigma_m = 0.05$ (for monthly returns).

**Geography effects:**
$$a_{k, g} \sim \mathcal{N}(0, \sigma_{a,k}^2), \quad g = 1, \ldots, G$$

**Sector-within-geography effects:**
$$b_{k, g, h} \sim \mathcal{N}(0, \sigma_{b,k}^2), \quad g = 1, \ldots, G; \, h = 1, \ldots, H$$

**Asset residuals (type-specific):**
$$u_{k,\ell} \sim \mathcal{N}(0, \sigma_{u,k,\text{type}(\ell)}^2)$$

where type(ℓ) ∈ {pub, priv}, so two variance parameters per regime: $\sigma_{u,k,\text{pub}}^2$ and $\sigma_{u,k,\text{priv}}^2$.

---

## 7. Hierarchical Covariance Specification

### Factor Structure

The covariance matrix in regime k:

$$\Sigma_k = B_k B_k^T + D_k$$

where:
- $B_k \in \mathbb{R}^{d \times r}$: factor loading matrix (d assets × r factors)
- $D_k \in \mathbb{R}^{d \times d}$: diagonal idiosyncratic variance matrix

**Interpretation:**
- $B_k B_k^T$: systemic risk (common factors explain co-movement)
- $D_k$: idiosyncratic risk (asset-specific uncertainty)

### Diagonal Idiosyncratic Variance

$$D_k = \text{diag}(\sigma_{k,1}^2, \sigma_{k,2}^2, \ldots, \sigma_{k,d}^2)$$

**Log-variance prior:**
$$\log \sigma_{k,\ell} \sim \mathcal{N}(\alpha_{k,\text{type}(\ell)}, \tau_{k,\text{type}(\ell)}^2)$$

where type(ℓ) ∈ {pub, priv}, giving two location and two scale parameters per regime.

### Hierarchical Factor Loadings

For row ℓ of B_k (the factor loadings for asset ℓ):

$$(B_k)_{\ell, :} = \beta_k + \beta_{k, g(\ell)} + \beta_{k, g(\ell), h(\ell)} + \eta_{k,\ell}$$

where:
- $\beta_k \in \mathbb{R}^r$: Global factor loadings in regime k
- $\beta_{k, g} \in \mathbb{R}^r$: Geography g adjustment to loadings
- $\beta_{k, g, h} \in \mathbb{R}^r$: Sector h adjustment (within geography g)
- $\eta_{k,\ell} \in \mathbb{R}^r$: Asset ℓ-specific loading residual

All terms are row vectors in $\mathbb{R}^r$ (one per latent factor).

### Priors for Factor Loadings

**Global loadings:**
$$\beta_k \sim \mathcal{N}_r(0, \sigma_{\beta,k}^2 I_r)$$

**Geography adjustments:**
$$\beta_{k, g} \sim \mathcal{N}_r(0, \sigma_{\beta g, k}^2 I_r)$$

**Sector-within-geography adjustments:**
$$\beta_{k, g, h} \sim \mathcal{N}_r(0, \sigma_{\beta g h, k}^2 I_r)$$

**Asset residuals (type-specific):**
$$\eta_{k,\ell} \sim \mathcal{N}_r(0, \sigma_{\eta, k, \text{type}(\ell)}^2 I_r)$$

### Tail Parameters

$$\nu_k \sim \text{LogNormal}(\mu_\nu, \sigma_\nu^2), \quad k = 1, \ldots, K$$

**Specification:** LogNormal parameterized by mean and SD of the log.

**Default values:** $\mu_\nu = \log(12)$, $\sigma_\nu = 0.4$ (centers prior around ν ≈ 12, allows range 5–30).

---

## 8. Mixed-Frequency Measurement Model

### Quarterly Aggregation from Latent Monthly Returns

Define the "true" quarterly aggregated return:

$$\tilde{y}_q^{(\text{priv})} = \sum_{m=0}^{2} x_{t(q) - m}^{(\text{priv})} = x_{3q-2}^{(\text{priv})} + x_{3q-1}^{(\text{priv})} + x_{3q}^{(\text{priv})}$$

**Interpretation:** Sum of the three months' latent private returns in quarter q.

### Smoothing & Measurement Noise Dynamics

For q ≥ 2:

$$y_q^{(\text{priv})} = \phi y_{q-1}^{(\text{priv})} + (1 - \phi) \tilde{y}_q^{(\text{priv})} + \varepsilon_q$$

where:
- $\phi \in [0, 1)$: Smoothing parameter (higher φ = more persistence, slower tracking of true returns)
- $\varepsilon_q \sim \mathcal{N}(0, R)$: Measurement noise

**Interpretation:**
- If φ = 0: No smoothing, y_q = \tilde{y}_q + ε_q (reported = aggregated monthly + noise)
- If φ = 0.5: Reported value is 50% lagged + 50% current aggregated
- If φ → 1: Reported values become very stale

### Initialization

For q = 1 (first quarter), we must specify y_0. **Choice for implementation: y_0 = 0**.

Then:
$$y_1^{(\text{priv})} = \phi \cdot 0 + (1 - \phi) \tilde{y}_1^{(\text{priv})} + \varepsilon_1 = (1 - \phi) \tilde{y}_1^{(\text{priv})} + \varepsilon_1$$

**Alternative:** Could set y_0 ~ N(ȳ_0, S_0) with fixed ȳ_0, S_0, but y_0 = 0 is simpler and interpretable (no prior reporting history).

### Smoothing Parameter Prior

$$\phi = \text{logistic}(\psi), \quad \psi \sim \mathcal{N}(\mu_\psi, \sigma_\psi^2)$$

**Default:** $\mu_\psi = 0$ (prior mean φ ≈ 0.5), $\sigma_\psi = 1.0$ (weak prior).

**Transformation ensures:** φ ∈ (0, 1) automatically via logistic function.

### Measurement Noise Covariance

$$R = \text{diag}(r_1^2, r_2^2, \ldots, r_{d_{\text{priv}}}^2)$$

**Prior on each diagonal element:**
$$\log r_j \sim \mathcal{N}(\mu_r, \sigma_r^2), \quad j = 1, \ldots, d_{\text{priv}}$$

**Default:** $\mu_r = -2$ (log-scale, means r_j ≈ 0.135, or ~13% noise), $\sigma_r = 0.5$ (weak prior).

### Handling Missing Quarters & Reporting Lag

**Missing quarters:** If quarter q is not in the observed set (Q_obs < Q), then y_q is **not conditioned on** during inference. It remains latent and is integrated out.

**Reporting lag L:** Quarter q becomes observable only at month t(q) + L. In the likelihood, condition only on y_q when t ≥ t(q) + L.

**Implementation detail:** When computing likelihood at month t:
- If a quarter's observation has arrived (t ≥ t(q) + L), include y_q in the likelihood
- Otherwise, omit it (latent)

---

## 9. Forward Algorithm (Marginalized HMM Likelihood)

### Objective

Compute the marginal likelihood P(r_1:T^(pub) | θ) by marginalizing over the latent regime sequence s_1:T:

$$P(r_{1:T}^{(\text{pub})} | \theta) = \sum_{s_{1:T}} P(r_{1:T}^{(\text{pub})} | s_{1:T}, \theta) P(s_{1:T} | \theta)$$

This is intractable to compute directly (K^T terms), so we use the forward algorithm.

### Forward Algorithm Recursion

**Initialization (t = 1):**

Define the forward probability:

$$\alpha_1(k) = \pi_{0,k} \cdot p(r_1^{(\text{pub})} | s_1 = k)$$

where $p(r_1^{(\text{pub})} | s_1 = k)$ is the likelihood of the first observation under regime k.

**Recursion (t = 2, ..., T):**

$$\alpha_t(k) = \left[ \sum_{k'=1}^K \alpha_{t-1}(k') P_{k',k} \right] \cdot p(r_t^{(\text{pub})} | s_t = k)$$

**Termination:**

$$P(r_{1:T}^{(\text{pub})} | \theta) = \sum_{k=1}^K \alpha_T(k)$$

### Log-Space Implementation (Numerical Stability)

To avoid underflow, work in log-probability space:

$$\log \alpha_t(k) = \text{LogSumExp}\left(\log \alpha_{t-1}(k') + \log P_{k',k} \quad \forall k'\right) + \log p(r_t^{(\text{pub})} | s_t = k)$$

**LogSumExp function:**

$$\text{LogSumExp}(a_1, \ldots, a_n) = \max_j a_j + \log \sum_i \exp(a_i - \max_j a_j)$$

This stabilizes by subtracting the maximum before exponentiating.

### Likelihood of Observation Under Regime

Given regime k, the observation r_t^(pub) comes from the marginal distribution:

$$p(r_t^{(\text{pub})} | s_t = k) = \int p(z_t | s_t = k) \, dz_t^{(\text{priv})}$$

where z_t = [r_t^(pub); z_t^(priv)], and we integrate out the private component.

Since z_t | s_t = k ~ MVT(μ_k, Σ_k, ν_k), the marginal for r_t^(pub) is obtained by:
1. Extract the sub-vector of μ_k corresponding to public indices: μ_k^(pub)
2. Extract the sub-matrix of Σ_k: Σ_k^(pub, pub)
3. Likelihood is MVT(μ_k^(pub), Σ_k^(pub, pub), ν_k)

### PyTensor.scan Implementation

The forward recursion is implemented using PyTensor.scan (looping construct):

```python
def forward_step(r_t, log_alpha_prev, P, mu_k, Sigma_k, nu_k):
    # Compute log-likelihood for each regime
    log_lik_k = logpdf_mvt(r_t, mu_k, Sigma_k, nu_k)  # shape (K,)
    
    # Update: log α_t(k) = LogSumExp(log α_{t-1} + log P^T) + log lik
    log_alpha_t = logsumexp(log_alpha_prev + log(P.T), axis=0) + log_lik_k
    
    return log_alpha_t

# Initialize
log_alpha_1 = log(pi0) + logpdf_mvt(r_data[0], mu_k, Sigma_k, nu_k)

# Scan
log_alphas, _ = scan(forward_step, sequences=r_data[1:], outputs_info=log_alpha_1,
                      non_sequences=[P, mu_k, Sigma_k, nu_k])

# Final likelihood
log_lik = logsumexp(log_alphas[-1])
```

---

## 10. Forward-Filter-Backward-Sampler (FFBS)

### Objective

After fitting the model (estimating θ), recover samples of the regime sequence s_1:T from the posterior p(s_1:T | r_1:T, θ).

### Algorithm Overview

**Step 1: Forward Filtering**
Run the forward algorithm to compute forward probabilities α_t(k) = p(s_t = k | r_1:t, θ).

**Step 2: Backward Sampling**
Sample regimes backward in time:

1. Initialize: Sample $s_T \sim \text{Categorical}(\tilde{\gamma}_T)$, where $\tilde{\gamma}_T(k) \propto \alpha_T(k)$
2. For t = T-1, ..., 1:
   - Compute backward weights: $\tilde{\gamma}_t(k) \propto \alpha_t(k) P_{k, s_{t+1}}$
   - Sample: $s_t \sim \text{Categorical}(\tilde{\gamma}_t)$

### Posterior Regime Probabilities

After forward filtering (computing α_t):

**Filtered probability** (incorporating data up to time t):

$$\gamma_t(k) = P(s_t = k | r_1:t, \theta) \propto \alpha_t(k)$$

**Smoothed probability** (incorporating all data):

$$\gamma_t(k) = P(s_t = k | r_1:T, \theta)$$

The backward sampler automatically produces samples from the smoothed distribution.

### Pseudo-code

```
# Forward pass (already done in forward algorithm)
for t = 1 to T:
    α_t(k) ← forward recursion (see Section 9)

# Backward sampling
γ_T(k) ← α_T(k) / sum_j α_T(j)  # Normalize
s_T ~ Categorical(γ_T)

for t = T-1 down to 1:
    γ_t(k) ← α_t(k) * P[k, s_{t+1}] / Z_t  where Z_t = sum_k' α_t(k') P[k', s_{t+1}]
    s_t ~ Categorical(γ_t)

return s_{1:T}  # regime sequence sample
```

---

## 11. Priors & Hyperpriors

### Complete Prior Specification

#### A. Regime Dynamics

**Transition matrix rows (sticky prior):**

For each i = 1, ..., K:
$$P_{i, \cdot} \sim \text{Dirichlet}(\alpha_{i,1}, \ldots, \alpha_{i,K})$$

**Sticky prior specification:**
- $\alpha_{i,i} = 20$ (diagonal: higher concentration toward staying in current regime)
- $\alpha_{i,j \neq i} = 2$ (off-diagonal: lower concentration toward switching)

**Initial distribution:**
$$\pi_0 \sim \text{Dirichlet}(\alpha_{\text{init},1}, \ldots, \alpha_{\text{init},K})$$

where $\alpha_{\text{init},i} = 1$ for all i (uninformative).

#### B. Hierarchical Means

**Global mean:**
$$m_k \sim \mathcal{N}(0, 0.05^2)$$

**Standard deviation of geography effects:**
$$\sigma_{a,k} \sim \text{HalfNormal}(0.05)$$

**Standard deviation of sector effects:**
$$\sigma_{b,k} \sim \text{HalfNormal}(0.05)$$

**Standard deviations of asset residuals (type-specific):**
$$\sigma_{u,k,\text{pub}} \sim \text{HalfNormal}(0.05)$$
$$\sigma_{u,k,\text{priv}} \sim \text{HalfNormal}(0.05)$$

#### C. Hierarchical Covariances

**Global factor loadings:**
$$\beta_k \sim \mathcal{N}_r(0, 0.1^2 I_r)$$

**Geography adjustment scales:**
$$\sigma_{\beta g,k} \sim \text{HalfNormal}(0.1)$$

**Sector adjustment scales:**
$$\sigma_{\beta g h,k} \sim \text{HalfNormal}(0.1)$$

**Asset residual scales (type-specific):**
$$\sigma_{\eta,k,\text{pub}} \sim \text{HalfNormal}(0.1)$$
$$\sigma_{\eta,k,\text{priv}} \sim \text{HalfNormal}(0.1)$$

**Idiosyncratic volatility hyperpriors:**
$$\alpha_{k,\text{pub}} \sim \mathcal{N}(0, 10^2)$$
$$\alpha_{k,\text{priv}} \sim \mathcal{N}(0, 10^2)$$
$$\tau_{k,\text{pub}} \sim \text{HalfNormal}(1)$$
$$\tau_{k,\text{priv}} \sim \text{HalfNormal}(1)$$

**Tail parameters:**
$$\nu_k \sim \text{LogNormal}(\log(12), 0.4^2)$$

#### D. Measurement Model

**Smoothing parameter:**
$$\psi \sim \mathcal{N}(0, 1^2)$$
$$\phi = \text{logistic}(\psi) \in (0, 1)$$

**Measurement noise scales:**
$$\mu_r \sim \mathcal{N}(-2, 1^2)$$
$$\sigma_r \sim \text{HalfNormal}(0.5)$$

$$\log r_j \sim \mathcal{N}(\mu_r, \sigma_r^2), \quad j = 1, \ldots, d_{\text{priv}}$$

### Hyperprior Rationale

- **HalfNormal(λ):** Encourages small variances, allows flexibility
- **λ = 0.05 for means:** Monthly returns typically 0–5% per month; scale 50 bps
- **λ = 0.1 for loadings:** Loadings bounded roughly by [-3, 3] in typical scenarios
- **LogNormal for ν:** Centers on ν ≈ 12, allows range 5–30 (fat tails in regimes, near-Gaussian overall)
- **Sticky Dirichlet:** Encourages regime persistence (realistic for economic regimes)

---

## 12. Graphical Model

### Plate Notation (Text Representation)

```
Regime Dynamics:
    π₀ ──→ s₁
           ↓
           v
    P ──→ s_t ──→ s_{t+1}
           (for t = 1, ..., T-1)

Emission Model:
    (μ_k, Σ_k, ν_k) ──→ z_t | s_t
    
    where z_t = [r_t^(pub); x_t^(priv)]
    
    - r_t^(pub) is OBSERVED (public returns)
    - x_t^(priv) is LATENT (private returns)

Hierarchical Means (for each regime k):
    m_k, σ_{a,k}, σ_{b,k}, σ_{u,k,*} ──→ μ_k ──→ z_t | s_t
    
Hierarchical Covariances (for each regime k):
    β_k, σ_{β*,k}, α_{k,*}, τ_{k,*} ──→ (B_k, D_k) ──→ Σ_k ──→ z_t | s_t

Measurement Model:
    φ, R ──→ y_q^(priv) | x_t^(priv) (latent-to-observed mapping)

Priors on All Parameters (implicit, see Section 11)
```

### Full Generative Model Flow

```
Sample π₀, P (regime dynamics)
↓
Sample m_k, a_{k,g}, b_{k,g,h}, u_{k,ℓ} for all k (means)
↓
Sample β_k, β_{k,g}, β_{k,g,h}, η_{k,ℓ}, {α_k, τ_k}, {log σ_k,ℓ} for all k (covariances)
↓
Sample ν_k for all k (tail parameters)
↓
Sample φ, R (measurement model)
↓
For t = 1 to T:
    Sample s_t ~ Categorical(P_{s_{t-1}, ·}) with s_1 ~ Categorical(π₀)
    Compute μ_{s_t}, Σ_{s_t} (regime-conditional, hierarchical)
    Sample z_t ~ MVT(μ_{s_t}, Σ_{s_t}, ν_{s_t})
    Extract r_t^(pub), x_t^(priv) from z_t

For q = 1 to Q:
    Compute \tilde{y}_q = sum of x_t^(priv) for months in quarter q
    Sample y_q ~ N(φ y_{q-1} + (1-φ)\tilde{y}_q, R)

OUTPUT: r_1:T^(pub), y_1:Q_obs^(priv), s_1:T, x_1:T^(priv) (ground truth for synthetic data)
```

---

## Summary: Sufficient Statistics for Implementation

**Continuous parameters to sample with NUTS:**
- P (transition matrix rows), π₀
- m_k, a_{k,g}, b_{k,g,h}, u_{k,ℓ}
- β_k, β_{k,g}, β_{k,g,h}, η_{k,ℓ}, α_{k,*}, τ_{k,*}
- ν_k
- φ, {log r_j}

**Discrete latent to recover post-hoc:**
- s_1:T (via FFBS)

**Continuous latent to recover post-hoc:**
- x_1:T^(priv) (via Kalman filter on measurement model)

**Likelihood:**
- Forward algorithm on r_1:T^(pub) (marginalizing s_1:T)
- Measurement model on y_1:Q_obs^(priv) (integrating over x_t^(priv) implicit in state space)

---

**End of Mathematical Specification**
