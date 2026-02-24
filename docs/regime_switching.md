# Regime-Switching Model: Mathematical Specification

## 1. Overview

A regime-switching model divides time into distinct **regimes** (market states) with different return characteristics. This is motivated by empirical evidence that financial markets exhibit structural breaks and state-dependent dynamics.

**Key insight:** Returns are not identically distributed over time. Instead, they depend on a latent regime variable $s_t \in \{1, …, K\}$.

---

## 2. Regime Dynamics: Markov Chain

### 2.1 Discrete-Time Markov Chain

The regime evolves as a first-order Markov chain:

$$s_t | s_{t-1} \sim \text{Categorical}(P_{s_{t-1},:})$$

Where **$P \in \mathbb{R}^{K \times K}$** is the **transition matrix**:

$$P = \begin{pmatrix}
P_{11} & P_{12} & \cdots & P_{1K} \\
P_{21} & P_{22} & \cdots & P_{2K} \\
\vdots & \vdots & \ddots & \vdots \\
P_{K1} & P_{K2} & \cdots & P_{KK}
\end{pmatrix}$$

**Constraints:**
- $P_{ij} \geq 0$ for all $i, j$
- $\sum_{j=1}^K P_{ij} = 1$ for all $i$ (row-stochastic)
- $P_{ij} = P(s_t = j | s_{t-1} = i)$ is the probability of transitioning from regime $i$ to regime $j$

### 2.2 Interpretation

- **Diagonal elements $P_{ii}$:** Persistence of regimes (probability of staying)
  - $P_{ii}$ close to 1 → regime is sticky (lasts long)
  - $P_{ii}$ close to 0 → frequent regime changes

- **Expected duration in regime $i$:**
$$E[\text{duration}_i] = \frac{1}{1 - P_{ii}}$$

**Example:** If $P_{11} = 0.95$, expected duration in regime 1 is $1/(1-0.95) = 20$ time steps.

### 2.3 Stationary Distribution

For an irreducible, aperiodic chain (typical), there exists a unique **stationary distribution** $\pi \in \mathbb{R}^K$ satisfying:

$$\pi P = \pi, \quad \sum_{i=1}^K \pi_i = 1$$

**Interpretation:** $\pi_i$ is the long-run frequency of regime $i$.

Computed as the left eigenvector of $P$ corresponding to eigenvalue 1:

$$\pi = \text{eig}_{\text{left}}(P; \lambda=1)$$

---

## 3. Return Dynamics: Regime-Conditional Student-t

### 3.1 Observation Model

Given regime $s_t = k$, returns $r_t \in \mathbb{R}^B$ follow a **multivariate Student-t distribution**:

$$r_t | s_t = k \sim \text{StudentT}_B(\nu_k, \mu_k, \Sigma_k)$$

Where:
- $\nu_k > 2$ — degrees of freedom (tail heaviness)
- $\mu_k \in \mathbb{R}^B$ — regime-specific mean return
- $\Sigma_k \in \mathbb{R}^{B \times B}$ — regime-specific covariance matrix

### 3.2 Probability Density

$$p(r_t | s_t = k) = \frac{\Gamma\left(\frac{\nu_k + B}{2}\right)}{\Gamma\left(\frac{\nu_k}{2}\right) (\pi \nu_k)^{B/2} |\Sigma_k|^{1/2}} \left(1 + \frac{1}{\nu_k}(r_t - \mu_k)^T \Sigma_k^{-1} (r_t - \mu_k)\right)^{-(\nu_k + B)/2}$$

### 3.3 Why Student-t Over Gaussian?

Financial returns exhibit **fat tails** (extreme moves more frequent than Gaussian predicts).

**Comparison:**

| Feature | Gaussian | Student-t (ν=4) |
|---------|----------|-----------------|
| P(return > 5σ) | ~0 | ~0.1% |
| Realistic for | Benign markets | Crisis/tail risk |
| Interpretation | Thin tails | Heavy tails (empirical) |

**Degrees of freedom interpretation:**
- $\nu = 3$ — very heavy tails (extreme events frequent)
- $\nu = 10$ — moderate tails
- $\nu \to \infty$ — approaches Gaussian

### 3.4 Scale-Mixture Representation

Equivalently, Student-t can be written as a scale mixture of normals:

$$r_t | s_t = k \sim \int_0^\infty \mathcal{N}(r_t | \mu_k, w \Sigma_k) \cdot p(w | \nu_k) dw$$

Where $w \sim \text{InverseGamma}\left(\frac{\nu_k}{2}, \frac{\nu_k}{2}\right)$.

This representation is useful for Gibbs sampling and can improve MCMC mixing.

---

## 4. Joint Likelihood

### 4.1 Complete Likelihood

Given observations $r_{1:T} = (r_1, …, r_T)$ and unobserved regimes $s_{1:T}$:

$$p(r_{1:T}, s_{1:T} | \theta) = p(s_1 | \pi) \prod_{t=2}^T p(s_t | s_{t-1}, P) \prod_{t=1}^T p(r_t | s_t, \mu_k, \Sigma_k, \nu_k)$$

Where:
- $p(s_1 | \pi)$ — initial regime distribution (typically stationary)
- $p(s_t | s_{t-1}, P)$ — transition probability
- $p(r_t | s_t, ...)$ — regime-conditional return likelihood

### 4.2 Marginal Likelihood (Filtered)

Integrating out regimes:

$$p(r_{1:T} | \theta) = \sum_{s_{1:T}} p(r_{1:T}, s_{1:T} | \theta)$$

This is computationally intractable for large $T$ and $K$, so we use **Bayesian inference** (MCMC) to sample from the posterior.

---

## 5. Priors

### 5.1 Transition Matrix Prior

On each row $i$ of $P$:

$$P_{i,:} \sim \text{Dirichlet}(\alpha_1, …, \alpha_K)$$

**Default (weakly informative):** $\alpha_k = 1$ for all $k$

**Interpretation:**
- Symmetric Dirichlet(1) treats all transitions equally likely a priori
- No regime favored over others
- Allows data to determine regime probabilities

**Alternative:** If domain knowledge suggests certain regimes are more persistent:
$$\alpha_{ii}^* > \alpha_{ij} \text{ for } j \neq i$$

### 5.2 Mean Returns Prior

$$\mu_{k,j} \sim \mathcal{N}(0, \sigma_\mu^2)$$

for each regime $k$ and asset $j$.

**Default:** $\sigma_\mu = 1$ (allows ±2% daily returns in 95% credible interval)

**Rationale:** Daily stock returns typically in range [-10%, +10%]; this prior is weakly informative.

### 5.3 Covariance Prior

Use **LKJ (Lewandowski-Kurowicka-Joe) prior** on correlation matrix plus HalfNormal on scales:

$$\Sigma_k = \text{diag}(\sigma_k) \Omega_k \text{diag}(\sigma_k)$$

Where:
- $\Omega_k \sim \text{LKJ}(\eta)$ — correlation matrix
- $\sigma_{k,j} \sim \text{HalfNormal}(\sigma_0)$ — marginal standard deviations

**LKJ parameter $\eta$:**
- $\eta = 1$ — uniform over correlation matrices
- $\eta = 2$ — slightly concentrated toward identity (default; weak correlation)
- $\eta > 2$ — strongly concentrated toward identity

**Scale prior:**
- Default: $\sigma_0 = 0.05$ (standard deviations typically 1-5% per day)

### 5.4 Degrees of Freedom Prior

$$\nu_k \sim \text{Exponential}(\lambda)$$

Truncated to $(\nu_\min, \nu_\max)$ with $\nu_\min = 2$ (required for Student-t).

**Default:** $\lambda = 0.1$ → $E[\nu_k] = 10$ (moderately heavy tails)

**Alternative settings:**
- $\lambda = 0.05$ → fatter tails, more extreme events
- $\lambda = 0.2$ → lighter tails, closer to normal

---

## 6. Graphical Model (DAG)

```
                    Dirichlet(1)
                         |
                         v
                    P (transition)
                    /         |
                   /          |
                  v           v
            s_{t-1}  ------>  s_t
                              |
                    +---------+---------+
                    |         |         |
                    v         v         v
            μ_k, Σ_k, ν_k     |       r_t (observed)
                    |         |
                    +------>--+
```

**Plate notation (for time series):**

```
For t = 1, ..., T:
    s_t ---> r_t
     |        |
     +-> [StudentT(ν_k, μ_k, Σ_k)]
```

With transitions:
```
s_{t-1} ---> s_t
    |
    +-> [Categorical(P_{s_{t-1}, :})]
```

---

## 7. Inference Algorithm

### 7.1 Hamiltonian MCMC (NUTS)

We use the **No-U-Turn Sampler (NUTS)**, an adaptive Hamiltonian Monte Carlo variant.

**Algorithm:**
1. Initialize parameters $\theta_0$
2. For iteration $i = 1, …, N$:
   - Sample momentum $p \sim \mathcal{N}(0, I)$
   - Run leapfrog integrator with adaptive step size
   - Accept/reject via Metropolis-Hastings
   - Return accepted $\theta_i$

**Diagnostics:**
- **R-hat:** Should be $< 1.01$ (chains mixed well)
- **ESS (Effective Sample Size):** Should be $> 400$ per chain
- **Divergences:** Should be $< 0.1\%$ (no bad geometry regions)

### 7.2 Convergence Checks

Before inference:
1. Run short chains (500 tune, 500 draw)
2. Check diagnostics
3. If good: run full inference
4. If bad: adjust priors or reparameterize

---

## 8. Implementation Notes

### 8.1 Identifiability

Regimes are **unidentifiable up to label permutation**. The model has the same likelihood under any relabeling of regimes.

**Solution:** Post-process posterior to label by mean return (regime 1 = lowest mean, etc.)

### 8.2 Computational Complexity

- **MCMC cost:** $O(T \times K^2 \times B^2)$ per iteration
- **Simulation cost:** $O(n_{\text{paths}} \times \text{horizon} \times K \times B)$

For typical sizes (T=500, K=2, B=5): ~10 min MCMC, ~1 sec simulation

### 8.3 Posterior Predictive Checks

After inference, validate model via:

$$r_t^{\text{pred}} \sim p(r_t | r_{1:T}) = \int p(r_t | \theta) p(\theta | r_{1:T}) d\theta$$

Compare observed vs. predicted:
- Tail behavior (Q-Q plot)
- Autocorrelation
- Regime persistence

---

## References

- Hamilton, J. D. (1989). "A new approach to the economic analysis of nonstationary time series." Econometrica, 57(2), 357-384.
- Frühwirth-Schnatter, S. (2006). *Finite Mixture and Markov Switching Models*. Springer.
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman and Hall.
