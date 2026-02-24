# Mathematical Foundations: Bayesian Statistics

This document covers the fundamental Bayesian concepts underlying the regime-switching framework.

## 1. Bayesian Inference

### 1.1 Bayes' Theorem

For observed data $y$ and unknown parameters $\theta$:

$$p(\theta | y) = \frac{p(y | \theta) p(\theta)}{p(y)}$$

Where:
- $p(\theta | y)$ = **posterior** (belief about $\theta$ after observing $y$)
- $p(y | \theta)$ = **likelihood** (probability of observing $y$ given $\theta$)
- $p(\theta)$ = **prior** (belief about $\theta$ before observing data)
- $p(y) = \int p(y | \theta) p(\theta) d\theta$ = **evidence** (normalizing constant)

### 1.2 Why Bayesian?

**Advantages for financial modeling:**

1. **Principled uncertainty quantification:** Parameters are distributions, not point estimates
2. **Incorporates prior knowledge:** Domain expertise formalized as priors
3. **Sequential learning:** Posterior from today becomes prior for tomorrow
4. **Hypothesis testing:** Formal comparison via Bayes factors
5. **Decision-making:** Direct probabilistic statements ("There's 85% probability regime switches")

**Disadvantages:**

1. Computationally expensive (MCMC sampling is slow)
2. Prior specification requires care
3. Interpretation more subtle than frequentist p-values

---

## 2. Prior Specification

Good priors are **weakly informative:** enough structure to stabilize inference without imposing unrealistic constraints.

### 2.1 Normal Priors

$$\theta \sim \text{Normal}(\mu, \sigma^2)$$

**Use for:** Means, returns, loadings

**Interpretation:** $\theta$ is centered at $\mu$ with 95% of mass in $[\mu - 2\sigma, \mu + 2\sigma]$

**Example:** For daily log-returns (typically -5% to +5%), use $\text{Normal}(0, 1)$ to allow wide range without favoring large values.

### 2.2 Half-Normal Priors

$$\sigma \sim \text{HalfNormal}(\sigma_0) = \text{Normal}^+(0, \sigma_0)$$

**Use for:** Standard deviations, volatilities (must be positive)

**Example:** Typical volatility 1-3% per day → use $\text{HalfNormal}(0.05)$ (5% scale)

### 2.3 LKJ Priors on Correlations

For correlation matrix $\Omega$ with shape parameter $\eta$:

$$\Omega \sim \text{LKJ}(\eta)$$

- $\eta = 1$: Uniform over correlation matrices (maximally uninformed)
- $\eta > 1$: Concentrates mass near identity (weak correlation)
- $\eta < 1$: Concentrates at extremes (strong correlation/anticorrelation)

**Why LKJ?** Ensures $\Omega$ is valid (positive definite, unit diagonal) and properly normalized.

### 2.4 Dirichlet Priors

For probabilities $p = (p_1, …, p_K)$ with $\sum p_k = 1$:

$$p \sim \text{Dirichlet}(\alpha_1, …, \alpha_K)$$

**Mean:** $E[p_k] = \alpha_k / \sum \alpha_j$

**Example:** Transition probabilities from regime 1 to {1, 2, 3} regimes:
$$P_{1,:} \sim \text{Dirichlet}(1, 1, 1)$$
This is symmetric, treating all transitions equally likely a priori.

---

## 3. Student-t Distribution

### 3.1 Motivation: Why Not Gaussian?

Financial returns exhibit **fat tails** — extreme moves happen more frequently than Gaussian model predicts.

**Gaussian:** Tail probability of > 5σ event ≈ 0.000000029 (never happens in 100 years)  
**Student-t(ν=4):** Tail probability of > 5σ event ≈ 0.001 (happens every 3 years)

Real data: > 5σ events happen roughly every 1-2 years. **Student-t is more realistic.**

### 3.2 Definition

For location $\mu$, scale $\sigma$, and degrees of freedom $\nu > 2$:

$$y \sim \text{StudentT}(\nu, \mu, \sigma)$$

**PDF:**
$$p(y | \nu, \mu, \sigma) \propto \left(1 + \frac{1}{\nu}\left(\frac{y-\mu}{\sigma}\right)^2\right)^{-(\nu+1)/2}$$

**Properties:**
- As $\nu \to \infty$: Becomes Gaussian
- $\nu = 3$: Heavy tails (typical for returns)
- $\nu = 1$: Cauchy (infinite mean, pathological)
- Tail behavior: Polynomial decay vs. Gaussian exponential decay

### 3.3 Prior on Degrees of Freedom

$$\nu \sim \text{Exponential}(\lambda)$$

**Why exponential?**
- $\nu > 2$ is constraint (tractable with truncation)
- $E[\nu] = 1/\lambda$ is interpretable
- Favors fatter tails (smaller $\nu$) as prior mean

**Default:** $\lambda = 0.1$ → $E[\nu] = 10$ (moderately heavy tails)  
**Fatter tails:** $\lambda = 0.05$ → $E[\nu] = 20$ (still fat, but less extreme)

---

## 4. MCMC & NUTS Sampling

### 4.1 Why MCMC?

Posterior $p(\theta | y)$ is typically intractable to compute analytically. MCMC generates samples from $p(\theta | y)$ without computing it explicitly.

### 4.2 No-U-Turn Sampler (NUTS)

**Algorithm:** Adaptive Hamiltonian Monte Carlo variant

**Key idea:** Use gradient information (derivatives of log-posterior) to propose moves efficiently

**Convergence checks:**
- **R-hat:** Should be < 1.01 (chains mixing well)
- **ESS (Effective Sample Size):** Should be > 400-500 per chain (enough independent samples)
- **Divergences:** Should be 0 or very rare (< 0.1% of draws)

---

## 5. Posterior Predictive Checks

After inference, validate whether model captures data structure.

**Procedure:**
1. Draw $\theta^{(s)}$ from posterior
2. For each sample, simulate $y^{(s)} \sim p(y | \theta^{(s)})$
3. Compare: Is $y^{(obs)}$ plausible under $y^{(s)}$ distribution?

**Visual checks:**
- Overlay observed data on predictive distribution
- Plot tail behavior (Q-Q plot)
- Check autocorrelation structure

---

## Further Reading

- [Gelman et al., Bayesian Data Analysis (3rd ed.)](https://stat.columbia.edu/~gelman/book/)
- [McElreath, Statistical Rethinking (2nd ed.)](https://xcelab.net/rm/statistical-rethinking/)
- [PyMC Documentation](https://www.pymc.io/welcome.html)
