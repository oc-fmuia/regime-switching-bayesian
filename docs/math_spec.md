# Mathematical Specification: Bayesian Regime-Switching Model for Equity Returns

**Date:** April 2, 2026
**Status:** Covers the implemented model (Notebook 01) and planned extensions (Notebooks 02--05)

---

## Table of Contents

1. [Introduction and Problem Setup](#1-introduction-and-problem-setup)
2. [Notation and Conventions](#2-notation-and-conventions)
3. [The Generative Model](#3-the-generative-model)
   - 3.1 [Regime Dynamics](#31-regime-dynamics)
   - 3.2 [Emission Distributions](#32-emission-distributions)
   - 3.3 [Covariance Parameterisation via LKJ Cholesky](#33-covariance-parameterisation-via-lkj-cholesky)
   - 3.4 [The Complete Generative Process](#34-the-complete-generative-process)
4. [Bayesian Inference via Marginalisation](#4-bayesian-inference-via-marginalisation)
   - 4.1 [Why Marginalise Over Regimes](#41-why-marginalise-over-regimes)
   - 4.2 [The Forward Algorithm](#42-the-forward-algorithm)
   - 4.3 [Normalised Forward Recursion and Numerical Stability](#43-normalised-forward-recursion-and-numerical-stability)
   - 4.4 [Connection to PyMC via pm.Potential](#44-connection-to-pymc-via-pmpotential)
   - 4.5 [Prior Specification and Hyperparameter Choices](#45-prior-specification-and-hyperparameter-choices)
5. [Regime Recovery: The Forward-Filter Backward-Sampler](#5-regime-recovery-the-forward-filter-backward-sampler)
   - 5.1 [Motivation](#51-motivation)
   - 5.2 [The Forward Filtering Pass](#52-the-forward-filtering-pass)
   - 5.3 [The Backward Sampling Pass](#53-the-backward-sampling-pass)
   - 5.4 [Smoothed versus Filtered Probabilities](#54-smoothed-versus-filtered-probabilities)
6. [Filtered Probabilities and Causal Allocation](#6-filtered-probabilities-and-causal-allocation)
7. [Label Switching](#7-label-switching)
   - 7.1 [The Symmetry Problem](#71-the-symmetry-problem)
   - 7.2 [Consequences for Multi-Chain Inference](#72-consequences-for-multi-chain-inference)
   - 7.3 [Resolution via Post-Hoc Permutation Alignment](#73-resolution-via-post-hoc-permutation-alignment)
8. [Planned Extensions](#8-planned-extensions)
   - 8.1 [Student-$t$ Emissions](#81-student-t-emissions)
   - 8.2 [Regime-Dependent Correlations](#82-regime-dependent-correlations)
   - 8.3 [Autoregressive Dynamics Within Regimes](#83-autoregressive-dynamics-within-regimes)
   - 8.4 [Time-Varying Transition Probabilities](#84-time-varying-transition-probabilities)
   - 8.5 [Walk-Forward Estimation](#85-walk-forward-estimation)
9. [References](#9-references)

---

## 1. Introduction and Problem Setup

The central problem in regime-aware portfolio management is this: observed equity returns exhibit pronounced time variation in their statistical properties (means, volatilities, and correlations all shift as markets move between expansionary and contractionary phases), yet a single multivariate distribution fitted to a long history of returns averages across these distinct environments. During calm markets such a model overstates risk; during crises it understates it. The root cause is that the data-generating process is not stationary but instead switches between latent regimes, each with its own characteristic distribution.

This observation motivates a model with two layers of structure. At the surface, we observe a $d$-dimensional time series of monthly equity returns $\mathbf{y}_1, \ldots, \mathbf{y}_T$. Beneath the surface, an unobserved discrete state $s_t \in \{0, 1, \ldots, K-1\}$ indexes the economic regime active at time $t$. Conditional on the regime, returns are drawn from a regime-specific distribution; the regimes themselves evolve as a Markov chain. This is the **Hidden Markov Model** (HMM), the natural formalism whenever one has a discrete latent process driving observable continuous data.

Hamilton (1989) introduced the Markov-switching framework for modelling business cycles, showing that a two-regime model could capture the asymmetric dynamics of US GDP growth. Ang and Bekaert (2002) extended this to international asset allocation, demonstrating that accounting for regimes materially changes optimal portfolio weights and that ignoring regime structure leads to substantial welfare losses. Ang and Bekaert (2004) showed further that the practical impact is largest in the tails, precisely where risk management matters most.

Our treatment is Bayesian: rather than seeking point estimates of the model parameters, we place priors on all unknowns and compute the full posterior distribution. This has three advantages for portfolio applications. First, posterior uncertainty about parameters propagates directly into uncertainty about regime probabilities and portfolio decisions, avoiding the false precision of maximum-likelihood estimates. Second, the Bayesian framework provides a natural mechanism for model comparison via criteria such as WAIC. Third, the hierarchical prior structure (particularly the sticky Dirichlet prior on transition probabilities) allows us to encode domain knowledge, namely that economic regimes persist for months or years rather than days, in a principled way.

The remainder of this document builds the mathematical framework from first principles. Section 2 establishes notation. Section 3 constructs the generative model layer by layer. Section 4 derives the marginalised likelihood and the forward algorithm that makes Bayesian inference tractable. Section 5 develops the forward-filter backward-sampler for regime recovery. Section 6 shows how filtered probabilities enable causal allocation rules. Section 7 addresses the label-switching symmetry. Section 8 formulates planned extensions.

---

## 2. Notation and Conventions

### Indices

| Symbol | Meaning | Range |
|--------|---------|-------|
| $t$ | Time (month) index | $1, 2, \ldots, T$ |
| $k, k', j$ | Regime index | $0, 1, \ldots, K-1$ |
| $i, \ell$ | Asset index | $1, 2, \ldots, d$ |

### Observed and Latent Variables

| Symbol | Dimension | Definition |
|--------|-----------|------------|
| $\mathbf{y}_t$ | $\mathbb{R}^d$ | Observed return vector at time $t$ |
| $\mathbf{y}_{1:T}$ | $\mathbb{R}^{T \times d}$ | Full observed return matrix |
| $s_t$ | $\{0, \ldots, K-1\}$ | Latent regime at time $t$ |
| $s_{1:T}$ | $\{0, \ldots, K-1\}^T$ | Complete latent regime sequence |

### Parameters

| Symbol | Dimension | Definition |
|--------|-----------|------------|
| $\mathbf{P}$ | $[0,1]^{K \times K}$ | Transition matrix; $P_{jk} = \Pr(s_t = k \mid s_{t-1} = j)$ |
| $\boldsymbol{\pi}_0$ | $\Delta^{K-1}$ | Initial state distribution; $\pi_{0,k} = \Pr(s_1 = k)$ |
| $\boldsymbol{\mu}_k$ | $\mathbb{R}^d$ | Mean return vector for regime $k$ |
| $\boldsymbol{\Sigma}_k$ | $\mathbb{S}^d_{++}$ | Covariance matrix for regime $k$ |
| $\boldsymbol{\sigma}_k$ | $\mathbb{R}^d_{>0}$ | Per-asset standard deviations for regime $k$ |
| $\mathbf{R}_k$ | Correlation matrix | Correlation matrix for regime $k$ |
| $\mathbf{L}_k$ | Lower triangular | Cholesky factor of $\boldsymbol{\Sigma}_k$ |
| $\mathbf{D}_k$ | Diagonal | $\operatorname{diag}(\boldsymbol{\sigma}_k)$ |

### Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Collective notation for all continuous parameters $(\mathbf{P}, \boldsymbol{\mu}_{0:K-1}, \boldsymbol{\Sigma}_{0:K-1})$ |
| $\Delta^{K-1}$ | The $(K-1)$-simplex $\{x \in \mathbb{R}^K_{\geq 0} : \sum_k x_k = 1\}$ |
| $\mathbb{S}^d_{++}$ | The cone of $d \times d$ symmetric positive-definite matrices |
| $\operatorname{LSE}$ | Log-sum-exp: $\operatorname{LSE}(a_1, \ldots, a_n) = \log \sum_{i=1}^n e^{a_i}$ |
| $\mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | $d$-variate Normal distribution |
| $\operatorname{Dir}(\boldsymbol{\alpha})$ | Dirichlet distribution with concentration $\boldsymbol{\alpha}$ |

We use 0-based regime indexing throughout to match the implementation. Rows of $\mathbf{P}$ are stochastic: $\sum_{k=0}^{K-1} P_{jk} = 1$ for each $j$.

---

## 3. The Generative Model

### 3.1 Regime Dynamics

#### The Markov Property

We model the regime sequence $s_1, s_2, \ldots, s_T$ as a **first-order homogeneous Markov chain**. This means:

$$
\Pr(s_t = k \mid s_{t-1}, s_{t-2}, \ldots, s_1) = \Pr(s_t = k \mid s_{t-1})
$$

for all $t \geq 2$ and all $k$, and moreover these transition probabilities do not depend on $t$.

Why is this a reasonable approximation for economic regimes? The Markov property says that, given the current regime, the future evolution of regimes is independent of how we arrived at the current state. This is a strong assumption (one could argue that a bull market that has lasted 5 years is more fragile than one that started last month), but it yields a model that is both tractable and empirically successful. Hamilton (1989) showed that a two-state Markov chain captures the essential asymmetry of US business cycles (long expansions, shorter contractions) without needing to model duration dependence explicitly. The persistence of regimes is controlled by the diagonal entries of the transition matrix rather than by explicit duration distributions.

Homogeneity (time-invariance of $\mathbf{P}$) is a further simplification that we relax in Section 8.4 via time-varying transition probabilities.

#### The Transition Matrix

The transition matrix $\mathbf{P} \in [0,1]^{K \times K}$ is a row-stochastic matrix:

$$
P_{jk} = \Pr(s_t = k \mid s_{t-1} = j), \qquad \sum_{k=0}^{K-1} P_{jk} = 1, \quad P_{jk} \geq 0
$$

Each row $\mathbf{P}_{j,\cdot}$ is a probability vector on the simplex $\Delta^{K-1}$ and specifies the distribution over next-period regimes given that the current regime is $j$. The diagonal entry $P_{jj}$ is the **self-transition** (persistence) probability; the expected duration of regime $j$, under the geometric distribution implied by the Markov property, is:

$$
\mathbb{E}[\text{duration of regime } j] = \frac{1}{1 - P_{jj}}
$$

For example, $P_{jj} = 0.95$ implies an expected duration of 20 months, while $P_{jj} = 0.90$ implies 10 months.

#### The Initial Distribution

The initial regime is drawn from a distribution $\boldsymbol{\pi}_0$ on the simplex:

$$
s_1 \sim \operatorname{Categorical}(\boldsymbol{\pi}_0), \qquad \pi_{0,k} = \Pr(s_1 = k)
$$

In the current implementation we fix $\boldsymbol{\pi}_0 = (1/K, \ldots, 1/K)$, the uniform distribution. This is a simplification: the data-generating process uses a non-uniform $\boldsymbol{\pi}_0$ (specifically $[0.8, 0.2]$ for the synthetic data), so for short series ($T < 50$) there is a mild prior-data mismatch at the first time step. For the series lengths of interest ($T \geq 100$), the influence of the initial distribution is negligible relative to the likelihood from 100+ observations, and the simplification avoids adding two more parameters to sample.

An alternative would be to set $\boldsymbol{\pi}_0$ equal to the **stationary distribution** of $\mathbf{P}$, defined as the left eigenvector of $\mathbf{P}$ associated with eigenvalue 1:

$$
\boldsymbol{\pi}^* \mathbf{P} = \boldsymbol{\pi}^*, \qquad \sum_k \pi^*_k = 1
$$

This choice couples $\boldsymbol{\pi}_0$ to $\mathbf{P}$ and encodes the belief that the process has been running long enough to reach stationarity before the observation window begins. The FFBS post-processing step does use the stationary distribution as the initial distribution for each posterior draw.

#### Joint Probability of the Regime Sequence

The joint probability of a complete regime sequence $s_{1:T}$ factors as:

$$
\Pr(s_{1:T} \mid \mathbf{P}, \boldsymbol{\pi}_0) = \pi_{0, s_1} \prod_{t=2}^{T} P_{s_{t-1}, s_t}
$$

This factorisation is the defining property of a Markov chain and is what makes the forward algorithm (Section 4.2) possible: the joint over $K^T$ sequences can be computed in $\mathcal{O}(TK^2)$ by exploiting the chain structure.

### 3.2 Emission Distributions

#### Conditional Independence Structure

The emission (observation) model specifies how observed returns are generated given the latent regime. The key structural assumption is **conditional independence**: given the regime sequence $s_{1:T}$ and the parameters, the returns at different times are independent:

$$
p(\mathbf{y}_{1:T} \mid s_{1:T}, \theta) = \prod_{t=1}^{T} p(\mathbf{y}_t \mid s_t, \theta)
$$

This is the standard HMM assumption and it means that all temporal dependence in the observed data arises through the latent regime chain, not through direct dependence between successive returns. Within a single regime, returns are i.i.d. draws from the regime-specific distribution. This is a simplification (real equity returns exhibit serial correlation even within regimes), but it keeps the model tractable and is the standard starting point in the literature. Section 8.3 discusses an AR extension that relaxes this assumption.

#### Multivariate Normal Emissions (Current Implementation)

Conditional on $s_t = k$, the $d$-dimensional return vector follows a multivariate Normal:

$$
\mathbf{y}_t \mid (s_t = k) \sim \mathcal{N}_d(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

The density is:

$$
p(\mathbf{y}_t \mid s_t = k) = (2\pi)^{-d/2} |\boldsymbol{\Sigma}_k|^{-1/2} \exp\!\left( -\tfrac{1}{2} (\mathbf{y}_t - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{y}_t - \boldsymbol{\mu}_k) \right)
$$

Each regime $k$ has its own mean vector $\boldsymbol{\mu}_k \in \mathbb{R}^d$ and covariance matrix $\boldsymbol{\Sigma}_k \in \mathbb{S}^d_{++}$, giving a total of $K(d + d(d+1)/2)$ emission parameters. For the current setting ($K = 2$, $d = 3$), this is $2(3 + 6) = 18$ parameters, though the identity-correlation simplification reduces the effective count to $2(3 + 3) = 12$.

The multivariate Normal is the natural starting point: it is conjugate to many standard priors, its log-density is differentiable everywhere (enabling gradient-based sampling), and the regime mixture already generates portfolio-level fat tails and time-varying volatility even when each regime individually is Gaussian. The limitation is that it cannot capture **intra-regime** excess kurtosis. Section 8.1 addresses this with a Student-$t$ extension.

### 3.3 Covariance Parameterisation via LKJ Cholesky

#### The Decomposition

Working directly with covariance matrices $\boldsymbol{\Sigma}_k$ as free parameters is problematic: one must enforce the positive-definiteness constraint $\boldsymbol{\Sigma}_k \succ 0$ during sampling, which is difficult in high dimensions. The standard Bayesian approach, implemented in PyMC's `LKJCholeskyCov`, decomposes the covariance matrix into scale and correlation components:

$$
\boldsymbol{\Sigma}_k = \mathbf{D}_k \, \mathbf{R}_k \, \mathbf{D}_k
$$

where:

- $\mathbf{D}_k = \operatorname{diag}(\sigma_{k,1}, \ldots, \sigma_{k,d})$ is a diagonal matrix of per-asset standard deviations ($\sigma_{k,i} > 0$),
- $\mathbf{R}_k$ is a $d \times d$ correlation matrix (symmetric, positive-definite with unit diagonal).

This decomposition separates the **scale** of each asset's volatility (the $\sigma_{k,i}$) from the **dependence structure** (the off-diagonal entries of $\mathbf{R}_k$), allowing separate priors on each component.

#### Why Cholesky?

Rather than parameterising $\mathbf{R}_k$ directly (which would require enforcing that all eigenvalues are positive and all diagonal entries equal 1), we work with its Cholesky factor. The full covariance Cholesky factor is:

$$
\boldsymbol{\Sigma}_k = \mathbf{L}_k \mathbf{L}_k^\top, \qquad \mathbf{L}_k = \mathbf{D}_k \, \mathbf{C}_k
$$

where $\mathbf{C}_k$ is the lower-triangular Cholesky factor of $\mathbf{R}_k$ (so $\mathbf{R}_k = \mathbf{C}_k \mathbf{C}_k^\top$). Working with $\mathbf{L}_k$ has three advantages:

1. **Guaranteed positive-definiteness.** For any lower-triangular $\mathbf{L}_k$ with positive diagonal entries, $\mathbf{L}_k \mathbf{L}_k^\top$ is automatically symmetric positive-definite. No additional constraints are needed during sampling.

2. **Unconstrained parameterisation.** The off-diagonal entries of $\mathbf{L}_k$ are unconstrained reals; the diagonal entries are constrained to be positive (enforced via a log-transform). NUTS operates on this unconstrained space.

3. **Efficient computation.** Sampling from $\mathcal{N}_d(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ requires only a triangular solve with $\mathbf{L}_k$, and the log-determinant is $\log|\boldsymbol{\Sigma}_k| = 2 \sum_i \log L_{k,ii}$.

#### The LKJ Prior on Correlations

The LKJ distribution (Lewandowski, Kurowicka, and Joe 2009) is a distribution over $d \times d$ correlation matrices parameterised by a single shape parameter $\eta > 0$. Its density is:

$$
p(\mathbf{R}_k \mid \eta) \propto |\mathbf{R}_k|^{\eta - 1}
$$

The parameter $\eta$ controls the concentration:

- $\eta = 1$: uniform distribution over all valid correlation matrices.
- $\eta > 1$: increasingly concentrated around the identity matrix $\mathbf{I}$ (shrinkage toward uncorrelated assets).
- $\eta < 1$: concentrated toward singular correlation matrices (extreme correlations).

For financial applications, $\eta = 2$ is a common default. It provides mild regularisation toward uncorrelated assets while allowing the data to drive the posterior toward whatever correlation structure is present. This is particularly important in regimes with few observations, where an unrestricted correlation matrix would be poorly identified.

#### Current Simplification: $\mathbf{R}_k = \mathbf{I}$

In the current implementation (Notebook 01), we fix $\mathbf{R}_k = \mathbf{I}$ for all regimes, so that:

$$
\boldsymbol{\Sigma}_k = \mathbf{D}_k^2 = \operatorname{diag}(\sigma_{k,1}^2, \ldots, \sigma_{k,d}^2)
$$

This implies **conditional independence given the regime**: $y_{t,i} \perp y_{t,j} \mid s_t$ for $i \neq j$. In practical terms, once we know the regime, knowing the return of one asset tells us nothing about the return of another beyond what the regime label itself implies. All observed cross-asset co-movement in the data arises from the shared latent regime, not from within-regime correlation.

This simplification reduces the number of covariance parameters from $K \cdot d(d+1)/2$ to $K \cdot d$ and keeps inference fast, allowing us to focus on learning the regime means, volatilities, and transition dynamics. Despite setting $\mathbf{R}_k = \mathbf{I}$ in the generative process, the model still uses `LKJCholeskyCov` in the prior specification (with the LKJ component active), so the infrastructure for regime-dependent correlations is already in place. Section 8.2 describes the extension to $\mathbf{R}_k \neq \mathbf{I}$.

### 3.4 The Complete Generative Process

Assembling the components above, the full generative model is:

**Prior draws:**

$$
\begin{aligned}
\mathbf{P}_{j,\cdot} &\sim \operatorname{Dir}(\boldsymbol{\alpha}_j), \quad j = 0, \ldots, K-1 \\
\mu_{k,i} &\sim \mathcal{N}(0, \sigma_\mu^2), \quad k = 0, \ldots, K-1, \; i = 1, \ldots, d \\
\sigma_{k,i} &\sim \operatorname{HalfNormal}(\tau_\sigma), \quad k = 0, \ldots, K-1, \; i = 1, \ldots, d \\
\mathbf{R}_k &\sim \operatorname{LKJ}(\eta), \quad k = 0, \ldots, K-1
\end{aligned}
$$

**Data generation:**

$$
\begin{aligned}
s_1 &\sim \operatorname{Categorical}(\boldsymbol{\pi}_0) \\
s_t \mid s_{t-1} &\sim \operatorname{Categorical}(\mathbf{P}_{s_{t-1}, \cdot}), \quad t = 2, \ldots, T \\
\mathbf{y}_t \mid s_t = k &\sim \mathcal{N}_d(\boldsymbol{\mu}_k, \mathbf{D}_k \mathbf{R}_k \mathbf{D}_k), \quad t = 1, \ldots, T
\end{aligned}
$$

The joint distribution of all variables (observed, latent, and parameters) is therefore:

$$
p(\mathbf{y}_{1:T}, s_{1:T}, \theta) = \underbrace{p(\theta)}_{\text{priors}} \cdot \underbrace{\pi_{0,s_1} \prod_{t=2}^T P_{s_{t-1}, s_t}}_{\text{regime chain}} \cdot \underbrace{\prod_{t=1}^T p(\mathbf{y}_t \mid s_t, \theta)}_{\text{emissions}}
$$

The inference task is to compute the posterior $p(\theta \mid \mathbf{y}_{1:T})$ and the regime posterior $p(s_{1:T} \mid \mathbf{y}_{1:T})$.

---

## 4. Bayesian Inference via Marginalisation

### 4.1 Why Marginalise Over Regimes

The posterior of interest is:

$$
p(\theta \mid \mathbf{y}_{1:T}) \propto p(\theta) \cdot p(\mathbf{y}_{1:T} \mid \theta)
$$

where the **marginalised likelihood** is obtained by summing over all possible regime sequences:

$$
p(\mathbf{y}_{1:T} \mid \theta) = \sum_{s_{1:T} \in \{0, \ldots, K-1\}^T} p(\mathbf{y}_{1:T}, s_{1:T} \mid \theta)
$$

This sum involves $K^T$ terms (for $K = 2$ and $T = 120$, this is $2^{120} \approx 10^{36}$), making direct enumeration impossible. However, the Markov structure of the regime chain makes it possible to compute this sum exactly in $\mathcal{O}(TK^2)$ time using the forward algorithm (Section 4.2).

Why marginalise rather than sampling the regime sequence directly? The reason is computational. We use the No-U-Turn Sampler (NUTS), a gradient-based Hamiltonian Monte Carlo method, to explore the posterior. NUTS requires the target density to be a differentiable function of **continuous** parameters. The discrete regime variables $s_t \in \{0, \ldots, K-1\}$ are not differentiable, so NUTS cannot sample them. There are three possible approaches:

1. **Gibbs sampling**: alternate between sampling $s_{1:T}$ conditional on $\theta$ (discrete full-conditional) and sampling $\theta$ conditional on $s_{1:T}$ (continuous full-conditional). This is conceptually simple but mixes poorly in practice: the discrete regime sequence and the continuous parameters are strongly coupled, so updating one while holding the other fixed leads to slow exploration.

2. **Marginalise and sample only $\theta$**: sum out $s_{1:T}$ analytically to obtain a smooth, differentiable marginal likelihood $p(\mathbf{y}_{1:T} \mid \theta)$ in the continuous parameters only, then use NUTS on this marginalised target. This is what we do.

3. **Marginalise at each time step** (online Rao-Blackwellisation): a hybrid approach used in particle MCMC methods.

Approach 2 is preferred because it yields a **smooth likelihood surface** in the continuous parameter space, allowing NUTS to compute efficient gradient-based proposals. The regime sequence is recovered post-hoc via the FFBS (Section 5), which is both exact and fast given a posterior draw of $\theta$.

### 4.2 The Forward Algorithm

The forward algorithm computes the marginalised likelihood by building up partial sums from left to right across the time series. Define the **forward variable**:

$$
\alpha_t(k) \triangleq p(\mathbf{y}_{1:t}, s_t = k \mid \theta)
$$

This is the joint probability of observing the first $t$ returns **and** being in regime $k$ at time $t$. The marginalised likelihood is obtained by summing the final forward variable over regimes:

$$
p(\mathbf{y}_{1:T} \mid \theta) = \sum_{k=0}^{K-1} \alpha_T(k)
$$

**Derivation of the recursion.** At $t = 1$:

$$
\alpha_1(k) = p(\mathbf{y}_1, s_1 = k \mid \theta) = p(s_1 = k) \cdot p(\mathbf{y}_1 \mid s_1 = k, \theta) = \pi_{0,k} \cdot f_k(\mathbf{y}_1)
$$

where $f_k(\mathbf{y}) \triangleq p(\mathbf{y} \mid s = k, \theta)$ is the emission density for regime $k$.

For $t \geq 2$, we marginalise over the previous regime:

$$
\begin{aligned}
\alpha_t(k) &= p(\mathbf{y}_{1:t}, s_t = k \mid \theta) \\
&= \sum_{k'=0}^{K-1} p(\mathbf{y}_{1:t}, s_{t-1} = k', s_t = k \mid \theta) \\
&= \sum_{k'=0}^{K-1} p(\mathbf{y}_{1:t-1}, s_{t-1} = k' \mid \theta) \cdot p(s_t = k \mid s_{t-1} = k') \cdot p(\mathbf{y}_t \mid s_t = k, \theta) \\
&= \left[\sum_{k'=0}^{K-1} \alpha_{t-1}(k') \, P_{k',k}\right] \cdot f_k(\mathbf{y}_t)
\end{aligned}
$$

The third line uses the conditional independence assumptions of the HMM: (i) $\mathbf{y}_t$ depends on $s_t$ but not on $s_{t-1}$ or $\mathbf{y}_{1:t-1}$ given $s_t$, and (ii) $s_t$ depends on $s_{t-1}$ but not on $\mathbf{y}_{1:t-1}$ given $s_{t-1}$.

In matrix form, defining $\boldsymbol{\alpha}_t = (\alpha_t(0), \ldots, \alpha_t(K-1))^\top$ and $\mathbf{f}_t = (f_0(\mathbf{y}_t), \ldots, f_{K-1}(\mathbf{y}_t))^\top$:

$$
\boldsymbol{\alpha}_t = \operatorname{diag}(\mathbf{f}_t) \, \mathbf{P}^\top \boldsymbol{\alpha}_{t-1}
$$

The total cost is $\mathcal{O}(TK^2)$ for the matrix-vector products, plus $\mathcal{O}(TKd^2)$ for evaluating the $K$ emission densities at each of $T$ time steps.

### 4.3 Normalised Forward Recursion and Numerical Stability

The forward variables $\alpha_t(k)$ are products of probabilities and densities. As $T$ grows, these products shrink toward zero exponentially fast, causing numerical underflow in floating-point arithmetic. Two techniques address this.

**Log-space computation.** We work with $\log \alpha_t(k)$ throughout and use the **log-sum-exp** (LSE) operation for the marginalisation step:

$$
\operatorname{LSE}(a_1, \ldots, a_n) = \log \sum_{i=1}^n e^{a_i} = a_{\max} + \log \sum_{i=1}^n e^{a_i - a_{\max}}
$$

where $a_{\max} = \max_i a_i$. Subtracting $a_{\max}$ before exponentiating ensures numerical stability: the largest term contributes $e^0 = 1$, and the smaller terms contribute values in $[0, 1]$.

The log-space recursion is:

$$
\log \alpha_t(k) = \operatorname{LSE}_{k'}\!\left(\log \alpha_{t-1}(k') + \log P_{k',k}\right) + \log f_k(\mathbf{y}_t)
$$

**Normalised recursion.** Even in log-space, the values $\log \alpha_t(k)$ drift to large negative values as $T$ grows (roughly $\mathcal{O}(-T)$), which can produce ill-conditioned gradients when differentiated through `pytensor.scan` for NUTS sampling. The solution is to **normalise** at each step.

Define the normalisation constant:

$$
c_t = \operatorname{LSE}_k\!\left(\log \alpha_t(k)\right) = \log \sum_k \alpha_t(k)
$$

and the normalised forward variable:

$$
\widetilde{\log \alpha}_t(k) = \log \alpha_t(k) - c_t
$$

After normalisation, $\sum_k \exp(\widetilde{\log \alpha}_t(k)) = 1$, so the normalised forward variables remain near zero in log-space regardless of $T$.

The recursion becomes:

1. **Initialise:** $\log \alpha_1(k) = \log \pi_{0,k} + \log f_k(\mathbf{y}_1)$; $c_1 = \operatorname{LSE}_k(\log \alpha_1(k))$; $\widetilde{\log \alpha}_1(k) = \log \alpha_1(k) - c_1$.

2. **Recurse** ($t = 2, \ldots, T$):

$$
\begin{aligned}
\log \alpha_t(k) &= \operatorname{LSE}_{k'}\!\left(\widetilde{\log \alpha}_{t-1}(k') + \log P_{k',k}\right) + \log f_k(\mathbf{y}_t) \\
c_t &= \operatorname{LSE}_k\!\left(\log \alpha_t(k)\right) \\
\widetilde{\log \alpha}_t(k) &= \log \alpha_t(k) - c_t
\end{aligned}
$$

3. **Terminate:** The marginalised log-likelihood is:

$$
\log p(\mathbf{y}_{1:T} \mid \theta) = \sum_{t=1}^{T} c_t + \operatorname{LSE}_k\!\left(\widetilde{\log \alpha}_T(k)\right)
$$

To see why this is correct, note that $\log \alpha_t(k) = \widetilde{\log \alpha}_t(k) + c_t$, so by induction $\log \alpha_t(k) = \widetilde{\log \alpha}_t(k) + \sum_{\tau=1}^t c_\tau$. The marginalised log-likelihood is $\operatorname{LSE}_k(\log \alpha_T(k)) = \operatorname{LSE}_k(\widetilde{\log \alpha}_T(k)) + \sum_{t=1}^T c_t$, since the constant $\sum_t c_t$ factors out of the LSE.

In practice, since $\widetilde{\log \alpha}_T$ sums to zero, the final $\operatorname{LSE}$ term is close to $\log K$ (exactly $\log K$ if all regimes are equally likely at $T$) and contributes a small correction.

### 4.4 Connection to PyMC via pm.Potential

PyMC builds a probabilistic model as a computational graph whose joint log-density is the sum of log-prior densities and log-likelihood contributions from observed random variables. The standard mechanism for adding an observed variable is `pm.Normal("obs", mu=..., observed=data)`, which adds $\sum_t \log p(y_t \mid \mu, \sigma)$ to the joint log-density.

In our model, the observed data do not enter through a standard PyMC distribution because the likelihood involves the forward algorithm over latent discrete states. Instead, we compute the scalar $\log p(\mathbf{y}_{1:T} \mid \theta)$ ourselves and add it to the model's joint log-density via `pm.Potential`:

```python
pm.Potential("hmm_loglik", total_ll)
```

A `pm.Potential` adds an arbitrary scalar to the log-density without defining a random variable. From NUTS's perspective, the model's unnormalised log-posterior is:

$$
\log \tilde{p}(\theta \mid \mathbf{y}_{1:T}) = \underbrace{\log p(\theta)}_{\text{sum of prior log-densities}} + \underbrace{\log p(\mathbf{y}_{1:T} \mid \theta)}_{\text{Potential}}
$$

The forward recursion is implemented via `pytensor.scan`, which compiles to `jax.lax.scan` under the NumPyro backend. This means:

- The recursion is JIT-compiled and runs as a native JAX loop, avoiding Python overhead.
- Automatic differentiation through the scan provides exact gradients of the log-likelihood with respect to all continuous parameters, which NUTS requires for its leapfrog integrator.
- The computational cost per gradient evaluation is $\mathcal{O}(TK^2)$, which is the cost of the forward pass itself (reverse-mode AD through a scan has the same asymptotic cost as the forward pass).

### 4.5 Prior Specification and Hyperparameter Choices

#### Transition Matrix: Sticky Dirichlet Prior

Each row of $\mathbf{P}$ is given an independent Dirichlet prior:

$$
\mathbf{P}_{j,\cdot} \sim \operatorname{Dir}(\alpha_{j,0}, \ldots, \alpha_{j,K-1})
$$

with a **sticky** (persistence-encouraging) concentration vector:

$$
\alpha_{j,k} = \begin{cases} \alpha_{\text{diag}} = 20 & \text{if } k = j \\ \alpha_{\text{off}} = 2 & \text{if } k \neq j \end{cases}
$$

The Dirichlet distribution on the $K$-simplex has density:

$$
p(\mathbf{p} \mid \boldsymbol{\alpha}) \propto \prod_{k=0}^{K-1} p_k^{\alpha_k - 1}
$$

The mean of the Dirichlet is $\mathbb{E}[p_k] = \alpha_k / \alpha_0$ where $\alpha_0 = \sum_k \alpha_k$. For our choice with $K = 2$:

$$
\mathbb{E}[P_{jj}] = \frac{20}{20 + 2} = \frac{20}{22} \approx 0.909
$$

This corresponds to a prior expected regime duration of $1/(1 - 0.909) \approx 11$ months. The total concentration $\alpha_0 = 22$ controls the prior's strength: it is equivalent to having observed 22 pseudo-transitions, of which 20 were self-transitions. This is moderately informative; it encodes the belief that regimes persist for multiple months, consistent with observed market dynamics where bull phases last years and bear episodes last months to quarters (Hamilton 1989). Nevertheless, the likelihood from 100+ actual observations will dominate.

Why **sticky** priors? Without the asymmetry ($\alpha_{\text{diag}} > \alpha_{\text{off}}$), a symmetric Dirichlet prior with moderate concentration would place substantial mass on transition matrices with $P_{jj} \approx 0.5$, implying regimes that last on average 2 months. This is inconsistent with economic reality and, more importantly, it creates an identifiability problem: if regimes switch every few months, they become hard to distinguish from a single-regime model with excess variance. The sticky prior anchors the model toward the persistent-regime region of parameter space, stabilising inference.

The prior is symmetric across regimes: both rows of $\mathbf{P}$ have the same concentration. This means the prior does not favour any particular regime ordering, which is related to the label-switching symmetry discussed in Section 7.

#### Regime Means: Normal Prior

$$
\mu_{k,i} \sim \mathcal{N}(0, \sigma_\mu^2), \qquad \sigma_\mu = 0.05
$$

independently for each regime $k$ and asset $i$. The prior is centered at zero monthly return, with a standard deviation of 5% monthly. This accommodates:

- Bull-regime drift: typically $+0.5\%$ to $+1.5\%$ monthly ($+6\%$ to $+18\%$ annualised), well within $\pm 1\sigma$.
- Bear-regime drift: typically $-0.5\%$ to $-3\%$ monthly, also within $\pm 1\sigma$.
- Extreme scenarios: $\pm 10\%$ monthly ($\pm 2\sigma$), corresponding to annualised returns of $\pm 120\%$.

The prior is weakly informative in the sense of Gelman et al. (2013, Ch. 5): it rules out implausible values (e.g., $+50\%$ monthly mean) while leaving plenty of room for the data to determine the posterior. The independence across assets and regimes is a simplification; a hierarchical prior that shares information across assets is a possible extension but is outside the scope of the current model.

#### Regime Covariances: LKJ + HalfNormal

The covariance prior is specified via PyMC's `LKJCholeskyCov`, which jointly specifies priors on the standard deviations and the correlation matrix:

$$
\begin{aligned}
\sigma_{k,i} &\sim \operatorname{HalfNormal}(\tau_\sigma), \qquad \tau_\sigma = 0.10 \\
\mathbf{R}_k &\sim \operatorname{LKJ}(\eta), \qquad \eta = 2
\end{aligned}
$$

**Standard deviations.** The $\operatorname{HalfNormal}(0.10)$ prior has its mode at 0 and a 95th percentile at approximately $0.10 \times 1.96 = 0.196$, i.e., about 20% monthly standard deviation or $20\% \times \sqrt{12} \approx 69\%$ annualised. This comfortably covers:

- Bull-regime volatilities: typically 3--5% monthly (10--17% annualised).
- Bear-regime volatilities: typically 6--12% monthly (21--42% annualised).

The prior places low density on monthly volatilities above 20%, which is appropriate for equity indices (even during the 2008 crisis, the S\&P 500's monthly realised volatility peaked around 15--20%).

**Correlations.** With $\eta = 2$, the LKJ prior shrinks correlations mildly toward zero (the identity matrix). For $d = 3$, the marginal prior on each off-diagonal correlation has approximately 90% of its mass in $[-0.7, 0.7]$. This allows the data to drive correlations toward whatever structure is present while providing regularisation that prevents poorly-identified correlation parameters from causing sampling difficulties.

#### Initial Distribution

As noted in Section 3.1, the initial distribution is fixed to uniform:

$$
\boldsymbol{\pi}_0 = (1/K, \ldots, 1/K)
$$

This is not a prior in the Bayesian sense (it is not updated by the data) but a fixed constant in the likelihood. For $T \geq 100$, the effect of $\boldsymbol{\pi}_0$ on the marginalised likelihood is negligible: the contribution of the first time step is $\mathcal{O}(1)$ while the remaining $T - 1$ steps contribute $\mathcal{O}(T)$.

---

## 5. Regime Recovery: The Forward-Filter Backward-Sampler

### 5.1 Motivation

The marginalised model samples the continuous parameters $\theta$ from the posterior $p(\theta \mid \mathbf{y}_{1:T})$ but does not produce samples of the regime sequence $s_{1:T}$, which was integrated out. To recover the regimes, we need a separate post-hoc step that, for each posterior draw $\theta^{(m)}$, samples:

$$
s_{1:T}^{(m)} \sim p(s_{1:T} \mid \mathbf{y}_{1:T}, \theta^{(m)})
$$

This is the **smoothed** regime posterior: the distribution over regime sequences conditional on the full data and a specific parameter draw. The collection of samples $\{(\theta^{(m)}, s_{1:T}^{(m)})\}_{m=1}^M$ then represents the full joint posterior over parameters and regimes.

The Forward-Filter Backward-Sampler (FFBS) accomplishes this in $\mathcal{O}(TK)$ time per posterior draw.

### 5.2 The Forward Filtering Pass

The forward pass is identical to the forward algorithm of Section 4.2, except that we store the normalised forward variables $\widetilde{\boldsymbol{\alpha}}_t$ for all $t$ (not just the final time step). After normalisation, these give the **filtered probabilities**:

$$
\gamma_t^{\text{filt}}(k) \triangleq p(s_t = k \mid \mathbf{y}_{1:t}, \theta) = \frac{\alpha_t(k)}{\sum_{k'} \alpha_t(k')}
$$

which is precisely $\exp(\widetilde{\log \alpha}_t(k))$ from the normalised recursion.

### 5.3 The Backward Sampling Pass

Given the stored filtered probabilities $\gamma_t^{\text{filt}}(k)$ for $t = 1, \ldots, T$, the backward pass samples the regime sequence in reverse order.

**Terminal step.** Sample the final regime from the filtered distribution at time $T$:

$$
s_T \sim \operatorname{Categorical}\!\left(\gamma_T^{\text{filt}}(0), \ldots, \gamma_T^{\text{filt}}(K-1)\right)
$$

**Backward recursion** ($t = T-1, T-2, \ldots, 1$). Given $s_{t+1}$, we need:

$$
p(s_t = k \mid s_{t+1}, \mathbf{y}_{1:T}, \theta)
$$

By the Markov property of the HMM and the fact that $\mathbf{y}_{t+1:T}$ is conditionally independent of $s_t$ given $s_{t+1}$ and $\theta$, this simplifies. Using Bayes' rule:

$$
\begin{aligned}
p(s_t = k \mid s_{t+1}, \mathbf{y}_{1:T}, \theta) &\propto p(s_{t+1} \mid s_t = k, \mathbf{y}_{1:t}, \theta) \cdot p(s_t = k \mid \mathbf{y}_{1:t}, \theta) \\
&= P_{k, s_{t+1}} \cdot \gamma_t^{\text{filt}}(k)
\end{aligned}
$$

The first line applies Bayes' rule with $p(s_t = k \mid \mathbf{y}_{1:t}, \theta)$ as the "prior" and $p(s_{t+1} \mid s_t = k)$ as the "likelihood" (the transition probability from the Markov chain). The second line uses the fact that, given $s_t$, the transition to $s_{t+1}$ depends only on $s_t$ (Markov property), not on $\mathbf{y}_{1:t}$.

Define the backward sampling weights:

$$
\tilde{\gamma}_t(k) \propto \gamma_t^{\text{filt}}(k) \cdot P_{k, s_{t+1}}
$$

In log-space:

$$
\log \tilde{\gamma}_t(k) = \widetilde{\log \alpha}_t(k) + \log P_{k, s_{t+1}} + \text{const.}
$$

Normalise and sample: $s_t \sim \operatorname{Categorical}(\tilde{\gamma}_t)$.

The complete FFBS algorithm is:

> **Algorithm: FFBS**
>
> **Input:** Data $\mathbf{y}_{1:T}$, parameters $\theta = (\mathbf{P}, \boldsymbol{\mu}_{0:K-1}, \boldsymbol{\Sigma}_{0:K-1})$
>
> **Forward pass:**
> 1. Compute $\log f_k(\mathbf{y}_t)$ for all $t, k$
> 2. Run the normalised forward recursion (Section 4.3) storing $\widetilde{\log \alpha}_t(k)$ for all $t$
>
> **Backward pass:**
> 3. Sample $s_T \sim \operatorname{Categorical}(\exp(\widetilde{\log \alpha}_T))$
> 4. For $t = T-1, T-2, \ldots, 1$:
>    - Compute $\log \tilde{\gamma}_t(k) = \widetilde{\log \alpha}_t(k) + \log P_{k, s_{t+1}}$
>    - Normalise: $\log \tilde{\gamma}_t \leftarrow \log \tilde{\gamma}_t - \operatorname{LSE}_k(\log \tilde{\gamma}_t)$
>    - Sample $s_t \sim \operatorname{Categorical}(\exp(\log \tilde{\gamma}_t))$
>
> **Output:** Regime sequence $s_{1:T}$

The cost is $\mathcal{O}(TK^2)$ for the forward pass (dominated by the matrix-vector products) and $\mathcal{O}(TK)$ for the backward pass (just vector operations at each step), giving $\mathcal{O}(TK^2)$ overall.

### 5.4 Smoothed versus Filtered Probabilities

The FFBS produces **smoothed** regime probabilities, while the forward pass alone produces **filtered** probabilities. These are fundamentally different quantities with different use cases.

**Filtered probability:**

$$
\gamma_t^{\text{filt}}(k) = p(s_t = k \mid \mathbf{y}_{1:t}, \theta)
$$

This conditions on data up to and including time $t$. It represents the best estimate of the current regime using only past and present information. No future data is used.

**Smoothed probability:**

$$
\gamma_t^{\text{smooth}}(k) = p(s_t = k \mid \mathbf{y}_{1:T}, \theta)
$$

This conditions on the **entire** data series, including observations after time $t$. It uses the future to refine the estimate of the past.

The smoothed probability is always at least as informative as the filtered probability (it conditions on a superset of the data), and in practice it is substantially more decisive near regime transition points. At a transition from bull to bear, the filtered probability may take several months to shift from low to high bear probability (it "sees" the transition only through accumulating evidence), while the smoothed probability shifts sharply because it also "sees" the bear-regime returns that follow.

The key distinction for applications:

| | Smoothed $\gamma_t^{\text{smooth}}$ | Filtered $\gamma_t^{\text{filt}}$ |
|---|---|---|
| **Information set** | $\mathbf{y}_{1:T}$ (full sample) | $\mathbf{y}_{1:t}$ (causal) |
| **Look-ahead bias** | Yes | No |
| **Use case** | Retrospective analysis, model validation, historical regime dating | Real-time allocation, out-of-sample decisions |
| **Computed by** | FFBS backward pass | Forward pass only |

For portfolio allocation, one **must** use filtered probabilities (Section 6). Smoothed probabilities are appropriate for understanding the historical regime structure but would introduce look-ahead bias if used for allocation decisions.

---

## 6. Filtered Probabilities and Causal Allocation

### Bayesian Model Averaging of Filtered Probabilities

A single posterior draw $\theta^{(m)}$ produces a single set of filtered probabilities $\gamma_t^{\text{filt}}(k; \theta^{(m)})$. To obtain a point estimate that integrates over parameter uncertainty, we average across posterior draws:

$$
\bar{\gamma}_t^{\text{filt}}(k) = \frac{1}{M} \sum_{m=1}^{M} \gamma_t^{\text{filt}}(k; \theta^{(m)})
$$

where $M$ is the number of posterior draws (possibly thinned). This is a Monte Carlo approximation to the Bayesian model average:

$$
p(s_t = k \mid \mathbf{y}_{1:t}) = \int p(s_t = k \mid \mathbf{y}_{1:t}, \theta) \, p(\theta \mid \mathbf{y}_{1:T}) \, d\theta
$$

Note the subtle point: the filtered probability conditions on $\mathbf{y}_{1:t}$, but the posterior over $\theta$ conditions on $\mathbf{y}_{1:T}$. In a true real-time setting, the posterior at time $t$ would condition only on $\mathbf{y}_{1:t}$, requiring re-estimation at each time step. The current approach, where parameters are estimated on the full sample and then filtered probabilities are computed, embeds an indirect form of in-sample information. Walk-forward re-estimation (Section 8.5) addresses this.

### The Allocation Rule

At each decision time $t$, the portfolio weight is determined by the filtered probability from the **previous** month (to avoid using $\mathbf{y}_t$ in the decision for time $t$):

$$
w_t^{\text{equity}} = \begin{cases}
0.80 & \text{if } \bar{\gamma}_{t-1}^{\text{filt}}(\text{Bear}) \leq 0.50 \\
0.20 & \text{if } \bar{\gamma}_{t-1}^{\text{filt}}(\text{Bear}) > 0.50
\end{cases}
$$

The complementary weight $1 - w_t^{\text{equity}}$ is allocated to a risk-free asset yielding a fixed rate of $r_f = 4\%$ per annum ($r_f^{\text{monthly}} = (1.04)^{1/12} - 1 \approx 0.327\%$). The portfolio return at time $t$ is:

$$
r_t^{\text{portfolio}} = w_t^{\text{equity}} \cdot r_t^{\text{equity}} + (1 - w_t^{\text{equity}}) \cdot r_f^{\text{monthly}}
$$

where $r_t^{\text{equity}}$ is the equal-weighted average of the $d$ equity index returns.

This is a **binary switch** rule: the equity allocation jumps between 80% and 20% depending on whether the bear probability exceeds a threshold. More sophisticated rules could use continuous functions of $\bar{\gamma}_t^{\text{filt}}$, but the binary rule has the virtue of transparency and is sufficient to demonstrate the value of regime information.

The rule uses $\bar{\gamma}_{t-1}^{\text{filt}}$ rather than $\bar{\gamma}_t^{\text{filt}}$ to ensure strict causality: the allocation for month $t$ depends only on information available at the end of month $t-1$, before the returns $\mathbf{y}_t$ are realised.

---

## 7. Label Switching

### 7.1 The Symmetry Problem

Label switching is a **fundamental symmetry of the likelihood** in all mixture and HMM models. The key observation is that the regime labels $\{0, 1, \ldots, K-1\}$ are arbitrary: if we permute the labels and simultaneously permute all associated parameters, the likelihood is unchanged.

Formally, let $\sigma: \{0, \ldots, K-1\} \to \{0, \ldots, K-1\}$ be any permutation. Define the permuted parameter:

$$
\sigma(\theta) = \left(\sigma(\mathbf{P}), \boldsymbol{\mu}_{\sigma(0)}, \ldots, \boldsymbol{\mu}_{\sigma(K-1)}, \boldsymbol{\Sigma}_{\sigma(0)}, \ldots, \boldsymbol{\Sigma}_{\sigma(K-1)}\right)
$$

where $[\sigma(\mathbf{P})]_{jk} = P_{\sigma^{-1}(j), \sigma^{-1}(k)}$ (permute both rows and columns). Then:

$$
p(\mathbf{y}_{1:T} \mid \theta) = p(\mathbf{y}_{1:T} \mid \sigma(\theta))
$$

This holds because the marginalised likelihood sums over all regime sequences, and the sum is invariant to relabelling. In the inner sum, replacing $s_t$ with $\sigma(s_t)$ and $\theta$ with $\sigma(\theta)$ leaves every term unchanged.

With $K$ regimes, there are $K!$ such permutations. If the prior is also invariant under permutations (which our symmetric sticky Dirichlet prior is), then the posterior has $K!$ equivalent modes:

$$
p(\theta \mid \mathbf{y}_{1:T}) = p(\sigma(\theta) \mid \mathbf{y}_{1:T}) \qquad \forall \sigma \in S_K
$$

### 7.2 Consequences for Multi-Chain Inference

The multimodality caused by label switching has practical consequences for MCMC inference:

1. **Within-chain mixing.** If a single chain explores multiple modes (i.e., switches between label assignments during sampling), the marginal posteriors for regime-specific parameters become multimodal mixtures that are difficult to interpret. In practice, with a reasonably informative likelihood and sticky priors, individual chains tend to stay in a single mode.

2. **Cross-chain diagnostics.** Different chains may converge to different modes. Chain 0 might learn "regime 0 = Bull, regime 1 = Bear" while chain 1 learns "regime 0 = Bear, regime 1 = Bull." Both are equally valid. But naively computing $\hat{R}$ across chains compares $\mu_{\text{Bull}}$ from chain 0 against $\mu_{\text{Bear}}$ from chain 1, producing a spuriously large $\hat{R}$ that incorrectly signals non-convergence.

3. **Posterior averaging.** Naively averaging the posterior across chains mixes the modes, yielding meaningless averaged parameters (e.g., an average of the bull and bear means).

### 7.3 Resolution via Post-Hoc Permutation Alignment

Our approach is to align labels across chains after sampling, using chain 0 as the reference. For each chain $c$, we find the permutation $\sigma_c$ that best aligns its posterior to chain 0.

**Parameter-space alignment.** For each chain $c$, compute the draw-averaged regime means:

$$
\bar{\boldsymbol{\mu}}^{(c)} = \frac{1}{M_c} \sum_{m=1}^{M_c} \boldsymbol{\mu}^{(c,m)} \in \mathbb{R}^{K \times d}
$$

Find the permutation minimising the sum of squared differences to the reference:

$$
\sigma_c = \arg\min_{\sigma \in S_K} \sum_{k=0}^{K-1} \sum_{i=1}^{d} \left(\bar{\mu}^{(c)}_{\sigma(k), i} - \bar{\mu}^{(0)}_{k, i}\right)^2
$$

For $K = 2$ this is a comparison of two permutations (identity and swap), so the cost is trivial. For general $K$ the cost is $\mathcal{O}(K! \cdot Kd)$, which is acceptable for small $K$.

Having found $\sigma_c$, we apply it to all draws in chain $c$: permute the rows (and columns, for $\mathbf{P}$) of every parameter sample. After alignment, cross-chain $\hat{R}$ and ESS are computed on the relabelled posterior.

**FFBS alignment.** For the regime-sequence samples, we align by maximising element-wise agreement with the reference chain:

$$
\sigma_c = \arg\max_{\sigma \in S_K} \frac{1}{M_c \cdot T} \sum_{m,t} \mathbf{1}[\sigma(s_t^{(c,m)}) = s_t^{(0,m)}]
$$

This is implemented efficiently by broadcasting all $K!$ permutation mappings at once.

Post-hoc relabelling is simple, exact for small $K$, and sufficient for the two-regime case. For larger $K$, more sophisticated methods, such as the probabilistic relabelling of Stephens (2000) or the pivotal reordering method of Marin, Mengersen, and Robert (2005), may be needed.

An alternative to post-hoc alignment is to impose an **ordering constraint** on the parameters during sampling, such as $\mu_{0,1} < \mu_{1,1} < \cdots < \mu_{K-1,1}$ (ordering by the mean of the first asset). This breaks the symmetry and confines each chain to a single mode. The implementation supports this via a soft constraint (`order_means=True`) that adds a penalty $-10^{10}$ when the ordering is violated. However, ordering constraints can distort the posterior geometry near the constraint boundary and slow down NUTS; the post-hoc approach avoids this.

---

## 8. Planned Extensions

### 8.1 Student-$t$ Emissions

The multivariate Normal emission model cannot capture excess kurtosis within a single regime. While the regime mixture generates portfolio-level fat tails (a draw from a mixture of Normals has heavier tails than either component), the returns within each regime are still Gaussian. Empirically, even conditional on a single regime, equity returns exhibit leptokurtosis.

The multivariate Student-$t$ distribution addresses this by introducing a regime-specific degrees-of-freedom parameter $\nu_k > 0$:

$$
\mathbf{y}_t \mid (s_t = k) \sim t_{\nu_k}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

The density is:

$$
p(\mathbf{y} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \nu_k) = \frac{\Gamma\!\left(\frac{\nu_k + d}{2}\right)}{\Gamma\!\left(\frac{\nu_k}{2}\right) (\nu_k \pi)^{d/2} |\boldsymbol{\Sigma}_k|^{1/2}} \left(1 + \frac{1}{\nu_k}(\mathbf{y} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{y} - \boldsymbol{\mu}_k)\right)^{-(\nu_k + d)/2}
$$

Key properties:

- **Tail behaviour.** The tails decay polynomially (as $\|\mathbf{y}\|^{-(\nu_k + d)}$) rather than exponentially. Smaller $\nu_k$ gives heavier tails.
- **Connection to Normal.** As $\nu_k \to \infty$, $t_{\nu_k}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \to \mathcal{N}_d(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$. The Normal is a special case.
- **Moments.** The mean is $\boldsymbol{\mu}_k$ (for $\nu_k > 1$). The covariance is $\frac{\nu_k}{\nu_k - 2} \boldsymbol{\Sigma}_k$ (for $\nu_k > 2$). The excess kurtosis of each marginal is $\frac{6}{\nu_k - 4}$ (for $\nu_k > 4$).
- **Scale-mixture representation.** The multivariate Student-$t$ admits a representation as a scale mixture of Normals: $\mathbf{y} = \boldsymbol{\mu}_k + \boldsymbol{\Sigma}_k^{1/2} \mathbf{z} / \sqrt{w}$ where $\mathbf{z} \sim \mathcal{N}_d(\mathbf{0}, \mathbf{I})$ and $w \sim \operatorname{Gamma}(\nu_k/2, \nu_k/2)$. This representation is useful for Gibbs sampling (introducing $w$ as an auxiliary variable) but is not needed for the marginalised approach.

For the prior on $\nu_k$, a common choice is $\nu_k \sim \operatorname{Gamma}(2, 0.1)$ (prior mean 20, allowing values down to around 3), or a shifted exponential $\nu_k - 2 \sim \operatorname{Exponential}(\lambda)$ to enforce $\nu_k > 2$ (ensuring finite variance).

In a regime-switching context, we expect bull regimes to have larger $\nu_k$ (closer to Gaussian) and bear regimes to have smaller $\nu_k$ (fatter tails), reflecting the empirical observation that crisis periods exhibit more extreme returns even after accounting for elevated volatility.

### 8.2 Regime-Dependent Correlations

The current model fixes $\mathbf{R}_k = \mathbf{I}$. Relaxing this to $\mathbf{R}_k \neq \mathbf{I}$ allows the model to capture one of the most important stylised facts in portfolio risk management: **correlations increase during crises**. In a bear regime, equities, which may be weakly correlated in normal times, tend to move together (the "correlation breakdown" or, more accurately, "correlation convergence toward one").

The extension is straightforward given the LKJ Cholesky infrastructure already in place:

$$
\mathbf{R}_k \sim \operatorname{LKJ}(\eta_k), \qquad k = 0, \ldots, K-1
$$

with the full covariance:

$$
\boldsymbol{\Sigma}_k = \mathbf{D}_k \, \mathbf{R}_k \, \mathbf{D}_k
$$

The number of free correlation parameters per regime is $d(d-1)/2$. For $d = 3$ this is 3 additional parameters per regime (6 total for $K = 2$), which is modest. For larger $d$, the number of parameters grows quadratically, and stronger priors (larger $\eta$) or structured correlation models may be needed.

The emission log-density becomes:

$$
\log f_k(\mathbf{y}_t) = -\frac{d}{2}\log(2\pi) - \sum_{i=1}^d \log L_{k,ii} - \frac{1}{2} \|\mathbf{L}_k^{-1}(\mathbf{y}_t - \boldsymbol{\mu}_k)\|^2
$$

where $\mathbf{L}_k$ is the Cholesky factor of $\boldsymbol{\Sigma}_k$ (i.e., $\mathbf{L}_k = \mathbf{D}_k \mathbf{C}_k$ with $\mathbf{C}_k = \operatorname{chol}(\mathbf{R}_k)$). The triangular solve $\mathbf{L}_k^{-1}(\mathbf{y}_t - \boldsymbol{\mu}_k)$ is $\mathcal{O}(d^2)$.

### 8.3 Autoregressive Dynamics Within Regimes

The current model assumes that, conditional on the regime, returns are i.i.d. This rules out within-regime serial dependence such as momentum (positive autocorrelation) or mean-reversion (negative autocorrelation). Hamilton's original (1989) model used AR(4) dynamics within each regime for quarterly GDP growth.

The AR($p$) extension replaces the emission mean with a dynamic conditional mean:

$$
\mathbf{y}_t \mid (s_t = k, \mathbf{y}_{t-1}, \ldots, \mathbf{y}_{t-p}) \sim \mathcal{N}_d\!\left(\boldsymbol{\mu}_k + \sum_{j=1}^{p} \mathbf{A}_{k,j}(\mathbf{y}_{t-j} - \boldsymbol{\mu}_k), \; \boldsymbol{\Sigma}_k\right)
$$

where $\mathbf{A}_{k,j} \in \mathbb{R}^{d \times d}$ are regime-specific AR coefficient matrices. The model reduces to the i.i.d. case when all $\mathbf{A}_{k,j} = \mathbf{0}$.

For monthly equity returns, $p = 1$ is typically sufficient. The forward algorithm generalises directly: at each step, the emission density $f_k(\mathbf{y}_t)$ now depends on the lagged returns, but the Markov structure of the regime chain is unchanged, so the recursion and its $\mathcal{O}(TK^2)$ cost are preserved.

The main challenge is the increased parameter count: each $\mathbf{A}_{k,j}$ adds $d^2$ parameters per regime per lag. For $K = 2$, $d = 3$, $p = 1$, this is $2 \times 9 = 18$ additional parameters. Shrinkage priors (e.g., $\operatorname{vec}(\mathbf{A}_{k,j}) \sim \mathcal{N}(\mathbf{0}, \tau_A^2 \mathbf{I})$ with $\tau_A$ small) help regularise.

### 8.4 Time-Varying Transition Probabilities

The constant transition matrix $\mathbf{P}$ assumes that regime-switching rates do not depend on observable economic conditions. The TVTP extension allows exogenous covariates $\mathbf{x}_t \in \mathbb{R}^q$ (e.g., VIX level, yield curve slope, credit spreads) to influence the transition probabilities at each time step.

The standard parameterisation uses a **multinomial logistic** (softmax) link:

$$
P_{jk}(t) = \Pr(s_t = k \mid s_{t-1} = j, \mathbf{x}_t) = \frac{\exp(\boldsymbol{\beta}_{jk}^\top \mathbf{x}_t)}{\sum_{k'=0}^{K-1} \exp(\boldsymbol{\beta}_{jk'}^\top \mathbf{x}_t)}
$$

where $\boldsymbol{\beta}_{jk} \in \mathbb{R}^q$ are regime-pair-specific coefficient vectors. For identifiability, one regime per origin state is taken as the reference ($\boldsymbol{\beta}_{jK-1} = \mathbf{0}$), giving $(K-1) \times q$ free parameters per origin state and $K(K-1)q$ total.

For $K = 2$ this simplifies to a logistic regression:

$$
P_{j0}(t) = \frac{1}{1 + \exp(-\boldsymbol{\beta}_j^\top \mathbf{x}_t)}, \qquad P_{j1}(t) = 1 - P_{j0}(t)
$$

with $\boldsymbol{\beta}_j \in \mathbb{R}^q$ for each origin state $j$.

The forward algorithm is unchanged in structure: the recursion $\alpha_t(k) = [\sum_{k'} \alpha_{t-1}(k') P_{k',k}(t)] \cdot f_k(\mathbf{y}_t)$ simply uses a time-varying transition matrix. The cost remains $\mathcal{O}(TK^2)$.

The TVTP model enables a richer form of scenario analysis: rather than asking "what happens if we are in a bear regime?", we can ask "what happens to regime probabilities if the VIX doubles?", connecting portfolio decisions to observable macroeconomic conditions.

Priors on the coefficients $\boldsymbol{\beta}_{jk}$ should be weakly informative (e.g., $\mathcal{N}(0, 1)$ or $\mathcal{N}(0, 2.5)$ on the logit scale), regularising toward the constant-$\mathbf{P}$ model when the covariates are not informative.

### 8.5 Walk-Forward Estimation

The current evaluation estimates parameters on the full sample and then computes filtered probabilities, which embeds in-sample information in the parameter estimates. A proper out-of-sample evaluation requires **walk-forward re-estimation**:

1. Fix an initial training window $[1, T_0]$.
2. Fit the Bayesian HMM on $\mathbf{y}_{1:T_0}$ to obtain the posterior $p(\theta \mid \mathbf{y}_{1:T_0})$.
3. Compute the filtered probability $\bar{\gamma}_{T_0}^{\text{filt}}(\text{Bear})$ and the allocation weight $w_{T_0+1}^{\text{equity}}$.
4. Observe $\mathbf{y}_{T_0+1}$ and record the portfolio return.
5. Expand the window to $[1, T_0 + 1]$ (expanding window) or shift to $[2, T_0 + 1]$ (rolling window) and repeat.

The sequence of out-of-sample portfolio returns $\{r_t^{\text{portfolio}}\}_{t=T_0+1}^{T}$ is then evaluated using standard performance metrics (Sharpe ratio, maximum drawdown, Calmar ratio) without any look-ahead bias.

The computational cost is substantial: the full Bayesian posterior must be re-estimated at each step (or, more practically, every $\Delta$ months). Warm-starting the sampler from the previous posterior can reduce the cost of the tuning phase.

---

## 9. References

- Ang, A. and Bekaert, G. (2002). "International Asset Allocation With Regime Shifts." *Review of Financial Studies*, 15(4), 1137--1187.
- Ang, A. and Bekaert, G. (2004). "How Regimes Affect Asset Allocation." *Financial Analysts Journal*, 60(2), 86--99.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman and Hall/CRC.
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357--384.
- Lewandowski, D., Kurowicka, D., and Joe, H. (2009). "Generating Random Correlation Matrices Based on Vines and Extended Onion Method." *Journal of Multivariate Analysis*, 100(9), 1989--2001.
- Marin, J.-M., Mengersen, K., and Robert, C. P. (2005). "Bayesian Modelling and Inference on Mixtures of Distributions." *Handbook of Statistics*, 25, 459--507.
- Stephens, M. (2000). "Dealing with Label Switching in Mixture Models." *Journal of the Royal Statistical Society B*, 62(4), 795--809.
- Vehtari, A., Gelman, A., and Gabry, J. (2017). "Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC." *Statistics and Computing*, 27(5), 1413--1432.
