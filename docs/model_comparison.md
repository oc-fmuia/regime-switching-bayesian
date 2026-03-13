# Model Specification: Original vs. Proposed Implementation

This document describes in full mathematical detail:

1. The **original model** as specified in the codebase (Section 1)
2. The **proposed alternative model** we will implement (Section 2)
3. **Why the original model cannot be implemented as a proper Bayesian model in PyMC**, and what implementation tradeoffs the alternative involves (Section 3)

---

## Notation

Throughout this document:

| Symbol | Meaning |
|--------|---------|
| $T$ | Number of time steps (observations) |
| $N$ | Number of assets |
| $K$ | Number of regimes |
| $M$ | Number of shock factors |
| $s_t \in \{0, \ldots, K-1\}$ | Hidden regime state at time $t$ |
| $\mathbf{r}_t \in \mathbb{R}^N$ | Vector of asset returns at time $t$ |
| $r_{t,n}$ | Return of asset $n$ at time $t$ |

---

## 1. The Original Model

The original model, as specified in the codebase (`src/inference/model_builder.py`, `src/returns/student_t.py`, `src/regimes/shocks.py`, `src/returns/covariance.py`), has three components: regime dynamics, a return model, and a shock propagation mechanism. We describe each in turn.

### 1.1 Regime Dynamics (Hidden Markov Chain)

The hidden regime state $s_t$ evolves as a discrete-time Markov chain with $K$ states. At each time step, the probability of transitioning from regime $i$ to regime $j$ is given by the entry $P_{ij}$ of a $K \times K$ **transition matrix** $\mathbf{P}$:

$$\Pr(s_t = j \mid s_{t-1} = i) = P_{ij}$$

Each row of $\mathbf{P}$ is a probability distribution over the $K$ regimes, meaning:

$$P_{ij} \geq 0 \quad \text{and} \quad \sum_{j=0}^{K-1} P_{ij} = 1 \quad \text{for each } i$$

**Priors.** Each row of the transition matrix receives an independent Dirichlet prior:

$$(P_{i,0},\; P_{i,1},\; \ldots,\; P_{i,K-1}) \sim \text{Dirichlet}(\alpha, \alpha, \ldots, \alpha) \quad \text{for each } i = 0, \ldots, K-1$$

where $\alpha > 0$ is a concentration parameter. With $\alpha = 1$ (the default), this is a uniform prior over all valid probability vectors — no transition pattern is favored a priori.

**Initial state.** In the original code, the initial state distribution is modeled as a separate Dirichlet random variable $\boldsymbol{\pi} \sim \text{Dirichlet}(\mathbf{1}_K)$, and each time step's regime is drawn independently from $\boldsymbol{\pi}$:

$$s_t \sim \text{Categorical}(\boldsymbol{\pi}) \quad \text{independently for each } t$$

This is a significant bug: it ignores the Markov chain structure entirely. Instead of each state depending on the previous state via $\mathbf{P}$, all states are drawn independently from a single distribution. The transition matrix $\mathbf{P}$ is defined in the model but never used.

### 1.2 Return Model (Multivariate Student-t)

Conditional on being in regime $k$, the $N$-dimensional return vector $\mathbf{r}_t$ is drawn from a **multivariate Student-t distribution**:

$$\mathbf{r}_t \mid s_t = k \;\sim\; \text{MvStudentT}(\nu_k,\; \boldsymbol{\mu}_k,\; \boldsymbol{\Sigma}_k)$$

where:

- $\boldsymbol{\mu}_k \in \mathbb{R}^N$ is the regime-conditional mean return vector
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{N \times N}$ is the regime-conditional covariance matrix (positive definite)
- $\nu_k > 2$ is the regime-conditional degrees of freedom (controls tail heaviness)

The density of the multivariate Student-t distribution in $N$ dimensions is:

$$p(\mathbf{r}_t \mid \nu_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \frac{\Gamma\!\bigl(\frac{\nu_k + N}{2}\bigr)}{\Gamma\!\bigl(\frac{\nu_k}{2}\bigr)\;(\nu_k \pi)^{N/2}\;|\boldsymbol{\Sigma}_k|^{1/2}} \left(1 + \frac{1}{\nu_k}(\mathbf{r}_t - \boldsymbol{\mu}_k)^{\!\top} \boldsymbol{\Sigma}_k^{-1}(\mathbf{r}_t - \boldsymbol{\mu}_k)\right)^{-(\nu_k + N)/2}$$

As $\nu_k \to \infty$, this converges to a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

**Priors on the return model parameters:**

$$\mu_{k,n} \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad \text{(each component of } \boldsymbol{\mu}_k\text{)}$$

$$\sigma_{k,n} \sim \text{HalfNormal}(\sigma_\text{vol}) \quad \text{(marginal volatilities)}$$

$$\nu_k \sim \text{Exponential}\!\left(\frac{1}{\nu_0}\right) \quad \text{(degrees of freedom)}$$

The default values are $\mu_0 = 0$, $\sigma_0 = 0.05$, $\sigma_\text{vol} = 0.1$, $\nu_0 = 10$.

### 1.3 Covariance Structure

The covariance matrix $\boldsymbol{\Sigma}_k$ is decomposed as the sum of a systematic (shock-driven) component and an idiosyncratic component:

$$\boldsymbol{\Sigma}_k = \underbrace{\mathbf{B}_k \mathbf{I}_M \mathbf{B}_k^\top}_{\text{systematic}} + \underbrace{\text{diag}(\boldsymbol{\sigma}_k)\;\boldsymbol{\Omega}_k\;\text{diag}(\boldsymbol{\sigma}_k)}_{\text{idiosyncratic}}$$

where:

- $\mathbf{B}_k \in \mathbb{R}^{N \times M}$ is the regime-conditional **loading matrix** (factor model). It maps $M$ shock factors to $N$ asset returns.
- $\mathbf{I}_M$ is the $M \times M$ identity matrix (shocks are assumed independent with unit variance).
- $\boldsymbol{\sigma}_k = (\sigma_{k,1}, \ldots, \sigma_{k,N})$ is the vector of marginal volatilities.
- $\text{diag}(\boldsymbol{\sigma}_k)$ is the $N \times N$ diagonal matrix with entries $\sigma_{k,n}$.
- $\boldsymbol{\Omega}_k \in \mathbb{R}^{N \times N}$ is the **correlation matrix** (unit diagonal, symmetric, positive definite).

**Priors on the covariance parameters:**

$$B_{k,n,m} \sim \mathcal{N}(0, \sigma_B^2) \quad \text{(each entry of the loading matrix)}$$

with default $\sigma_B = 0.5$.

The specification calls for an **LKJ prior** on the correlation matrix:

$$\boldsymbol{\Omega}_k \sim \text{LKJ}(\eta)$$

with default $\eta = 2$ (weakly favoring correlations near zero). However, in the actual code, the correlation matrix is hardcoded to the identity: $\boldsymbol{\Omega}_k = \mathbf{I}_N$. The LKJ prior is declared in `PriorSpec` but never used.

### 1.4 Shock Propagation

The shock model (`src/regimes/shocks.py`) defines the return-generating process:

$$\mathbf{r}_t = \boldsymbol{\mu}_{s_t} + \mathbf{B}_{s_t}\,\mathbf{u}_t + \boldsymbol{\varepsilon}_t$$

where:

- $\mathbf{u}_t \in \mathbb{R}^M$ is a vector of $M$ independent shock factors, drawn as $\mathbf{u}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_M)$ for Monte Carlo, or set to deterministic values for stress testing
- $\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{s_t}^\text{idio})$ is the idiosyncratic noise

This is a **factor model**: the shock loading matrix $\mathbf{B}_{s_t}$ maps common risk factors into asset-specific returns, with the mapping depending on the current regime.

### 1.5 Summary of Original Model

Collecting all the pieces:

$$\begin{aligned}
&\textbf{Transition matrix:} & (P_{i,0}, \ldots, P_{i,K-1}) &\sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K) &&\text{for each row } i \\[4pt]
&\textbf{Regime dynamics:} & s_t &\sim \text{MarkovChain}(\mathbf{P}, \boldsymbol{\pi}_0) &&\text{hidden states} \\[4pt]
&\textbf{Mean returns:} & \mu_{k,n} &\sim \mathcal{N}(\mu_0,\, \sigma_0^2) &&\text{for each regime } k,\text{ asset } n \\[4pt]
&\textbf{Volatilities:} & \sigma_{k,n} &\sim \text{HalfNormal}(\sigma_\text{vol}) &&\text{for each regime } k,\text{ asset } n \\[4pt]
&\textbf{Correlations:} & \boldsymbol{\Omega}_k &\sim \text{LKJ}(\eta) &&\text{for each regime } k \\[4pt]
&\textbf{Degrees of freedom:} & \nu_k &\sim \text{Exponential}(1/\nu_0) &&\text{for each regime } k \\[4pt]
&\textbf{Loading matrices:} & B_{k,n,m} &\sim \mathcal{N}(0,\, \sigma_B^2) &&\text{for each regime } k,\text{ asset } n,\text{ factor } m \\[4pt]
&\textbf{Covariance:} & \boldsymbol{\Sigma}_k &= \mathbf{B}_k \mathbf{B}_k^\top + \text{diag}(\boldsymbol{\sigma}_k)\,\boldsymbol{\Omega}_k\,\text{diag}(\boldsymbol{\sigma}_k) \\[4pt]
&\textbf{Returns:} & \mathbf{r}_t \mid s_t\!=\!k &\sim \text{MvStudentT}(\nu_k,\; \boldsymbol{\mu}_k,\; \boldsymbol{\Sigma}_k) \\[4pt]
\end{aligned}$$

### 1.6 Bugs in the Original Implementation

The original implementation has **two critical bugs** that make it non-functional:

**Bug 1: No Markov chain dynamics.** The code draws each $s_t$ independently from $\boldsymbol{\pi}$ instead of using the transition matrix $\mathbf{P}$. This means the regime sequence has no temporal structure — the model does not capture persistence or switching patterns.

**Bug 2: Hardcoded regime index.** In the likelihood section of `build()`, the code sets `regime_idx = 0` and uses only regime 0's parameters for all observations:

```python
regime_idx = 0  # Placeholder
B_k = B[regime_idx, :, :]
sigma_k = vols[regime_idx, :]
# ...
```

This means the model is not regime-switching at all: every observation is modeled as coming from a single regime. Parameters for regimes $1, \ldots, K-1$ are sampled from their priors but never appear in the likelihood, so the posterior for those parameters equals their prior (i.e., no learning occurs).

---

## 2. The Proposed Alternative Model

The alternative model retains the core idea — a Hidden Markov Model with regime-conditional return distributions — but simplifies the emission model to make Bayesian inference tractable in PyMC. The reasons for these simplifications are explained in detail in Section 3.

### 2.1 Regime Dynamics (Proper Markov Chain)

Identical structure to Section 1.1, but now properly implemented.

**Transition matrix prior** (same as before):

$$(P_{i,0},\; P_{i,1},\; \ldots,\; P_{i,K-1}) \sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K) \quad \text{for each } i$$

**Initial state distribution:** Uniform over all regimes:

$$\Pr(s_0 = k) = \frac{1}{K} \quad \text{for all } k$$

**Markov transition** (the key fix — this was missing from the original):

$$\Pr(s_t = j \mid s_{t-1} = i) = P_{ij} \quad \text{for } t = 1, \ldots, T-1$$

### 2.2 Return Model (Independent Normal per Asset)

Instead of a multivariate Student-t, each asset's return is modeled independently conditional on the regime:

$$r_{t,n} \mid s_t = k \;\sim\; \mathcal{N}(\mu_{k,n},\; \sigma_{k,n}^2) \quad \text{independently for each asset } n$$

The joint log-density for the full return vector $\mathbf{r}_t$ given regime $k$ factorizes as a product over assets:

$$\log p(\mathbf{r}_t \mid s_t = k) = \sum_{n=1}^{N} \log p(r_{t,n} \mid s_t = k) = \sum_{n=1}^{N} \left[ -\frac{1}{2}\log(2\pi) - \log(\sigma_{k,n}) - \frac{(r_{t,n} - \mu_{k,n})^2}{2\sigma_{k,n}^2} \right]$$

**Priors** (same as before for means and volatilities):

$$\mu_{k,n} \sim \mathcal{N}(\mu_0,\; \sigma_0^2) \quad \text{for each regime } k, \text{ asset } n$$

$$\sigma_{k,n} \sim \text{HalfNormal}(\sigma_\text{vol}) \quad \text{for each regime } k, \text{ asset } n$$

### 2.3 No Shock Loadings, No Correlations, No Degrees of Freedom

Compared to the original model, the proposed alternative removes:

- **Loading matrices $\mathbf{B}_k$**: No factor model / shock propagation
- **Correlation matrices $\boldsymbol{\Omega}_k$**: No cross-asset correlations within a regime
- **Degrees of freedom $\nu_k$**: No heavy-tailed emissions within a regime

The rationale for each removal is discussed in Section 3. Cross-asset dependence is captured implicitly through the shared hidden state $s_t$: when the regime switches from "normal" to "stressed," all assets simultaneously shift to higher-volatility, lower-mean parameters.

### 2.4 Summary of Proposed Model

$$\begin{aligned}
&\textbf{Transition matrix:} & (P_{i,0}, \ldots, P_{i,K-1}) &\sim \text{Dirichlet}(\alpha \cdot \mathbf{1}_K) &&\text{for each row } i \\[4pt]
&\textbf{Initial state:} & s_0 &\sim \text{Categorical}\!\left(\tfrac{1}{K} \cdot \mathbf{1}_K\right) \\[4pt]
&\textbf{Regime dynamics:} & \Pr(s_t = j \mid s_{t-1} = i) &= P_{ij} &&\text{for } t \geq 1 \\[4pt]
&\textbf{Mean returns:} & \mu_{k,n} &\sim \mathcal{N}(\mu_0,\, \sigma_0^2) &&\text{for each regime } k,\text{ asset } n \\[4pt]
&\textbf{Volatilities:} & \sigma_{k,n} &\sim \text{HalfNormal}(\sigma_\text{vol}) &&\text{for each regime } k,\text{ asset } n \\[4pt]
&\textbf{Returns:} & r_{t,n} \mid s_t\!=\!k &\sim \mathcal{N}(\mu_{k,n},\; \sigma_{k,n}^2) &&\text{independently per asset}
\end{aligned}$$

### 2.5 Parameter Count Comparison

For $K = 2$ regimes, $N = 3$ assets, $M = 2$ shock factors:

| Parameter | Original Model | Proposed Model |
|-----------|---------------|----------------|
| Transition matrix $\mathbf{P}$ | $K \times K = 4$ (but $K$ row constraints, so $K(K-1) = 2$ free) | Same: 2 free |
| Mean returns $\boldsymbol{\mu}_k$ | $K \times N = 6$ | Same: 6 |
| Volatilities $\boldsymbol{\sigma}_k$ | $K \times N = 6$ | Same: 6 |
| Correlation matrices $\boldsymbol{\Omega}_k$ | $K \times N(N-1)/2 = 6$ | **Removed**: 0 |
| Degrees of freedom $\nu_k$ | $K = 2$ | **Removed**: 0 |
| Loading matrices $\mathbf{B}_k$ | $K \times N \times M = 12$ | **Removed**: 0 |
| **Total free continuous parameters** | **34** | **14** |

The proposed model has 14 continuous parameters. The original had 34 (plus $T$ discrete hidden states in both cases). Fewer parameters means fewer identifiability issues and faster, more reliable MCMC convergence.

---

## 3. Why the Original Model Cannot Be Implemented in PyMC, and Implementation Details of the Alternative

### 3.1 The Fundamental Problem: Discrete Hidden States in MCMC

The regime sequence $s_0, s_1, \ldots, s_{T-1}$ is a set of $T$ discrete latent variables. PyMC's primary sampler is **NUTS (No-U-Turn Sampler)**, which is a gradient-based method: it computes derivatives of the log-posterior with respect to all parameters and uses these gradients to propose efficient moves through parameter space.

**Gradients are not defined for discrete variables.** The regime state $s_t$ takes integer values $\{0, 1, \ldots, K-1\}$. You cannot differentiate $\log p(\text{data} \mid s_t)$ with respect to $s_t$ because $s_t$ is not a continuous variable — there is no meaningful "direction" in which to perturb it.

This means we cannot simply put the $T$ discrete variables and the continuous parameters (transition matrix, means, volatilities, etc.) into a single model and run NUTS. We need a strategy to handle the discrete variables.

### 3.2 Three Strategies for Discrete Latent Variables in PyMC

#### Strategy A: Gibbs-within-NUTS (PyMC's default for discrete variables)

PyMC can handle **small numbers** of discrete variables by using a compound sampler: NUTS for continuous parameters and a Metropolis (or CategoricalGibbsMetropolis) step for each discrete variable.

For our model, this would require $T$ separate Gibbs steps (one per time step) at each MCMC iteration. With $T = 250$ and $K = 2$, this means 250 discrete Metropolis updates interleaved with NUTS. In practice this is:

- **Extremely slow**: Each Gibbs step requires evaluating the full model likelihood.
- **Poorly mixing**: Metropolis steps for discrete variables in HMMs get stuck because changing a single $s_t$ requires coordinated changes with its neighbors $s_{t-1}$ and $s_{t+1}$.
- **Not scalable**: Runtime grows linearly with $T$, and mixing deteriorates with longer sequences.

**Verdict: Not practical for HMMs with more than a handful of time steps.**

#### Strategy B: Marginalization via `pymc-extras`

The `pymc-extras` library provides `DiscreteMarkovChain` and `pmx.marginalize()`, which analytically integrate out the discrete states using the **forward algorithm** (see Section 3.4 below). After marginalization, the model contains only continuous parameters and can be sampled with pure NUTS.

However, `pmx.marginalize()` imposes a strict technical constraint: the relationship between the discrete variable and the observed data must be **element-wise**. Formally, if $s_t$ is the discrete variable and $\mathbf{r}_t$ is the observed variable, then the emission probability must factorize as:

$$p(\mathbf{r}_t \mid s_t = k, \theta) = \prod_{n} f(r_{t,n} \mid g_n(k, \theta))$$

where each $f$ depends on $s_t$ only through an **indexing** operation $g_n(k, \theta) = \theta_{k,n}$.

The **multivariate Student-t** violates this constraint because its density involves the **full covariance matrix** $\boldsymbol{\Sigma}_k$ and the quadratic form $(\mathbf{r}_t - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{r}_t - \boldsymbol{\mu}_k)$, which couples all $N$ components of $\mathbf{r}_t$ together. This coupling means the emission does not factorize element-wise over assets.

Furthermore, `pymc-extras` version 0.2.7 is **incompatible with PyMC 5.25.1** (the version installed in this project). Specifically, `pymc-extras` attempts to import `CustomProgress` from `pymc.util`, which was removed in recent PyMC versions. While a monkeypatch can fix the initial import, the incompatibility propagates through multiple internal modules, leading to a subtly broken marginalization that produces 100% divergent MCMC samples.

**Verdict: Not viable due to both the element-wise constraint and version incompatibility.**

#### Strategy C: Custom Forward Algorithm (Proposed Approach)

We implement the forward algorithm ourselves using PyTensor's `scan` operation, which provides automatic differentiation. The discrete states are never represented as model variables — they are analytically summed out inside a custom log-likelihood function, which is added to the model as a `pm.Potential`.

The resulting model contains **only continuous parameters** ($\mathbf{P}$, $\boldsymbol{\mu}_k$, $\boldsymbol{\sigma}_k$) and can be sampled with pure NUTS. This approach:

- Has no dependency on `pymc-extras`
- Works with any emission distribution (Normal, Student-t, multivariate — any distribution for which we can compute a log-density)
- Provides automatic differentiation via PyTensor (no manual gradient code)
- Is the standard textbook approach for HMMs in probabilistic programming

**Verdict: This is the approach we will implement.** The rest of Section 3 describes it in detail.

### 3.3 Mathematical Foundation: The Forward Algorithm

The forward algorithm computes the **marginal likelihood** of the observed data given only the continuous parameters, with the discrete states analytically integrated out.

#### 3.3.1 What We Want to Compute

We want:

$$p(\mathbf{r}_{0:T-1} \mid \mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\sigma}) = \sum_{s_0, s_1, \ldots, s_{T-1}} p(\mathbf{r}_{0:T-1}, s_{0:T-1} \mid \mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\sigma})$$

The sum is over all possible regime sequences — there are $K^T$ of them, so brute-force summation is intractable.

#### 3.3.2 Factorization of the Joint Probability

Using the chain rule and the Markov property, the joint probability factorizes as:

$$p(\mathbf{r}_{0:T-1}, s_{0:T-1}) = p(s_0)\,p(\mathbf{r}_0 \mid s_0) \prod_{t=1}^{T-1} p(s_t \mid s_{t-1})\,p(\mathbf{r}_t \mid s_t)$$

where we drop the conditioning on $\mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\sigma}$ for brevity, and:

- $p(s_0 = k) = \pi_k = 1/K$ is the initial state probability
- $p(s_t = j \mid s_{t-1} = i) = P_{ij}$ is the transition probability
- $p(\mathbf{r}_t \mid s_t = k) = \prod_{n=1}^{N} \mathcal{N}(r_{t,n} \mid \mu_{k,n}, \sigma_{k,n}^2)$ is the emission probability

#### 3.3.3 The Forward Recursion

Define the **forward variable** $\alpha_t(k)$ as the joint probability of observing returns up to time $t$ and being in state $k$ at time $t$:

$$\alpha_t(k) \;\equiv\; p(\mathbf{r}_0, \mathbf{r}_1, \ldots, \mathbf{r}_t,\; s_t = k)$$

**Base case** ($t = 0$):

$$\alpha_0(k) = p(s_0 = k)\;p(\mathbf{r}_0 \mid s_0 = k) = \pi_k \cdot p(\mathbf{r}_0 \mid s_0 = k)$$

**Recursion** ($t \geq 1$):

$$\alpha_t(j) = p(\mathbf{r}_t \mid s_t = j) \sum_{i=0}^{K-1} \alpha_{t-1}(i)\;P_{ij}$$

The inner sum is a matrix-vector product: it takes the previous forward variables $\alpha_{t-1}(i)$ and propagates them through the transition matrix. Then the result is multiplied by the emission probability at the current time step.

**Marginal likelihood:**

$$p(\mathbf{r}_{0:T-1}) = \sum_{k=0}^{K-1} \alpha_{T-1}(k)$$

#### 3.3.4 Log-Space Implementation

In practice, the forward variables $\alpha_t(k)$ become extremely small for long sequences (they are products of probabilities), so we work in log-space. Define:

$$\ell_t(k) \equiv \log \alpha_t(k)$$

**Base case:**

$$\ell_0(k) = \log \pi_k + \log p(\mathbf{r}_0 \mid s_0 = k)$$

**Recursion:**

$$\ell_t(j) = \log p(\mathbf{r}_t \mid s_t = j) + \text{logsumexp}_{i}\!\left(\ell_{t-1}(i) + \log P_{ij}\right)$$

where the **logsumexp** operation is defined as:

$$\text{logsumexp}_i(x_i) \equiv \log\!\left(\sum_i \exp(x_i)\right) = x_{\max} + \log\!\left(\sum_i \exp(x_i - x_{\max})\right)$$

and $x_{\max} = \max_i x_i$. The second form is numerically stable because all exponents are $\leq 0$.

**Marginal log-likelihood:**

$$\log p(\mathbf{r}_{0:T-1}) = \text{logsumexp}_{k}\!\left(\ell_{T-1}(k)\right)$$

**Computational complexity:** $O(T \cdot K^2)$ — linear in the number of time steps, quadratic in the number of regimes. For $T = 250$ and $K = 2$, this is 1000 operations. The brute-force sum would require $K^T = 2^{250} \approx 10^{75}$ operations.

### 3.4 Post-hoc State Recovery: The Forward-Backward Algorithm

After MCMC sampling, we want to know which regime the model thinks each time step belongs to. For a given set of continuous parameter values $(\mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\sigma})$, we want the **smoothed state probabilities**:

$$\gamma_t(k) \equiv p(s_t = k \mid \mathbf{r}_{0:T-1},\; \mathbf{P},\; \boldsymbol{\mu},\; \boldsymbol{\sigma})$$

This requires both the **forward** variables $\alpha_t(k)$ (Section 3.3.3) and **backward** variables $\beta_t(k)$.

#### 3.4.1 The Backward Recursion

Define:

$$\beta_t(k) \equiv p(\mathbf{r}_{t+1}, \ldots, \mathbf{r}_{T-1} \mid s_t = k)$$

**Base case** ($t = T-1$):

$$\beta_{T-1}(k) = 1 \quad \text{(no future observations)}$$

**Recursion** (backwards from $t = T-2$ to $t = 0$):

$$\beta_t(i) = \sum_{j=0}^{K-1} P_{ij}\;p(\mathbf{r}_{t+1} \mid s_{t+1} = j)\;\beta_{t+1}(j)$$

#### 3.4.2 Smoothed Probabilities

Combining forward and backward:

$$\gamma_t(k) = \frac{\alpha_t(k)\;\beta_t(k)}{\sum_{j=0}^{K-1} \alpha_t(j)\;\beta_t(j)}$$

In log-space:

$$\log \gamma_t(k) = \ell_t^{\alpha}(k) + \ell_t^{\beta}(k) - \text{logsumexp}_{j}\!\left(\ell_t^{\alpha}(j) + \ell_t^{\beta}(j)\right)$$

and then $\gamma_t(k) = \exp(\log \gamma_t(k))$.

This computation is done **in NumPy** (not PyTensor) after sampling, once per posterior draw. For each MCMC draw $(\mathbf{P}^{(s)}, \boldsymbol{\mu}^{(s)}, \boldsymbol{\sigma}^{(s)})$, we run the forward-backward algorithm and get $\gamma_t^{(s)}(k)$. Averaging over draws gives the **posterior mean state probability**:

$$\bar{\gamma}_t(k) = \frac{1}{S}\sum_{s=1}^{S} \gamma_t^{(s)}(k)$$

### 3.5 Implementation in PyTensor / PyMC

The implementation translates the log-space forward algorithm into PyTensor operations, which allows PyMC to automatically compute gradients for NUTS.

#### 3.5.1 Model Structure

```python
with pm.Model() as model:
    # --- Priors (continuous parameters only) ---
    P = pm.Dirichlet("P", a=alpha * np.ones((K, K)), shape=(K, K))
    regime_means = pm.Normal("regime_means", mu=0.0, sigma=0.05, shape=(K, N))
    volatilities = pm.HalfNormal("volatilities", sigma=0.1, shape=(K, N))

    # --- Compute log emission probabilities ---
    # log p(r_t | s_t=k) = sum_n log N(r_{t,n} | mu_{k,n}, sigma_{k,n})
    # Shape: (T, K)
    log_emission = _compute_log_emission(returns_data, regime_means, volatilities)

    # --- Forward algorithm (integrates out discrete states) ---
    log_P = pt.log(P)
    log_init = pt.log(pt.ones(K) / K)
    log_lik = _forward_logp(log_emission, log_P, log_init)

    # --- Add to model as a Potential ---
    pm.Potential("hmm_loglik", log_lik)
```

There are no discrete variables in this model. The `pm.Potential` adds $\log p(\mathbf{r}_{0:T-1} \mid \mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\sigma})$ to the log-posterior. PyMC's NUTS sampler then explores the joint posterior over $(\mathbf{P}, \boldsymbol{\mu}_k, \boldsymbol{\sigma}_k)$.

#### 3.5.2 Computing Log Emission Probabilities

The function `_compute_log_emission(returns_data, regime_means, volatilities)` computes a matrix of shape $(T, K)$ where entry $(t, k)$ is:

$$\text{log\_emission}[t, k] = \log p(\mathbf{r}_t \mid s_t = k) = \sum_{n=1}^{N} \log \mathcal{N}(r_{t,n} \mid \mu_{k,n}, \sigma_{k,n})$$

In PyTensor, this is computed by broadcasting returns over regimes:

```python
def _compute_log_emission(returns_data, regime_means, volatilities):
    # returns_data: (T, N) observed data (constant)
    # regime_means: (K, N) PyTensor variable
    # volatilities: (K, N) PyTensor variable
    #
    # For each regime k and time t, compute:
    #   sum_n [ -0.5*log(2*pi) - log(sigma_{k,n}) - 0.5*((r_{t,n} - mu_{k,n})/sigma_{k,n})^2 ]
    #
    # returns_data[:, None, :] has shape (T, 1, N) — broadcasts over K
    # regime_means[None, :, :] has shape (1, K, N) — broadcasts over T
    # Result before summing: shape (T, K, N)
    # After summing over N: shape (T, K)

    r = returns_data[:, None, :]                # (T, 1, N)
    mu = regime_means[None, :, :]               # (1, K, N)
    sigma = volatilities[None, :, :]            # (1, K, N)

    log_emission = -0.5 * pt.log(2 * np.pi) - pt.log(sigma) - 0.5 * ((r - mu) / sigma) ** 2
    return log_emission.sum(axis=-1)            # (T, K)
```

#### 3.5.3 Forward Algorithm in `pt.scan`

PyTensor's `scan` operation implements a sequential loop with automatic differentiation. It takes a step function and iterates it over a sequence:

```python
def _forward_logp(log_emission, log_P, log_init):
    # log_emission: (T, K)
    # log_P: (K, K) — log of the transition matrix
    # log_init: (K,) — log of the initial state distribution

    def step(log_emit_t, log_alpha_prev):
        # log_alpha_prev: (K,) — log forward variables at time t-1
        # log_emit_t: (K,) — log emission probabilities at time t
        #
        # For each target state j, compute:
        #   log_alpha_t(j) = log_emit_t(j) + logsumexp_i(log_alpha_prev(i) + log_P(i, j))
        #
        # log_alpha_prev[:, None] has shape (K, 1) — broadcasts over target states
        # log_P has shape (K, K) where log_P[i, j] = log P(i -> j)
        # Sum is over source states (axis=0)

        log_alpha_t = log_emit_t + pt.logsumexp(log_alpha_prev[:, None] + log_P, axis=0)
        return log_alpha_t

    # Base case: log_alpha_0(k) = log_init(k) + log_emission(0, k)
    log_alpha_0 = log_init + log_emission[0]

    # Iterate over t = 1, ..., T-1
    log_alphas, _ = pytensor.scan(
        fn=step,
        sequences=[log_emission[1:]],
        outputs_info=[log_alpha_0],
    )

    # Marginal log-likelihood: logsumexp over final forward variables
    # log_alphas has shape (T-1, K); last row is log_alpha_{T-1}
    return pt.logsumexp(log_alphas[-1])
```

The `scan` unrolls the forward recursion over time. PyTensor automatically differentiates through the entire computation, producing gradients of $\log p(\mathbf{r}_{0:T-1})$ with respect to $\mathbf{P}$, $\boldsymbol{\mu}_k$, and $\boldsymbol{\sigma}_k$. These gradients are what NUTS needs to sample efficiently.

### 3.6 Summary of Tradeoffs

| Aspect | Original Model | Proposed Model |
|--------|---------------|----------------|
| **Emission distribution** | Multivariate Student-t | Independent Normal per asset |
| **Heavy tails within regime** | Yes (via $\nu_k$) | No (but mixture of regimes creates heavy tails at portfolio level) |
| **Cross-asset correlations** | Yes (via $\boldsymbol{\Omega}_k$ and $\mathbf{B}_k$) | Implicit only (shared regime state) |
| **Factor model (shocks)** | Yes ($\mathbf{B}_k$ loading matrices) | No |
| **Regime dynamics** | Broken (iid draws) | Proper Markov chain |
| **Bayesian inference** | Broken (hardcoded regime) | Full MCMC via forward algorithm |
| **Number of parameters** | 34 (for $K\!=\!2$, $N\!=\!3$, $M\!=\!2$) | 14 |
| **MCMC method** | N/A (broken) | Pure NUTS (no discrete variables) |
| **PyMC dependency** | pymc-extras (incompatible) | PyMC + PyTensor only |

**What we gain:**
- A working regime-switching model with proper Bayesian inference
- Correct Markov chain dynamics (temporal structure in regimes)
- Efficient NUTS sampling with automatic differentiation
- No external dependencies beyond PyMC
- Posterior regime probabilities via forward-backward

**What we lose:**
- Heavy-tailed emissions within each regime (Student-t). However, the mixture of Normals across regimes produces heavier tails at the portfolio level than a single Gaussian. Also, adding Student-t emissions back is straightforward with this implementation — it only requires changing the log emission computation (see below).
- Explicit cross-asset correlations and factor structure. In practice, for 2-3 assets, the shared regime state captures the dominant source of co-movement (all assets shift together when regime changes). For larger portfolios (10+ assets), adding explicit correlations would become important.

### 3.7 Path to Extending the Model

The custom forward algorithm approach is **not** limited to independent Normal emissions. Any emission distribution can be used, as long as we can compute its log-density in PyTensor. Concretely:

**Adding Student-t emissions** requires only changing the log emission computation:

$$\log p(r_{t,n} \mid s_t = k) = \log \Gamma\!\left(\frac{\nu_k + 1}{2}\right) - \log \Gamma\!\left(\frac{\nu_k}{2}\right) - \frac{1}{2}\log(\nu_k \pi) - \log(\sigma_{k,n}) - \frac{\nu_k + 1}{2}\log\!\left(1 + \frac{1}{\nu_k}\left(\frac{r_{t,n} - \mu_{k,n}}{\sigma_{k,n}}\right)^2\right)$$

This adds $K$ parameters ($\nu_k$ per regime) and requires no changes to the forward algorithm itself.

**Adding multivariate emissions** (e.g., multivariate Normal with a full covariance matrix) replaces the sum-over-assets with a single multivariate log-density:

$$\log p(\mathbf{r}_t \mid s_t = k) = -\frac{N}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol{\Sigma}_k| - \frac{1}{2}(\mathbf{r}_t - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\mathbf{r}_t - \boldsymbol{\mu}_k)$$

This requires parameterizing $\boldsymbol{\Sigma}_k$ (e.g., via Cholesky decomposition with LKJ priors) and again requires no changes to the forward algorithm.

Both extensions are purely changes to the `_compute_log_emission` function; the `_forward_logp` function and the rest of the model stay exactly the same.
