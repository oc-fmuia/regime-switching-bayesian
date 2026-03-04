# Finance Specification: Interpretation & Applications

**Version:** 1.0  
**Date:** March 4, 2026  
**Audience:** Portfolio managers, risk officers, practitioners

---

## Table of Contents

1. [Core Financial Concepts](#1-core-financial-concepts)
2. [Regime Interpretation](#2-regime-interpretation)
3. [Private Returns & Reporting Distortions](#3-private-returns--reporting-distortions)
4. [Quarterly Aggregation Effects](#4-quarterly-aggregation-effects)
5. [Risk Decomposition: Systemic vs. Idiosyncratic](#5-risk-decomposition-systemic-vs-idiosyncratic)
6. [Portfolio Analysis](#6-portfolio-analysis)
7. [Scenario Analysis Framework](#7-scenario-analysis-framework)
8. [Real-Time Filtering & Monitoring](#8-real-time-filtering--monitoring)

---

## 1. Core Financial Concepts

### Return Definition

All returns in this model are **log-returns** (continuously compounded):

$$r_t = \log(P_t / P_{t-1})$$

**Why log-returns?**
- Additivity: Monthly log-returns sum to period returns
- Mathematical convenience: MVT assumes log-scale
- Empirically justified: Real financial data closer to normal in log-space

### Public vs. Private Assets

**Public Assets** (d_pub):
- Liquid, frequently traded, prices available daily
- Examples: stocks, bonds, commodity futures, indices
- **Observed directly:** r_t^(pub) is the monthly log-return
- No reporting lag or smoothing

**Private Assets** (d_priv):
- Illiquid, difficult to price, infrequently valued
- Examples: real estate, private equity, hedge funds, insurance liabilities
- **NOT directly observed:** Only quarterly reported values y_q^(priv)
- Reported values contain smoothing and measurement error (see Section 3)

### Time Frequency Mismatch

The model handles a critical challenge: **public data monthly, private data quarterly**.

```
Timeline:

Month:    1   2   3   4   5   6   7   8   9  10  11  12 ...
Quarter:           1           2           3           4  ...
Public:   pub pub pub pub pub pub pub pub pub pub pub pub
Private:           y₁              y₂              y₃
           (observed at         (observed at    (observed at
            month 3+L)          month 6+L)      month 9+L)
```

The model jointly infers:
- **Latent monthly private returns** x_t^(priv) (what the assets actually returned each month)
- **Quarterly reported smoothed values** y_q^(priv) (what was reported, with lag and smoothing)

---

## 2. Regime Interpretation

### What is a Regime?

A **regime** is a persistent state of the market characterized by:
- Different expected returns (means)
- Different volatility structure (covariances)
- Different tail behavior (degrees of freedom)
- Different cross-asset correlation patterns

### Typical Three-Regime Structure

**Regime 1: GROWTH**
- Positive expected returns across assets
- Low volatility (σ ≈ 8–15% annualized)
- Near-normal tail behavior (ν ≈ 15–20, kurtosis ≈ 3.3)
- Low/positive correlations between public & private
- Duration: weeks to months, typically longest regime

**Regime 2: CRISIS / STRESS**
- Negative or near-zero expected returns
- High volatility (σ ≈ 25–50% annualized)
- Heavy tails (ν ≈ 3–5, kurtosis ≈ 3–6)
- Correlations spike: flight-to-quality, liquidation
- Duration: days to weeks, transition phase

**Regime 3: RECOVERY**
- Intermediate returns and volatility
- Tail behavior between growth and crisis
- Correlations gradually normalizing
- Duration: weeks to months, unstable bridge state

### Regime Probability Evolution

At any time t, the filter computes:

$$P(s_t = k | r_1:t^{(\text{pub})}, y_{\text{obs}}^{(\text{priv})}, \theta)$$

**Interpretation:**
- Probability regime k is active at time t, given observed data
- As new monthly data arrives, these probabilities update
- Sharp increases in P(s_t = crisis) signal market stress

### Why Regimes Matter for Risk

**Single-regime model (naive):**
- Assumes constant mean and covariance
- Computes VaR, CVaR assuming fixed distributions
- Fails when actual regime changes (worst time to be wrong)

**Regime-switching model (this framework):**
- Tracks probabilities of each regime in real-time
- Scenario analysis accounts for regime transitions
- Hedging strategies conditional on regime
- Risk metrics weighted by regime probability

---

## 3. Private Returns & Reporting Distortions

### The Core Problem: Smooth Valuation

Private assets are valued infrequently and often with discretion. Reported quarterly values exhibit:

1. **Smoothing:** Valuations lag true market values
2. **Stale Pricing:** Quarter q reported at month t(q) + L (lag L)
3. **Measurement Noise:** Valuation uncertainty, rounding
4. **Autocorrelation:** Returns artificially smooth, volatility understated

### Measurement Model Explained

The reported quarterly value:

$$y_q^{(\text{priv})} = \phi y_{q-1}^{(\text{priv})} + (1 - \phi) \tilde{y}_q^{(\text{priv})} + \varepsilon_q$$

**Component breakdown:**

**Φ y_{q-1}^(priv)**: "Carryover"
- Prior quarter's reported value (inertia)
- Higher φ → more weight on lagged value
- φ = 0: No carryover, fresh valuation each quarter
- φ = 0.5: 50% of prior quarter's report persists

**(1-φ) \tilde{y}_q^(priv)**: "True aggregated return"
- \tilde{y}_q = sum of three months' true returns
- (1-φ) weight on actual period performance
- If φ high, this term gets suppressed

**ε_q ~ N(0, R)**: Measurement noise
- Valuation uncertainty, rounding, timing
- Uncorrelated across quarters
- Diagonal R: independent noise per asset

### Parameter Interpretation: Φ (Smoothing)

**If φ = 0 (mark-to-market):**
- Reported values = true aggregated monthly returns + noise
- Captures real performance each quarter
- Typical for liquid/transparent assets

**If φ = 0.3 (moderate smoothing):**
- 30% carryover from prior quarter, 70% new performance
- Realistic for many alternative assets
- Reported values lag true values by ~1–2 months

**If φ = 0.7 (heavy smoothing):**
- 70% carryover, 30% new performance
- Highly stale pricing, infrequent revaluations
- Typical for private equity, some real estate

### Example: Real Estate Fund

Suppose true monthly returns are: [+5%, -2%, +3%] in quarter 1.

**Aggregated true quarterly:** \tilde{y}_1 = +5% - 2% + 3% = +6%

**With different φ values:**

| φ | Interpretation | y_1^(priv) = (1-φ)·6% + noise | Observed |
|---|---|---|---|
| 0.0 | Fresh valuation | 6% + ε | Tracks true return |
| 0.3 | Moderate smoothing | 4.2% + ε | Somewhat damped |
| 0.7 | Heavy smoothing | 1.8% + ε | Heavily damped |

**Key insight:** When φ is high, reported quarterly returns understate true volatility. A quarter with large true moves appears muted in reported data. The model recovers the true x_t^(priv) by un-smoothing.

### Measurement Noise: R

$$R = \text{diag}(r_1^2, \ldots, r_{d_{\text{priv}}}^2)$$

**Typical values** (for monthly/quarterly data):
- r_j ≈ 0.5–2% (half to two percent per quarter)
- Accounts for valuation uncertainty, mark-to-model error

**Financial interpretation:**
- Valuation committees might agree within ±0.5–1% of "true" value
- Independent across assets (no systemic valuation bias)

---

## 4. Quarterly Aggregation Effects

### Why Monthly Latent Aggregate to Quarterly Reported

The private assets are valued quarterly, but we infer monthly latent returns because:

1. **Public assets are monthly:** We observe public returns monthly, want consistent framework
2. **Regime changes intra-quarter:** Crisis can hit in month 2 of a quarter; monthly granularity captures it
3. **Latent monthly is more flexible:** Can model month-to-month correlation; quarterly-only would miss structure

### Aggregation Formula

True quarterly aggregated return:

$$\tilde{y}_q^{(\text{priv})} = x_{3q-2}^{(\text{priv})} + x_{3q-1}^{(\text{priv})} + x_{3q}^{(\text{priv})}$$

**Months in quarter q:** {3q-2, 3q-1, 3q}
- Quarter 1: {1, 2, 3}
- Quarter 2: {4, 5, 6}
- Quarter 3: {7, 8, 9}

### Reporting Lag: L Months

Quarter q is reported at month t(q) + L, where t(q) = 3q.

**Example:** L = 1 month delay
- Q1 (ends month 3) reported at month 4
- Q2 (ends month 6) reported at month 7
- Q3 (ends month 9) reported at month 10

**Likelihood conditioning:**
- At month t, only condition on quarters where t ≥ t(q) + L
- Future quarters remain unobserved (integrated out)

### Quarterly vs. Monthly Volatility

**Observed quarterly reported volatility:** SD(y_q)
- Appears lower due to smoothing (high φ)
- Understates true underlying risk

**Inferred monthly volatility:** SD(x_t)
- Recovers true monthly risk
- Accounts for aggregation (3 months' uncertainty compounds)

**Relationship:**
If monthly returns were i.i.d. with std σ_m, then quarterly aggregated std would be:

$$\sigma_q^{(\text{unsmoothed})} \approx \sqrt{3} \cdot \sigma_m$$

But with smoothing parameter φ:

$$\sigma_q^{(\text{smoothed})} \approx (1 - \phi) \sqrt{3} \cdot \sigma_m + \phi \cdot \sigma_q^{(\text{prior})}$$

The model estimates φ and σ_m jointly to recover true risk.

---

## 5. Risk Decomposition: Systemic vs. Idiosyncratic

### Factor Model Interpretation

The covariance matrix in regime k:

$$\Sigma_k = B_k B_k^T + D_k$$

decomposes total risk into:
- **Systemic risk:** B_k B_k^T (common factors explain co-movement)
- **Idiosyncratic risk:** D_k (asset-specific, uncorrelated)

### Systemic Risk: B_k B_k^T

**Interpretation:**
- Driven by r latent risk factors (r ≪ d, typically r ≈ 2–3)
- Factors could represent:
  - Market factor (aggregate stock/bond returns)
  - Volatility regime (VIX-like)
  - Credit conditions (spread levels)
  - Real estate cycle
  - Liquidity conditions

**Factor loadings:** (B_k)_ℓ,: ∈ ℝ^r
- How much asset ℓ is exposed to each factor
- Regime-dependent: crisis exposure ≠ growth exposure
- Hierarchical: global factors + geography adjustments + sector adjustments + asset residuals

**Example (r = 2 factors, growth regime):**

```
Factor 1: "Aggregate market factor"
Factor 2: "Volatility/risk-off"

Asset: Large-cap growth stock
  Loading_growth = [0.8, 0.3]
  (80% market exposure, 30% vol exposure)

Asset: Government bonds
  Loading_growth = [-0.1, -0.8]
  (negative market exposure, strong vol-hedging)

Asset: Private equity
  Loading_growth = [0.5, 0.1]
  (50% market, low vol sensitivity)
```

### Idiosyncratic Risk: D_k

**Interpretation:**
- Asset-specific risk, uncorrelated across assets
- Diagonal matrix: independent noise per asset
- Volatility per asset σ_{k,ℓ}

**Sources:**
- Company-specific news (for stocks)
- Fund manager performance variation
- Valuation uncertainty (especially for private assets)
- Measurement error

### Risk Decomposition at Asset Level

For asset ℓ in regime k:

$$\sigma_{k,\ell}^2 = \sum_{i=1}^r \left[(B_k)_{\ell,i}\right]^2 \cdot \text{Var}(\text{Factor}_i) + \sigma_{k,\ell,\text{idio}}^2$$

(Assuming factors have unit variance.)

**Portfolio risk implication:**
- High factor loadings → high correlation with other assets
- High idiosyncratic → diversifiable through many holdings
- Crisis regime: factor loadings often increase (higher systemic risk)

---

## 6. Portfolio Analysis

### Portfolio Return Distribution

Given weights w ∈ ℝ^d (sum to 1), the portfolio return at time t:

$$r_t^{\text{port}} = w^T z_t = w^T [r_t^{(\text{pub})}; x_t^{(\text{priv})}]$$

The distribution of r_t^port conditional on regime k:

$$r_t^{\text{port}} | (s_t = k) \sim \text{StudentT}(\nu_k, w^T \mu_k, w^T \Sigma_k w)$$

**Parameters:**
- Mean: w^T μ_k (portfolio-level expected return)
- Variance: w^T Σ_k w (portfolio-level variance)
- Tail parameter: ν_k (regime's tail thickness)

### Risk Metrics

#### Value-at-Risk (VaR)

VaR at confidence α (e.g., α = 0.95 for 95% VaR):

$$\text{VaR}_\alpha = w^T \mu_k + \sqrt{w^T \Sigma_k w} \cdot t_{\nu_k, 1-\alpha}$$

where t_{ν,α} is the α-quantile of the Student-t distribution with ν degrees of freedom.

**Regime-specific:** VaR_crisis > VaR_growth (heavier tails, higher volatility).

#### Expected Shortfall (CVaR)

CVaR = expected return in the worst α% of outcomes:

$$\text{CVaR}_\alpha = \mathbb{E}[r_t^{\text{port}} | r_t^{\text{port}} \leq \text{VaR}_\alpha]$$

**For Student-t:** Closed form exists; typically 10–30% worse than VaR.

#### Sharpe Ratio

$$\text{Sharpe} = \frac{w^T \mu_k - r_f}{\sqrt{w^T \Sigma_k w}}$$

where r_f is risk-free rate (typically 0 for excess returns).

**Regime-dependent:** Growth regime Sharpe > Crisis regime Sharpe.

### Maximum Drawdown

Over a horizon H months:

$$\text{MDD} = \min_{h \in [1,H]} \left( \max_{h' \leq h} \sum_{t=1}^{h'} r_t^{\text{port}} - \sum_{t=1}^h r_t^{\text{port}} \right)$$

**Scenario output:** Distribution of MDD across 1000s simulations and regime paths.

---

## 7. Scenario Analysis Framework

### Baseline Scenario

**Definition:** No interventions, pure filtered future conditional on posterior beliefs.

**Process (per Monte Carlo path):**
1. Sample posterior regime sequence s_{T+1:T+H} from filtered distribution at T
2. For each month t ∈ [T+1, T+H]:
   - Sample z_t ~ MVT(μ_{s_t}, Σ_{s_t}, ν_{s_t})
   - Extract r_t^(pub), x_t^(priv)
3. For each quarter q in [T+1, T+H]:
   - Compute y_q^(priv) via measurement model
4. Compute portfolio metrics (return, VaR, CVaR, MDD, etc.)

**Output:** Distribution of outcomes under filtered beliefs (regime probabilities from today's data).

### Scenario A: Regime Pinning

**Intervention:** Force regime k* for months [T+1, T+W].

**Interpretation:** "What if we stay in crisis for the next 6 months?"

**Mechanics:**
- s_t = k* for t ∈ [T+1, T+W]
- Freely transition after month T+W
- Volatilities, correlations, returns all follow regime k*

**Risk question answered:** Portfolio loss if crisis persists? Downside under extended stress?

### Scenario B: Mean Shift

**Intervention:** Add Δμ to mean in regime k for window [T+1, T+W].

**Specification:** Δμ can target:
- Global: Δμ applied to all assets
- Geographic: Δμ to assets in geography g
- Sectoral: Δμ to assets in sector h
- Asset-specific: Δμ to specific asset ℓ

**Interpretation:** "What if growth stalls (lower returns) globally for 6 months?"

**Mechanics:**
- z_t ~ MVT(μ_{s_t} + Δμ, Σ_{s_t}, ν_{s_t}) for t ∈ [T+1, T+W]
- Measurement model for y_q^(priv) uses shifted x_t^(priv)

**Risk question answered:** Portfolio return impact of sector/geography underperformance?

### Scenario C: Covariance Shock

**Intervention:** Scale factor loadings B_k or idiosyncratic scales D_k.

**Specification:**
- Increase vol: Σ_k ← λ_vol · Σ_k (e.g., λ_vol = 1.5 for 50% vol increase)
- Increase correlation: Scale B_k by λ_B, keeps D_k fixed (makes factors more important)
- Increase idiosyncratic: Scale D_k by λ_D, keeps B_k fixed (more asset-specific risk)

**Interpretation:** "What if volatility spikes 50% for 3 months?" (market stress, liquidity event)

**Mechanics:**
- Modified Σ_k used in sampling z_t, then in measurement model for y_q
- Regime tail parameter ν_k unchanged

**Risk question answered:** Portfolio sensitivity to vol spikes, correlation compression?

### Scenario D: Tail Shock

**Intervention:** Reduce tail parameter ν_k to ν_k' < ν_k.

**Interpretation:** "What if tails get fatter (more extreme tail risk)?"

**Example:** ν_crisis = 4 → ν_crisis' = 2.5 (much fatter tails, kurtosis 6 → 12)

**Mechanics:**
- z_t ~ MVT(μ_{s_t}, Σ_{s_t}, ν_k') with reduced ν
- Same means and covariances, but distribution has much heavier tails

**Risk question answered:** VaR and CVaR under extreme tail scenarios?

### Scenario E: Measurement Model Shock

**Intervention:** Change smoothing parameter φ or measurement noise R.

**Specification:**
- Increase smoothing: φ → φ' > φ (reported values become MORE stale)
- Increase measurement noise: R → λ_R · R (more valuation uncertainty)

**Interpretation:** "What if private asset valuations become unreliable?" (mark-to-market freeze, forced marking to model, fund illiquidity)

**Mechanics:**
- y_q^(priv) = φ' y_{q-1}^(priv) + (1-φ') \tilde{y}_q^(priv) + ε_q'

**Risk question answered:** Portfolio impact if private asset values uncertain or stale? How much does true risk exceed reported smoothed risk?

### Scenario Comparison & Reporting

For each scenario:
1. Run Monte Carlo (e.g., 1000 paths)
2. Compute portfolio metrics per path (return, VaR, CVaR, MDD)
3. Report:
   - Mean, median, std of each metric
   - Percentiles (5th, 25th, 50th, 75th, 95th)
   - Probability of loss > threshold
   - Comparison to baseline: "5% worse VaR than baseline", etc.
4. Visualize: Distribution plots, heatmaps of metric changes

---

## 8. Real-Time Filtering & Monitoring

### The Monthly Update Workflow

**At end of month T:**
1. Observe r_T^(pub) (public returns for month T)
2. Possibly observe y_q^(priv) (if quarter q just ended and lag L = 0)
3. Use filter to update:
   - P(s_T | data up to T) (current regime probabilities)
   - P(x_T^(priv) | data up to T) (current private return estimate)
   - Posterior θ is NOT refitted (uses saved posterior from calibration)

### Mathematical Filter Update

**State-space formulation:**

**State:** s_T (regime), x_T^(priv) (latent private returns)

**Observation:** r_T^(pub) and possibly y_q^(priv)

**Update equation:**

$$P(\text{state}_T | \text{obs}_T) \propto P(\text{obs}_T | \text{state}_T) \cdot P(\text{state}_T | \text{data}_{1:T-1})$$

**Likelihood of new observation:**

For new r_T^(pub):
$$P(r_T^{(\text{pub})} | s_T = k) = \int P(z_T | s_T = k) \, dx_T^{(\text{priv})} = \text{MVT}_k^{(\text{pub})}(r_T^{(\text{pub})})$$

For new y_q (if arrived):
$$P(y_q | x_{...}^{(\text{priv})}) = \mathcal{N}(\phi y_{q-1} + (1-\phi) \tilde{y}_q, R)$$

### Regime Probability Update Example

**Before month T:** P(s_T = crisis) = 0.10 (10% chance in crisis)

**Observe r_T^(pub):** Highly negative returns, high volatility

**Likelihood ratio:** P(r_T | crisis) / P(r_T | growth) = 50× higher in crisis

**After update:** P(s_T = crisis | r_T) ≈ 0.83 (83% chance, strong signal)

**Implication for portfolio:** Risk metrics shift sharply (higher VaR, CVaR).

### Private Return Estimation

The filter also updates the estimate of x_T^(priv):

$$x_T^{(\text{priv})} | \text{data}_{1:T} \sim \mathcal{N}(\hat{x}_T^{(\text{priv})}, \text{Cov}(x_T^{(\text{priv})}))$$

(More precisely, mixture distribution over sampled regimes.)

**Use case:** Manager sees y_q^(priv) reported, but x_T^(priv) gives monthly granularity.

**Example:** y_2 (Q2 reported) arrived, shows +8% quarterly. Filter estimates:
- x_4^(priv) (month 4): +3%
- x_5^(priv) (month 5): +2%
- x_6^(priv) (month 6): +3%

These can differ significantly if months had disparate returns (x_4 strong, x_5 weak) that average to reported y_2.

### Monitoring Alerts

**Real-time signals from filter:**

| Signal | Interpretation | Action |
|--------|---|---|
| P(s_T = crisis) jumps from 10% → 60% | Possible regime shift | Increase monitoring, review hedges |
| x_T^(priv) < expected by 500 bps | Private assets underperforming | Check fund performance, NAV trends |
| φ estimate > 0.7 | Heavy smoothing detected | Private valuations increasingly stale |
| R increases (larger diag) | Measurement noise rising | Valuation uncertainty elevated |

---

## Summary: Model & Reality

| Model Component | Financial Reality |
|---|---|
| s_t (regime) | Market state (growth, crisis, recovery, etc.) |
| μ_k (regime mean) | Expected returns conditional on market state |
| Σ_k (regime covariance) | Risk structure, correlation regime |
| ν_k (tail parameter) | Fat-tail risk; crisis has fatter tails |
| φ (smoothing) | Illiquidity, infrequent valuation updates |
| R (measurement noise) | Valuation uncertainty |
| r_t^(pub) | Observed market prices (liquid, transparent) |
| x_t^(priv) (latent) | True economic returns of illiquid assets |
| y_q^(priv) (reported) | Fund reports, NAVs, etc. (smoothed, lagged) |
| B_k B_k^T (systemic) | Common risk factors (market, liquidity, credit, cycle) |
| D_k (idiosyncratic) | Diversifiable asset-specific risk |
| Forward algorithm | Inference of current regime from history |
| FFBS | Backward-looking regime identification |
| Filter | Real-time regime and state update (no refitting) |
| Scenario analysis | "What-if" risk questions for hedging/allocation |

---

**End of Finance Specification**
