# Finance Specification v0: Regime-Switching for Public Assets

**Version:** 0.1
**Date:** March 2026
**Scope:** Minimal use case — public assets only, two regimes, monthly data

---

## 1. Use Case

A portfolio manager holds a small portfolio of broad public equity indices — for example, US Large Cap, International Developed, and Emerging Markets. Historical experience shows that markets alternate between two qualitatively different environments:

- **Calm periods** with positive average returns, low volatility, and moderate cross-asset correlation.
- **Stress periods** with negative or near-zero average returns, elevated volatility, and sharply increased correlation (assets fall together).

The manager wants to:

1. Estimate the parameters of each regime from historical data.
2. Infer which regime was active at each point in history.
3. Assess regime-conditional portfolio risk (e.g., VaR).

This v0 model addresses exactly these three questions and nothing more.

---

## 2. Data

- **Assets:** d public equity indices (d typically 3–5).
- **Frequency:** Monthly log-returns.
- **Observation:** Fully observed, no missing data.
- **Horizon:** T months of history (e.g., T = 120 for 10 years).

Log-returns are used because they are additive over time and empirically closer to symmetric distributions than simple returns.

---

## 3. Regime Interpretation

### Regime 1: Bull / Growth

- Positive expected monthly returns (e.g., +0.5% to +1.5% per month).
- Low volatility (e.g., 3–5% monthly standard deviation, roughly 10–17% annualized).
- Moderate cross-asset correlations (e.g., 0.3–0.6).
- Typically the dominant regime — markets spend most of the time here.

### Regime 2: Bear / Stress

- Negative or near-zero expected monthly returns (e.g., -1% to 0%).
- High volatility (e.g., 6–12% monthly standard deviation, roughly 20–40% annualized).
- Elevated cross-asset correlations (e.g., 0.6–0.9) — diversification breaks down.
- Shorter in duration but high impact on portfolio losses.

### Key Insight

A single-regime model (constant mean and covariance) averages over these two environments. It underestimates risk in stress periods and overestimates it in calm periods. The regime-switching model captures this asymmetry.

---

## 4. What the Model Answers

**Q1: What regime are we in?**
The forward algorithm produces filtered regime probabilities P(s_t = k | data up to t) at each month. A sharp rise in P(s_t = Bear) signals a regime shift.

**Q2: How do returns differ by regime?**
Posterior estimates of mu_k and Sigma_k reveal the return and risk profile of each regime. The model quantifies how much worse the Bear regime is relative to Bull.

**Q3: What is the regime-conditional portfolio risk?**
For a portfolio with weights w, the regime-conditional return is w^T mu_k with variance w^T Sigma_k w. VaR and other risk metrics can be computed per regime, or as a weighted average across regimes using the filtered probabilities.

---

## 5. Scope Boundaries

This v0 deliberately excludes:

- **Private assets and mixed-frequency data.** All assets are public and monthly.
- **Fat tails (Student-t).** Returns are modeled as multivariate Normal. Student-t can be added as a later extension.
- **Hierarchical mean structure.** Each regime has a simple mean vector — no geography, sector, or asset-level decomposition.
- **Factor covariance structure.** Each regime has a full covariance matrix. Factor decomposition (B_k B_k^T + D_k) is only needed when d is large.
- **Scenario analysis and real-time filtering.** These are downstream applications built on top of the fitted model.

Each of these can be layered on independently once the v0 model is working and validated.

---

**End of Finance Specification v0**
