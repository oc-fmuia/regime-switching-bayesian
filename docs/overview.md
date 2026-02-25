# Regime-Switching Bayesian Framework: Overview

**⚠️ START HERE: Read `problem_and_solution.md` first.** That document explains the concrete financial problem, shows what goes wrong with standard approaches, and walks through how each component solves it.

This page provides the organizational overview and tool justification. You should read it *after* understanding the financial problem.

## The Problem We're Solving

Financial markets don't operate in a single stable state—they shift between **regimes** (bull/bear, normal/crisis, low-vol/high-vol) with fundamentally different statistical properties.

**Consequence:** Standard statistical models fail because they:
- Underestimate tail risk during regime shifts
- Assume constant correlations (they spike in crises)
- Miss the shock structure that drives regime-dependent impacts
- Ignore estimation uncertainty

**See `problem_and_solution.md` for concrete examples** (2008 crisis, pension fund scenario, VaR failures).

This framework solves these problems by building a **probabilistic model that explicitly accounts for regime switching**, allowing practitioners to:
1. Infer current regime (+ uncertainty) from data
2. Model regime-dependent returns, correlations, tail risk
3. Understand shocks and their propagation by regime
4. Simulate forward-looking scenarios with full uncertainty quantification
5. Make portfolio decisions based on realistic risk metrics

## Why Bayesian?

We chose a **Bayesian approach** rather than frequentist methods for three fundamental reasons:

### 1. **Regime Uncertainty Is the Core Problem**

Frequentist approaches treat regimes as either:
- **Fixed in advance** (regime-switching GARCH assumes known structure)
- **Unknown but estimated pointwise** (single regime estimate via maximum likelihood)

Neither captures the **epistemic uncertainty** inherent in real decisions. In the Bayesian framework:
- We quantify a **posterior distribution over regimes** given current data
- Uncertainty propagates through the full inference chain
- Forecasts naturally incorporate regime identification error

### 2. **Prior Knowledge Improves Estimates**

Bayesian priors encode domain knowledge:
- **Regime persistence**: We expect regimes to last weeks/months, not days. The Dirichlet prior on transition matrices naturally encodes this
- **Return distributions**: Based on historical regime characteristics, we set priors on Student-t parameters (tail index, volatility)
- **Shock structure**: We specify expected impact sizes via hierarchical priors on loading matrices

Frequentist approaches ignore this information, requiring larger samples and producing overconfident intervals.

### 3. **Decisions Require Full Uncertainty Quantification**

Portfolio optimization under regime switching requires:
- **Posterior samples** from the full joint distribution (regimes, returns, shocks)
- **Propagation of uncertainty** into portfolio metrics (VaR, expected return, drawdown)
- **Sensitivity analysis** (how do results change with different priors?)

The Bayesian posterior is exactly the right object for these tasks. Frequentist confidence intervals don't naturally propagate through complex decision functions.

### Bayesian vs. Frequentist in Practice

| Aspect | Frequentist | Bayesian |
|--------|-----------|----------|
| **Regime identification** | Point estimate (MLE) | Full posterior distribution |
| **Uncertainty** | Confidence intervals (misaligned with decision-making) | Posterior credible intervals (direct probability statements) |
| **Prior knowledge** | Ignored or ad-hoc (constraints) | Formally incorporated via prior distributions |
| **Forward simulation** | Difficult (how to propagate MLE uncertainty?) | Natural (sample from posterior, propagate to predictions) |
| **Scalability** | Fast (single optimization) | Slower (MCMC), but parallelizable |
| **Interpretation** | "If we repeated the experiment infinitely..." | "Given observed data, what do we believe?" |

For regime-switching portfolio analysis, Bayesian is not optional—it's necessary.

---

## Our Architecture

### Core Components (Commits 1–7)

```
regime-switching-bayesian/
├── src/
│   ├── regimes/           # Regime dynamics (Markov chains, shocks)
│   ├── returns/           # Return models (Student-t, covariance)
│   └── inference/         # Bayesian inference (PyMC, NUTS, diagnostics)
│   └── simulation/        # Forward-looking scenarios (Monte Carlo)
├── docs/                  # Mathematical foundations and guides
├── notebooks/             # Educational and interactive demonstrations
└── tests/                 # 130+ unit tests (100% pass rate)
```

### Tool Choices Explained

#### **PyMC** (Probabilistic Programming)

Why PyMC instead of Stan or other frameworks?
- **Pythonic**: Natural integration with NumPy/SciPy ecosystem
- **Flexible**: Can express complex hierarchical structures (regime-dependent priors)
- **NUTS sampler built-in**: No external dependencies
- **ArviZ integration**: Automatic convergence diagnostics (Rhat, ESS, divergence)
- **Posterior predictive**: Straightforward model checking

#### **NUTS Sampler** (Hamiltonian MCMC)

Why not Gibbs or variational inference?
- **Efficient**: O(d^0.25) scaling vs O(d^0.5) for simpler methods (important as model grows)
- **Robust**: Works well for correlated posteriors (regime identification creates strong correlations)
- **Diagnostics**: Divergence detection catches inference failures automatically
- **No tuning**: NUTS adapts step size and tree depth automatically

#### **Student-t Returns** (Fat Tails)

Why not Gaussian?
- **Real data**: Financial returns exhibit heavy tails (kurtosis >> 3)
- **Gaussian misspecifies risk**: Underestimates extreme losses by orders of magnitude
- **Regime switching amplifies the problem**: Tail index (ν) varies by regime
- **Student-t captures this**: Hierarchical priors on ν per regime

#### **Shock Propagation** (Factor Model)

Why explicit shocks?
- **Interpretability**: Each shock has a meaning (yield curve shift, VIX spike, credit spread)
- **Regime-dependent loading**: Same shock hits differently in different regimes
- **Variance decomposition**: Understand systematic vs. idiosyncratic risk
- **Stress testing**: Apply custom scenarios deterministically

---

## Documentation Map

This framework is documented across multiple files, each serving a specific purpose:

### 1. **overview.md** (this file)
   - **What it covers**: Big picture, motivation, tool justification, Bayesian reasoning
   - **When to read**: First—to understand *why* we built this way
   - **Audience**: Stakeholders, decision-makers, practitioners new to Bayesian regime-switching

### 2. **math_foundations.md**
   - **What it covers**: Mathematical preliminaries (Bayesian inference, MCMC basics, notation)
   - **When to read**: Before diving into component docs if you're unfamiliar with Bayesian notation
   - **Audience**: Practitioners comfortable with probability but new to Bayesian inference

### 3. **regime_switching.md**
   - **What it covers**: Markov chain theory, stationary distributions, regime interpretation
   - **When to read**: If you want to understand the regime dynamics layer
   - **Maps to code**: `src/regimes/markov.py` (MarkovChain class)
   - **Audience**: Data scientists, portfolio managers understanding regime identification

### 4. **shock_propagation.md**
   - **What it covers**: Shock model mathematics, factor loadings, variance decomposition
   - **When to read**: If you need to understand how shocks affect returns and how to use them for stress testing
   - **Maps to code**: `src/regimes/shocks.py` (ShockModel, ReturnWithShocks)
   - **Audience**: Risk managers, practitioners building stress tests

### 5. **bayesian_model_builder.md**
   - **What it covers**: Full model specification, prior choices, PyMC assembly
   - **When to read**: If you want to customize priors or understand the full joint distribution
   - **Maps to code**: `src/inference/model_builder.py` (PriorSpec, ModelBuilder)
   - **Audience**: Bayesian practitioners, anyone tuning the model

### 6. **inference_and_diagnostics.md**
   - **What it covers**: NUTS sampler, convergence diagnostics (Rhat, ESS), posterior predictive checks
   - **When to read**: If you need to run inference, diagnose convergence problems, or validate your model
   - **Maps to code**: `src/inference/sampler.py` (NUTSSampler, DiagnosticsComputer)
   - **Audience**: Practitioners running inference, model validators

### 7. **monte_carlo_simulation.md**
   - **What it covers**: Forward-looking scenario generation, portfolio metrics, VaR/CVaR computation
   - **When to read**: If you want to generate paths from posterior samples or compute portfolio statistics
   - **Maps to code**: `src/simulation/simulator.py` (MonteCarloSimulator)
   - **Audience**: Risk managers, portfolio analysts, practitioners building dashboards

### 8. **Jupyter Notebooks** (coming in Commit 8)
   - **01_model_building.ipynb**: Walkthrough of data generation → inference → diagnostics
   - **02_interactive_exploration.ipynb**: Hands-on tuning and scenario exploration
   - **When to read**: After understanding the individual components, to see them working together

---

## Workflow: From Data to Decisions

Here's how to use this framework end-to-end:

### Step 1: Model Building
```
regimes/ (Markov chain)
+ returns/ (Student-t model)
+ shocks/ (Shock structure)
→ model_builder.py assembles in PyMC
```
**Read**: regime_switching.md, shock_propagation.md, bayesian_model_builder.md

### Step 2: Inference
```
PyMC model
+ observed returns
+ NUTS sampler
→ posterior samples
→ convergence diagnostics
```
**Read**: inference_and_diagnostics.md

### Step 3: Scenario Analysis
```
posterior samples (regimes, returns, shock parameters)
+ Monte Carlo simulator
→ forward-looking paths
→ portfolio metrics (VaR, CVaR, Sharpe, drawdown)
```
**Read**: monte_carlo_simulation.md

### Step 4: Decision Making
```
Scenario results
+ business logic
→ portfolio optimization
→ risk reports
→ stress testing
```
**Maps to**: Your application code (not in this framework)

---

## Key Design Decisions

### 1. **Modular Architecture**

Each component (regimes, returns, shocks) is standalone and testable. This means:
- You can use the MarkovChain class independently
- You can plug in a different return model (e.g., Gaussian)
- You can run inference without forward simulation

**Benefit**: Flexibility and reusability.

### 2. **Progressive Test Sizing**

Tests use small (n=10), medium (n=50), and large (n=100) datasets. This ensures:
- Unit tests complete in <5 seconds total
- Edge cases caught at scale without waiting
- Numerical stability verified across regimes

**Benefit**: Fast iteration + confidence in scaling.

### 3. **Type Hints + Docstrings on Every Function**

Every function has:
- Full type annotations (inputs/outputs)
- Mathematical docstrings with formulas
- Usage examples

**Benefit**: Self-documenting code, IDE autocompletion, fewer bugs.

### 4. **No External Data Required**

All tests and examples use synthetic data. This means:
- Reproducible, no data-access issues
- Clear assumptions (you see exactly what the model assumes)
- Easy to extend with real data

**Benefit**: Transparency and ease of deployment.

---

## When to Use This Framework

### ✅ Good Use Cases

- **Portfolio risk analysis** under regime shifts
- **Asset allocation** accounting for correlation regimes
- **Stress testing** with shock scenarios
- **Derivative pricing** under regime-dependent volatility
- **Regulatory reporting** with proper uncertainty quantification
- **Research** into Bayesian regime-switching methods

### ❌ Not Suitable For

- **High-frequency trading** (regime dynamics operate at daily/weekly scale)
- **Real-time inference** (MCMC is too slow; use variational inference or priors instead)
- **Single-regime data** (use simpler frequentist models)

---

## Next Steps

1. **Start here**: Read this file (overview.md) — you're doing it ✓
2. **Big picture**: Skim math_foundations.md to orient yourself
3. **Deep dive**: Choose a component:
   - Curious about regimes? → regime_switching.md + regimes/markov.py
   - Curious about shocks? → shock_propagation.md + regimes/shocks.py
   - Curious about inference? → bayesian_model_builder.md + inference_and_diagnostics.md
4. **Hands-on**: Run the notebooks (coming Commit 8)
5. **Customize**: Modify priors in PriorSpec, add your own shocks, build on top

---

## Questions This Framework Answers

| Question | Where to Find the Answer |
|----------|--------------------------|
| What regime are we in now? | inference_and_diagnostics.md → regime posterior samples |
| How uncertain are we? | inference_and_diagnostics.md → Rhat, ESS, credible intervals |
| How do shocks propagate? | shock_propagation.md → factor loading matrices, variance decomposition |
| What's the portfolio VaR? | monte_carlo_simulation.md → portfolio metrics function |
| How do I customize priors? | bayesian_model_builder.md → PriorSpec examples |
| Does my model fit the data? | inference_and_diagnostics.md → posterior predictive checks |
| How sensitive are results to priors? | bayesian_model_builder.md → sensitivity section |

---

## Summary

This is a **production-ready Bayesian regime-switching framework** for multi-asset returns. It combines:
- **Markov chains** for regime dynamics
- **Student-t distributions** for fat tails
- **Shock propagation** for factor structure
- **NUTS inference** for scalable Bayesian computation
- **Monte Carlo simulation** for portfolio analysis

The Bayesian approach is essential because regime identification inherently uncertain—we don't just want a point estimate, we want the full posterior distribution to propagate through our decisions.

All components are modular, well-tested, and documented. Start with this overview, pick your entry point (a specific component or a use case), and dive in.

**Questions?** See the relevant documentation page linked above.
