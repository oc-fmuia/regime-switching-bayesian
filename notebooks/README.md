# Jupyter Notebooks: Bayesian Regime-Switching Model

This directory contains two comprehensive Jupyter notebooks that demonstrate the complete regime-switching Bayesian modeling workflow.

## Notebooks

### 1. `01_model_building.ipynb` — Model Building & Inference
**Duration**: 30-45 minutes  
**Level**: Intermediate

Demonstrates the fundamentals of Bayesian regime-switching models:

- **Data Generation**: Create synthetic multi-asset returns with:
  - Markov regime switches (Normal vs Stressed market)
  - Regime-conditional means and volatilities
  - Shock-driven (factor) components
  - Realistic financial dynamics

- **Model Construction**: Build PyMC Bayesian model with:
  - Dirichlet priors on transition matrices
  - Normal priors on regime means
  - HalfNormal priors on volatilities
  - Prior sensitivity analysis

- **Inference Setup**: Prepare NUTS sampling:
  - Model compilation and validation
  - Parameter dimensionality
  - Computational requirements

- **Posterior Analysis**:
  - Convergence diagnostics (Rhat, ESS)
  - Posterior distributions visualization
  - Parameter recovery from data

**Key Insights**:
- Regime identification is tractable from returns alone
- Posterior uncertainty quantifies parameter estimates
- Two-regime model captures market dynamics well
- Convergence diagnostics validate inference

---

### 2. `02_interactive_exploration.ipynb` — Scenario Analysis & Portfolio Metrics
**Duration**: 20-30 minutes  
**Level**: Applied

Leverages posterior inference to generate forward-looking scenarios:

- **Monte Carlo Paths**: Generate 1000+ scenarios with:
  - Regime-switching dynamics from posterior
  - Realistic return distribution
  - Shock-factor propagation
  - 1-year horizon (252 trading days)

- **Portfolio Analytics**:
  - Expected return & volatility
  - Sharpe ratio (risk-adjusted return)
  - Value-at-Risk (VaR) — downside tail
  - Conditional VaR (CVaR) — expected loss beyond VaR
  - Maximum drawdown — worst peak-to-trough decline

- **Scenario Analysis**:
  - Performance by regime (Normal vs Stressed)
  - Regime frequency and conditional statistics
  - Regime exposure vs portfolio return correlation
  - Regime timing impact on outcomes

- **Stress Testing**:
  - Worst-case scenarios (bottom 5%)
  - Best-case scenarios (top 5%)
  - Risk-return trade-offs
  - Allocation comparisons

**Key Insights**:
- Regime exposure is the dominant return driver
- Conditional statistics differ dramatically by regime
- Downside risk (CVaR) exceeds simple variance assumptions
- Portfolio allocation interacts with regime switching

---

## How to Use

### Prerequisites
```bash
# Install dependencies
pip install numpy pandas matplotlib seaborn jupyter pymc arviz

# Or use the project requirements
pip install -r ../requirements.txt
```

### Running the Notebooks

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open `01_model_building.ipynb`**:
   - Run cells sequentially (Shift+Enter)
   - Inspect data generation and model structure
   - Review posterior estimates and diagnostics
   - Estimated runtime: 5-10 minutes

3. **Open `02_interactive_exploration.ipynb`**:
   - Load posterior samples (from Notebook 1 or simulated)
   - Generate Monte Carlo scenarios
   - Compute portfolio metrics
   - Analyze by regime and stress scenarios
   - Estimated runtime: 10-15 minutes

---

## Code Organization

Each notebook is self-contained but builds on Notebook 1's results:

```
Notebook 1: Data → Model → Inference
    ↓
Notebook 2: Posterior Samples → Scenarios → Analysis
```

You can also run Notebook 2 standalone with simulated posterior samples.

---

## Key Functions Used

### From `src/`

**Regimes & Data** (Commits 2-4):
- `MarkovChain`: Regime-switching dynamics
- `StudentTReturnModel`: Student-t returns with fat tails
- `ShockModel` / `ReturnWithShocks`: Factor-driven components

**Inference** (Commits 5-6):
- `ModelBuilder`: Construct PyMC model
- `NUTSSampler`: NUTS sampling orchestration
- `DiagnosticsComputer`: Convergence checks (Rhat, ESS)

**Simulation** (Commit 7):
- `MonteCarloSimulator`: Generate scenarios from posterior
- Path generation, analytics, scenario decomposition

---

## What Each Notebook Teaches

### Notebook 1: Conceptual Foundation
✅ How regime-switching models work  
✅ Bayesian inference from financial data  
✅ Convergence diagnostics  
✅ Posterior interpretation  

### Notebook 2: Practical Application
✅ Portfolio optimization with uncertainty  
✅ Risk quantification (VaR, CVaR)  
✅ Scenario analysis and stress testing  
✅ Regime exposure attribution  
✅ What-if analysis for decision-making  

---

## Real-World Extensions

These notebooks form the foundation for:

1. **Adaptive Portfolio Management**
   - Regime-timely allocation adjustments
   - Dynamic hedging strategies
   - Real-time risk monitoring

2. **Risk Management**
   - Enterprise VaR reporting
   - Stress test automation
   - Scenario generation for decision-makers

3. **Research Applications**
   - Publish findings on regime-switching models
   - Compare with traditional approaches
   - Extend to high-dimensional problems

4. **Production Deployment**
   - Automated monthly/quarterly rebalancing
   - Real-time posterior updates
   - API wrapper for portfolio systems

---

## Troubleshooting

### Slow Model Building (Notebook 1)
- This is normal for PyMC on CPU
- Consider GPU acceleration for production

### Memory Issues with Large Scenarios (Notebook 2)
- Reduce `n_scenarios` from 1000 to 500
- Or reduce `n_steps` from 252 to 100

### Import Errors
```bash
# Ensure src is in Python path
export PYTHONPATH=/path/to/regime-switching-bayesian/src:$PYTHONPATH
```

---

## References

**Key Papers**:
- Hamilton (1989) — Regime-switching models
- Guidolin & Timmermann (2007) — Multi-asset regime switching
- Glasserman (2004) — Monte Carlo methods in finance

**Libraries**:
- PyMC 5.x: https://docs.pymc.io
- ArviZ: https://arviz-devs.github.io
- NumPy/Pandas: Scientific computing stack

---

## License

MIT License. See `../LICENSE` for details.

---

**Questions?** See the parent documentation:
- `../docs/bayesian_model_builder.md` — Model theory
- `../docs/inference_and_diagnostics.md` — Inference details
- `../docs/monte_carlo_simulation.md` — Scenario generation
