# Tutorial Notebook Plan (Marimo) -- v0

**Date:** March 2026
**Prerequisites:** All modules from `implement_v0.md` Steps 1-6 passing.

---

## Goal

A single marimo notebook that walks a reader through the full v0 regime-switching pipeline,
from synthetic data to posterior interpretation. The notebook is designed to double as a
draft blog post: every code cell is preceded by narrative prose, and every figure is captioned
with financial intuition.

## File

```
notebooks/regime_switching_v0.py
```

Marimo notebooks are plain Python files, so they diff cleanly in git and need no special
tooling to review.

## Dependency

`marimo` is added as a `notebook` optional-dependency group in `pyproject.toml` with a
corresponding pixi environment.

---

## Section Outline

### Section 1 -- Setup and Imports

- `import marimo as mo`
- Import `src.data_gen`, `src.model`, `src.inference`, `src.plotting`
- Brief `mo.md()` cell: one-paragraph elevator pitch for the problem
  ("Equity markets alternate between calm growth and volatile drawdowns.
  A regime-switching model discovers these states from return data alone.")

### Section 2 -- The Story

Narrative-only cell (`mo.md()`). Two paragraphs covering:

- **Why regime-switching?** Traditional models assume a single return distribution;
  this misses the clustering of volatility and mean shifts that portfolio managers
  see in practice.
- **What this notebook demonstrates.** We build a 2-regime Hidden Markov Model for
  3 equity indices, fit it with Bayesian inference (PyMC + NUTS), and recover the
  most likely regime sequence. Everything is synthetic so we can check our answers.

### Section 3 -- Generate Synthetic Data

- Call `generate_hmm_data(T=120, K=2, d=3)` with default parameters.
- Display the default parameters in a formatted table (`mo.ui.table` or `mo.md()` with
  a markdown table).
- **Figure 1 -- Synthetic returns:** three-panel time-series plot of asset returns.
  Background shading indicates the true regime (light green = Bull, light red = Bear).
  Use `matplotlib` with `fig, axes = plt.subplots(d, 1, sharex=True)`.
- Caption: explain what the reader should notice (higher vol in shaded Bear periods,
  positive drift in Bull).

### Section 4 -- Specify the Bayesian Model

- Call `build_model(data["returns"], K=2)` to get `(model, model_marg)`.
- **Figure 2 -- Model DAG:** render `pm.model_to_graphviz(model)` as an image.
  Explain each node in prose:
  - `P` -- the transition matrix (sticky Dirichlet prior encourages persistence)
  - `mu` -- per-regime mean returns
  - `chol_cov_k` -- per-regime covariance via LKJ
  - `chain` -- the latent regime sequence (marginalized out for NUTS)
  - `obs` -- observed returns
- Short prose block: why marginalization matters (NUTS needs continuous parameters;
  we integrate out the discrete chain analytically via the forward algorithm, which
  `pymc_extras.marginalize` does internally).

### Section 5 -- Fit with NUTS

- Call `fit(model_marg, draws=2000, tune=2000, chains=4, seed=42)`.
- Display `check_diagnostics(idata)` as a summary table.
- **Figure 3 -- Trace plots:** `az.plot_trace(idata, var_names=["P", "mu"])`.
- Prose: what to look for in traces (mixing, stationarity, no divergences).
- If diagnostics fail, add a `mo.callout()` warning.

### Section 6 -- Recover Regimes via FFBS

- Prose: explain the `recover_marginals` limitation discovered in our empirical tests
  and why we fall back to a custom NumPy FFBS.
- Call `run_ffbs(idata, data["returns"], seed=42)`.
- Compute `P(s_t = 1)` across all posterior draws (mean over chains and draws).
- **Figure 4 -- Regime probabilities:** call `plot_regime_probabilities(regime_samples,
  true_regimes=data["regimes"])`. Overlay the true regime labels as a step function.
- Caption: discuss how well the model recovers regimes and where uncertainty is highest
  (transition points).

### Section 7 -- Posterior Interpretation

- **Figure 5 -- Parameter recovery:** call `plot_posterior_summary(idata,
  true_params=data["params"])`. Forest plot with true values marked as vertical lines.
- Prose discussion:
  - Are the true means inside the 94% HDI?
  - Does the posterior transition matrix reflect the true persistence?
  - What would a portfolio manager learn from the Bear-regime covariance?
- **Figure 6 -- Regime-conditional return distributions:** for each regime, plot the
  marginal return distribution (posterior predictive histogram or KDE per asset),
  overlaid with the true density. Use the posterior mean of `mu` and `chol_cov` per
  regime.

### Section 8 -- Next Steps

Narrative-only cell. Tease v1 extensions:

- Multivariate Student-t emissions (fat tails)
- Hierarchical priors across asset classes
- Mixed-frequency data (quarterly private assets)
- Factor structure for covariance
- Real data application

---

## Notebook Conventions

- **No logic duplication.** Every computation calls a function from `src.*`. The notebook
  contains only glue code, calls, and narrative.
- **Reproducibility.** All random seeds are set explicitly. The notebook can be run
  end-to-end with `marimo run notebooks/regime_switching_v0.py`.
- **Reactivity.** Leverage marimo's reactive execution: changing a parameter in Section 3
  (e.g. `T` or `seed`) automatically reruns downstream sections.
- **Figure style.** Use `matplotlib` with `seaborn` defaults (`sns.set_theme()`). All
  figures should have titles, axis labels, and legends.
- **Marimo UI elements.** Use `mo.ui.slider` or `mo.ui.number` for interactive parameter
  exploration where it aids understanding (e.g. number of time steps, seed). Keep it
  minimal -- the primary audience is a blog reader, not a dashboard user.

---

## Implementation Notes

- This is Step 7 in `implement_v0.md`, built after the integration test (Step 6) passes.
- Estimated effort: Medium (mostly prose and plotting, since all logic lives in `src/*`).
- The notebook should be testable by running `marimo run notebooks/regime_switching_v0.py`
  and verifying it completes without error.

---

**End of Tutorial Notebook Plan**
