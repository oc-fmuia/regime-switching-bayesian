import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    from pathlib import Path

    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    import arviz as az
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")

    from src.data_gen import generate_hmm_data
    from src.inference import (
        align_regime_samples,
        check_diagnostics_label_aware,
        fit,
        run_ffbs,
    )
    from src.model import build_model
    from src.plotting import (
        plot_posterior_summary,
        plot_regime_probabilities,
        plot_returns_with_regimes,
    )

    return (
        align_regime_samples,
        az,
        build_model,
        check_diagnostics_label_aware,
        fit,
        generate_hmm_data,
        np,
        plot_posterior_summary,
        plot_regime_probabilities,
        plot_returns_with_regimes,
        plt,
        run_ffbs,
        sns,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian Regime-Switching for Equity Returns

    Equity markets alternate between calm growth and volatile drawdowns.
    A single-distribution model — one mean vector, one covariance matrix —
    cannot capture this structure: it overestimates volatility in calm periods
    and underestimates tail risk during crises.

    A **regime-switching model** discovers these hidden market states from
    return data alone, giving portfolio managers a probabilistic,
    time-varying view of the current environment and its likely persistence.

    This notebook builds a Bayesian Hidden Markov Model (HMM) in PyMC,
    fits it via NUTS on a marginalised likelihood, and recovers the latent
    regime sequence using a forward-filter backward-sampler (FFBS).
    Because the data are synthetic, we can verify that the model recovers
    the true parameters and regimes.

    **References.**
    Hamilton (1989), *Econometrica*;
    Ang & Bekaert (2002), *Review of Financial Studies*;
    Gelman et al. (2013), *Bayesian Data Analysis* 3rd ed. (BDA3).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## The statistical question

    > **Can we identify structurally distinct market regimes from equity
    > return data alone, and if so, what do the regime-conditional parameters
    > imply for portfolio construction?**

    Traditional portfolio models assume returns are drawn from a single
    multivariate distribution — one mean vector $\boldsymbol{\mu}$ and one
    covariance matrix $\boldsymbol{\Sigma}$.  This assumption fails when the
    data-generating process switches between distinct economic environments.
    Hamilton (1989) introduced the Markov-switching framework for business-cycle
    analysis; Ang & Bekaert (2002) showed that accounting for regimes
    materially changes optimal asset allocation and risk measurement.

    A **Hidden Markov Model (HMM)** with $K$ latent states provides a principled
    framework: conditioned on the state, returns follow a regime-specific
    multivariate Normal, and the states evolve as a first-order Markov chain.
    Bayesian inference propagates uncertainty about *which regime is active at
    each point in time* directly into posterior quantities, avoiding the false
    precision of point estimates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Synthetic data

    We work with three synthetic equity indices — **US Equity**, **EAFE
    Equity** (developed international), and **EM Equity** (emerging
    markets).  All three share the same two-regime structure: a
    **bull** regime with moderate positive drift and low volatility, and a
    **bear** regime with negative drift and elevated volatility.

    The synthetic data are generated from a known $K = 2$ HMM with
    identity correlation matrices (assets are conditionally uncorrelated
    given the regime), allowing us to verify that the model recovers the
    true parameters and regimes in a clean setting.  All parameters below
    are calibrated to realistic monthly magnitudes; annualised equivalents
    are shown for interpretation.

    A subsequent notebook in this series extends the framework to a
    multi-asset universe (equities, bonds, gold) with regime-dependent
    correlations — a materially harder inference problem.
    """)
    return


@app.cell
def _():
    asset_names = ["US Equity", "EAFE Equity", "EM Equity"]
    return (asset_names,)


@app.cell
def _(mo):
    T_slider = mo.ui.slider(
        start=60, stop=360, value=120, step=12,
        label="T (months)",
    )
    seed_number = mo.ui.number(
        start=0, stop=9999, value=42,
        label="Random seed",
    )
    sampler_dropdown = mo.ui.dropdown(
        options={"NumPyro (JAX, default)": "numpyro", "PyMC": "pymc"},
        value="NumPyro (JAX, default)",
        label="NUTS sampler",
    )
    mo.hstack([T_slider, seed_number, sampler_dropdown], justify="start")
    return T_slider, sampler_dropdown, seed_number


@app.cell
def _(asset_names, mo):
    bull_mean_sliders = [
        mo.ui.slider(start=0.0, stop=0.05, value=v, step=0.001,
                      label=f"Bull μ – {a}")
        for a, v in zip(asset_names, [0.010, 0.008, 0.012])
    ]
    bear_mean_sliders = [
        mo.ui.slider(start=-0.05, stop=0.0, value=v, step=0.001,
                      label=f"Bear μ – {a}")
        for a, v in zip(asset_names, [-0.005, -0.008, -0.003])
    ]
    bull_vol_sliders = [
        mo.ui.slider(start=0.01, stop=0.10, value=v, step=0.005,
                      label=f"Bull σ – {a}")
        for a, v in zip(asset_names, [0.040, 0.035, 0.045])
    ]
    bear_vol_sliders = [
        mo.ui.slider(start=0.02, stop=0.20, value=v, step=0.005,
                      label=f"Bear σ – {a}")
        for a, v in zip(asset_names, [0.080, 0.090, 0.100])
    ]

    mo.md("### Regime parameters (data generation)")
    mo.vstack([
        mo.md("**Bull (growth) regime**"),
        mo.hstack(bull_mean_sliders, justify="start"),
        mo.hstack(bull_vol_sliders, justify="start"),
        mo.md("**Bear (stress) regime**"),
        mo.hstack(bear_mean_sliders, justify="start"),
        mo.hstack(bear_vol_sliders, justify="start"),
    ])
    return (
        bear_mean_sliders,
        bear_vol_sliders,
        bull_mean_sliders,
        bull_vol_sliders,
    )


@app.cell
def _(
    T_slider,
    bear_mean_sliders,
    bear_vol_sliders,
    bull_mean_sliders,
    bull_vol_sliders,
    generate_hmm_data,
    np,
    seed_number,
):
    _bull_mu = np.array([s.value for s in bull_mean_sliders])
    _bear_mu = np.array([s.value for s in bear_mean_sliders])
    _bull_vol = np.array([s.value for s in bull_vol_sliders])
    _bear_vol = np.array([s.value for s in bear_vol_sliders])

    data = generate_hmm_data(
        T=T_slider.value, K=2, d=3, seed=seed_number.value,
        mus=np.vstack([_bull_mu, _bear_mu]),
        sigmas=np.vstack([_bull_vol, _bear_vol]),
    )
    return (data,)


@app.cell
def _(asset_names, data, mo, np):
    _p = data["params"]
    _c = data["config"]
    _mus_ann = _p["mus"] * 12 * 100
    _vols_ann = _p["sigmas"] * np.sqrt(12) * 100
    _dur = [1.0 / (1.0 - _p["P"][k, k]) for k in range(2)]

    _rows = []
    for _i, _name in enumerate(asset_names):
        _rows.append(
            f"| Mean: {_name} | {_mus_ann[0, _i]:+.1f}% | {_mus_ann[1, _i]:+.1f}% |"
        )
    for _i, _name in enumerate(asset_names):
        _rows.append(
            f"| Volatility: {_name} | {_vols_ann[0, _i]:.1f}% | {_vols_ann[1, _i]:.1f}% |"
        )
    _rows.append(f"| Expected duration | {_dur[0]:.1f} months | {_dur[1]:.1f} months |")

    _table = "\n".join(_rows)

    mo.md(
        f"""
### Generating parameters (annualised)

| Parameter | Bull (regime 0) | Bear (regime 1) |
|---|---|---|
{_table}

**T** = {_c['T']} months, **d** = {_c['d']} assets, **seed** = {_c['seed']}.

The bull regime has moderate positive drift and low volatility.  The bear
regime features negative drift across all equities and roughly doubled
volatility — the classic growth/stress dichotomy.  Assets are conditionally
uncorrelated given the regime (identity correlation matrices); a later
notebook in this series introduces regime-dependent cross-asset correlations.
        """
    )
    return


@app.cell
def _(asset_names, data, plot_returns_with_regimes):
    fig_returns = plot_returns_with_regimes(
        data["returns"], data["regimes"], asset_names=asset_names,
    )
    fig_returns
    return


@app.cell
def _(mo):
    mo.md(r"""
    The green-shaded periods are **bull** (regime 0) and the red-shaded
    periods are **bear** (regime 1).  Notice the visibly higher volatility
    and negative drift during bear episodes across all three equity indices.
    This is the regime-conditional structure our model will attempt to
    recover.
    """)
    return


@app.cell
def _(asset_names, data, plt, sns):
    _returns = data["returns"]
    _regimes = data["regimes"]
    _d = _returns.shape[1]

    fig_dist, axes_dist = plt.subplots(1, _d, figsize=(14, 4), sharey=True)
    for _i in range(_d):
        _ax = axes_dist[_i]
        for _k, (_label, _color) in enumerate(
            zip(["Bull", "Bear"], [sns.color_palette()[2], sns.color_palette()[3]])
        ):
            _vals = _returns[_regimes == _k, _i]
            if len(_vals) > 2:
                sns.kdeplot(_vals, ax=_ax, color=_color, label=_label, fill=True, alpha=0.3)
        _ax.set_title(asset_names[_i])
        _ax.set_xlabel("Monthly return")
        if _i == 0:
            _ax.legend()

    fig_dist.suptitle("Regime-Conditional Return Distributions (true labels)", y=1.02)
    fig_dist.tight_layout()
    fig_dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    These regime-conditional marginals show the return separation that the
    model will attempt to recover.  The mean shift (leftward) and
    volatility expansion (wider spread) in the bear regime are clearly
    visible for all three equity indices.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bayesian model specification

    We seek the joint posterior over all continuous parameters given the
    observed returns:

    $$
    p(\mathbf{P}, \boldsymbol{\mu}_{0:K-1}, \boldsymbol{\Sigma}_{0:K-1}
    \mid \mathbf{y}_{1:T})
    \;\propto\;
    p(\mathbf{y}_{1:T} \mid \mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\Sigma})
    \;\cdot\;
    p(\mathbf{P})\, p(\boldsymbol{\mu})\, p(\boldsymbol{\Sigma})
    $$

    where the likelihood $p(\mathbf{y}_{1:T} \mid \cdot)$ is the
    marginalised HMM likelihood obtained by summing over all $K^T$ possible
    regime sequences via the forward algorithm.

    ### Prior choices

    Each prior is weakly informative and calibrated to monthly financial
    return magnitudes:

    - **Transition matrix** $\mathbf{P}$: sticky Dirichlet prior with
      $\alpha_{\text{diag}} = 20$, $\alpha_{\text{off}} = 2$.  The implied
      prior mean self-transition probability is $20/22 \approx 0.91$,
      encouraging regime persistence consistent with observed market dynamics
      where bull phases last years and bear episodes last months (Hamilton, 1989).

    - **Regime means** $\boldsymbol{\mu}_k \sim \mathcal{N}(\mathbf{0},\,
      0.05\,\mathbf{I})$: centered at zero monthly return, with a prior
      standard deviation of $5\%$ monthly ($\approx \pm 17\%$ annualised at
      $1\sigma$).  This accommodates both growth-regime drift ($\sim +1\%$
      monthly) and crisis-regime drawdowns ($\sim -2\%$ monthly) without
      excluding extreme scenarios (BDA3, Ch. 5).

    - **Regime covariances**: parameterised via `LKJCholeskyCov` with
      $\eta = 2$ (mild shrinkage toward uncorrelated assets) and
      `HalfNormal(0.10)` priors on the per-asset standard deviations.  At
      $\eta = 2$ the LKJ prior places most mass on correlation matrices with
      moderate off-diagonal entries, reflecting the belief that assets are
      neither perfectly correlated nor perfectly independent.  The
      `HalfNormal(0.10)` prior on monthly standard deviations encodes the
      belief that per-asset monthly volatilities are typically below $\sim
      35\%$ annualised at $2\sigma$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Marginalisation and the forward algorithm

    The discrete regime sequence $s_{1:T}$ is **analytically marginalised**
    via the forward algorithm (Hamilton, 1989).  This is essential: the
    No-U-Turn Sampler (NUTS) requires a differentiable, continuous parameter
    space and cannot sample discrete variables directly.

    The forward recursion computes the log-forward probabilities
    $\log \alpha_{t,k} = \log p(\mathbf{y}_{1:t},\, s_t = k \mid \theta)$
    via

    $$
    \log \alpha_{t,k} =
    \log \sum_{k'} \exp\!\bigl(\log \alpha_{t-1,k'} + \log P_{k',k}\bigr)
    \;+\; \log p(\mathbf{y}_t \mid s_t = k)
    $$

    with $\log \alpha_{1,k} = \log \pi_{0,k} + \log p(\mathbf{y}_1 \mid
    s_1 = k)$.  The marginalised log-likelihood is
    $\log p(\mathbf{y}_{1:T} \mid \theta) = \text{logsumexp}_k\,
    \log \alpha_{T,k}$.

    The recursion is implemented via `pytensor.scan`, which compiles to
    `jax.lax.scan` under the NumPyro backend for efficient execution.
    """)
    return


@app.cell
def _(build_model, data):
    model = build_model(data["returns"], K=2)
    return (model,)


@app.cell
def _(mo, model):
    import pymc as pm

    try:
        graph = pm.model_to_graphviz(model)
        _dag_msg = (
            "### Model DAG\n\n"
            "The graph below shows the model's dependency structure.  The "
            "discrete regime chain does not appear as a node — it has been "
            "analytically marginalised.  The `hmm_loglik` Potential encodes "
            "the forward-algorithm log-likelihood as a scalar contribution "
            "to the joint log-density."
        )
    except Exception:
        graph = None
        _dag_msg = "*(graphviz not available — skipping DAG render)*"
    mo.md(_dag_msg)
    return (graph,)


@app.cell
def _(graph):
    graph
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Model assumptions

    The following assumptions are built into the current specification.
    Each represents a modelling choice with known limitations:

    1. **Gaussian emissions.** Returns are conditionally Normal given the
       regime.  This captures regime-level mean and volatility shifts but
       cannot model intra-regime fat tails (excess kurtosis within a single
       regime).  A multivariate Student-$t$ extension addresses this
       limitation.

    2. **Time-homogeneous transition matrix.**  $\mathbf{P}$ is constant
       across time — regime-switching probabilities do not depend on
       observable covariates (e.g. VIX, yield curve slope).  Time-varying
       transition probabilities (TVTP) relax this.

    3. **$K = 2$ regimes.**  The number of regimes is fixed a priori,
       not selected from data.  Bayesian model comparison via WAIC can
       evaluate whether $K = 2$ is adequate relative to $K = 1$ or $K = 3$.

    4. **Conditional independence across time.**  Given the regime,
       $\mathbf{y}_t$ is independent of $\mathbf{y}_{t-1}$.  This rules out
       within-regime momentum or mean-reversion effects (Hamilton's original
       1989 model used AR(4) dynamics within each regime).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Posterior inference

    > **Question:** Given the observed returns and our prior beliefs, what
    > are the posterior distributions over the transition matrix, regime means,
    > and regime covariances?

    We draw samples from the marginalised posterior using the No-U-Turn Sampler
    (NUTS) with 4 independent chains, 2000 tuning steps, and 2000 posterior
    draws per chain (8000 draws total).  The NumPyro/JAX backend is used by
    default for faster compilation and sampling.

    > **Note:** The first run involves PyTensor → JAX compilation and may take
    > a few minutes.  Subsequent runs with the same model graph are faster.
    """)
    return


@app.cell
def _(data, fit, model, sampler_dropdown):
    _mus_init = data["params"]["mus"]
    idata = fit(
        model, draws=2000, tune=2000, chains=4, seed=42,
        nuts_sampler=sampler_dropdown.value,
        initvals={"mu": _mus_init},
    )
    return (idata,)


@app.cell
def _(check_diagnostics_label_aware, idata, mo):
    diag = check_diagnostics_label_aware(idata, K=2)
    _status = lambda v: "pass" if v else "**FAIL**"
    _ls = diag["label_switching_detected"]
    _ls_msg = (
        f"Yes — naive max R-hat was {diag['naive_max_rhat']:.3f}, "
        f"improved to {diag['max_rhat']:.3f} after relabeling "
        f"(best permutations: {diag['best_permutations']})"
        if _ls
        else "No"
    )
    mo.md(
        f"""
### Convergence diagnostics

| Check | Value | Status |
|-------|-------|--------|
| Divergences | {diag['n_divergences']} | {_status(diag['no_divergences'])} |
| max R-hat | {diag['max_rhat']:.3f} | {_status(diag['rhat_ok'])} |
| min ESS (bulk) | {diag['min_ess_bulk']:.0f} | {_status(diag['ess_ok'])} |
| Label switching | {_ls_msg} | |

**Interpreting the diagnostics:**

- **Divergences** indicate regions of high curvature in the posterior
  geometry where the leapfrog integrator's discrete steps introduce
  unacceptable bias.  Zero divergences is the target.
- **R-hat** $< 1.01$ indicates that all chains have converged to the
  same stationary distribution.  Values above 1.01 in a regime-switching
  model may reflect **label switching** (a symmetry of the likelihood,
  not a sampling failure) rather than genuine non-convergence.
- **ESS (bulk)** $> 400$ ensures enough effective independent draws for
  reliable 94% credible intervals (BDA3, Ch. 11).
- **Label switching** is diagnosed by comparing per-chain posterior means
  of $\\boldsymbol{{\\mu}}$ and testing whether a permutation of regime
  labels improves R-hat.
        """
    )
    return


@app.cell
def _(az, idata, plt):
    fig_trace, _ = plt.subplots()
    plt.close(fig_trace)
    axes_trace = az.plot_trace(idata, var_names=["P", "mu"], figsize=(14, 8))
    fig_trace = axes_trace[0, 0].figure
    fig_trace.tight_layout()
    fig_trace
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading the trace plots:**  The left column shows the marginal
    posterior density for each parameter element; the right column shows the
    sampled values across iterations.  Look for:

    1. **Overlapping chains** in the trace (right) — indicates good mixing
       and convergence to the same stationary distribution.
    2. **Absence of stuck segments or spikes** — no divergences or
       trapping in local modes.
    3. **Stable marginal densities** across chains (left) — each chain
       explores the same posterior region.

    With $K = 2$ regimes and no hard ordering constraint, some chains may
    **label-switch**: one chain calls the high-mean regime "regime 0" while
    another calls it "regime 1".  This produces bimodal marginals in the
    trace plots but is cosmetic — it does not affect per-draw regime recovery
    (each draw is internally consistent).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Label switching: diagnosis and resolution

    Label switching is a **symmetry of the likelihood**: if we permute the
    regime indices and simultaneously permute $\mathbf{P}$,
    $\boldsymbol{\mu}$, and $\boldsymbol{\Sigma}$, the likelihood is
    unchanged.  The posterior is therefore invariant under label permutations,
    and different MCMC chains may explore different "copies" of the same mode.

    Our label-aware diagnostic aligns chains by comparing draw-averaged
    $\boldsymbol{\mu}$ values.  For each chain, it finds the permutation of
    regime indices that minimises the sum-of-squared differences to chain 0's
    mean, then recomputes R-hat on the relabeled posterior.  If relabeling
    improves R-hat below the 1.01 threshold, the original exceedance was due
    to label switching rather than genuine non-convergence.

    For the FFBS regime recovery that follows, label switching is handled
    separately: the regime-sequence samples are aligned across chains by
    maximising element-wise agreement.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regime recovery: forward-filter backward-sampler

    > **Question:** Given the fitted posterior, what is the posterior
    > probability that the market was in a bear regime at each point in time?

    The marginalised model integrates out the regime sequence during sampling,
    so we recover it post-hoc using a **forward-filter backward-sampler
    (FFBS)**.  For each posterior draw of $(\mathbf{P}, \boldsymbol{\mu}_k,
    \boldsymbol{\Sigma}_k)$:

    1. **Forward pass:** compute $\log \alpha_{t,k}$ using the draw's
       parameters and the observed returns.
    2. **Backward sampling:** sample $s_T$ from the normalised $\alpha_T$,
       then for $t = T{-}1, \ldots, 1$, sample $s_t \mid s_{t+1}$ from
       $\alpha_t \odot \mathbf{P}_{:,\, s_{t+1}}$.

    This produces one complete regime-sequence sample per posterior draw,
    preserving the full joint uncertainty over both parameters *and* regimes.
    The collection of regime-sequence samples yields a posterior probability
    $P(s_t = k \mid \mathbf{y}_{1:T})$ at every time step.
    """)
    return


@app.cell
def _(align_regime_samples, data, idata, run_ffbs):
    _raw = run_ffbs(idata, data["returns"], seed=42)
    regime_samples = align_regime_samples(_raw, K=2)
    return (regime_samples,)


@app.cell
def _(data, plot_regime_probabilities, plt, regime_samples):
    fig_regimes, ax_regimes = plt.subplots(figsize=(14, 3.5))
    plot_regime_probabilities(
        regime_samples, true_regimes=data["regimes"], ax=ax_regimes,
    )
    fig_regimes.tight_layout()
    fig_regimes
    return


@app.cell
def _(mo):
    mo.md(r"""
    The stacked bands show $P(s_t = k \mid \mathbf{y}_{1:T})$ at every
    time step, aggregated over all posterior draws.  The dashed black line
    is the true generating regime.  Posterior uncertainty concentrates at
    **transition points** — exactly where the model should be least certain.

    **Portfolio implication.**  A portfolio manager can use these posterior
    regime probabilities to dynamically tilt allocations: when
    $P(\text{Bear} \mid \text{data})$ exceeds $0.5$, the model signals
    elevated downside risk.  The width of the probability bands at transition
    points communicates signal uncertainty — exactly the information needed
    to size positions responsibly.  Unlike a point-estimate classifier,
    the Bayesian approach produces a *probability* of regime change, not a
    binary signal.
    """)
    return


@app.cell
def _(data, mo, np, regime_samples):
    flat = regime_samples.reshape(-1, regime_samples.shape[-1])
    modal = np.array(
        [np.bincount(flat[:, t], minlength=2).argmax() for t in range(flat.shape[1])]
    )
    acc_direct = np.mean(modal == data["regimes"])
    acc_flipped = np.mean((1 - modal) == data["regimes"])
    accuracy = max(acc_direct, acc_flipped)

    mo.md(
        f"""
### Regime recovery accuracy

Modal regime accuracy (best of direct / label-flipped):
**{accuracy:.1%}**

This metric is available only because we use synthetic data with known
ground-truth regimes.  With real market data, regime accuracy cannot be
directly computed — which is why the posterior probability bands above
are the primary output: they express the model's belief about the
current regime *and* its uncertainty, without requiring knowledge of
the true state.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Posterior parameter recovery

    > **Question:** Do the posterior distributions concentrate around the true
    > generating parameters, and what do the posterior credible intervals imply
    > for portfolio decisions?

    Since we generated the data from known parameters, we can verify whether
    the 94% highest density intervals (HDI) contain the true values.  The
    forest plots below show 94% HDI intervals for the regime means
    $\boldsymbol{\mu}_k$ and the transition matrix $\mathbf{P}$.
    Red diamonds mark the true generating values.
    """)
    return


@app.cell
def _(data, idata, mo, plot_posterior_summary):
    fig_mu = plot_posterior_summary(
        idata, var_name="mu", true_values=data["params"]["mus"],
        title="Posterior: regime means (μ)",
    )
    fig_P = plot_posterior_summary(
        idata, var_name="P", true_values=data["params"]["P"],
        title="Posterior: transition matrix (P)",
    )
    mo.vstack([fig_mu, fig_P])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interpreting the posterior in financial terms

    **Regime means.**  The posterior mean of $\mu_{k,i}$ (monthly) maps to
    an annualised expected return of $12 \times \mu_{k,i}$.  For example, if
    the posterior mean of the bull-regime equity mean is $+0.010$, this
    corresponds to $+12\%$ annualised.  The width of the 94% credible
    interval, similarly annualised, represents genuine parameter uncertainty
    — not sampling variability in a frequentist sense, but the posterior
    belief about the parameter's plausible range given $T$ observations.

    **Transition matrix.**  $P_{kk}$ determines the expected regime duration
    via $\mathbb{E}[\text{duration}_k] = 1 / (1 - P_{kk})$.  For the
    generating parameters: $P_{00} = 0.95$ implies a bull regime persisting
    for an expected 20 months ($\approx 1.7$ years), while $P_{11} = 0.90$
    implies bear episodes lasting 10 months on average.  The posterior
    credible interval around $P_{kk}$ translates directly into uncertainty
    about regime persistence — a key input for tactical allocation timing.

    **Decision-theoretic interpretation.**  A Bayesian portfolio optimiser
    would integrate over this posterior uncertainty when computing optimal
    weights, producing allocations that are robust to parameter estimation
    error.  The credible interval widths quantify how much the data *can*
    tell us — and, equally important, where substantial uncertainty remains.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Looking ahead: regime-conditional correlations

    In this notebook the generating process uses **identity correlation
    matrices** — assets are conditionally independent given the regime.
    The model's `LKJCholeskyCov` parameterisation *can* learn non-trivial
    correlations; we have simply not exercised that capability yet.

    A subsequent notebook in this series introduces a multi-asset universe
    (equities, bonds, gold) with **regime-dependent correlations** — the
    correlation-masking problem central to portfolio risk management.
    The demonstrated ability to recover distinct regime means and
    transition dynamics here is a necessary prerequisite for that harder
    problem.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model limitations

    The results above demonstrate that the Bayesian HMM correctly recovers
    the generating parameters and regime sequence from synthetic data.
    Before applying this framework to real portfolios, several limitations
    should be acknowledged:

    - **Gaussian emissions** may underestimate tail risk within regimes.
      Financial returns exhibit excess kurtosis even conditional on the
      regime; the current model can only produce portfolio-level fat tails
      through the regime mixture.

    - **Constant transition probabilities** assume that regime-switching
      rates do not depend on economic conditions.  A VIX spike or yield
      curve inversion does not influence $\mathbf{P}$ in this model.

    - **$K = 2$ is assumed, not tested.**  We have not compared the
      2-regime model against alternatives ($K = 1$ or $K = 3$) using
      a formal Bayesian model comparison criterion such as WAIC.

    - **Synthetic data.**  The model has been validated on data it was
      designed to fit.  Out-of-sample performance on real market data
      may differ due to non-stationarity, structural breaks, and
      violations of the conditional Gaussian assumption.

    - **No posterior predictive checks.**  We have not verified whether
      the fitted model reproduces key features of the observed data
      (e.g. marginal return distributions, autocorrelation structure).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Series roadmap

    This notebook is the first in a series on Bayesian regime-switching
    models for portfolio management:

    1. **The Problem and the Model** — *this notebook.*  Build a 2-regime
       HMM for equity returns, fit it with NUTS, recover regimes via FFBS.
    2. **Model Validation** — prior and posterior predictive checks.
       Verify that the priors produce plausible data and that the fitted
       model reproduces observed return features (Bayesian p-values,
       BDA3 Ch. 6).
    3. **Multi-Asset Extension** — extend to an equity-bond-gold universe
       with regime-dependent correlations.  Address the correlation-masking
       problem and its implications for hedge ratios.
    4. **Portfolio Analytics** — translate posterior uncertainty into
       VaR, CVaR, regime-conditional correlations, and hedge ratio
       implications.
    5. **Model Comparison** — $K = 1$ vs $K = 2$ vs $K = 3$ via WAIC.
       Introduce the 3-regime scenario and discuss honest uncertainty
       (Vehtari, Gelman & Gabry, 2017).
    6. **Real Data Application** — apply the pipeline to S&P 500,
       long-duration Treasuries, and gold (2000–2025).
    7. **Extensions** — Student-$t$ emissions, factor covariance
       structure, time-varying transition probabilities.

    ---

    **References**

    - Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of
      Nonstationary Time Series and the Business Cycle." *Econometrica*,
      57(2), 357–384.
    - Ang, A. and Bekaert, G. (2002). "International Asset Allocation With
      Regime Shifts." *Review of Financial Studies*, 15(4), 1137–1187.
    - Ang, A. and Bekaert, G. (2004). "How Regimes Affect Asset Allocation."
      *Financial Analysts Journal*, 60(2), 86–99.
    - Gelman, A. et al. (2013). *Bayesian Data Analysis*, 3rd ed. Chapman
      and Hall/CRC.
    - Vehtari, A., Gelman, A. and Gabry, J. (2017). "Practical Bayesian
      Model Evaluation Using Leave-One-Out Cross-Validation and WAIC."
      *Statistics and Computing*, 27(5), 1413–1432.
    """)
    return


if __name__ == "__main__":
    app.run()
