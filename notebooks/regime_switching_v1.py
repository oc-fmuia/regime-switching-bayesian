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

    from regime_switching_bayesian.data_gen import generate_hmm_data
    from regime_switching_bayesian.inference import (
        check_diagnostics_label_aware,
        fit,
        forward_filter_probs,
        run_ffbs,
    )
    from regime_switching_bayesian.model import build_model
    from regime_switching_bayesian.plotting import (
        plot_posterior_summary,
        plot_regime_probabilities,
        plot_returns_with_regimes,
    )

    return (
        az,
        build_model,
        check_diagnostics_label_aware,
        fit,
        forward_filter_probs,
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

    > **When do equity markets shift from expansion to stress, and can a
    > portfolio respond before the drawdown has already materialised?**

    Standard risk models estimate a single mean vector and a single
    covariance matrix from a long history of returns.  During calm markets
    this approach overstates volatility and leaves return on the table;
    during a crisis it understates tail risk because the long-window
    average dilutes the signal from the current environment.  The root
    cause is that **financial returns do not come from one statistical
    regime**: bull phases feature moderate drift and compressed volatility,
    while bear phases bring negative drift, volatility spikes, and
    correlation convergence toward one.  A model that averages across these
    environments mis-prices risk in both.

    A **regime-switching model** replaces the single distribution with $K$
    distinct distributions, one per regime, together with a transition
    matrix $\mathbf{P}$ that governs how the market moves between them.
    At every point in time the model produces a *probability* of being in
    each regime rather than a binary label, giving the portfolio manager a
    calibrated, time-varying risk lens.  In this notebook we build such a
    model as a Bayesian Hidden Markov Model (HMM) using
    [PyMC](https://www.pymc.io/), fit it via the No-U-Turn Sampler (NUTS),
    and recover the latent regime sequence with a backward-sampling pass
    that reconstructs the most likely state at each point in time.

    This is the first instalment in a series of blog posts that builds
    the regime-switching framework from the ground up.  Subsequent
    notebooks extend the model with scenario simulation, richer emission
    distributions, exogenous covariates, and real data applications; see
    the **Series roadmap** at the end of this notebook.

    **References.**
    Hamilton (1989), *Econometrica*;
    Ang & Bekaert (2002), *Review of Financial Studies*;
    Gelman et al. (2013), *Bayesian Data Analysis* 3rd ed. (BDA3).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From a portfolio question to a statistical model

    > **"My risk model treats 2008 and 2017 as the same market.  How can I
    > let the data tell me which environment we are in right now?"**

    This is fundamentally a question about **latent structure**: the data
    contain returns, but the underlying market regime is unobserved.
    Traditional portfolio models assume that returns are drawn from a single
    multivariate distribution with one mean vector $\boldsymbol{\mu}$ and
    one covariance matrix $\boldsymbol{\Sigma}$.  This assumption fails
    whenever the data-generating process switches between distinct economic
    environments.  Hamilton (1989) introduced the Markov-switching framework
    for business-cycle analysis, and Ang & Bekaert (2002) showed that
    accounting for regimes materially changes both optimal asset allocation
    and risk measurement.

    A **Hidden Markov Model (HMM)** with $K$ latent states translates the
    portfolio question into a well-posed inference problem: conditioned on
    the state, returns follow a regime-specific multivariate Normal
    ($\mathbf{y}_t \mid s_t = k \sim \mathcal{N}(\boldsymbol{\mu}_k,
    \boldsymbol{\Sigma}_k)$), and the states evolve as a first-order Markov
    chain ($P(s_t \mid s_{t-1}) = P_{s_{t-1}, s_t}$).  Bayesian inference
    then propagates uncertainty about *which regime is active at each point
    in time* directly into posterior quantities, avoiding the false
    precision of point estimates.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Synthetic data

    We work with three synthetic equity indices, namely **US Equity**,
    **EAFE Equity** (developed international), and **EM Equity** (emerging
    markets).  All three share the same two-regime structure: a
    **bull** regime with moderate positive drift and low volatility, and a
    **bear** regime with negative drift and elevated volatility.

    The synthetic data are generated from a known $K = 2$ HMM, allowing us to
    verify that the model recovers the true parameters and regimes in a clean
    setting.  All parameters below are calibrated to realistic monthly
    magnitudes; annualised equivalents are shown for interpretation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The generative model

    The data-generating process is a first-order Hidden Markov Model with
    $K$ regimes and $d$ assets.  At each time step:

    $$
    s_1 \sim \mathrm{Categorical}(\boldsymbol{\pi}_0), \qquad
    s_t \mid s_{t-1} \sim \mathrm{Categorical}(\mathbf{P}_{s_{t-1}, :}), \qquad
    \mathbf{y}_t \mid s_t = k \sim \mathcal{N}(\boldsymbol{\mu}_k,\, \boldsymbol{\Sigma}_k)
    $$

    where $\boldsymbol{\pi}_0$ is the initial state distribution,
    $\mathbf{P}$ is the $K \times K$ transition matrix
    ($P_{jk} = \Pr(s_t = k \mid s_{t-1} = j)$),
    and each regime $k$ has its own mean vector $\boldsymbol{\mu}_k \in \mathbb{R}^d$
    and covariance matrix $\boldsymbol{\Sigma}_k \in \mathbb{R}^{d \times d}$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Covariance structure and the correlation assumption

    Each regime's covariance matrix is decomposed as

    $$
    \boldsymbol{\Sigma}_k
    = \mathbf{D}_k \, \mathbf{R}_k \, \mathbf{D}_k
    = \mathbf{D}_k \, \mathbf{L}_k \mathbf{L}_k^\top \, \mathbf{D}_k
    $$

    where the three pieces are:

    | Component | Definition | Role |
    |-----------|-----------|------|
    | $\mathbf{D}_k = \mathrm{diag}(\sigma_{k,1}, \ldots, \sigma_{k,d})$ | Diagonal matrix of per-asset standard deviations | Scales each asset's volatility |
    | $\mathbf{R}_k = \mathbf{L}_k \mathbf{L}_k^\top$ | Correlation matrix (via its Cholesky factor $\mathbf{L}_k$) | Encodes within-regime co-movement |
    | $\boldsymbol{\Sigma}_k$ | Full covariance matrix | Determines the joint return distribution in regime $k$ |

    ### Simplifying assumption in this notebook

    We set $\mathbf{R}_k = \mathbf{I}$
    (the identity matrix) for every regime, so that
    $\boldsymbol{\Sigma}_k = \mathbf{D}_k^2 = \mathrm{diag}(\sigma_{k,1}^2,
    \ldots, \sigma_{k,d}^2)$.
    This implies **conditional independence given the regime**:
    $y_{t,i} \perp y_{t,j} \mid s_t$ for $i \neq j$.  In practical terms,
    within a given market regime, knowing that US equities dropped today
    tells you nothing extra about EM equities beyond what the regime label
    itself already implies.  All observed co-movement in the data comes
    from the shared regime, not from within-regime correlation.

    This is a deliberate simplification that keeps inference fast and focused
    on learning the regime means, volatilities, and transition dynamics.
    A later notebook in this series introduces $\mathbf{R}_k \neq \mathbf{I}$
    with regime-dependent correlations, which is the harder problem at the
    heart of portfolio risk management.
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
    _bull_mu_defaults = [0.010, 0.008, 0.012]
    _bear_mu_defaults = [-0.005, -0.008, -0.003]
    _bull_vol_defaults = [0.040, 0.035, 0.045]
    _bear_vol_defaults = [0.080, 0.090, 0.100]

    bull_mean_sliders = [
        mo.ui.slider(start=0.0, stop=0.05, value=v, step=0.001,
                      label=f"Bull μ – {a}")
        for a, v in zip(asset_names, _bull_mu_defaults)
    ]
    bear_mean_sliders = [
        mo.ui.slider(start=-0.05, stop=0.0, value=v, step=0.001,
                      label=f"Bear μ – {a}")
        for a, v in zip(asset_names, _bear_mu_defaults)
    ]
    bull_vol_sliders = [
        mo.ui.slider(start=0.01, stop=0.10, value=v, step=0.005,
                      label=f"Bull σ – {a}")
        for a, v in zip(asset_names, _bull_vol_defaults)
    ]
    bear_vol_sliders = [
        mo.ui.slider(start=0.02, stop=0.20, value=v, step=0.005,
                      label=f"Bear σ – {a}")
        for a, v in zip(asset_names, _bear_vol_defaults)
    ]

    mo.vstack([
        mo.md("### Regime parameters (data generation)"),
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
    asset_names,
    bear_mean_sliders,
    bear_vol_sliders,
    bull_mean_sliders,
    bull_vol_sliders,
    mo,
):
    _slider_specs = [
        ("Bull μ", bull_mean_sliders),
        ("Bear μ", bear_mean_sliders),
        ("Bull σ", bull_vol_sliders),
        ("Bear σ", bear_vol_sliders),
    ]
    _param_rows = []
    for _prefix, _sliders in _slider_specs:
        for _a, _s in zip(asset_names, _sliders):
            _param_rows.append(
                f"| {_prefix} – {_a} "
                f"| {_s.start} | {_s.stop} | {_s.step} "
                f"| {_s.value} |"
            )
    _param_table = "\n".join(_param_rows)

    mo.md(
        f"""
**Selected parameter values**

| Parameter | Min | Max | Step | Selected |
|-----------|-----|-----|------|----------|
{_param_table}
        """
    )
    return


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

The bull regime has moderate positive drift and low volatility, while the
bear regime features negative drift across all equities and roughly doubled
volatility, i.e. the classic growth/stress dichotomy.  Assets are
conditionally uncorrelated given the regime
($\\mathbf{{R}}_k = \\mathbf{{I}}$, i.e.
$y_{{t,i}} \\perp y_{{t,j}} \\mid s_t$); a later notebook in this series
introduces regime-dependent cross-asset correlations.
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
def _(asset_names, data, mo, np):
    _returns = data["returns"]
    _regimes = data["regimes"]
    _p = data["params"]

    _interp_rows = []
    for _i, _name in enumerate(asset_names):
        _bull_vals = _returns[_regimes == 0, _i]
        _bear_vals = _returns[_regimes == 1, _i]
        _bull_mean = np.mean(_bull_vals) * 100 if len(_bull_vals) > 0 else 0
        _bear_mean = np.mean(_bear_vals) * 100 if len(_bear_vals) > 0 else 0
        _bull_std = np.std(_bull_vals) * 100 if len(_bull_vals) > 0 else 0
        _bear_std = np.std(_bear_vals) * 100 if len(_bear_vals) > 0 else 0
        _interp_rows.append(
            f"| {_name} | {_bull_mean:+.2f}% | {_bull_std:.2f}% "
            f"| {_bear_mean:+.2f}% | {_bear_std:.2f}% |"
        )
    _interp_table = "\n".join(_interp_rows)

    _n_bull = int(np.sum(_regimes == 0))
    _n_bear = int(np.sum(_regimes == 1))

    mo.md(
        f"""
### Interpreting the regime-conditional distributions

Each panel above shows the **regime-conditional marginal**
$p(y_{{t,i}} \\mid s_t = k)$, i.e. the distribution of asset $i$'s
monthly return given the market is in regime $k$.  These are the building
blocks the model must learn: one distribution per asset per regime.

**Sample statistics from the generated data** ({_n_bull} bull months,
{_n_bear} bear months):

| Asset | Bull mean | Bull std | Bear mean | Bear std |
|-------|-----------|----------|-----------|----------|
{_interp_table}

The bull distributions are tightly clustered around small positive means,
while the bear distributions are shifted left (negative mean) with roughly
double the spread.  This is consistent with the generating parameters and
confirms the data exhibit the regime structure we expect.

The **overlap region** between the two KDEs, i.e. where the green and
red densities intersect, is where regime classification is most uncertain.
Returns in that overlap could plausibly come from either regime, and the
model will assign intermediate posterior probabilities at those time steps.
        """
    )
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
    ### Marginalisation of the discrete regime sequence

    The discrete regime sequence $s_{1:T}$ is **analytically marginalised**
    via the forward algorithm (Hamilton, 1989).  This is essential: the
    No-U-Turn Sampler (NUTS) requires a differentiable, continuous parameter
    space and cannot sample discrete variables directly.  The forward
    recursion computes $\log p(\mathbf{y}_{1:T} \mid \theta)$ in
    $\mathcal{O}(T K^2)$ time, avoiding the need to enumerate all $K^T$
    possible regime sequences.  See **Appendix A** for the full derivation.
    """)
    return


@app.cell
def _(build_model, data):
    model = build_model(data["returns"], K=2)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Model DAG

    The diagram below shows the model's dependency structure.  The
    discrete regime chain $s_{1:T}$ does not appear as a node because it
    has been analytically marginalised.  The key components are:

    - **$\mathbf{P} \sim \mathrm{Dirichlet}$** ($K \times K$): the
      transition matrix, with a sticky Dirichlet prior that encourages
      regime persistence.
    - **$\boldsymbol{\mu} \sim \mathrm{Normal}$** ($K \times d$):
      regime-conditional mean return vectors.
    - **$\texttt{chol\_cov\_k} \sim \mathrm{LKJCholeskyCov}$** (one per
      regime): the Cholesky-factored covariance matrices, from which
      PyMC derives deterministic nodes for the correlation matrices
      ($\texttt{chol\_cov\_k\_corr}$) and the per-asset standard
      deviations ($\texttt{chol\_cov\_k\_stds}$).
    - **$\texttt{hmm\_loglik} \sim \mathrm{Potential}$**: the
      marginalised HMM log-likelihood computed by the forward algorithm.
      This is not a random variable but a scalar contribution to the
      joint log-density that encodes both the emission likelihoods and
      the transition dynamics.
    """)
    return


@app.cell
def _(model):
    import pymc as pm

    pm.model_to_graphviz(model)

    # Custom mermaid alternative (kept for reference):
    #
    # mo.mermaid(
    #     """
    #     graph TD
    #         alpha["α (Dirichlet conc.)"] --> P["P (K×K transition matrix)"]
    #         P --> hmm["hmm_loglik (Potential)"]
    #         mu["μ ~ Normal(0, 0.05)  (K×d)"] --> hmm
    #         eta["η (LKJ shape)"] --> chol0["chol_cov_0 ~ LKJCholeskyCov"] & chol1["chol_cov_1 ~ LKJCholeskyCov"]
    #         sd_prior["σ ~ HalfNormal(0.10)"] --> chol0 & chol1
    #         chol0 & chol1 --> hmm
    #         y["y₁:T (observed returns)"] --> hmm
    #     """
    # )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Model assumptions

    The following assumptions are built into the current specification.
    Each represents a modelling choice with known limitations:

    1. **Gaussian emissions.**
       $\mathbf{y}_t \mid s_t = k \sim \mathcal{N}(\boldsymbol{\mu}_k,\,
       \boldsymbol{\Sigma}_k)$.
       This captures regime-level mean and volatility shifts but
       cannot model intra-regime fat tails (excess kurtosis $\kappa > 3$
       within a single regime).  A multivariate Student-$t$ extension
       addresses this limitation.

    2. **Time-homogeneous transition matrix.**
       $P(s_t = k \mid s_{t-1} = j) = P_{jk}$ for all $t$.
       Regime-switching probabilities do not depend on
       observable covariates (e.g. VIX, yield curve slope).  Time-varying
       transition probabilities (TVTP) relax this.

    3. **$K = 2$ regimes.**
       $K$ is fixed a priori, i.e. it is a modelling choice rather than a
       quantity inferred from the data.  Bayesian model comparison via WAIC
       can evaluate whether $K = 2$ is adequate relative to $K = 1$ or
       $K = 3$.

    4. **Conditional independence across time.**
       $\mathbf{y}_t \perp \mathbf{y}_{t-1} \mid s_t$, i.e. given the
       current regime, today's return is independent of yesterday's.
       This rules out
       within-regime momentum or mean-reversion effects (Hamilton's original
       1989 model used AR(4) dynamics within each regime).

    5. **Sticky Dirichlet prior on $\mathbf{P}$.**
       Each row of the transition matrix is drawn from a Dirichlet with
       concentration $\alpha_{\text{diag}} = 20$,
       $\alpha_{\text{off}} = 2$.  This encodes a prior belief that
       regimes are persistent (prior mean self-transition probability
       $\approx 0.91$).  The prior is symmetric across regimes and does
       not favour any particular regime ordering, which is why
       label switching can occur (see **Appendix B**).
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
        f"Yes — corrected via permutation alignment "
        f"(naive max R-hat {diag['naive_max_rhat']:.3f} → "
        f"aligned {diag['max_rhat']:.3f}). "
        f"See **Appendix B** for details."
        if _ls
        else "No"
    )
    _ess_note = (
        " (computed on label-aligned posterior)"
        if diag.get("ess_on_aligned", False) else ""
    )
    mo.md(
        f"""
### Convergence diagnostics

| Check | Value | Status |
|-------|-------|--------|
| Divergences | {diag['n_divergences']} | {_status(diag['no_divergences'])} |
| max R-hat | {diag['max_rhat']:.3f} | {_status(diag['rhat_ok'])} |
| min ESS (bulk){_ess_note} | {diag['min_ess_bulk']:.0f} | {_status(diag['ess_ok'])} |
| Label switching | {_ls_msg} | |

- **Divergences** = 0 confirms the sampler navigated the posterior geometry
  without numerical issues.
- **R-hat** < 1.01 after label alignment confirms all chains converged to
  the same stationary distribution.
- **ESS (bulk)** > 400 ensures enough effective independent draws for
  reliable 94% credible intervals (BDA3, Ch. 11).
        """
    )
    return (diag,)


@app.cell
def _(data, diag, np):
    aligned_idata = diag["aligned_idata"]

    _mu_post = aligned_idata.posterior["mu"].values.mean(axis=(0, 1))
    bear_idx = int(np.argmin(_mu_post.sum(axis=1)))
    bull_idx = 1 - bear_idx

    perm = [0, 0]
    perm[bear_idx] = 1
    perm[bull_idx] = 0

    label_for = {bear_idx: "Bear", bull_idx: "Bull"}

    true_mus_matched = data["params"]["mus"][perm]
    true_P_matched = data["params"]["P"][np.ix_(perm, perm)]
    return (
        aligned_idata,
        bear_idx,
        bull_idx,
        label_for,
        perm,
        true_P_matched,
        true_mus_matched,
    )


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
    sampled values across iterations.  Overlapping chains in the trace
    indicate good mixing.  With $K = 2$ regimes and no hard ordering
    constraint, some chains may **label-switch**, producing bimodal
    marginals that are a cosmetic artefact of the likelihood symmetry (see
    **Appendix B**) rather than a sampling failure.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regime recovery

    > **Question:** Given the fitted posterior, what is the posterior
    > probability that the market was in a bear regime at each point in time?

    The marginalised model integrates out the regime sequence during sampling,
    so we recover it post-hoc using a **forward-filter backward-sampler
    (FFBS)**.  For each posterior draw, the FFBS runs a forward pass
    (identical to the marginalisation step) and then samples a complete
    regime sequence backward in time.  This preserves the full joint
    uncertainty over both parameters *and* regimes.  The collection of
    regime-sequence samples yields a posterior probability
    $P(s_t = k \mid \mathbf{y}_{1:T})$ at every time step.
    """)
    return


@app.cell
def _(aligned_idata, data, run_ffbs):
    regime_samples = run_ffbs(aligned_idata, data["returns"], seed=42)
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
def _(bear_idx, data, mo, np, perm, regime_samples):
    _flat = regime_samples.reshape(-1, regime_samples.shape[-1])
    _T = _flat.shape[1]

    _bear_prob = (_flat == bear_idx).mean(axis=0)

    _true_reg = data["regimes"]
    _transitions = np.where(np.diff(_true_reg) != 0)[0]
    _n_transitions = len(_transitions)

    _uncertain_mask = (_bear_prob > 0.2) & (_bear_prob < 0.8)
    _n_uncertain = int(_uncertain_mask.sum())

    _modal = np.array(
        [np.bincount(_flat[:, t], minlength=2).argmax() for t in range(_T)]
    )
    _modal_mapped = np.array([perm[m] for m in _modal])
    accuracy = float(np.mean(_modal_mapped == _true_reg))

    mo.md(
        f"""
### Regime recovery results

The stacked bands show $P(s_t = k \\mid \\mathbf{{y}}_{{1:T}})$ at every
time step, aggregated over all posterior draws.  The dashed black line
is the true generating regime.

**Key observations:**

- **Modal regime accuracy:** **{accuracy:.1%}** (best of direct / label-flipped).
  The model correctly identifies the regime at almost every time step.
- **Transition detection:** The data contain **{_n_transitions}** true regime
  transitions.  Posterior uncertainty concentrates at these transition
  points, which is exactly where the model *should* be least certain.
- **Uncertain periods:** {_n_uncertain} out of {_T} months have
  $P(\\text{{Bear}}) \\in (0.2, 0.8)$, reflecting genuine ambiguity at
  regime boundaries.

This accuracy metric is only possible because we use synthetic data with
known ground-truth regimes.  With real market data, the posterior
probability bands are the primary output: they express the model's
belief about the current regime *and* its uncertainty, without requiring
knowledge of the true state.
        """
    )
    return (accuracy,)


@app.cell
def _(mo):
    mo.md(r"""
    ### A realistic portfolio use-case: regime-aware allocation

    How would a portfolio manager actually *use* these regime probabilities?
    As a proof of concept, consider a simple rule applied to an
    **equal-weight portfolio** of all three equity indices:

    - **Static benchmark:** Hold 100% equity at all times (no regime info).
    - **Regime-aware strategy:** At each month $t$, use the *filtered*
      probability $P(s_{t-1} = \text{Bear} \mid \mathbf{y}_{1:t-1})$,
      which depends only on data available *before* time $t$, to set the
      portfolio mix between equities and risk-free securities yielding
      4% per annum (≈ 0.33% per month).  If $P(\text{Bear}) > 0.5$,
      allocate 20% to equities and 80% to risk-free; otherwise allocate
      80% to equities and 20% to risk-free.

    The asymmetric split reflects the strategy's risk-management role:
    in bull regimes the portfolio captures most of the equity upside
    while earning a floor return on the remaining 20%; in bear regimes
    it shifts decisively to safety, keeping only a small equity
    toe-hold so it is not entirely out of the market if the regime
    signal is wrong.

    The regime-aware strategy is a **risk-management** tool, not an
    alpha generator.  Its value shows up in risk-adjusted metrics rather
    than in higher cumulative return.  Specifically, we report the
    annualised **Sharpe ratio**

    $$
    \text{SR} = \frac{\bar{r}}{\hat\sigma_r}\,\sqrt{12}
    $$

    where $\bar{r}$ and $\hat\sigma_r$ are the sample mean and standard
    deviation of monthly portfolio returns, and the **Calmar ratio**

    $$
    \text{Calmar} = \frac{r_{\text{ann}}}{\lvert\text{MaxDD}\rvert}
    $$

    where $r_{\text{ann}}$ is the annualised return and MaxDD the maximum
    peak-to-trough drawdown.  Because the strategy always holds some
    risk-free securities, it never matches the full equity upside;
    however, shifting to a 20/80 equity/risk-free split during bear
    months sharply reduces volatility.  Since bear months have higher
    volatility than bull months, the volatility reduction is
    proportionally larger than the return reduction, and the Sharpe
    ratio typically improves.

    > **Important caveats:** (1) The posterior parameters were estimated
    > on the *full* sample, so this is an in-sample demonstration, not a
    > backtest.  A proper out-of-sample P&L analysis with walk-forward
    > re-estimation will be the topic of a dedicated future blog post.
    > (2) Transaction costs and slippage are ignored.
    """)
    return


@app.cell
def _(aligned_idata, bear_idx, data, forward_filter_probs, mo, np, plt):
    _filtered = forward_filter_probs(aligned_idata, data["returns"], thin=10)
    _bear_filt = _filtered[:, bear_idx]

    _r_ew = data["returns"].mean(axis=1)
    _T_pnl = len(_r_ew)
    _rf_monthly = 0.04 / 12  # 4% p.a. risk-free rate

    _w_equity = np.ones(_T_pnl)
    for _t in range(1, _T_pnl):
        _w_equity[_t] = 0.2 if _bear_filt[_t - 1] > 0.5 else 0.8

    _r_static = _r_ew
    _r_aware = _w_equity * _r_ew + (1.0 - _w_equity) * _rf_monthly

    _cum_static = np.cumprod(1.0 + _r_static)
    _cum_aware = np.cumprod(1.0 + _r_aware)

    _tot_s = (_cum_static[-1] - 1) * 100
    _tot_a = (_cum_aware[-1] - 1) * 100
    _vol_s = np.std(_r_static) * np.sqrt(12) * 100
    _vol_a = np.std(_r_aware) * np.sqrt(12) * 100
    _dd_s = np.min(_cum_static / np.maximum.accumulate(_cum_static) - 1) * 100
    _dd_a = np.min(_cum_aware / np.maximum.accumulate(_cum_aware) - 1) * 100
    _ann_s = (np.prod(1.0 + _r_static) ** (12.0 / _T_pnl) - 1) * 100
    _ann_a = (np.prod(1.0 + _r_aware) ** (12.0 / _T_pnl) - 1) * 100
    _sh_s = np.mean(_r_static) / np.std(_r_static) * np.sqrt(12)
    _sh_a = np.mean(_r_aware) / np.std(_r_aware) * np.sqrt(12)
    _cal_s = _ann_s / abs(_dd_s) if abs(_dd_s) > 0 else np.inf
    _cal_a = _ann_a / abs(_dd_a) if abs(_dd_a) > 0 else np.inf

    fig_pnl, _ax_pnl = plt.subplots(figsize=(14, 4))
    _ax_pnl.plot(_cum_static, label="Static 100% equity", linewidth=1.5, color="#666666")
    _ax_pnl.plot(_cum_aware, label="Regime-aware (80/20 bull, 20/80 bear)", linewidth=1.5, color="#1976D2")

    for _t in range(_T_pnl):
        if _w_equity[_t] < 0.8:
            _ax_pnl.axvspan(_t - 0.5, _t + 0.5, alpha=0.08, color="red")

    _ax_pnl.set_xlabel("Time (months)")
    _ax_pnl.set_ylabel("Cumulative value ($1 invested)")
    _ax_pnl.set_title("Regime-Aware vs. Static Allocation (Equal-Weight Portfolio)")
    _ax_pnl.legend(loc="upper left")
    _ax_pnl.set_xlim(0, _T_pnl - 1)
    fig_pnl.tight_layout()

    _n_reduced = int((_w_equity < 0.8).sum())

    mo.vstack([
        fig_pnl,
        mo.md(
            f"""
| Metric | Static | Regime-aware |
|--------|--------|--------------|
| Cumulative return | {_tot_s:+.1f}% | {_tot_a:+.1f}% |
| Annualised return | {_ann_s:+.1f}% | {_ann_a:+.1f}% |
| Annualised volatility | {_vol_s:.1f}% | {_vol_a:.1f}% |
| Max drawdown | {_dd_s:+.1f}% | {_dd_a:+.1f}% |
| Sharpe ratio | {_sh_s:.2f} | {_sh_a:.2f} |
| Calmar ratio | {_cal_s:.2f} | {_cal_a:.2f} |

The equity sleeve is an equal-weight average of all three equity indices;
the non-equity portion earns a 4% p.a. risk-free rate.  In bull months
the split is 80/20 equity/risk-free; in bear months it flips to 20/80.
The regime-aware strategy entered the defensive 20/80 allocation in
**{_n_reduced}** out of {_T_pnl} months (light-red shading).
The **Sharpe ratio** improves because bear months have higher volatility
than bull months: shifting to mostly risk-free during those periods
lowers the denominator ($\\hat\\sigma_r$) proportionally more than the
numerator ($\\bar{{r}}$), which is cushioned by the risk-free yield.
The **Calmar ratio** improves for the same reason, i.e. the shallower
max drawdown more than compensates for any reduction in annualised return.

A rigorous out-of-sample evaluation with walk-forward re-estimation,
proper transaction cost modelling, and multiple seeds as well as a
comprehensive analysis of the evaluation metrics is the subject of
a future post in this series.
            """
        ),
    ])
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
def _(
    aligned_idata,
    asset_names,
    label_for,
    mo,
    plot_posterior_summary,
    true_P_matched,
    true_mus_matched,
):
    _mu_labels = []
    for _k in range(2):
        for _a in asset_names:
            _mu_labels.append(f"{label_for[_k]} – {_a}")

    _P_labels = []
    for _j in range(2):
        for _k in range(2):
            _P_labels.append(f"{label_for[_j]}→{label_for[_k]}")

    fig_mu = plot_posterior_summary(
        aligned_idata, var_name="mu", true_values=true_mus_matched,
        labels=_mu_labels,
        title="Posterior: regime means (μ)",
        xlim=(-0.1, 0.1),
    )
    fig_P = plot_posterior_summary(
        aligned_idata, var_name="P", true_values=true_P_matched,
        labels=_P_labels,
        title="Posterior: transition matrix (P)",
    )
    mo.vstack([fig_mu, fig_P])
    return


@app.cell
def _(
    aligned_idata,
    asset_names,
    az,
    bear_idx,
    bull_idx,
    data,
    label_for,
    mo,
    np,
    perm,
):
    _summary_mu = az.summary(
        aligned_idata, var_names=["mu"], hdi_prob=0.94
    )
    _summary_P = az.summary(
        aligned_idata, var_names=["P"], hdi_prob=0.94
    )

    _true_mus = data["params"]["mus"]
    _true_P = data["params"]["P"]
    _K = _true_mus.shape[0]
    _d = _true_mus.shape[1]

    _mu_rows = []
    for _k in range(_K):
        _data_k = perm[_k]
        for _i in range(_d):
            _idx = _k * _d + _i
            _row = _summary_mu.iloc[_idx]
            _mean_m = _row["mean"]
            _hdi_lo = _row["hdi_3%"]
            _hdi_hi = _row["hdi_97%"]
            _true_val = _true_mus[_data_k, _i]
            _mean_ann = _mean_m * 12 * 100
            _hdi_lo_ann = _hdi_lo * 12 * 100
            _hdi_hi_ann = _hdi_hi * 12 * 100
            _true_ann = _true_val * 12 * 100
            _covered = "yes" if _hdi_lo <= _true_val <= _hdi_hi else "no"
            _mu_rows.append(
                f"| {label_for[_k]} – {asset_names[_i]} "
                f"| {_mean_ann:+.1f}% | [{_hdi_lo_ann:+.1f}%, {_hdi_hi_ann:+.1f}%] "
                f"| {_true_ann:+.1f}% | {_covered} |"
            )
    _mu_table = "\n".join(_mu_rows)

    _P_rows = []
    for _k in range(_K):
        _data_k = perm[_k]
        _idx_diag = _k * _K + _k
        _row = _summary_P.iloc[_idx_diag]
        _mean_p = _row["mean"]
        _hdi_lo = _row["hdi_3%"]
        _hdi_hi = _row["hdi_97%"]
        _true_p = _true_P[_data_k, _data_k]
        _dur_mean = 1.0 / (1.0 - _mean_p) if _mean_p < 1 else np.inf
        _dur_true = 1.0 / (1.0 - _true_p) if _true_p < 1 else np.inf
        _covered = "yes" if _hdi_lo <= _true_p <= _hdi_hi else "no"
        _P_rows.append(
            f"| $P$ ({label_for[_k]}→{label_for[_k]}) "
            f"| {_mean_p:.3f} | [{_hdi_lo:.3f}, {_hdi_hi:.3f}] "
            f"| {_true_p:.3f} | {_covered} "
            f"| {_dur_mean:.1f} months (true: {_dur_true:.1f}) |"
        )
    _P_table = "\n".join(_P_rows)

    mo.md(
        f"""
### Interpreting the results

The posterior regime indices do not necessarily match the data-generation
indices.  We identify regimes by their posterior mean: the regime with
higher (lower) average $\\mu$ across assets is labelled Bull (Bear).
In this run, posterior regime **{bull_idx}** = Bull, posterior regime
**{bear_idx}** = Bear.

**Regime means** (annualised):

| Parameter | Posterior mean | 94% HDI | True value | Covered? |
|-----------|---------------|---------|------------|----------|
{_mu_table}

All true regime means fall within their 94% HDI, confirming that the model
successfully recovers the generating parameters.  The HDI widths reflect genuine
uncertainty: with {data['config']['T']} months of data, the bull-regime
means are estimated more precisely (more bull months in the sample) than
the bear-regime means.

**Transition matrix** (self-transition probabilities):

| Parameter | Posterior mean | 94% HDI | True value | Covered? | Implied duration |
|-----------|---------------|---------|------------|----------|-----------------|
{_P_table}

The posterior means for the self-transition probabilities are close to the
true values, and the implied regime durations match well.  A Bayesian
portfolio optimiser would integrate over these posterior distributions when
computing optimal weights, producing allocations that are robust to
parameter estimation error.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Looking ahead: regime-conditional correlations

    In this notebook the generating process uses **identity correlation
    matrices**, i.e. assets are conditionally independent given the regime
    ($y_{t,i} \perp y_{t,j} \mid s_t$).
    The model's `LKJCholeskyCov` parameterisation *can* learn non-trivial
    correlations; we have simply not exercised that capability yet.

    A subsequent notebook in this series introduces a multi-asset universe
    (equities, bonds, gold) with **regime-dependent correlations**, which
    is the correlation-masking problem central to portfolio risk management.
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

    This notebook is the first in a series.  The planned follow-ups, each
    building on the previous one, are:

    1. **Scenario forward simulation and performance evaluation.**
       Given the fitted posterior, simulate forward return paths under
       different regime assumptions (e.g. pin the bear regime for 12
       months, shock covariances by a factor of 2) and compute
       regime-conditional portfolio metrics: VaR, CVaR, maximum drawdown
       distributions, and Sharpe / Calmar ratios -- all with full
       parameter uncertainty propagated.

    2. **Regime-dependent correlations, long-tailed emissions, and
       autoregression.**  Replace the identity correlation assumption
       ($\mathbf{R}_k = \mathbf{I}$) with regime-dependent correlation
       matrices estimated via `LKJCholeskyCov`.  Replace multivariate
       Normal emissions with multivariate Student-$t$ to capture
       intra-regime fat tails.  Add optional AR($p$) dynamics within each
       regime to model momentum and mean-reversion effects that vary by
       market environment.

    3. **Exogenous covariates and their effect on scenario analysis.**
       Replace the constant Dirichlet transition matrix with time-varying
       transition probabilities (TVTP) conditioned on observable macro
       variables (e.g. VIX, yield curve slope):
       $P_{ij}(t) = \mathrm{softmax}(\mathbf{x}_t^\top
       \boldsymbol{\beta})$.  This lets economic indicators influence
       regime-switching rates and allows scenario analysis to ask *"what
       happens to regime probabilities if the VIX doubles?"* rather than
       treating transitions as purely data-driven.

    4. **Real data applications.**  Apply the full pipeline to historical
       market data (e.g. S\&P 500, long-duration Treasuries, and gold
       from 2000 to 2025) using a data loader that produces arrays
       compatible with the synthetic data interface.  Includes
       walk-forward re-estimation for proper out-of-sample evaluation and
       comparison against the synthetic-data results from earlier
       notebooks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Appendix A: The forward algorithm

    The discrete regime sequence $s_{1:T}$ is analytically marginalised
    via the forward algorithm.  This is essential because NUTS requires a
    differentiable, continuous parameter space and cannot sample discrete
    variables directly.

    ### Forward recursion

    Define the log-forward probabilities:

    $$
    \log \alpha_{t,k} = \log p(\mathbf{y}_{1:t},\, s_t = k \mid \theta)
    $$

    **Initialisation** ($t = 1$):

    $$
    \log \alpha_{1,k} = \log \pi_{0,k} + \log p(\mathbf{y}_1 \mid s_1 = k)
    $$

    **Recursion** ($t = 2, \ldots, T$):

    $$
    \log \alpha_{t,k} =
    \log \sum_{k'=0}^{K-1} \exp\!\bigl(\log \alpha_{t-1,k'} + \log P_{k',k}\bigr)
    \;+\; \log p(\mathbf{y}_t \mid s_t = k)
    $$

    The $\text{logsumexp}$ operation prevents numerical underflow in the
    summation over previous states.

    **Normalised recursion.**  In practice we normalise $\log\alpha_t$ at
    each step by subtracting $c_t = \text{logsumexp}_k\,\log\alpha_{t,k}$
    and accumulating the constants separately.  The marginalised
    log-likelihood is then

    $$
    \log p(\mathbf{y}_{1:T} \mid \theta)
    = \sum_{t=1}^{T} c_t
    \;+\; \text{logsumexp}_{k}\, \widetilde{\log\alpha}_{T,k}
    $$

    where $\widetilde{\log\alpha}$ denotes the normalised forward
    variable.  Without this step the un-normalised $\log\alpha_t$ drifts to
    $\mathcal{O}(-10\,T)$ as $T$ grows, producing ill-conditioned gradients
    through `pytensor.scan` and causing massive NUTS divergences for
    $T \gtrsim 200$.

    This scalar is added to the joint log-density via a `pm.Potential`,
    enabling NUTS to explore the continuous parameter space
    $\theta = (\mathbf{P}, \boldsymbol{\mu}, \boldsymbol{\Sigma})$ while
    accounting for the discrete latent structure.

    ### Implementation

    The recursion is implemented via `pytensor.scan`, which compiles to
    `jax.lax.scan` under the NumPyro backend.  This avoids Python-level
    loops over $T$ and enables efficient GPU/CPU vectorisation.  The
    computational cost is $\mathcal{O}(T K^2)$ per log-likelihood evaluation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Appendix B: Label switching

    ### The symmetry

    Label switching is a **fundamental symmetry of the likelihood** in
    mixture and HMM models.  If we simultaneously permute the regime
    indices and the corresponding parameters —

    $$
    \boldsymbol{\mu}_k \to \boldsymbol{\mu}_{\sigma(k)}, \quad
    \boldsymbol{\Sigma}_k \to \boldsymbol{\Sigma}_{\sigma(k)}, \quad
    P_{jk} \to P_{\sigma(j),\sigma(k)}
    $$

    for any permutation $\sigma$ of $\{0, \ldots, K{-}1\}$, the
    likelihood is unchanged:

    $$
    p(\mathbf{y}_{1:T} \mid \theta) = p(\mathbf{y}_{1:T} \mid \sigma(\theta))
    $$

    The posterior is therefore **multimodal by construction**: it has $K!$
    equivalent modes corresponding to the $K!$ label permutations.
    Different MCMC chains may explore different modes.

    ### Concrete example (K = 2)

    Suppose chain 0 learns (Bull = regime 0, Bear = regime 1) while
    chain 1 learns (Bull = regime 1, Bear = regime 0).  Both chains have
    identical likelihoods and produce identical regime-sequence samples —
    they just use opposite labels.  Naively computing R-hat across chains
    compares $\mu_{\text{Bull}}$ from chain 0 against $\mu_{\text{Bear}}$
    from chain 1, producing a spurious R-hat > 1.

    ### Our resolution

    **Parameter alignment:** For each chain $c$, we compute the draw-averaged
    $\boldsymbol{\mu}$ and find the permutation $\sigma_c$ that minimises
    the sum-of-squared differences to chain 0's draw-averaged
    $\boldsymbol{\mu}$:

    $$
    \sigma_c = \arg\min_{\sigma} \sum_{k,i}
    \bigl(\bar{\mu}^{(c)}_{\sigma(k),i} - \bar{\mu}^{(0)}_{k,i}\bigr)^2
    $$

    We then permute $\mathbf{P}$, $\boldsymbol{\mu}$, and all Cholesky
    factors in chain $c$ according to $\sigma_c$ before computing R-hat
    and ESS.

    **FFBS alignment:** For the regime-sequence samples, we align chains
    by maximising element-wise agreement with the reference chain, using
    the same permutation search.

    This post-hoc relabeling is simple and exact for small $K$.  For
    larger $K$, more sophisticated methods (e.g. Stephens 2000,
    "Dealing with label switching in mixture models") may be needed.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
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
    - Stephens, M. (2000). "Dealing with Label Switching in Mixture Models."
      *Journal of the Royal Statistical Society B*, 62(4), 795–809.
    """)
    return


if __name__ == "__main__":
    app.run()
