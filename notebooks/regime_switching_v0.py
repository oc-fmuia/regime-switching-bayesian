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
        check_diagnostics,
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
    # Bayesian Regime-Switching for Equity Returns (v0)

    Equity markets alternate between calm growth and volatile drawdowns.
    A **regime-switching model** discovers these hidden states from return
    data alone, giving portfolio managers a probabilistic view of the
    current market environment and its likely persistence.

    This notebook walks through a minimal but complete implementation:
    synthetic data generation, Bayesian model specification in PyMC,
    posterior sampling with NUTS, and regime recovery via the
    forward-filter backward-sampler (FFBS).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why regime-switching?

    Traditional portfolio models assume a **single** return distribution --
    one mean, one covariance.  In reality, financial returns cluster into
    distinct regimes: a *Bull/Growth* regime with moderate positive drift
    and low volatility, and a *Bear/Stress* regime with negative drift and
    elevated volatility.  Ignoring this structure underestimates tail risk
    and over-smooths expected returns.

    A **Hidden Markov Model (HMM)** captures this by positing $K$ latent
    states that switch according to a Markov chain.  Conditioned on the
    state, returns are drawn from a regime-specific multivariate Normal.
    Bayesian inference lets us estimate all parameters jointly while
    propagating uncertainty about *which regime is active at each point
    in time*.

    In this notebook we build a 2-regime HMM for 3 synthetic equity
    indices, fit it with **PyMC + NUTS**, and recover the most likely
    regime sequence.  Because the data is synthetic, we can verify that
    the model recovers the true parameters and regimes.
    """)
    return


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
def _(mo):
    _asset_labels = ["US Equity", "EAFE Equity", "EM Equity"]

    bull_mean_sliders = [
        mo.ui.slider(start=0.0, stop=0.05, value=v, step=0.001,
                      label=f"Bull μ – {a}")
        for a, v in zip(_asset_labels, [0.010, 0.008, 0.012])
    ]
    bear_mean_sliders = [
        mo.ui.slider(start=-0.05, stop=0.0, value=v, step=0.001,
                      label=f"Bear μ – {a}")
        for a, v in zip(_asset_labels, [-0.005, -0.008, -0.003])
    ]
    bull_vol_sliders = [
        mo.ui.slider(start=0.01, stop=0.10, value=v, step=0.005,
                      label=f"Bull σ – {a}")
        for a, v in zip(_asset_labels, [0.040, 0.035, 0.045])
    ]
    bear_vol_sliders = [
        mo.ui.slider(start=0.02, stop=0.20, value=v, step=0.005,
                      label=f"Bear σ – {a}")
        for a, v in zip(_asset_labels, [0.080, 0.090, 0.100])
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
def _(data, mo):
    params = data["params"]
    cfg = data["config"]
    mo.md(
        f"""
        ### Default generating parameters

        | Parameter | Regime 0 (Bear) | Regime 1 (Bull) |
        |-----------|-----------------|-----------------|
        | Monthly mean | `{params['mus'][0]}` | `{params['mus'][1]}` |
        | Monthly vol  | `{params['sigmas'][0]}` | `{params['sigmas'][1]}` |
        | Transition row | `{params['P'][0]}` | `{params['P'][1]}` |

        **T** = {cfg['T']} months, **d** = {cfg['d']} assets, **seed** = {cfg['seed']}.
        """
    )
    return


@app.cell
def _(data, plot_returns_with_regimes):
    asset_names = ["US Equity", "EAFE Equity", "EM Equity"]
    fig_returns = plot_returns_with_regimes(
        data["returns"], data["regimes"], asset_names=asset_names,
    )
    fig_returns
    return


@app.cell
def _(mo):
    mo.md(r"""
    The green-shaded periods are **Bull** (regime 1) and the red-shaded
    periods are **Bear** (regime 0).  Notice the visibly higher volatility
    and negative drift during Bear episodes.  This is the structure our
    model will try to recover.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bayesian Model Specification

    We define a Hidden Markov Model with:

    - **Transition matrix** $\mathbf{P}$ with a *sticky Dirichlet* prior
      that encourages regime persistence.
    - **Regime means** $\boldsymbol{\mu}_k \sim \mathcal{N}(0, 0.05)$ --
      weakly informative, centered at zero.
    - **Regime covariances** parameterised via `LKJCholeskyCov` with
      $\eta = 2$ (mild shrinkage toward independence) and
      `HalfNormal(0.10)` priors on the standard deviations.
    - **Observations** $\mathbf{y}_t \mid s_t = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

    The discrete chain $s_{1:T}$ is **marginalized out** via a manual
    forward algorithm (`pytensor.scan`) so that NUTS only sees continuous
    parameters.  Under the NumPyro/JAX backend (the default) this
    compiles to `jax.lax.scan`, giving a significant speed-up over the
    PyMC C backend.
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
        mo.md(
            "### Model DAG\n\n"
            "The discrete chain has been analytically marginalised; "
            "the `hmm_loglik` Potential encodes the forward algorithm."
        )
    except Exception:
        graph = None
        mo.md("*(graphviz not available -- skipping DAG render)*")
    return (graph,)


@app.cell
def _(graph):
    graph
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Posterior Sampling with NUTS

    We sample from the marginalized model.  Because the discrete chain
    has been analytically integrated out, NUTS operates on a purely
    continuous parameter space: $\mathbf{P}$, $\boldsymbol{\mu}$, and
    the Cholesky factors of each regime's covariance.

    > **Note:** Sampling takes a few minutes on the first run due to
    > PyTensor compilation.  Subsequent runs with the same model graph
    > are faster.
    """)
    return


@app.cell
def _(data, fit, model, sampler_dropdown):
    _mus_init = data["params"]["mus"]  # (K, d) from the sliders
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
        f"Yes -- naive max R-hat was {diag['naive_max_rhat']:.3f}, "
        f"improved to {diag['max_rhat']:.3f} after relabeling "
        f"(best permutations: {diag['best_permutations']})"
        if _ls
        else "No"
    )
    mo.md(
        f"""
        ### Diagnostics

        | Check | Value | Status |
        |-------|-------|--------|
        | Divergences | {diag['n_divergences']} | {_status(diag['no_divergences'])} |
        | max R-hat | {diag['max_rhat']:.3f} | {_status(diag['rhat_ok'])} |
        | min ESS (bulk) | {diag['min_ess_bulk']:.0f} | {_status(diag['ess_ok'])} |
        | Label switching | {_ls_msg} | |
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
    Look for: (1) chains that overlap and explore the same region -- good
    mixing; (2) absence of stuck segments or spikes -- no divergences;
    (3) stable marginal densities across chains.

    With 2 regimes and no ordering constraint, you may see **label
    switching** where chains flip which regime is "0" vs "1".  This is
    cosmetic and does not affect per-draw regime recovery below.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Regime Recovery: Forward-Filter Backward-Sampler

    `pymc_extras.recover_marginals` does **not** support
    `DiscreteMarkovChain` (it raises `NotImplementedError`).  We
    therefore implement a custom **FFBS** in NumPy that runs
    post-hoc on each posterior draw:

    1. **Forward pass**: compute $\log \alpha_{t,k}$ via the standard
       HMM forward recursion using the draw's $\mathbf{P}$,
       $\boldsymbol{\mu}_k$, and $\boldsymbol{\Sigma}_k$.
    2. **Backward sampling**: sample $s_T$ from the normalised
       $\alpha_T$, then for $t = T{-}1, \ldots, 1$, sample
       $s_t \mid s_{t+1}$ using $\alpha_t$ and the transition column
       $\mathbf{P}_{:, s_{t+1}}$.

    This gives one regime-sequence sample per posterior draw, preserving
    full posterior uncertainty about both parameters *and* regimes.
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
    The stacked bands show the posterior probability of each regime at
    every time step, aggregated over all NUTS draws.  The dashed black
    line is the true generating regime.  Uncertainty concentrates at
    **transition points** -- exactly where you would expect the model to
    be least certain.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Posterior Parameter Recovery

    Since we generated the data, we can check whether the posterior
    concentrates around the true values.  The forest plot below shows
    94% HDI intervals for the transition matrix $\mathbf{P}$ and the
    regime means $\boldsymbol{\mu}$.
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

        A portfolio manager can use the posterior regime probabilities to
        tilt allocations dynamically -- e.g. reducing equity exposure when
        $P(\\text{{Bear}}) > 0.5$ -- while the width of the probability
        bands communicates how confident the signal is.
        """
    )
    return


@app.cell
def _(data, plt, sns):
    _returns = data["returns"]
    _regimes = data["regimes"]
    _asset_names = ["US Equity", "EAFE Equity", "EM Equity"]
    _d = _returns.shape[1]

    fig_dist, axes_dist = plt.subplots(1, _d, figsize=(14, 4), sharey=True)
    for i in range(_d):
        ax = axes_dist[i]
        for k, (label, color) in enumerate(
            zip(["Bear", "Bull"], [sns.color_palette()[3], sns.color_palette()[2]])
        ):
            vals = _returns[_regimes == k, i]
            if len(vals) > 2:
                sns.kdeplot(vals, ax=ax, color=color, label=label, fill=True, alpha=0.3)
        ax.set_title(_asset_names[i])
        ax.set_xlabel("Monthly return")
        if i == 0:
            ax.legend()

    fig_dist.suptitle("Regime-Conditional Return Distributions (true labels)", y=1.02)
    fig_dist.tight_layout()
    fig_dist
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Next Steps

    This v0 implementation establishes the core pipeline.  Planned
    extensions for v1 and beyond include:

    - **Multivariate Student-$t$ emissions** -- heavier tails to capture
      extreme drawdowns without inflating the normal volatility estimate.
    - **Hierarchical priors** -- share strength across asset classes for
      means and covariances.
    - **Mixed-frequency data** -- incorporate quarterly private-asset
      returns alongside monthly public returns.
    - **Factor structure for covariance** -- decompose risk into
      systematic and idiosyncratic components.
    - **Real data application** -- replace synthetic returns with actual
      index data and evaluate out-of-sample regime detection.
    """)
    return


if __name__ == "__main__":
    app.run()
