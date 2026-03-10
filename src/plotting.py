"""Plotting utilities for the regime-switching HMM (v0)."""

import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_regime_probabilities(
    regime_samples: np.ndarray,
    true_regimes: np.ndarray | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot P(s_t = k) over time as stacked filled bands.

    Parameters
    ----------
    regime_samples : (n_chains, n_draws, T) or (n_samples, T) integer array
    true_regimes : (T,) optional ground-truth regime labels
    ax : matplotlib axes; created if None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    flat = regime_samples.reshape(-1, regime_samples.shape[-1])
    T = flat.shape[1]
    K = int(flat.max()) + 1

    probs = np.zeros((T, K))
    for k in range(K):
        probs[:, k] = (flat == k).mean(axis=0)

    t_axis = np.arange(T)
    colors = ["#4CAF50", "#F44336", "#2196F3", "#FF9800"][:K]
    labels = ["Bull/Growth", "Bear/Stress"] if K == 2 else [f"Regime {k}" for k in range(K)]

    ax.stackplot(t_axis, probs.T, labels=labels, colors=colors, alpha=0.6)

    if true_regimes is not None:
        ax.step(
            t_axis, true_regimes, where="mid",
            color="black", linewidth=1.5, linestyle="--", label="True regime",
        )

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("P(regime)")
    ax.set_title("Regime Probabilities Over Time")
    ax.legend(loc="upper right")
    ax.set_xlim(0, T - 1)
    return ax


def plot_returns_with_regimes(
    returns: np.ndarray,
    regimes: np.ndarray,
    asset_names: list[str] | None = None,
) -> matplotlib.figure.Figure:
    """
    Multi-panel time-series of returns with regime background shading.

    Parameters
    ----------
    returns : (T, d) observed returns
    regimes : (T,) regime labels
    asset_names : optional list of asset names
    """
    T, d = returns.shape
    if asset_names is None:
        asset_names = [f"Asset {i}" for i in range(d)]

    fig, axes = plt.subplots(d, 1, sharex=True, figsize=(12, 2.5 * d))
    if d == 1:
        axes = [axes]

    t_axis = np.arange(T)
    regime_colors = {0: "#C8E6C9", 1: "#FFCDD2"}

    for i, ax in enumerate(axes):
        ax.plot(t_axis, returns[:, i], linewidth=0.8, color="#333333")

        start = 0
        for t in range(1, T):
            if regimes[t] != regimes[t - 1] or t == T - 1:
                end = t if t < T - 1 else T
                color = regime_colors.get(regimes[start], "#E0E0E0")
                ax.axvspan(start, end, alpha=0.3, color=color)
                start = t

        ax.set_ylabel(asset_names[i])
        ax.set_xlim(0, T - 1)

    axes[-1].set_xlabel("Time (months)")
    fig.suptitle("Synthetic Returns by Regime", fontsize=13)
    fig.tight_layout()
    return fig


def plot_posterior_summary(
    idata: az.InferenceData,
    var_name: str = "mu",
    true_values: np.ndarray | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> matplotlib.figure.Figure:
    """
    Forest plot for a single variable with per-row true-value markers.

    Parameters
    ----------
    idata : inference data from NUTS sampling
    var_name : variable to plot (e.g. "mu", "P")
    true_values : array whose flattened values match the forest-plot rows
        (bottom to top).  For mu (K,d) pass ``params["mus"]``; for P (K,K)
        pass ``params["P"]``.
    title : optional title; defaults to "Posterior: {var_name}"
    figsize : figure size
    """
    axes = az.plot_forest(
        idata, var_names=[var_name], combined=True, figsize=figsize,
    )
    ax = axes[0]
    fig = ax.figure

    if true_values is not None:
        flat = np.asarray(true_values).ravel()
        y_ticks = ax.get_yticks()
        for i, y in enumerate(y_ticks):
            if i < len(flat):
                ax.plot(
                    flat[i], y, marker="d", color="red", markersize=8,
                    zorder=10, label="True value" if i == 0 else None,
                )

    ax.set_title(title or f"Posterior: {var_name}")
    ax.legend(loc="best", framealpha=0.8)
    fig.tight_layout()
    return fig
