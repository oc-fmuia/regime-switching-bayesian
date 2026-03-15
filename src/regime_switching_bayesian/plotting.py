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
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    xlim: tuple[float, float] | None = None,
    hdi_prob: float = 0.94,
) -> matplotlib.figure.Figure:
    """
    Forest plot with explicit labels and correctly-placed true-value markers.

    Parameters
    ----------
    idata: az.InferenceData
        Inference data from NUTS sampling.
    var_name: str
        Variable to plot (e.g. "mu", "P").
    true_values: np.ndarray | None
        Array whose flattened shape matches the variable's trailing dimensions.
    labels: list[str] | None
        One label per flattened parameter row, in the same order as ravel().
    title: str | None
        Optional title; defaults to "Posterior: {var_name}".
    figsize: tuple[float, float]
        Figure size.
    xlim: tuple[float, float] | None
        Optional x-axis limits.
    hdi_prob: float
        HDI probability (default 0.94).

    Returns
    -------
    matplotlib.figure.Figure
        The forest plot figure.
    """
    samples = idata.posterior[var_name].values
    flat_samples = samples.reshape(-1, *samples.shape[2:])
    flat_all = flat_samples.reshape(flat_samples.shape[0], -1)
    n_params = flat_all.shape[1]

    means = flat_all.mean(axis=0)
    alpha = (1 - hdi_prob) / 2
    hdi_lo = np.percentile(flat_all, alpha * 100, axis=0)
    hdi_hi = np.percentile(flat_all, (1 - alpha) * 100, axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(n_params)

    for i in range(n_params):
        ax.plot([hdi_lo[i], hdi_hi[i]], [i, i], color="#2196F3", linewidth=2.5)
        ax.plot(means[i], i, "o", color="#2196F3", markersize=5, zorder=5)

    if true_values is not None:
        flat_true = np.asarray(true_values).ravel()
        for i in range(min(n_params, len(flat_true))):
            ax.plot(
                flat_true[i], i, marker="d", color="red", markersize=8,
                zorder=10, label="True value" if i == 0 else None,
            )

    if labels is not None:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels[:n_params])
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{var_name}[{i}]" for i in range(n_params)])

    ax.invert_yaxis()
    ax.set_title(title or f"Posterior: {var_name}")
    ax.legend(loc="best", framealpha=0.8)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    fig.tight_layout()
    return fig
