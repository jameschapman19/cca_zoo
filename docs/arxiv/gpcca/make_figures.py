"""Render fig1_interaction.pdf, fig2_scaling.pdf and fig3_uncertainty.pdf
from the experiment JSON.

Sized to sit comfortably in a two-column-ish arxiv figure slot (~6.5in wide).
Palette follows the dataviz-skill categorical guidance: a small set of
distinguishable, colorblind-safe hues, consistent ordering across figures.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

OUT = "."

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "figure.dpi": 150,
    }
)

COLORS = {
    "rCCA": "#8a8d91",
    "TreeCCA": "#e07b39",
    "GAMCCA": "#3b78b0",
    "GaussianProcessCCA": "#1f9e6e",
}


def fig1() -> None:
    with open(f"{OUT}/results_interaction.json") as f:
        rows = json.load(f)
    methods = ["rCCA", "TreeCCA", "GAMCCA", "GaussianProcessCCA"]
    labels = ["rCCA\n(linear)", "TreeCCA\n(trees)", "GAMCCA\n(additive)", "GaussianProcessCCA\n(joint kernel)"]
    means = [np.mean([r[m] for r in rows]) for m in methods]
    stds = [np.std([r[m] for r in rows]) for m in methods]

    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    x = np.arange(len(methods))
    bars = ax.bar(
        x, means, yerr=stds, capsize=4,
        color=[COLORS[m] for m in methods],
        edgecolor="white", linewidth=0.8, width=0.6,
        error_kw=dict(elinewidth=1.1, ecolor="#333333"),
    )
    for xi, m, s in zip(x, means, stds):
        label = "0.00" if abs(m) < 0.005 else f"{m:.2f}"
        ax.text(xi, m + s + 0.03, label, ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out canonical correlation")
    ax.set_ylim(-0.25, 1.05)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Recovering a genuine feature interaction ($u \\cdot v$)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_interaction.pdf")
    plt.close(fig)


def fig2() -> None:
    with open(f"{OUT}/results_scaling.json") as f:
        data = json.load(f)
    sparse = data["sparse"]
    exact = data["exact_n800"]

    m_vals = [r["n_inducing"] for r in sparse]
    corrs = [r["test_corr"] for r in sparse]
    times = [r["fit_time"] for r in sparse]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))

    ax1.plot(m_vals, corrs, "o-", color=COLORS["GaussianProcessCCA"], label="sparse (n=4000)")
    ax1.axhline(
        exact["test_corr"], color=COLORS["GAMCCA"], linestyle="--",
        label=f"exact (n={exact['n_train']})",
    )
    ax1.set_xlabel("n_inducing")
    ax1.set_ylabel("Held-out canonical correlation")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("Approximation quality")

    ax2.plot(m_vals, times, "o-", color=COLORS["GaussianProcessCCA"], label="sparse (n=4000)")
    ax2.axhline(
        exact["fit_time"], color=COLORS["GAMCCA"], linestyle="--",
        label=f"exact (n={exact['n_train']})",
    )
    ax2.set_xlabel("n_inducing")
    ax2.set_ylabel("Fit time (s)")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_title("Fit cost")

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_scaling.pdf")
    plt.close(fig)


def fig3() -> None:
    with open(f"{OUT}/results_uncertainty.json") as f:
        d = json.load(f)

    x_grid = np.array(d["x_grid"])
    mean = np.array(d["mean"])
    std = np.array(d["std"])
    x_train = np.array(d["x_train"])
    train_proj = np.array(d["train_proj"])
    sample_paths = np.array(d["sample_paths"])

    color = COLORS["GaussianProcessCCA"]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))

    ax.fill_between(
        x_grid, mean - 1.96 * std, mean + 1.96 * std,
        color=color, alpha=0.18, linewidth=0, label="95% credible band",
    )
    for i in range(sample_paths.shape[1]):
        ax.plot(
            x_grid, sample_paths[:, i], color=color, alpha=0.35, linewidth=0.8,
            label="posterior samples" if i == 0 else None,
        )
    ax.plot(x_grid, mean, color=color, linewidth=1.8, label="posterior mean")
    ax.scatter(
        x_train, train_proj, color="#333333", s=18, zorder=5,
        label="training points",
    )

    ax.set_xlabel("view 1 input $x$")
    ax.set_ylabel("encoder output $f_1(x)$")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Predictive uncertainty: view 1's fitted encoder")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_uncertainty.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("figures written")
