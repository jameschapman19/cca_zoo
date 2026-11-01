"""Figures for the missing-view benchmark, sized for a slide's content box.

python plot.py output/results.json output/results_mofa.json output/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Categorical slots 1-6 of the validated default palette, in order, with a marker
# shape per series so identity does not rest on colour alone.
STYLE = {
    "IncompleteGCCA": ("#2a78d6", "o"),
    "CCAEY (masked)": ("#eb6834", "s"),
    "MOFA+ (default)": ("#1baf7a", "D"),
    "GFA, zero-filled": ("#eda100", "^"),
    "GCCA, zero-filled": ("#e87ba4", "v"),
    "GCCA, complete cases": ("#008300", "P"),
    "MOFA+ (best of 5 random)": ("#6250d6", "X"),
}
RANDOM_INIT = "MOFA+ (best of 5 random)"
SIZE = (11.0, 5.4)  # inches: a 16:9 slide's content area under a title
DPI = 200

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
    }
)

DENSITY = {1.0: "dense loadings", 0.2: "sparse loadings"}


def _legend(fig: plt.Figure, names: list[str]) -> None:
    handles = [
        Line2D(
            [],
            [],
            color=STYLE[n][0],
            marker=STYLE[n][1],
            markersize=7,
            linewidth=2.2,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
            label=n,
        )
        for n in names
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        handlelength=1.6,
        columnspacing=1.4,
    )


def _lines(ax: plt.Axes, data: pd.DataFrame, metric: str, ticks: list[float]) -> None:
    """Mean over seeds with its standard error, one line per method."""
    methods = [m for m in STYLE if m in set(data["method"])]
    for k, method in enumerate(methods):
        color, marker = STYLE[method]
        g = data[data["method"] == method].groupby("missing_rate")[metric]
        mean, se = g.mean(), g.std() / np.sqrt(g.count())
        x = mean.index.to_numpy() + (k - (len(methods) - 1) / 2) * 0.006  # dodge
        ax.errorbar(
            x,
            mean,
            yerr=se,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.2,
            elinewidth=1.0,
            capsize=2,
            markeredgecolor=SURFACE,
            markeredgewidth=1.2,
        )
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def facets(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    columns: str,
    rows: str,
    column_labels: dict,
    row_labels: dict,
    path: Path,
    ylim: tuple[float, float],
) -> None:
    """A grid of panels: ``rows`` by ``columns``, the metric against missing rate."""
    row_values, column_values = (
        sorted(df[rows].unique(), reverse=True),
        sorted(df[columns].unique()),
    )
    fig, axes = plt.subplots(
        len(row_values),
        len(column_values),
        figsize=SIZE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    ticks = sorted(df["missing_rate"].unique())
    for i, r in enumerate(row_values):
        for j, c in enumerate(column_values):
            ax = axes[i, j]
            _lines(ax, df[(df[rows] == r) & (df[columns] == c)], metric, ticks)
            ax.set_ylim(*ylim)
            ax.set_title(
                f"{column_labels.get(c, c)}\n{row_labels.get(r, r)}", fontsize=10.5
            )
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(row_values) - 1:
                ax.set_xlabel("samples missing each view")
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold")
    _legend(fig, [m for m in STYLE if m in set(df["method"])])
    fig.tight_layout(rect=(0, 0.12, 1, 0.96))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def stability(df: pd.DataFrame, path: Path) -> None:
    """Per-seed recovery of IncompleteGCCA and MOFA+ from one or five inits."""
    names = ["IncompleteGCCA", "MOFA+ (default)", RANDOM_INIT]
    colors = {
        "IncompleteGCCA": STYLE["IncompleteGCCA"][0],
        "MOFA+ (default)": STYLE["MOFA+ (default)"][0],
        RANDOM_INIT: "#8a8984",
    }
    data = df[
        (df["mechanism"] == "mcar")
        & (df["missing_rate"] == 0.4)
        & (df["shared"] == 0.1)
    ]
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    rng = np.random.default_rng(0)
    for ax, density in zip(axes, [1.0, 0.2]):
        for k, name in enumerate(names):
            values = data[(data["method"] == name) & (data["density"] == density)][
                "recovery"
            ]
            ax.scatter(
                k + rng.uniform(-0.12, 0.12, len(values)),
                values,
                s=64,
                color=colors[name],
                edgecolor=SURFACE,
                linewidth=1.2,
                zorder=3,
            )
            ax.hlines(
                values.median(), k - 0.28, k + 0.28, color=INK, linewidth=2, zorder=4
            )
        ax.set_xticks(
            range(3),
            ["Incomplete\nGCCA", "MOFA+\ndefault", "MOFA+ best of 5\nrandom inits"],
        )
        ax.set_title(DENSITY[density], fontsize=10.5)
        ax.set_ylim(0, 1.02)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("shared-factor recovery")
    fig.suptitle(
        "MOFA+ collapses on some datasets, from any start (40% missing; bar = median)",
        x=0.01,
        ha="left",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def runtime(df: pd.DataFrame, path: Path) -> None:
    """Median seconds per fit, ranked, in one hue."""
    seconds = df.groupby("method")["seconds"].median().sort_values()
    fig, ax = plt.subplots(figsize=SIZE)
    bars = ax.barh(
        seconds.index, seconds.values, height=0.5, color=STYLE["IncompleteGCCA"][0]
    )
    ax.bar_label(bars, labels=[f"{s:.1f} s" for s in seconds], padding=4, color=INK)
    ax.set_xlabel("median seconds per fit, including cross-validation of shrinkage")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, seconds.max() * 1.15)
    fig.suptitle(
        "Fit time (150 samples, 180 features)",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main() -> None:
    """Draw the figures of each regime in the results files into a directory."""
    *results, out = (Path(a) for a in sys.argv[1:])
    everything = pd.DataFrame(
        [row for r in results for row in json.loads(r.read_text())]
    )
    for regime, df in everything.groupby("regime"):
        shared = {
            s: f"shared signal {s:.1%} of variance"
            for s in sorted(df["shared"].unique())
        }
        mcar = df[df["mechanism"] == "mcar"]
        facets(
            mcar,
            "recovery",
            "shared-factor recovery",
            f"Recovering the shared factors, views missing at random ({regime} regime)",
            "shared",
            "density",
            shared,
            DENSITY,
            out / f"recovery_mcar_{regime}.png",
            (0.0, 1.02),
        )
        facets(
            mcar[mcar["missing_rate"] > 0],
            "imputation_skill",
            "imputation skill",
            f"Filling in a missing view; 0 = no better than its mean, axis cut at -1 ({regime})",
            "shared",
            "density",
            shared,
            DENSITY,
            out / f"imputation_mcar_{regime}.png",
            (-1.0, 1.0),
        )
        facets(
            df[df["shared"] == max(shared)],
            "recovery",
            "shared-factor recovery",
            f"Missingness tied to the biology (MNAR) vs random (MCAR), "
            f"{max(shared):.1%} shared ({regime} regime)",
            "mechanism",
            "density",
            {"mcar": "MCAR", "mnar": "MNAR"},
            DENSITY,
            out / f"recovery_mnar_{regime}.png",
            (0.0, 1.02),
        )
        if regime == "base":
            stability(df, out / "stability.png")
            runtime(mcar, out / "runtime.png")


if __name__ == "__main__":
    main()
