"""Figures for the JIVEEY paper, sized for a PowerPoint content box (10 x 4.5 in).

python plot.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import HERE

OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
COLORS = {
    "JIVEEY (selected)": "#1f4e9c",
    "JIVEEY (true ranks)": "#6fa0e0",
    "AJIVE (true ranks)": "#d97b1c",
    "AJIVE (auto ranks)": "#f0b27a",
    "MOFA+": "#4c9a4c",
    "IncompleteGCCA": "#8a8a8a",
    "CCA-EY (no individual)": "#b05fb0",
}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def e1() -> None:
    """Shared and individual recovery against the fraction of views missing."""
    d = pd.read_json(HERE / "output" / "e1.json")
    for column, name in (("shared", "joint"), ("private", "individual")):
        fig, axes = plt.subplots(1, 3, figsize=(10, 4.5), sharey=True)
        for ax, regime in zip(axes, ("moderate", "faint", "fainter")):
            for method, color in COLORS.items():
                s = d[(d.regime == regime) & (d.method == method)].dropna(subset=[column])
                if s.empty:
                    continue
                g = s.groupby("rate")[column].agg(["mean", "sem"])
                ax.errorbar(g.index * 100, g["mean"], g["sem"], label=method, color=color, marker="o", ms=4, lw=1.8, capsize=2)
            ax.set_title({"moderate": "10% shared", "faint": "2% shared", "fainter": "1% shared"}[regime])
            ax.set_xlabel("views missing (%)")
            ax.set_xticks([0, 30, 50])
        axes[0].set_ylabel(f"{name} recovery")
        axes[0].set_ylim(0, 1.02)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        fig.savefig(OUT / f"e1_{name}.png", dpi=200)
        plt.close(fig)


def e2() -> None:
    """Estimated variance shares against the truth."""
    d = pd.read_json(HERE / "output" / "e2.json")
    fig, axes = plt.subplots(1, 4, figsize=(10, 4.5), sharey="row")
    for ax, ((regime, rate), s) in zip(axes, d.groupby(["regime", "rate"], sort=False)):
        methods = ["truth", "JIVEEY, in sample", "JIVEEY, cross-fitted", "AJIVE, in sample"]
        m = s.groupby("method")[["joint", "individual", "noise"]].mean().loc[methods]
        e = s.groupby("method")[["joint"]].sem().loc[methods]
        x = np.arange(len(methods))
        ax.bar(x, m["joint"], yerr=e["joint"], color=["#444", "#1f4e9c", "#6fa0e0", "#d97b1c"], capsize=2)
        ax.set_xticks(x, ["truth", "JIVEEY\nin sample", "JIVEEY\ncross-fit", "AJIVE\nin sample"], fontsize=8)
        ax.set_title(f"{regime}, {int(rate * 100)}% missing", fontsize=10)
    axes[0].set_ylabel("joint share of variance")
    fig.tight_layout()
    fig.savefig(OUT / "e2_joint_share.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    e1()
    e2() if (HERE / "output" / "e2.json").exists() else None
