"""Phase 2 kill gate: joint EY+NG vs CCA->FastICA in the targeted regime.

Run:  python scripts/phase2_gate.py   (about 10 minutes on 4 cores)
Writes figures/phase2_amari.png, figures/phase2_paired.png, results/phase2.npz.
Gate seeds are 100-109; the mu pilot used seeds 0-2.
"""

from __future__ import annotations

import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ieye import methods as M
from ieye.data import TwoViewModel
from ieye.metrics import amari_to_sources

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(100, 110))
NS = [500, 1000, 2000, 5000]
K, P = 5, 10
CONFIGS = {
    "mixed kurtosis (+2/-1), rho gap 0.05": ([0.80, 0.75, 0.70, 0.65, 0.60], [2, -1, 2, -1, 2]),
    "all super-Gaussian (+2), rho gap 0.05": ([0.80, 0.75, 0.70, 0.65, 0.60], [2] * 5),
    "mixed kurtosis (+2/-1), rho gap 0.01": ([0.80, 0.79, 0.78, 0.77, 0.76], [2, -1, 2, -1, 2]),
}
METHODS = {
    "CCA": M.cca,
    "CCA→FastICA": M.cca_fastica,
    "CCA→MultiViewICA": M.multiview_ica,
    "CCA→CF rotation": partial(M.rotation_only, restarts=4, max_iter=200),
    "joint EY+NG (μ=1)": partial(M.joint, mu=1.0, restarts=4, max_iter=200),
    "joint EY+NG (μ=10)": partial(M.joint, mu=10.0, restarts=4, max_iter=200),
}


def run(job: tuple[str, int, int]) -> tuple[str, int, int, dict[str, float]]:
    config, n, seed = job
    torch.set_num_threads(1)
    rho, kurtosis = CONFIGS[config]
    rng = np.random.default_rng(seed)
    model = TwoViewModel.make(rho, kurtosis, P, rng)
    _, views = model.sample(n, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = {name: amari_to_sources(model, f(views, K, seed)) for name, f in METHODS.items()}
    return config, n, seed, scores


def main() -> None:
    jobs = [(c, n, s) for c in CONFIGS for n in NS for s in SEEDS]
    with Pool(4) as pool:
        results = pool.map(run, jobs, chunksize=1)
    amari = {
        (c, m): np.array([[r[3][m] for r in results if r[0] == c and r[1] == n] for n in NS])
        for c in CONFIGS for m in METHODS
    }  # (n_ns, n_seeds)
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results/phase2.npz", **{f"{c}|{m}": a for (c, m), a in amari.items()},
             ns=NS, seeds=SEEDS)

    colours = dict(zip(METHODS, ["#999999", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]))
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.5), sharey=True)
    for ax, config in zip(axes, CONFIGS):
        for name in METHODS:
            a = amari[(config, name)]
            ax.plot(NS, a.mean(1), "-o", ms=3, color=colours[name], label=name)
            ax.fill_between(NS, a.min(1), a.max(1), color=colours[name], alpha=0.15, lw=0)
        ax.set(xscale="log", xlabel="n samples", title=config.replace(", ", "\n"))
        ax.set_xticks(NS, [str(n) for n in NS])
    axes[0].set_ylabel("Amari distance to sources (lower is better)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Phase 2: mean and min-max over 10 seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/phase2_amari.png", dpi=150)

    fig, axes = plt.subplots(1, 3, figsize=(10, 5.5), sharey=True)
    for ax, config in zip(axes, CONFIGS):
        for j, name in enumerate(["CCA→CF rotation", "joint EY+NG (μ=1)", "joint EY+NG (μ=10)"]):
            diff = amari[(config, name)] - amari[(config, "CCA→FastICA")]
            for i, n in enumerate(NS):
                ax.scatter(np.full(len(SEEDS), i + 0.22 * (j - 1)), diff[i], s=10,
                           color=colours[name], label=name if i == 0 else None)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(NS)), [str(n) for n in NS])
        ax.set(xlabel="n samples", title=config.replace(", ", "\n"))
    axes[0].set_ylabel("Amari(method) − Amari(CCA→FastICA)\n(below 0: method wins)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Phase 2 paired by seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/phase2_paired.png", dpi=150)

    print("gate: joint wins at a (config, n) iff its worst seed beats FastICA's best seed")
    for config in CONFIGS:
        for i, n in enumerate(NS):
            ref = amari[(config, "CCA→FastICA")][i]
            line = [f"{config[:34]:34s} n={n:5d} fastica={ref.mean():.3f}±{ref.std():.3f}"]
            for name in list(METHODS)[3:]:
                a = amari[(config, name)][i]
                line.append(f"{name}={a.mean():.3f} sep={a.max() < ref.min()}")
            print("  ".join(line))


if __name__ == "__main__":
    main()
