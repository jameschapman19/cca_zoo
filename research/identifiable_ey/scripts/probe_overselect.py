"""Probe: can non-Gaussianity choose the shared subspace out of 3k CCA components?

Run:  python scripts/probe_overselect.py   Writes figures/probe_overselect.png, results/probe_overselect.npz.
Seeds 300-307.
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
from ieye.metrics import amari_to_sources, subspace_angle
from probe_lowrho import CONFIGS, K, ORACLE, P, oracle

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(300, 308))
NS = [1000, 2000, 5000]
METHODS = {
    "CCA top-k→FastICA": M.cca_fastica,
    "CCA top-3k→FastICA, keep k": partial(M.overselect_fastica, m=3 * K),
    "CCA top-3k→CF pursuit": partial(M.overselect_pursuit, m=3 * K),
}


def run(job: tuple[str, int, int]) -> tuple[str, int, int, dict[str, tuple[float, float]]]:
    config, n, seed = job
    torch.set_num_threads(1)
    rho, kurtosis = CONFIGS[config]
    rng = np.random.default_rng(seed)
    model = TwoViewModel.make(rho, kurtosis, P, rng)
    _, views = model.sample(n, rng)
    fits = {**METHODS, ORACLE: lambda v, k, s: oracle(model, v, s)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = {}
        for name, f in fits.items():
            w = f(views, K, seed)
            out[name] = (amari_to_sources(model, w), subspace_angle(model, w))
    return config, n, seed, out


def main() -> None:
    jobs = [(c, n, s) for c in CONFIGS for n in NS for s in SEEDS]
    with Pool(4) as pool:
        results = pool.map(run, jobs, chunksize=1)
    names = [*METHODS, ORACLE]
    table = {
        (c, m, i): np.array([[r[3][m][i] for r in results if r[0] == c and r[1] == n] for n in NS])
        for c in CONFIGS for m in names for i in (0, 1)
    }
    np.savez(ROOT / "results/probe_overselect.npz", **{f"{c}|{m}|{'amari' if i == 0 else 'angle'}": a
                                                      for (c, m, i), a in table.items()})
    colours = dict(zip(names, ["#0072B2", "#E69F00", "#D55E00", "#000000"]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True, sharey=True)
    for ax, config in zip(axes.ravel(), CONFIGS):
        for name in names:
            a = table[(config, name, 0)]
            ax.plot(NS, a.mean(1), "--" if name == ORACLE else "-o", ms=3, color=colours[name], label=name)
            ax.fill_between(NS, a.min(1), a.max(1), color=colours[name], alpha=0.12, lw=0)
        ax.set(xscale="log", title=config, xticks=NS, xticklabels=[str(n) for n in NS])
    for ax in axes[1]:
        ax.set_xlabel("n samples")
    for ax in axes[:, 0]:
        ax.set_ylabel("Amari distance")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Overselection: mean and min–max over 8 seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/probe_overselect.png", dpi=150)
    for config in CONFIGS:
        for i, n in enumerate(NS):
            cells = [f"{m}: {table[(config, m, 0)][i].mean():.3f}/{table[(config, m, 1)][i].mean():.0f}°" for m in names]
            print(f"{config:32s} n={n:5d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
