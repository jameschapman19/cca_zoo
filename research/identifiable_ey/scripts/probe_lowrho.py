"""Probe: weak shared signal (low rho, p=30), where subspace and noise matter.

Idea 2 (noise-aware contrast) vs the plain CF contrast and CCA->FastICA, with an
oracle-subspace FastICA as the floor set by rotation error alone.

Run:  python scripts/probe_lowrho.py     Writes figures/probe_lowrho.png, results/probe_lowrho.npz.
Seeds 200-207.
"""

from __future__ import annotations

import warnings
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import FastICA

from ieye import methods as M
from ieye.data import TwoViewModel
from ieye.metrics import amari_to_sources, subspace_angle

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(200, 208))
NS = [500, 2000, 5000]
K, P = 5, 30
CONFIGS = {
    "rho 0.5–0.3, mixed kurtosis": ([0.5, 0.45, 0.4, 0.35, 0.3], [2, -1, 2, -1, 2]),
    "rho 0.5–0.3, super-Gaussian": ([0.5, 0.45, 0.4, 0.35, 0.3], [3] * 5),
    "rho 0.35–0.15, mixed kurtosis": ([0.35, 0.3, 0.25, 0.2, 0.15], [2, -1, 2, -1, 2]),
    "rho 0.35–0.15, super-Gaussian": ([0.35, 0.3, 0.25, 0.2, 0.15], [3] * 5),
}
FAST = dict(restarts=4, max_iter=200)
METHODS = {
    "CCA": M.cca,
    "CCA→FastICA": M.cca_fastica,
    "CCA→CF rotation": partial(M.rotation_only, **FAST),
    "CCA→CF rotation, denoised": partial(M.rotation_only, denoise=True, bandwidth=2.0, **FAST),
    "joint EY+NG": partial(M.joint, mu=1.0, **FAST),
    "joint EY+NG, denoised": partial(M.joint, mu=1.0, denoise=True, bandwidth=2.0, **FAST),
}
ORACLE = "oracle subspace→FastICA"


def oracle(model: TwoViewModel, views: list[np.ndarray], seed: int) -> list[np.ndarray]:
    weights = [model.canonical_weights(v) for v in range(2)]
    shared = np.mean([(x - x.mean(0)) @ w for x, w in zip(views, weights)], axis=0)
    ica = FastICA(n_components=K, whiten="unit-variance", random_state=seed, max_iter=1000).fit(shared)
    return [w @ ica.components_.T for w in weights]


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
    }  # i=0 Amari, i=1 angle; each (n_ns, n_seeds)
    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results/probe_lowrho.npz", **{f"{c}|{m}|{'amari' if i == 0 else 'angle'}": a
                                                  for (c, m, i), a in table.items()})
    colours = dict(zip(names, ["#999999", "#0072B2", "#E69F00", "#D55E00", "#009E73", "#CC79A7", "#000000"]))
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
    fig.suptitle("Weak shared signal, p=30: mean and min–max over 8 seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/probe_lowrho.png", dpi=150)
    for config in CONFIGS:
        for i, n in enumerate(NS):
            ref = table[(config, "CCA→FastICA", 0)][i]
            cells = [f"{m}={table[(config, m, 0)][i].mean():.3f}" for m in names[1:]]
            ang = table[(config, "CCA", 1)][i].mean()
            print(f"{config:32s} n={n:5d} angle={ang:5.1f}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
