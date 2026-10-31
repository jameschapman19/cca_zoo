"""Synthetic sweep: batch size x embedding dimension for EY and LeJEPA arms.

Linear two-view data (p=256, 8 shared Gaussian sources, rho 0.5-0.2), 20000
training samples, linear encoders, 3000 Adam steps. Seeds 400-403.

Run:  python scripts/sweep_batch_dim.py [--plot-only]
Writes results/sweep_batch_dim.npz and figures/sweep_batch_dim.png.
"""

from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ieye import ssl
from ieye.data import TwoViewModel

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/sweep_batch_dim.npz"
SEEDS = list(range(400, 404))
DIMS = [16, 64, 128]
BATCHES = [8, 16, 32, 64, 128]
P, R, N_TRAIN, STEPS, LR = 256, 8, 20000, 3000, 3e-3
RHO = np.linspace(0.5, 0.2, R)
METRICS = ["capture", "probe_r2", "variance_ratio"]


def run(job: tuple[str, int, int, int]) -> tuple[tuple, list[float]]:
    arm, d, batch, seed = job
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = TwoViewModel.make(RHO, np.zeros(R), P, rng)
    _, views = model.sample(N_TRAIN, rng)
    try:
        w = ssl.train(arm, views, d, batch, STEPS, LR, np.random.default_rng(seed + 1))
    except FloatingPointError:  # a diverged run is an outcome, recorded as NaN
        return job, [np.nan] * len(METRICS)
    _, test = model.sample(20000, np.random.default_rng(seed + 2))
    target = np.sum(1 + np.pad(RHO, (0, max(d - R, 0)))[:d])
    variance = np.mean([np.trace(np.cov((x @ e).T)) for x, e in zip(test, w)])
    return job, [
        ssl.subspace_capture(model, w),
        ssl.probe_r2(model, w, np.random.default_rng(seed + 3)),
        variance / target,
    ]


def reference() -> float:
    """Probe R^2 at the population CCA weights, mean over seeds."""
    scores = []
    for seed in SEEDS:
        model = TwoViewModel.make(RHO, np.zeros(R), P, np.random.default_rng(seed))
        scores.append(ssl.probe_r2(model, [model.canonical_weights(v) for v in range(2)],
                                   np.random.default_rng(seed + 3)))
    return float(np.mean(scores))


def sweep() -> None:
    jobs = [(a, d, b, s) for a in ssl.ARMS for d in DIMS for b in BATCHES for s in SEEDS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Pool(4) as pool:
            results = pool.map(run, jobs, chunksize=1)
    table = {m: np.full((len(ssl.ARMS), len(DIMS), len(BATCHES), len(SEEDS)), np.nan) for m in METRICS}
    arms = list(ssl.ARMS)
    for (arm, d, b, s), values in results:
        index = (arms.index(arm), DIMS.index(d), BATCHES.index(b), SEEDS.index(s))
        for m, v in zip(METRICS, values):
            table[m][index] = v
    np.savez(RESULT, **table, oracle_probe_r2=reference())


def plot() -> None:
    data = np.load(RESULT)
    arms = list(ssl.ARMS)
    colours = dict(zip(arms, ["#D55E00", "#0072B2", "#999999", "#009E73"]))
    titles = {
        "capture": "Fraction of CCA subspace in learned span",
        "probe_r2": "Linear-probe test R² (dashed: population CCA)",
        "variance_ratio": "Embedding variance / EY optimum (EY arms)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 5.5))
    for ax, metric in zip(axes, METRICS):
        for a, arm in enumerate(arms):
            if metric == "variance_ratio" and "LeJEPA" in arm:
                continue
            xs, ys, lo, hi = [], [], [], []
            for i, d in enumerate(DIMS):
                for j, b in enumerate(BATCHES):
                    v = data[metric][a, i, j]
                    xs.append(d / b)
                    ys.append(np.nanmean(v))
                    lo.append(np.nanmin(v))
                    hi.append(np.nanmax(v))
            xs, ys, lo, hi = map(np.array, (xs, ys, lo, hi))
            order = np.argsort(xs)
            ax.errorbar(xs[order], ys[order], yerr=[(ys - lo)[order], (hi - ys)[order]], fmt="o",
                        ms=3, lw=0.8, color=colours[arm], label=arm, alpha=0.8)
        if metric == "probe_r2":
            ax.axhline(float(data["oracle_probe_r2"]), ls="--", color="k", lw=0.8)
        if metric == "variance_ratio":
            ax.axhline(1.0, ls="--", color="k", lw=0.8)
        ax.set(xscale="log", xlabel="embedding dim / batch size", title=titles[metric])
        ax.title.set_fontsize(8)
    axes[0].set_ylabel("mean and min–max over 4 seeds, one point per (dim, batch)")
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/sweep_batch_dim.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        sweep()
    plot()
