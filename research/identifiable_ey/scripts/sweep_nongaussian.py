"""Do heavy-tailed sources hurt SIGReg's subspace recovery? Same toy as sweep_batch_dim.

Paired grid: excess kurtosis in {0, 6}, d in {16, 64}, batch in {32, 128}, 4 seeds
(400-403), plus LeJEPA at lambda=0.5 as an amplified diagnostic.

Run:  python scripts/sweep_nongaussian.py [--plot-only]
Writes results/sweep_nongaussian.npz and figures/sweep_nongaussian.png.
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
RESULT = ROOT / "results/sweep_nongaussian.npz"
SEEDS = list(range(400, 404))
KURTOSIS = [0.0, 6.0]
DIMS = [16, 64]
BATCHES = [32, 128]
P, R, N_TRAIN, STEPS, LR = 256, 8, 20000, 3000, 3e-3
RHO = np.linspace(0.5, 0.2, R)
ARMS = {  # name -> (ssl arm, lambda)
    "EY single-batch": ("EY single-batch", None),
    "EY split-batch (unbiased)": ("EY split-batch (unbiased)", None),
    "LeJEPA λ=0.05": ("LeJEPA (SIGReg)", 0.05),
    "LeJEPA λ=0.5": ("LeJEPA (SIGReg)", 0.5),
}
METRICS = ["capture", "probe_r2"]


def run(job: tuple[str, float, int, int, int]) -> tuple[tuple, list[float]]:
    name, kappa, d, batch, seed = job
    torch.set_num_threads(1)
    arm, lam = ARMS[name]
    if lam is not None:
        ssl.LAMBDA = lam
    rng = np.random.default_rng(seed)
    model = TwoViewModel.make(RHO, np.full(R, kappa), P, rng)
    _, views = model.sample(N_TRAIN, rng)
    w = ssl.train(arm, views, d, batch, STEPS, LR, np.random.default_rng(seed + 1))
    return job, [ssl.subspace_capture(model, w),
                 ssl.probe_r2(model, w, np.random.default_rng(seed + 3))]


def sweep() -> None:
    jobs = [(a, k, d, b, s) for a in ARMS for k in KURTOSIS for d in DIMS for b in BATCHES for s in SEEDS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Pool(4) as pool:
            results = pool.map(run, jobs, chunksize=1)
    shape = (len(ARMS), len(KURTOSIS), len(DIMS), len(BATCHES), len(SEEDS))
    table = {m: np.full(shape, np.nan) for m in METRICS}
    arms = list(ARMS)
    for (a, k, d, b, s), values in results:
        index = (arms.index(a), KURTOSIS.index(k), DIMS.index(d), BATCHES.index(b), SEEDS.index(s))
        for m, v in zip(METRICS, values):
            table[m][index] = v
    np.savez(RESULT, **table)


def plot() -> None:
    data = np.load(RESULT)
    arms = list(ARMS)
    colours = dict(zip(arms, ["#D55E00", "#0072B2", "#999999", "#000000"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    labels = [f"d={d}\nbatch {b}" for d in DIMS for b in BATCHES]
    for ax, metric in zip(axes, METRICS):
        for a, arm in enumerate(arms):
            for k, kappa in enumerate(KURTOSIS):
                v = data[metric][a, k].reshape(len(DIMS) * len(BATCHES), -1)
                x = np.arange(len(labels)) + 0.09 * (a - 1.5) + (0.0 if k == 0 else 0.0)
                ax.errorbar(x, v.mean(1), yerr=[v.mean(1) - v.min(1), v.max(1) - v.mean(1)],
                            fmt="o" if kappa == 0 else "s", ms=4, lw=0.8, color=colours[arm],
                            mfc=colours[arm] if kappa else "white",
                            label=f"{arm}, kurtosis {kappa:g}")
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_title({"capture": "Fraction of CCA subspace in learned span",
                      "probe_r2": "Linear-probe test R²"}[metric], fontsize=9)
    axes[0].set_ylabel("mean and min–max over 4 seeds (hollow: Gaussian, filled: kurtosis 6)")
    axes[1].legend(fontsize=5, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/sweep_nongaussian.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        sweep()
    plot()
