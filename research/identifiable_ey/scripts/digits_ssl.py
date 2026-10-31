"""Nonlinear encoders on real data, cheapest rung: sklearn digits (8x8, 1797 images).

Shared MLP encoder, two augmented views (random +-1 pixel shift, Gaussian noise,
pixel dropout), the four SSL arms of ``ieye.ssl``, linear and kNN probes on
un-augmented embeddings of 797 held-out digits. Seeds 500-504.

Run:  python scripts/digits_ssl.py [--plot-only]
Writes results/digits_ssl.npz and figures/digits_ssl.png.
"""

from __future__ import annotations

import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from ieye import ssl

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/digits_ssl.npz"
SEEDS = list(range(500, 503))
BATCHES = [64, 256]
D, N_TRAIN, LR = 32, 1000, 1e-3
CHECKPOINTS = [250, 1000, 3000, 8000]  # training steps at which the probes are scored
ARMS = [*ssl.ARMS, "random encoder (no training)"]
SHIFTS = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random +-1 pixel shift (zero filled), Gaussian noise and 20% pixel dropout."""
    n = x.shape[0]
    img = F.pad(x.view(n, 1, 8, 8), (1, 1, 1, 1))
    shifted = torch.stack([img[:, :, 1 + i : 9 + i, 1 + j : 9 + j] for i, j in SHIFTS])
    pick = torch.randint(len(SHIFTS), (n,), generator=gen)
    out = shifted[pick, torch.arange(n)].reshape(n, 64)
    out = out + 0.1 * torch.randn(out.shape, generator=gen)
    return out * (torch.rand(out.shape, generator=gen) > 0.2)


def run(job: tuple[str, int, int]) -> tuple[tuple, list[list[float]]]:
    arm, batch, seed = job
    torch.set_num_threads(1)
    digits = load_digits()
    x = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    y = digits.target
    split = np.random.default_rng(seed).permutation(len(y))
    train, test = split[:N_TRAIN], split[N_TRAIN:]
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    encoder = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, D))
    def probes() -> list[float]:
        with torch.no_grad():
            emb = encoder(x).numpy()
        mean, std = emb[train].mean(0), emb[train].std(0) + 1e-8
        emb = (emb - mean) / std
        linear = LogisticRegression(max_iter=3000).fit(emb[train], y[train]).score(emb[test], y[test])
        knn = KNeighborsClassifier(10).fit(emb[train], y[train]).score(emb[test], y[test])
        return [linear, knn]

    if arm not in ssl.ARMS:
        return job, [probes()] * len(CHECKPOINTS)
    opt = torch.optim.Adam(encoder.parameters(), lr=LR)
    scores = []
    for step in range(1, max(CHECKPOINTS) + 1):
        idx = torch.tensor(train)[torch.randint(N_TRAIN, (batch,), generator=gen)]
        z = [encoder(augment(x[idx], gen)) for _ in range(2)]
        loss = ssl.ARMS[arm](z, step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{arm}: non-finite loss at step {step}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in CHECKPOINTS:
            scores.append(probes())
    return job, scores


def sweep() -> None:
    jobs = [(a, b, s) for a in ARMS for b in BATCHES for s in SEEDS]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with Pool(4) as pool:
            results = pool.map(run, jobs, chunksize=1)
    table = np.full((len(CHECKPOINTS), 2, len(ARMS), len(BATCHES), len(SEEDS)), np.nan)
    for (a, b, s), values in results:
        table[:, :, ARMS.index(a), BATCHES.index(b), SEEDS.index(s)] = values
    np.savez(RESULT, accuracy=table)


def plot() -> None:
    acc = np.load(RESULT)["accuracy"]  # (checkpoint, probe, arm, batch, seed)
    colours = ["#D55E00", "#0072B2", "#999999", "#009E73", "#000000"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True, sharey=True)
    for (b, k), ax in zip([(b, k) for b in range(len(BATCHES)) for k in range(2)], axes.ravel()):
        for a, arm in enumerate(ARMS):
            v = acc[:, k, a, b]  # (checkpoint, seed)
            ax.plot(CHECKPOINTS, v.mean(1), "-o", ms=3, color=colours[a], label=arm)
            ax.fill_between(CHECKPOINTS, v.min(1), v.max(1), color=colours[a], alpha=0.15, lw=0)
        ax.set(xscale="log", title=f"{['Linear', 'kNN (k=10)'][k]} probe, batch {BATCHES[b]}")
        ax.title.set_fontsize(9)
    for ax in axes[1]:
        ax.set_xlabel("training steps")
    for ax in axes[:, 0]:
        ax.set_ylabel("held-out accuracy")
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/digits_ssl.png", dpi=150)


if __name__ == "__main__":
    if "--plot-only" not in sys.argv:
        sweep()
    plot()
