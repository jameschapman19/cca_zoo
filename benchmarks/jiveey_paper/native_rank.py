"""A joint-rank rule native to GCCA with missing views: AJIVE's random-direction bound, on the same missingness pattern.

The stack of each view's orthonormal signal bases (zero where the view is missing, weighted by the
number of views each sample has, as IncompleteGCCA does) has singular values; those above what random
orthonormal bases of the same shapes give at the 95th percentile are joint.
"""

from __future__ import annotations

import numpy as np


def _stack(bases: list[np.ndarray], observed: np.ndarray) -> np.ndarray:
    counts = observed.sum(1)
    blocks = []
    for b, o in zip(bases, observed.T):
        block = np.zeros((len(o), b.shape[1]))
        block[o] = b
        blocks.append(block)
    return np.concatenate(blocks, axis=1) / np.sqrt(counts)[:, None]


def joint_rank_random_direction(views, observed, signal, n_draws=100, level=0.95, seed=0) -> int:
    """Number of singular values of the stacked signal bases above the random-direction bound."""
    rng = np.random.default_rng(seed)
    bases = []
    for v, o, s in zip(views, observed.T, signal):
        x = v[o] - v[o].mean(0)
        bases.append(np.linalg.svd(x, full_matrices=False)[0][:, :s])
    sv = np.linalg.svd(_stack(bases, observed), compute_uv=False)
    top = []
    for _ in range(n_draws):
        rand = [np.linalg.qr(rng.standard_normal(b.shape))[0] for b in bases]
        top.append(np.linalg.svd(_stack(rand, observed), compute_uv=False)[0])
    bound = np.quantile(top, level)
    return int(np.sum(sv > bound))
