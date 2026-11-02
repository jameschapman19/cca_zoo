"""Rank selection that uses what makes a column joint: it predicts the other views on held-out samples.

The joint rank K is the number of canonical correlations between the views' held-out joint scores
that beat a permutation null; the individual rank is then chosen at that K by the one-standard-error
rule on the held-out loss. Replaces the joint-by-individual grid, where the held-out loss is flat.
"""

from __future__ import annotations

import numpy as np
from common import JIVEEY, _folds, heldout_loss


def _canonical(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Canonical correlations of two score blocks (centred, orthonormalised)."""
    qa = np.linalg.qr(a - a.mean(0))[0]
    qb = np.linalg.qr(b - b.mean(0))[0]
    return np.sort(np.linalg.svd(qa.T @ qb, compute_uv=False))[::-1]


def _statistic(model: JIVEEY, views, rng: np.random.Generator, permute: bool) -> np.ndarray:
    """Mean over view pairs of the k-th canonical correlation of held-out joint scores."""
    k = model.n_joint
    seen = np.stack([~np.isnan(v).any(1) for v in views], axis=1)
    scores = [z[:, :k] for z in model._projections(views)]
    out = []
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            both = np.flatnonzero(seen[:, i] & seen[:, j])
            if len(both) < 2 * k:
                continue
            a, b = scores[i][both], scores[j][both]
            if permute:
                b = b[rng.permutation(len(both))]
            out.append(_canonical(a, b))
    return np.mean(out, axis=0) if out else np.full(k, np.nan)


def select_joint_rank(views, k_max=4, r_fit=4, shrinkage=0.5, seed=0, n_perm=40, level=0.95):
    """Number of joint components whose held-out cross-view correlation beats chance."""
    rng = np.random.default_rng(seed)
    folds = list(_folds(views, seed))
    stats, nulls = [], []
    for train, test in folds:
        model = JIVEEY(k_max, r_fit, shrinkage).fit([v[train] for v in views], seed=seed, n_init=1)
        held = [v[test] for v in views]
        stats.append(_statistic(model, held, rng, False))
        nulls.append([_statistic(model, held, rng, True) for _ in range(n_perm)])
    observed = np.nanmean(stats, axis=0)
    null = np.nanmean(np.array(nulls), axis=0)  # (n_perm, k_max)
    threshold = np.quantile(null, level, axis=0)
    above = observed > threshold
    k = 0
    while k < k_max and above[k]:
        k += 1
    return k


def select_individual_rank(views, k, shrinkage=0.5, individual=(0, 1, 2, 3, 4, 5), seed=0, n_splits=3):
    """The smallest individual rank within one standard error of the best held-out loss at joint rank k."""
    losses = np.zeros((len(individual), n_splits))
    for g, r in enumerate(individual):
        for f, (train, test) in enumerate(_folds(views, seed, n_splits)):
            m = JIVEEY(k, r, shrinkage).fit([v[train] for v in views], seed=seed, n_init=1)
            losses[g, f] = heldout_loss(m, [v[test] for v in views])
    mean = losses.mean(1)
    best = int(np.argmin(mean))
    se = losses[best].std(ddof=1) / np.sqrt(n_splits)
    return min(r for g, r in enumerate(individual) if mean[g] <= mean[best] + se)


def select_ranks2(views, shrinkage=0.5, seed=0):
    """(K, r) by the held-out cross-view correlation rule, then the one-SE rule."""
    k = max(select_joint_rank(views, shrinkage=shrinkage, seed=seed), 1)
    return k, select_individual_rank(views, k, shrinkage, seed=seed)
