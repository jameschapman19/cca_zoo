"""AJIVE for views that are missing for whole samples: complete them by low-rank EM, then run AJIVE.

The EM rank is chosen by hiding views of samples that have several and scoring how well each rank
fills them in; AJIVE then picks its own signal, joint and individual ranks.
"""

from __future__ import annotations

import numpy as np
from common import AJIVE, Views, em_complete


def choose_em_rank(views: Views, observed: np.ndarray, grid=range(2, 19, 2), seed: int = 0, fraction: float = 0.2) -> int:
    """The EM rank that best fills in views hidden from samples that have at least two."""
    rng = np.random.default_rng(seed)
    hidden = np.zeros_like(observed)
    for i in np.flatnonzero(observed.sum(1) >= 2):
        if rng.random() < fraction:
            hidden[i, rng.choice(np.flatnonzero(observed[i]))] = True
    kept = observed & ~hidden
    masked = [np.where(k[:, None], v, np.nan) for v, k in zip(views, kept.T)]
    errors = []
    for rank in grid:
        done = em_complete(masked, kept, rank)
        err = sum(np.sum((d[h] - v[h]) ** 2) for d, v, h in zip(done, views, hidden.T) if h.any())
        errors.append(err)
    return list(grid)[int(np.argmin(errors))]


def ajive_em(views: Views, observed: np.ndarray, seed: int = 0) -> tuple[AJIVE, Views, int]:
    """AJIVE fitted to EM-completed views with every rank chosen automatically."""
    rank = choose_em_rank(views, observed, seed=seed)
    done = em_complete(views, observed, rank)
    return AJIVE(n_components="auto").fit(done), done, rank


def observed_signal_ranks(views: Views, observed: np.ndarray) -> list[int]:
    """Each view's signal rank from the singular values of its observed rows only (Gavish-Donoho)."""
    from cca_zoo.linear._ajive import _optimal_rank

    ranks = []
    for v, o in zip(views, observed.T):
        x = v[o] - v[o].mean(0)
        sv = np.linalg.svd(x, compute_uv=False)
        ranks.append(max(_optimal_rank(sv, *x.shape), 1))
    return ranks


def ajive_em_observed_ranks(views: Views, observed: np.ndarray, seed: int = 0, em_rank: str = "signal"):
    """AJIVE on EM-completed views, with signal ranks from the observed rows and the joint rank automatic.

    The EM rank is the sum of the signal ranks (an upper bound on the joint plus individual rank)
    or, with ``em_rank="cv"``, chosen by hiding views.
    """
    signal = observed_signal_ranks(views, observed)
    rank = sum(signal) if em_rank == "signal" else choose_em_rank(views, observed, seed=seed)
    done = em_complete(views, observed, rank)
    try:
        model = AJIVE(n_components="auto", signal_ranks=signal).fit(done)
    except ValueError:
        return None, done, rank, signal
    return model, done, rank, signal


def ajive_em_final(views: Views, observed: np.ndarray):
    """AJIVE for views missing from whole samples.

    1. Each view's signal rank comes from the singular values of its observed rows (Gavish-Donoho).
    2. Missing views are filled by rank-(sum of signal ranks) EM on the concatenated views.
    3. AJIVE on the completed views picks the joint rank and gives the joint scores.
    4. Each view's individual scores come from its observed rows alone: the joint scores are
       projected out and the top (signal rank - joint rank) singular directions of the rest kept.

    Returns the joint scores (n by K, NaN-free), a list of individual scores (zero for samples that
    lack the view), and the ranks, or None when AJIVE finds no joint structure.
    """
    signal = observed_signal_ranks(views, observed)
    done = em_complete(views, observed, sum(signal))
    try:
        model = AJIVE(n_components="auto", signal_ranks=signal).fit(done)
    except ValueError:
        return None
    joint = np.mean(model.transform(done), axis=0)
    k = joint.shape[1]
    individual = []
    for x, o, s in zip(views, observed.T, signal):
        xo = x[o] - x[o].mean(0)
        q = np.linalg.qr(joint[o] - joint[o].mean(0))[0]
        u, sv, _ = np.linalg.svd(xo - q @ (q.T @ xo), full_matrices=False)
        r = max(s - k, 0)
        scores = np.zeros((len(x), r))
        scores[o] = u[:, :r] * sv[:r]
        individual.append(scores)
    return joint, individual, dict(signal=signal, joint=k)
