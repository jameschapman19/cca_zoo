"""How well a method recovers the shared factors and fills in missing views."""

from __future__ import annotations

import numpy as np


def subspace_recovery(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared cosine of the principal angles between the two column spaces.

    1 when the estimated factors span the true ones, near 0 when unrelated; free
    of the rotation and sign of the estimated factors.
    """
    qa = np.linalg.qr(estimated - estimated.mean(axis=0))[0]
    qb = np.linalg.qr(truth - truth.mean(axis=0))[0]
    return float(np.mean(np.linalg.svd(qa.T @ qb, compute_uv=False) ** 2))


def imputation_r2(
    reconstruction: list[np.ndarray],
    signal: list[np.ndarray],
    observed: np.ndarray,
) -> float:
    """R squared of the reconstruction of the shared signal of withheld views.

    Pooled over the views and the samples that lack them, against each view's
    noise-free shared part. The private factors and noise cannot be predicted
    from the other views, so they are not asked for: 1 is a perfect fill-in of
    what the other views can tell, 0 no better than the mean.
    """
    sse = sst = 0.0
    for rec, true, seen in zip(reconstruction, signal, observed.T):
        missing = ~seen
        sse += np.sum((rec[missing] - true[missing]) ** 2)
        sst += np.sum(true[missing] ** 2)
    return float(1.0 - sse / sst)


def imputation_skill(
    reconstruction: list[np.ndarray],
    signal: list[np.ndarray],
    masked: list[np.ndarray],
    observed: np.ndarray,
) -> float:
    """Skill over predicting each view's observed mean, for the withheld shared signal.

    ``1 - SSE / SSE_mean``: 0 is no better than filling in the mean of the samples
    that have the view, 1 is perfect, below 0 is worse. The mean is the fair
    baseline, because when the shared signal is a small share of the variance the
    feature means, estimated from the observed samples, are about as noisy as the
    signal, and no method can do better than predicting them.
    """
    sse = sse_mean = 0.0
    for rec, true, view, seen in zip(reconstruction, signal, masked, observed.T):
        missing = ~seen
        sse += np.sum((rec[missing] - true[missing]) ** 2)
        sse_mean += np.sum((np.nanmean(view, axis=0) - true[missing]) ** 2)
    return float(1.0 - sse / sse_mean)
