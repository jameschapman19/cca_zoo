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
