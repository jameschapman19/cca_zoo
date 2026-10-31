"""Recovery metrics against a :class:`~ieye.data.TwoViewModel`."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ieye.data import TwoViewModel


def amari(t: np.ndarray) -> float:
    """Amari distance of a square matrix to a scaled permutation, in [0, 1]."""
    a = np.abs(t)
    k = a.shape[0]
    rows = (a / a.max(axis=1, keepdims=True)).sum(axis=1) - 1
    cols = (a / a.max(axis=0, keepdims=True)).sum(axis=0) - 1
    return float((rows.sum() + cols.sum()) / (2 * k * (k - 1)))


def amari_to_sources(model: TwoViewModel, weights: list[np.ndarray]) -> float:
    """Mean over views of the Amari distance of ``W_v' P_v[:, :k]``."""
    return float(
        np.mean(
            [amari(w.T @ model.view_signal_mixing(v)) for v, w in enumerate(weights)]
        )
    )


def source_r2(
    model: TwoViewModel, weights: list[np.ndarray], n_test: int, rng: np.random.Generator
) -> float:
    """Mean over sources of the squared correlation with the best-matched component.

    The shared variate ``mean_v x_v W_v`` is scored on fresh samples, with
    components matched to sources one-to-one by the Hungarian algorithm.
    """
    s, views = model.sample(n_test, rng)
    z = np.mean([x @ w for x, w in zip(views, weights)], axis=0)
    r2 = np.corrcoef(s.T, z.T)[: model.k, model.k :] ** 2
    rows, cols = linear_sum_assignment(-r2)
    return float(r2[rows, cols].mean())


def subspace_angle(model: TwoViewModel, weights: list[np.ndarray]) -> float:
    """Mean over views of the largest principal angle (degrees) to the CCA subspace.

    Angles are measured in the population-whitened metric, where view ``v``'s
    CCA subspace is the span of the first ``k`` coordinates.
    """
    angles = []
    for v, w in enumerate(weights):
        q, _ = np.linalg.qr(model.mixings[v].T @ w)
        cosines = np.linalg.svd(q[: model.k], compute_uv=False)
        angles.append(np.degrees(np.arccos(np.clip(cosines.min(), -1.0, 1.0))))
    return float(np.mean(angles))
