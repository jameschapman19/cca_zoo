"""Canonical-correlation metrics shared by every fitted multiview model.

Every function here operates on already-computed arrays (latent scores, a
correlation matrix) rather than on a fitted model, mirroring
``sklearn.metrics``'s own convention of taking computed values (e.g.
``y_true``/``y_pred``) rather than an estimator. ``BaseModel``'s
``pairwise_correlations``/``average_pairwise_correlations``/
``get_factor_loadings`` (and the probabilistic module's
``PosteriorMeanTransformMixin``, which needs a different per-view
projection) are thin wrappers around these: they compute the per-view
latent scores their own way and delegate the actual metric computation
here, so the math is written -- and tested -- exactly once.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def pairwise_correlations(transformed: Sequence[ArrayLike]) -> np.ndarray:
    """Full pairwise Pearson correlation matrix between views' latent scores.

    Args:
        transformed: List of arrays, each of shape (n_samples,
            latent_dimensions) -- one per view's own canonical variate
            (e.g. the output of a fitted model's ``transform``).

    Returns:
        Array of shape (n_views, n_views, latent_dimensions) where entry
        ``[i, j, d]`` is the Pearson correlation between view i's and view
        j's d-th canonical variate.
    """
    T = np.stack([np.asarray(t) for t in transformed], axis=0)
    T = T - T.mean(axis=1, keepdims=True)
    norms = np.sqrt((T**2).sum(axis=1, keepdims=True))
    T_norm = T / np.where(norms > 1e-12, norms, 1.0)
    corrs: np.ndarray = np.einsum("isd,jsd->ijd", T_norm, T_norm)
    return corrs


def average_pairwise_correlations(correlations: ArrayLike) -> np.ndarray:
    """Mean off-diagonal pairwise correlation per canonical dimension.

    Args:
        correlations: Array of shape (n_views, n_views, latent_dimensions)
            -- typically the output of :func:`pairwise_correlations`.

    Returns:
        Array of shape (latent_dimensions,) with the average off-diagonal
        pairwise correlation for each canonical dimension.
    """
    corrs = np.asarray(correlations)
    n_views = corrs.shape[0]
    off_diag_sum: np.ndarray = corrs.sum(axis=(0, 1)) - sum(
        corrs[i, i, :] for i in range(n_views)
    )
    n_pairs = n_views * (n_views - 1)
    result: np.ndarray = off_diag_sum / n_pairs
    return result


def factor_loadings(
    views: Sequence[ArrayLike], transformed: Sequence[ArrayLike]
) -> list[np.ndarray]:
    """Pearson correlation between each raw feature and its view's own variate.

    Args:
        views: List of arrays, each of shape (n_samples, n_features_i).
        transformed: List of arrays, each of shape (n_samples,
            latent_dimensions), aligned with ``views`` -- view i's own
            canonical variate.

    Returns:
        List of arrays, each of shape (n_features_i, latent_dimensions),
        where entry ``[j, d]`` is the correlation between feature j of
        view i and the d-th canonical variate of view i.
    """
    loadings = []
    for view, variate in zip(views, transformed):
        v = np.asarray(view)
        t = np.asarray(variate)
        v_c = v - v.mean(axis=0)
        t_c = t - t.mean(axis=0)
        cov = v_c.T @ t_c / (v.shape[0] - 1)
        std_v = np.maximum(v_c.std(axis=0, ddof=1), 1e-12)
        std_t = np.maximum(t_c.std(axis=0, ddof=1), 1e-12)
        loadings.append(cov / np.outer(std_v, std_t))
    return loadings
