"""Correlation metrics on latent scores, as returned by ``transform``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def pairwise_correlations(transformed: Sequence[ArrayLike]) -> np.ndarray:
    """Pearson correlation between every pair of views' latent scores.

    Args:
        transformed: Scores of shape (n_samples, n_components), one per view.

    Returns:
        Shape (n_views, n_views, n_components); entry ``[i, j, d]``
        correlates the d-th scores of views i and j.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.metrics import pairwise_correlations
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> t2 = 0.8 * t1 + 0.2 * rng.standard_normal((20, 1))
        >>> round(float(pairwise_correlations([t1, t2])[0, 1, 0]), 2)
        0.98
    """
    T = np.stack([np.asarray(t) for t in transformed], axis=0)
    T = T - T.mean(axis=1, keepdims=True)
    norms = np.sqrt((T**2).sum(axis=1, keepdims=True))
    T_norm = T / np.where(norms > 1e-12, norms, 1.0)
    corrs: np.ndarray = np.einsum("isd,jsd->ijd", T_norm, T_norm)
    return corrs


def average_pairwise_correlations(correlations: ArrayLike) -> np.ndarray:
    """Mean correlation over pairs of distinct views, per dimension.

    Args:
        correlations: Output of :func:`pairwise_correlations`.

    Returns:
        Shape (n_components,).
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
    """Correlation of each feature with its own view's latent scores.

    Args:
        views: Arrays of shape (n_samples, n_features_i), one per view.
        transformed: Scores of shape (n_samples, n_components), one per view.

    Returns:
        One array of shape (n_features_i, n_components) per view.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.metrics import factor_loadings
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> view1 = np.column_stack(
        ...     [t1[:, 0] + 0.3 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> round(float(factor_loadings([view1], [t1])[0][0, 0]), 2)
        0.97
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
