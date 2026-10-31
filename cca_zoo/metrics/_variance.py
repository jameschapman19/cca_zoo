"""Variance explained by each latent dimension, as in factor-analysis reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def variance_explained(
    views: Sequence[ArrayLike], transformed: Sequence[ArrayLike]
) -> np.ndarray:
    """Fraction of each view's variance explained by each latent dimension alone.

    The R squared of regressing the view's centred columns on one dimension's
    scores, as in MOFA's per-factor variance explained. Dimensions that are
    correlated explain overlapping variance, so a view's fractions add up to
    at most 1 only for uncorrelated scores. Rows with a ``NaN`` score or
    feature, such as a sample that lacks the view, are left out of that view.

    Args:
        views: Arrays of shape (n_samples, n_features_i), one per view.
        transformed: Scores of shape (n_samples, n_components), one per view.

    Returns:
        Shape (n_views, n_components).

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.metrics import variance_explained
        >>> rng = np.random.default_rng(0)
        >>> t = rng.standard_normal((200, 1))
        >>> view = np.column_stack([t[:, 0], rng.standard_normal(200)])
        >>> round(float(variance_explained([view], [t])[0, 0]), 1)
        0.5
    """
    fractions = []
    for view, scores in zip(views, transformed, strict=True):
        v, t = np.asarray(view, dtype=float), np.asarray(scores, dtype=float)
        seen = np.isfinite(v).all(axis=1) & np.isfinite(t).all(axis=1)
        v, t = v[seen] - v[seen].mean(axis=0), t[seen] - t[seen].mean(axis=0)
        # Regressing on a single centred column: ||t_k' V||^2 / ||t_k||^2.
        explained = np.sum((t.T @ v) ** 2, axis=1) / np.maximum(
            np.sum(t**2, axis=0), 1e-12
        )
        fractions.append(explained / np.sum(v**2))
    return np.stack(fractions)
