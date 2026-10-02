"""Redundancy analysis: how much of each view's own variance the variates explain."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def adequacy_coefficient(loadings: Sequence[ArrayLike]) -> list[np.ndarray]:
    """Variance of each view extracted by its own canonical variates.

    The mean squared factor loading per dimension, the variance-extracted
    term of Stewart and Love's (1968) redundancy index.

    Args:
        loadings: Loadings of shape (n_features_i, n_components), one per
            view, as from :func:`~cca_zoo.metrics.factor_loadings`.

    Returns:
        One array of shape (n_components,) per view.

    References:
        Stewart, D., & Love, W. (1968). A general canonical correlation index.
        Psychological Bulletin, 70(3), 160-163.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.metrics import adequacy_coefficient, factor_loadings
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> view1 = np.column_stack(
        ...     [t1[:, 0] + 0.3 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> adequacy = adequacy_coefficient(factor_loadings([view1], [t1]))
        >>> round(float(adequacy[0][0]), 2)
        0.48
    """
    return [np.mean(np.asarray(loading) ** 2, axis=0) for loading in loadings]


def redundancy_index(
    loadings: Sequence[ArrayLike], correlations: ArrayLike
) -> np.ndarray:
    """Stewart-Love redundancy: variance of view i explained through view j.

    View i's adequacy times the squared canonical correlation between the
    views' variates, per dimension. Asymmetric in ``i`` and ``j``.

    Args:
        loadings: Loadings of shape (n_features_i, n_components), one per
            view, as from :func:`~cca_zoo.metrics.factor_loadings`.
        correlations: Shape (n_views, n_views, n_components), as from
            :func:`~cca_zoo.metrics.pairwise_correlations`.

    Returns:
        Shape (n_views, n_views, n_components); the diagonal is each view's
        adequacy.

    References:
        Stewart, D., & Love, W. (1968). A general canonical correlation index.
        Psychological Bulletin, 70(3), 160-163.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.metrics import (
        ...     factor_loadings,
        ...     pairwise_correlations,
        ...     redundancy_index,
        ... )
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> t2 = 0.8 * t1 + 0.2 * rng.standard_normal((20, 1))
        >>> view1 = np.column_stack(
        ...     [t1[:, 0] + 0.1 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> view2 = np.column_stack(
        ...     [t2[:, 0] + 0.1 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> redundancy = redundancy_index(
        ...     factor_loadings([view1, view2], [t1, t2]),
        ...     pairwise_correlations([t1, t2]),
        ... )
        >>> round(float(redundancy[0, 1, 0]), 2)
        0.52
    """
    adequacy = np.stack(adequacy_coefficient(loadings), axis=0)  # (n_views, k)
    corrs = np.asarray(correlations)
    result: np.ndarray = adequacy[:, np.newaxis, :] * corrs**2
    return result


def total_redundancy(redundancy: ArrayLike) -> np.ndarray:
    """Redundancy summed over dimensions, shape (n_views, n_views).

    Args:
        redundancy: Output of :func:`redundancy_index`.

    Returns:
        Entry ``[i, j]`` is the variance of view i explained through view j.
    """
    result: np.ndarray = np.asarray(redundancy).sum(axis=-1)
    return result
