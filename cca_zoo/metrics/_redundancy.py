"""Redundancy-analysis metrics (Stewart & Love, 1968; Cramer & Nicewander, 1979).

These quantify how much of one view's *own* variance is captured by the
shared canonical variates -- something a canonical correlation alone
doesn't say: two views can be highly canonically correlated on a dimension
that explains almost none of either view's own variance, if that
dimension happens to pick out a thin, high-correlation sliver of each
view's feature space. Redundancy answers "how useful is this canonical
variate for reconstructing this view's own features", which canonical
correlation alone does not.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def adequacy_coefficient(loadings: Sequence[ArrayLike]) -> list[np.ndarray]:
    """Own-set variance each view's canonical variates extract from that view.

    Also known as the adequacy coefficient (Cramer & Nicewander, 1979) or
    per-dimension communality: the mean squared factor loading, i.e. the
    average proportion of a view's own feature variance captured by each
    of its canonical variates.

    Args:
        loadings: List of arrays, each of shape (n_features_i,
            latent_dimensions) -- typically the output of
            :func:`~cca_zoo.metrics.factor_loadings`.

    Returns:
        List of arrays, each of shape (latent_dimensions,): view i's own
        variance extracted by each of its canonical dimensions.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.metrics import factor_loadings
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> view1 = np.column_stack(
        ...     [t1[:, 0] + 0.3 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> loadings = factor_loadings([view1], [t1])
        >>> adequacy = adequacy_coefficient(loadings)
        >>> adequacy[0].shape
        (1,)
        >>> round(float(adequacy[0][0]), 2)
        0.48
    """
    return [np.mean(np.asarray(loading) ** 2, axis=0) for loading in loadings]


def redundancy_index(
    loadings: Sequence[ArrayLike], correlations: ArrayLike
) -> np.ndarray:
    """Stewart & Love (1968) redundancy: variance in view i explained via view j.

    For each ordered pair of views and canonical dimension, the proportion
    of view i's own variance that is both captured by its d-th canonical
    variate (:func:`adequacy_coefficient`) *and* shared with view j (the
    squared canonical correlation between their d-th variates). Unlike a
    canonical correlation, redundancy is asymmetric: view i's redundancy
    given view j need not equal view j's redundancy given view i, since
    each view's own adequacy can differ.

    Args:
        loadings: List of arrays, each of shape (n_features_i,
            latent_dimensions) -- typically the output of
            :func:`~cca_zoo.metrics.factor_loadings`.
        correlations: Array of shape (n_views, n_views, latent_dimensions)
            -- typically the output of
            :func:`~cca_zoo.metrics.pairwise_correlations`.

    Returns:
        Array of shape (n_views, n_views, latent_dimensions), where entry
        ``[i, j, d]`` is the redundancy of view i given view j's d-th
        canonical variate. The diagonal ``[i, i, d]`` equals view i's own
        adequacy coefficient, since a view's correlation with itself is 1.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.metrics import factor_loadings, pairwise_correlations
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> t2 = 0.8 * t1 + 0.2 * rng.standard_normal((20, 1))
        >>> view1 = np.column_stack(
        ...     [t1[:, 0] + 0.1 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> view2 = np.column_stack(
        ...     [t2[:, 0] + 0.1 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> loadings = factor_loadings([view1, view2], [t1, t2])
        >>> corrs = pairwise_correlations([t1, t2])
        >>> redundancy = redundancy_index(loadings, corrs)
        >>> redundancy.shape
        (2, 2, 1)
        >>> round(float(redundancy[0, 1, 0]), 2)
        0.52
    """
    adequacy = np.stack(adequacy_coefficient(loadings), axis=0)  # (n_views, k)
    corrs = np.asarray(correlations)
    result: np.ndarray = adequacy[:, np.newaxis, :] * corrs**2
    return result


def total_redundancy(redundancy: ArrayLike) -> np.ndarray:
    """Cumulative redundancy across every retained canonical dimension.

    Args:
        redundancy: Array of shape (n_views, n_views, latent_dimensions) --
            typically the output of :func:`redundancy_index`.

    Returns:
        Array of shape (n_views, n_views): entry ``[i, j]`` is the total
        proportion of view i's variance explained by view j's canonical
        variates, summed over every retained dimension.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.metrics import factor_loadings, pairwise_correlations
        >>> rng = np.random.default_rng(0)
        >>> t1 = rng.standard_normal((20, 1))
        >>> t2 = 0.8 * t1 + 0.2 * rng.standard_normal((20, 1))
        >>> view1 = np.column_stack(
        ...     [t1[:, 0] + 0.1 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> view2 = np.column_stack(
        ...     [t2[:, 0] + 0.1 * rng.standard_normal(20), rng.standard_normal(20)]
        ... )
        >>> loadings = factor_loadings([view1, view2], [t1, t2])
        >>> corrs = pairwise_correlations([t1, t2])
        >>> redundancy = redundancy_index(loadings, corrs)
        >>> total = total_redundancy(redundancy)
        >>> total.shape
        (2, 2)
        >>> round(float(total[0, 1]), 2)
        0.52
    """
    result: np.ndarray = np.asarray(redundancy).sum(axis=-1)
    return result
