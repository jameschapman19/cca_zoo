"""Permutation tests for multiview models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, validate_params
from sklearn.utils.parallel import Parallel, delayed

from cca_zoo._utils._validation import validate_views
from cca_zoo.metrics import (
    average_pairwise_correlations,
    factor_loadings,
    pairwise_correlations,
)


@dataclass
class PermutationTestResult:
    """Result of :func:`permutation_test_significance`.

    Attributes:
        correlations: Observed canonical correlation per dimension, shape (k,).
        null_correlations: Permuted correlations, shape (n_permutations, k).
        p_values: P-value per dimension, shape (k,).
        loadings: Observed factor loadings, shape (n_features_i, k) per view.
        null_loadings: Permuted loadings aligned to ``loadings``, shape
            (n_permutations, n_features_i, k) per view.
        loading_p_values: P-value per feature and dimension, shape
            (n_features_i, k) per view.
    """

    correlations: np.ndarray
    null_correlations: np.ndarray
    p_values: np.ndarray
    loadings: list[np.ndarray]
    null_loadings: list[np.ndarray]
    loading_p_values: list[np.ndarray]


def _correlations_and_loadings(
    model: BaseEstimator, views: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per-dimension canonical correlations and factor loadings of a fit."""
    # Arrays whatever the output container set by set_output or set_config.
    scores = [np.asarray(s) for s in model.transform(views)]
    correlations = average_pairwise_correlations(pairwise_correlations(scores))
    return correlations, factor_loadings(views, scores)


@validate_params(
    {
        "estimator": [HasMethods(["fit", "transform"])],
        "views": [list],
        "n_permutations": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=False,
)
def permutation_test_significance(
    estimator: BaseEstimator,
    views: list[ArrayLike],
    n_permutations: int = 1000,
    random_state: int | np.random.RandomState | None = None,
    n_jobs: int | None = None,
) -> PermutationTestResult:
    """Permutation test of the canonical correlations and factor loadings.

    Refits ``estimator`` on data with the rows of every view but the first
    shuffled independently, which keeps each view's covariance and breaks
    their correspondence. Each permuted fit's loadings are aligned to the
    observed ones by orthogonal Procrustes over all views before comparison,
    since permuted fits can rotate or reflect near-tied dimensions.

    Only the first dimension's p-value is a valid test. Each later dimension
    is compared with the permuted fits' dimension of the same rank, which
    were fitted without removing the earlier dimensions' signal; Winkler et
    al. (2020) test them step-down, residualising each earlier dimension
    first.

    Args:
        estimator: An unfitted multiview estimator.
        views: Arrays of shape (n_samples, n_features_i), one per view.
        n_permutations: Number of permutations. Default is 1000.
        random_state: Seed for the permutations. Default is None.
        n_jobs: Number of parallel jobs. Default is None.

    Returns:
        The observed and null statistics.

    Raises:
        ValueError: If ``n_permutations`` is not positive.

    References:
        Winkler, A. M., Renaud, O., Smith, S. M., & Nichols, T. E. (2020).
        Permutation inference for canonical correlation analysis.
        NeuroImage, 220, 117065.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import permutation_test_significance
        >>> rng = np.random.default_rng(0)
        >>> z = rng.standard_normal((40, 1))
        >>> X1 = z @ rng.standard_normal((1, 5)) + 0.1 * rng.standard_normal((40, 5))
        >>> X2 = z @ rng.standard_normal((1, 4)) + 0.1 * rng.standard_normal((40, 4))
        >>> result = permutation_test_significance(
        ...     CCA(), [X1, X2], n_permutations=49, random_state=0
        ... )
        >>> float(result.p_values[0])
        0.02
    """
    arrays = validate_views(views)
    n_views = len(arrays)

    fitted = clone(estimator).fit(arrays)
    true_corr, true_loadings = _correlations_and_loadings(fitted, arrays)
    true_stack = np.vstack(true_loadings)  # (sum(n_features_i), k)
    split_points = np.cumsum([loading.shape[0] for loading in true_loadings[:-1]])

    seeds = check_random_state(random_state).randint(
        np.iinfo(np.int32).max, size=n_permutations
    )

    def _one_permutation(seed: int) -> tuple[np.ndarray, np.ndarray]:
        local_rng = np.random.default_rng(seed)
        permuted = [arrays[0]] + [local_rng.permutation(v, axis=0) for v in arrays[1:]]
        model = clone(estimator).fit(permuted)
        corr, loadings = _correlations_and_loadings(model, permuted)
        stack = np.vstack(loadings)
        rotation = scipy.linalg.orthogonal_procrustes(stack, true_stack)[0]
        aligned_stack: np.ndarray = stack @ rotation
        return corr, aligned_stack

    results = Parallel(n_jobs=n_jobs)(delayed(_one_permutation)(seed) for seed in seeds)
    null_correlations = np.stack([corr for corr, _ in results])  # (n_perm, k)
    null_stacks = np.stack([stack for _, stack in results])  # (n_perm, sum_p, k)
    null_loadings = list(np.split(null_stacks, split_points, axis=1))

    p_values = (1 + (null_correlations >= true_corr).sum(axis=0)) / (1 + n_permutations)
    loading_p_values = [
        (1 + (np.abs(null_loadings[i]) >= np.abs(true_loadings[i])).sum(axis=0))
        / (1 + n_permutations)
        for i in range(n_views)
    ]

    return PermutationTestResult(
        correlations=true_corr,
        null_correlations=null_correlations,
        p_values=p_values,
        loadings=true_loadings,
        null_loadings=null_loadings,
        loading_p_values=loading_p_values,
    )
