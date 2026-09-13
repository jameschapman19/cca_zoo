"""Permutation-based significance testing for multiview CCA models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils.parallel import Parallel, delayed

from cca_zoo._utils._validation import validate_views


def procrustes_rotation(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Solve the orthogonal Procrustes problem aligning ``target`` to ``reference``.

    Finds the orthogonal matrix ``R`` minimising ``||reference - target @
    R||`` (Frobenius norm), via the classical SVD solution (Schönemann,
    1966): writing the SVD of ``target.T @ reference`` as ``U @ S @ Vt``,
    the optimum is ``R = U @ Vt``. ``R`` is a general orthogonal matrix, not
    restricted to a proper (determinant +1) rotation, so it also captures
    axis reflections (sign flips) -- both are needed when matching
    permutation- or bootstrap-resampled canonical variates back to a
    reference fit, since resampling can induce either (Xia et al., 2018,
    *Nat. Commun.*; McIntosh & Lobaugh, 2004, *NeuroImage*).

    Args:
        reference: Array of shape (n, k).
        target: Array of shape (n, k), matched row-for-row with
            ``reference`` (e.g. the same features/variables in the same
            order), but not necessarily in the same column (component)
            order or sign.

    Returns:
        Orthogonal matrix of shape (k, k) such that ``target @ R`` is
        optimally aligned to ``reference`` in the least-squares sense.

    Raises:
        ValueError: If ``reference`` and ``target`` don't have the same
            shape.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.model_selection import procrustes_rotation
        >>> rng = np.random.default_rng(0)
        >>> reference = rng.standard_normal((20, 3))
        >>> true_rotation, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        >>> target = reference @ true_rotation.T
        >>> recovered = procrustes_rotation(reference, target)
        >>> np.allclose(target @ recovered, reference, atol=1e-8)
        True
    """
    if reference.shape != target.shape:
        raise ValueError(
            "reference and target must have the same shape, got "
            f"{reference.shape} and {target.shape}."
        )
    m = target.T @ reference
    u, _, vt = np.linalg.svd(m)
    rotation: np.ndarray = u @ vt
    return rotation


@dataclass
class PermutationTestResult:
    """Result of :func:`permutation_test_significance`.

    Attributes:
        correlations_: Observed per-dimension average pairwise canonical
            correlations, shape (k,).
        null_correlations_: Per-dimension correlations from each
            permutation, shape (n_permutations, k).
        p_values_: Per-dimension permutation p-value for the canonical
            correlations, shape (k,).
        loadings_: Observed factor loadings, one array of shape
            (n_features_i, k) per view (see
            :meth:`~cca_zoo._base.BaseModel.get_factor_loadings`).
        null_loadings_: Permuted factor loadings, realigned to
            ``loadings_`` via :func:`procrustes_rotation`, one array of
            shape (n_permutations, n_features_i, k) per view.
        loading_p_values_: Per-feature, per-dimension permutation p-value
            for the factor loadings, one array of shape (n_features_i, k)
            per view.
    """

    correlations_: np.ndarray
    null_correlations_: np.ndarray
    p_values_: np.ndarray
    loadings_: list[np.ndarray]
    null_loadings_: list[np.ndarray]
    loading_p_values_: list[np.ndarray]


def permutation_test_significance(
    estimator: BaseEstimator,
    views: list[ArrayLike],
    n_permutations: int = 1000,
    random_state: int | np.random.Generator | None = None,
    n_jobs: int | None = None,
) -> PermutationTestResult:
    """Permutation test for canonical correlation and feature-loading significance.

    Fits a clone of ``estimator`` on ``views``, then repeatedly refits a
    fresh clone on data where every view except the first has had its rows
    *independently* shuffled -- destroying the true cross-view
    correspondence while preserving each view's own covariance structure --
    to build a null distribution.

    Canonical-correlation significance (``p_values_``) compares each
    dimension's observed correlation directly to its permuted
    counterparts: since both the observed and permuted fits rank
    dimensions by correlation strength, the d-th dimension of a permuted
    fit is already the right null comparison for the d-th observed
    dimension, with no realignment needed.

    Feature-loading significance (``loading_p_values_``) is subtler: a
    permuted refit is not guaranteed to recover canonical variates in the
    same order or with the same sign as the observed fit, since
    permutation can induce an arbitrary rotation or reflection of
    near-tied dimensions (Xia et al., 2018, *Nat. Commun.*; McIntosh &
    Lobaugh, 2004, *NeuroImage*). Each permutation's loadings are
    therefore realigned to the observed loadings via
    :func:`procrustes_rotation` (fit jointly across all views' stacked
    loadings, since the rotation ambiguity is shared across views) before
    being compared feature-by-feature and dimension-by-dimension.

    Args:
        estimator: An unfitted multiview CCA/PLS estimator implementing
            the :class:`~cca_zoo._base.BaseModel` interface.
        views: List of arrays, each of shape (n_samples, n_features_i).
        n_permutations: Number of permutations to draw. Default 1000.
        random_state: Seed or ``numpy.random.Generator`` for reproducible
            permutations.
        n_jobs: Number of permutations to fit in parallel (forwarded to
            :class:`joblib.Parallel` via ``sklearn.utils.parallel``).
            Default ``None`` (sequential).

    Returns:
        PermutationTestResult with the observed and null statistics.

    Raises:
        ValueError: If fewer than 2 views are provided, or
            ``n_permutations`` is not positive.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import permutation_test_significance
        >>> rng = np.random.default_rng(0)
        >>> z = rng.standard_normal((40, 1))
        >>> X1 = z @ rng.standard_normal((1, 5)) + 0.1 * rng.standard_normal((40, 5))
        >>> X2 = z @ rng.standard_normal((1, 4)) + 0.1 * rng.standard_normal((40, 4))
        >>> result = permutation_test_significance(
        ...     CCA(latent_dimensions=1), [X1, X2], n_permutations=49, random_state=0
        ... )
        >>> result.p_values_.shape
        (1,)
    """
    arrays = validate_views(views)
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be positive, got {n_permutations}.")
    n_views = len(arrays)

    fitted = clone(estimator).fit(arrays)
    true_corr = np.asarray(fitted.score(arrays))
    true_loadings = fitted.get_factor_loadings(arrays)
    true_stack = np.vstack(true_loadings)  # (sum(n_features_i), k)
    split_points = np.cumsum([loading.shape[0] for loading in true_loadings[:-1]])

    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, np.iinfo(np.int32).max, size=n_permutations)

    def _one_permutation(seed: int) -> tuple[np.ndarray, np.ndarray]:
        local_rng = np.random.default_rng(seed)
        permuted = [arrays[0]] + [local_rng.permutation(v, axis=0) for v in arrays[1:]]
        model = clone(estimator).fit(permuted)
        corr = np.asarray(model.score(permuted))
        loadings = model.get_factor_loadings(permuted)
        stack = np.vstack(loadings)
        rotation = procrustes_rotation(true_stack, stack)
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
        correlations_=true_corr,
        null_correlations_=null_correlations,
        p_values_=p_values,
        loadings_=true_loadings,
        null_loadings_=null_loadings,
        loading_p_values_=loading_p_values,
    )
