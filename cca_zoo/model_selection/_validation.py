"""Cross-validation of multiview estimators, as in :mod:`sklearn.model_selection`."""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn
import sklearn.model_selection as skms
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from cca_zoo.model_selection._search import _PARAM_PREFIX, _unwrapped_scoring, _wrap


def cross_val_score(
    estimator: BaseEstimator, views: list[ArrayLike], **kwargs: Any
) -> np.ndarray:
    """Score a multiview estimator by cross-validation.

    :func:`sklearn.model_selection.cross_val_score` on a list of views.

    Args:
        estimator: A multiview estimator.
        views: Arrays of shape (n_samples, n_features_i), one per view.
        **kwargs: Passed to :func:`sklearn.model_selection.cross_val_score`,
            such as ``cv``, ``groups`` and ``n_jobs``. A ``scoring`` callable
            is called as ``scoring(estimator, views)``.

    Returns:
        The score of each fold, by default the mean canonical correlation.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import cross_val_score
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 5)), rng.standard_normal((50, 4))
        >>> cross_val_score(CCA(), [X1, X2], cv=3).shape
        (3,)
    """
    wrapper, X = _wrap(estimator, views)
    scores: np.ndarray = skms.cross_val_score(
        wrapper, X, scoring=_unwrapped_scoring(kwargs.pop("scoring", None)), **kwargs
    )
    return scores


def cross_validate(
    estimator: BaseEstimator, views: list[ArrayLike], **kwargs: Any
) -> dict[str, Any]:
    """Evaluate a multiview estimator by cross-validation.

    :func:`sklearn.model_selection.cross_validate` on a list of views.

    Args:
        estimator: A multiview estimator.
        views: Arrays of shape (n_samples, n_features_i), one per view.
        **kwargs: Passed to :func:`sklearn.model_selection.cross_validate`,
            such as ``cv``, ``return_train_score`` and ``return_estimator``.
            A ``scoring`` callable, or each in a dict, is called as
            ``scoring(estimator, views)``.

    Returns:
        sklearn's result dictionary; with ``return_estimator``, the fitted
        multiview estimators.
    """
    wrapper, X = _wrap(estimator, views)
    results: dict[str, Any] = skms.cross_validate(
        wrapper, X, scoring=_unwrapped_scoring(kwargs.pop("scoring", None)), **kwargs
    )
    if "estimator" in results:
        results["estimator"] = [fitted.estimator_ for fitted in results["estimator"]]
    return results


def cross_val_predict(
    estimator: BaseEstimator, views: list[ArrayLike], **kwargs: Any
) -> list[np.ndarray]:
    """Out-of-fold scores of a multiview estimator.

    :func:`sklearn.model_selection.cross_val_predict` with ``method="transform"``
    on a list of views: each sample is transformed by the model fitted without
    its fold. Each fold's model fixes its own sign and order of components, so
    compare views within a component (e.g. with
    :func:`cca_zoo.metrics.pairwise_correlations`) rather than scores across folds.

    Args:
        estimator: A multiview estimator.
        views: Arrays of shape (n_samples, n_features_i), one per view.
        **kwargs: Passed to :func:`sklearn.model_selection.cross_val_predict`,
            such as ``cv``, ``groups`` and ``n_jobs``.

    Returns:
        Arrays of shape (n_samples, n_components), one per view.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import cross_val_predict
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 5)), rng.standard_normal((50, 4))
        >>> [s.shape for s in cross_val_predict(CCA(n_components=2), [X1, X2], cv=3)]
        [(50, 2), (50, 2)]
    """
    wrapper, X = _wrap(estimator, views)
    # sklearn's validation insists on ``predict`` even with method="transform".
    with sklearn.config_context(skip_parameter_validation=True):
        scores = skms.cross_val_predict(wrapper, X, method="transform", **kwargs)
    return np.split(scores, len(views), axis=1)


def learning_curve(
    estimator: BaseEstimator, views: list[ArrayLike], **kwargs: Any
) -> tuple[np.ndarray, ...]:
    """Cross-validated scores of a multiview estimator against training set size.

    :func:`sklearn.model_selection.learning_curve` on a list of views.

    Args:
        estimator: A multiview estimator.
        views: Arrays of shape (n_samples, n_features_i), one per view.
        **kwargs: Passed to :func:`sklearn.model_selection.learning_curve`,
            such as ``train_sizes`` and ``cv``.

    Returns:
        sklearn's ``(train_sizes, train_scores, test_scores, ...)``.
    """
    wrapper, X = _wrap(estimator, views)
    curve: tuple[np.ndarray, ...] = skms.learning_curve(
        wrapper,
        X,
        None,
        scoring=_unwrapped_scoring(kwargs.pop("scoring", None)),
        **kwargs,
    )
    return curve


def validation_curve(
    estimator: BaseEstimator,
    views: list[ArrayLike],
    param_name: str,
    param_range: ArrayLike,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-validated scores of a multiview estimator across one parameter's values.

    :func:`sklearn.model_selection.validation_curve` on a list of views.
    ``param_name`` may set one view's value of a per-view parameter, e.g.
    ``"c__0"``.

    Args:
        estimator: A multiview estimator.
        views: Arrays of shape (n_samples, n_features_i), one per view.
        param_name: Parameter to vary.
        param_range: Values to try.
        **kwargs: Passed to :func:`sklearn.model_selection.validation_curve`,
            such as ``cv``.

    Returns:
        ``(train_scores, test_scores)``, each of shape (n_values, n_folds).

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import validation_curve
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 5)), rng.standard_normal((50, 4))
        >>> train, test = validation_curve(
        ...     RidgeCCA(), [X1, X2], "c__0", [0.0, 0.5, 1.0], cv=3
        ... )
        >>> test.shape
        (3, 3)
    """
    wrapper, X = _wrap(estimator, views)
    train_scores, test_scores = skms.validation_curve(
        wrapper,
        X,
        None,
        param_name=_PARAM_PREFIX + param_name,
        param_range=param_range,
        scoring=_unwrapped_scoring(kwargs.pop("scoring", None)),
        **kwargs,
    )
    return train_scores, test_scores
