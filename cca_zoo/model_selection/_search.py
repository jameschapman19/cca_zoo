"""Sklearn-idiomatic hyperparameter search for multiview CCA models.

Every multiview estimator in cca_zoo already subclasses
:class:`sklearn.base.BaseEstimator`, so ``get_params``/``set_params``/``clone``
work out of the box. The one thing standing between these models and sklearn's
model-selection tools is the calling convention: ``fit``/``transform``/``score``
take a ``list[ArrayLike]`` of per-view arrays, not a single 2-D ``X``, so
:class:`sklearn.model_selection.GridSearchCV` and friends can't split folds
correctly on it.

:class:`MultiviewWrapper` bridges that gap: it concatenates the views into one
2-D array (so sklearn can index rows/folds normally) and splits them back
before delegating to the wrapped multiview estimator. It is a plain
``BaseEstimator``, so once wrapped, *any* sklearn tool applies directly:
``GridSearchCV``, ``RandomizedSearchCV``, ``HalvingGridSearchCV``,
``cross_val_score``, ``cross_validate``, ``learning_curve``, ``Pipeline``, etc.

:class:`GridSearchCV` and :class:`RandomizedSearchCV` below are thin
convenience wrappers that do this concatenation automatically and delegate
the actual search to :class:`sklearn.model_selection.GridSearchCV` /
:class:`sklearn.model_selection.RandomizedSearchCV`, so all of sklearn's
search machinery (parallelism, scoring, ``cv_results_``, multimetric support,
...) is reused rather than reimplemented.
"""

from __future__ import annotations

import re
from typing import Any, cast

import numpy as np
import sklearn.model_selection as skms
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone

_PARAM_PREFIX = "estimator__"
_VIEW_PARAM_RE = re.compile(r"^(.+)__(\d+)$")


class MultiviewWrapper(BaseEstimator):
    """Adapt a multiview estimator to sklearn's single-``X`` estimator API.

    Sklearn's model-selection tools (``GridSearchCV``, ``cross_val_score``,
    ``Pipeline``, ...) require an estimator whose ``fit``/``score`` accept
    ``(X, y)`` with ``X`` a single 2-D array, so they can index and split rows
    into folds. This wrapper concatenates the views' feature axes into one
    array on the way in and splits them back into views on the way out, so
    any sklearn tool that only ever sees the concatenated array can be used
    unmodified with a cca_zoo multiview estimator.

    Args:
        estimator: A multiview CCA estimator (e.g. :class:`~cca_zoo.linear.CCA`).
        split_indices: Number of features in each view, in order. Used to
            split the concatenated array back into views.

    Example:
        >>> import numpy as np
        >>> from sklearn.model_selection import cross_val_score
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import MultiviewWrapper
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 5)), rng.standard_normal((50, 4))
        >>> wrapper = MultiviewWrapper(CCA(), split_indices=[5, 4])
        >>> scores = cross_val_score(wrapper, np.hstack([X1, X2]), cv=3)

    Per-view hyperparameters (a per-view CCA model accepts a scalar,
    broadcast to every view, or an explicit list with one value per view --
    e.g. ``KCCA(c=[0.01, 0.1])``) can be searched independently per view
    through ``set_params`` using a ``name__<view index>`` suffix, e.g.
    ``estimator__c__0``. This is mainly useful for grid/randomized search: a
    param grid of ``{"c__0": [0.01, 0.1], "c__1": [0.1, 1.0]}`` makes
    sklearn's ``ParameterGrid`` search the Cartesian product of the two
    views' values. Indices left unset keep the estimator's current value for
    that view (broadcast if it was a scalar).
    """

    def __init__(self, estimator: BaseEstimator, split_indices: list[int]) -> None:
        self.estimator = estimator
        self.split_indices = split_indices

    def _split_views(self, X: np.ndarray) -> list[np.ndarray]:
        """Split a concatenated matrix back into individual views."""
        views = []
        start = 0
        for p in self.split_indices:
            views.append(X[:, start : start + p])
            start += p
        return views

    def set_params(self, **params: Any) -> MultiviewWrapper:
        """Set parameters, honouring per-view ``name__<view index>`` overrides.

        Args:
            **params: Parameter names/values. A key of the form
                ``estimator__<name>__<index>`` sets view ``<index>``'s
                value of the wrapped estimator's ``<name>`` parameter
                without disturbing the other views.
        """
        own: dict[str, Any] = {}
        inner: dict[str, Any] = {}
        for key, value in params.items():
            if key.startswith(_PARAM_PREFIX):
                inner[key[len(_PARAM_PREFIX) :]] = value
            else:
                own[key] = value
        if own:
            super().set_params(**own)
        if inner:
            self._set_inner_params(**inner)
        return self

    def _set_inner_params(self, **inner_params: Any) -> None:
        """Apply the wrapped estimator's params, expanding per-view keys."""
        direct: dict[str, Any] = {}
        per_view: dict[str, dict[int, Any]] = {}
        for key, value in inner_params.items():
            match = _VIEW_PARAM_RE.match(key)
            if match:
                name, current = match.group(1), getattr(
                    self.estimator, match.group(1), None
                )
                if not hasattr(current, "set_params"):
                    per_view.setdefault(name, {})[int(match.group(2))] = value
                    continue
            direct[key] = value

        if direct:
            self.estimator.set_params(**direct)

        n_views = len(self.split_indices)
        for name, overrides in per_view.items():
            bad = sorted(idx for idx in overrides if idx >= n_views)
            if bad:
                raise ValueError(
                    f"Per-view parameter '{name}' has index/indices {bad} but "
                    f"there are only {n_views} views."
                )
            current = getattr(self.estimator, name)
            values = (
                list(current) if isinstance(current, list) else [current] * n_views
            )
            for idx, value in overrides.items():
                values[idx] = value
            self.estimator.set_params(**{name: values})

    def fit(
        self, X: np.ndarray, y: None = None, **fit_params: Any
    ) -> MultiviewWrapper:
        """Fit the wrapped estimator on the concatenated multiview data."""
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._split_views(X), **fit_params)
        return self

    def score(self, X: np.ndarray, y: None = None) -> float:
        """Mean canonical correlation over all latent dimensions."""
        scores: np.ndarray = self.estimator_.score(self._split_views(X))
        return float(scores.mean())

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform and re-concatenate, so the wrapper composes with Pipeline."""
        transformed = self.estimator_.transform(self._split_views(X))
        return np.hstack(transformed)


def _wrap_param_space(
    param_space: dict[str, list[Any]] | list[dict[str, list[Any]]],
) -> dict[str, list[Any]] | list[dict[str, list[Any]]]:
    """Prefix a param_grid/param_distributions' keys with ``estimator__``."""
    if isinstance(param_space, dict):
        return {f"{_PARAM_PREFIX}{k}": v for k, v in param_space.items()}
    return [
        {f"{_PARAM_PREFIX}{k}": v for k, v in space.items()} for space in param_space
    ]


def _unprefix(key: str) -> str:
    return key[len(_PARAM_PREFIX) :] if key.startswith(_PARAM_PREFIX) else key


def _unwrap_cv_results(cv_results: dict[str, Any]) -> dict[str, Any]:
    """Strip the ``estimator__`` prefix from ``cv_results_`` keys/params.

    Without this, ``cv_results_["param_c"]`` doesn't exist -- only
    ``cv_results_["param_estimator__c"]`` does -- which is inconsistent with
    the unprefixed keys in ``best_params_``.
    """
    unwrapped: dict[str, Any] = {}
    for key, value in cv_results.items():
        if key.startswith(f"param_{_PARAM_PREFIX}"):
            unwrapped[f"param_{_unprefix(key[len('param_') :])}"] = value
        elif key == "params":
            unwrapped[key] = [
                {_unprefix(k): v for k, v in params.items()} for params in value
            ]
        else:
            unwrapped[key] = value
    return unwrapped


def _copy_fitted_attrs(target: Any, inner: BaseEstimator) -> None:
    """Copy every fitted (trailing-underscore) attribute from ``inner``.

    ``cv_results_``, ``best_params_`` and ``best_estimator_`` need their
    ``estimator__`` prefix undone; every other fitted attribute (``best_score_``,
    ``best_index_``, ``scorer_``, ``n_splits_``, ``refit_time_``,
    ``multimetric_``, ...) is forwarded as-is, so new sklearn attributes are
    picked up automatically instead of needing to be listed by hand here.
    """
    special = {"cv_results_", "best_params_", "best_estimator_"}
    for attr, value in vars(inner).items():
        if attr in special or not attr.endswith("_") or attr.endswith("__"):
            continue
        setattr(target, attr, value)

    if "cv_results_" in vars(inner):
        target.cv_results_ = _unwrap_cv_results(inner.cv_results_)
    if "best_params_" in vars(inner):
        target.best_params_ = {_unprefix(k): v for k, v in inner.best_params_.items()}
    if "best_estimator_" in vars(inner):
        target.best_estimator_ = inner.best_estimator_.estimator_


class _MultiviewSearchMixin:
    """Shared ``transform``/``score`` for the wrapped multiview search classes."""

    refit: bool
    _inner_cv: BaseEstimator
    best_estimator_: BaseEstimator

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Transform multiview data using the best estimator.

        Raises:
            AttributeError: If ``refit=False``, so no ``best_estimator_``
                was fitted.
        """
        if not self.refit:
            raise AttributeError(
                "`transform` is not available when `refit=False`; no "
                "`best_estimator_` was fitted. Set `refit=True` to use it."
            )
        return cast(list[np.ndarray], self.best_estimator_.transform(views))

    def score(self, views: list[ArrayLike], y: None = None) -> float:
        """Score the best estimator on held-out multiview data."""
        x_concat = np.hstack([np.asarray(v) for v in views])
        return float(self._inner_cv.score(x_concat, y))


class GridSearchCV(_MultiviewSearchMixin, BaseEstimator):
    """Exhaustive grid search with cross-validation for multiview CCA models.

    A thin multiview adapter around
    :class:`sklearn.model_selection.GridSearchCV`: views are horizontally
    stacked into one array via :class:`MultiviewWrapper`, and the actual
    search (candidate generation, parallel fold evaluation, ``cv_results_``,
    refitting, ...) is entirely sklearn's.

    Args:
        estimator: A multiview CCA estimator (e.g.
            :class:`~cca_zoo.linear.CCA`).
        param_grid: Dictionary or list of dictionaries with parameter
            names as keys and lists of parameter settings as values.
        cv: Number of cross-validation folds or a cross-validation
            splitter.  Default is 5.
        scoring: Scoring strategy.  When ``None`` the estimator's
            default :meth:`score` method is used.
        n_jobs: Number of jobs to run in parallel. Default is ``None``
            (sequential).
        refit: Whether to refit the best estimator on the full dataset.
            Default is ``True``.
        verbose: Verbosity level. Default is 0.
        pre_dispatch: Controls the number of jobs dispatched during
            parallel execution, forwarded to sklearn's ``GridSearchCV``.
        error_score: Value to assign to the score if fitting a candidate
            raises an exception, forwarded to sklearn's ``GridSearchCV``.
        return_train_score: If ``True``, ``cv_results_`` also includes
            training scores.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import GridSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> gs = GridSearchCV(
        ...     CCA(), param_grid={"latent_dimensions": [1, 2]}, cv=2
        ... )
        >>> gs = gs.fit([X1, X2])
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
        *,
        cv: int | Any = 5,
        scoring: str | None = None,
        n_jobs: int | None = None,
        refit: bool = True,
        verbose: int = 0,
        pre_dispatch: str | int = "2*n_jobs",
        error_score: float = np.nan,
        return_train_score: bool = False,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        **fit_params: Any,
    ) -> GridSearchCV:
        """Run grid search with cross-validation on multiview data.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
                All arrays must have the same number of rows.
            y: Ignored.
            **fit_params: Additional keyword arguments forwarded to the
                estimator's ``fit`` method during each fold.

        Returns:
            self: Fitted grid search object.
        """
        arrays = [np.asarray(v) for v in views]
        wrapped_estimator = MultiviewWrapper(
            estimator=self.estimator,
            split_indices=[a.shape[1] for a in arrays],
        )
        self._inner_cv = skms.GridSearchCV(
            estimator=wrapped_estimator,
            param_grid=_wrap_param_space(self.param_grid),
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )
        self._inner_cv.fit(np.hstack(arrays), y, **fit_params)
        _copy_fitted_attrs(self, self._inner_cv)
        return self


class RandomizedSearchCV(_MultiviewSearchMixin, BaseEstimator):
    """Randomized search with cross-validation for multiview CCA models.

    Samples ``n_iter`` parameter settings from ``param_distributions``
    instead of exhaustively trying every combination in a grid -- useful
    when a hyperparameter (e.g. ``c``) is continuous, or when the grid is
    too large to search exhaustively. A thin multiview adapter around
    :class:`sklearn.model_selection.RandomizedSearchCV`, following the same
    :class:`MultiviewWrapper` pattern as :class:`GridSearchCV`.

    Args:
        estimator: A multiview CCA estimator (e.g.
            :class:`~cca_zoo.linear.CCA`).
        param_distributions: Dictionary (or list of dictionaries) with
            parameter names as keys and either a list of values to sample
            from, or a distribution (anything with a ``rvs`` method, e.g.
            ``scipy.stats.loguniform``).
        n_iter: Number of parameter settings sampled. Default is 10.
        cv: Number of cross-validation folds or a cross-validation
            splitter.  Default is 5.
        scoring: Scoring strategy.  When ``None`` the estimator's
            default :meth:`score` method is used.
        n_jobs: Number of jobs to run in parallel. Default is ``None``
            (sequential).
        refit: Whether to refit the best estimator on the full dataset.
            Default is ``True``.
        verbose: Verbosity level. Default is 0.
        random_state: Controls the randomness of the parameter sampling.
        pre_dispatch: Controls the number of jobs dispatched during
            parallel execution, forwarded to sklearn's ``RandomizedSearchCV``.
        error_score: Value to assign to the score if fitting a candidate
            raises an exception, forwarded to sklearn's ``RandomizedSearchCV``.
        return_train_score: If ``True``, ``cv_results_`` also includes
            training scores.

    Example:
        >>> import numpy as np
        >>> from scipy.stats import loguniform
        >>> from cca_zoo.linear import rCCA
        >>> from cca_zoo.model_selection import RandomizedSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> rs = RandomizedSearchCV(
        ...     rCCA(),
        ...     param_distributions={"c": loguniform(1e-3, 1.0)},
        ...     n_iter=5,
        ...     cv=2,
        ...     random_state=0,
        ... )
        >>> rs = rs.fit([X1, X2])
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any] | list[dict[str, Any]],
        *,
        n_iter: int = 10,
        cv: int | Any = 5,
        scoring: str | None = None,
        n_jobs: int | None = None,
        refit: bool = True,
        verbose: int = 0,
        random_state: int | Any = None,
        pre_dispatch: str | int = "2*n_jobs",
        error_score: float = np.nan,
        return_train_score: bool = False,
    ) -> None:
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        **fit_params: Any,
    ) -> RandomizedSearchCV:
        """Run randomized search with cross-validation on multiview data.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
                All arrays must have the same number of rows.
            y: Ignored.
            **fit_params: Additional keyword arguments forwarded to the
                estimator's ``fit`` method during each fold.

        Returns:
            self: Fitted randomized search object.
        """
        arrays = [np.asarray(v) for v in views]
        wrapped_estimator = MultiviewWrapper(
            estimator=self.estimator,
            split_indices=[a.shape[1] for a in arrays],
        )
        self._inner_cv = skms.RandomizedSearchCV(
            estimator=wrapped_estimator,
            param_distributions=_wrap_param_space(self.param_distributions),
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            verbose=self.verbose,
            random_state=self.random_state,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )
        self._inner_cv.fit(np.hstack(arrays), y, **fit_params)
        _copy_fitted_attrs(self, self._inner_cv)
        return self
