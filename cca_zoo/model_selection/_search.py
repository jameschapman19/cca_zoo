"""Hyperparameter search for multiview estimators.

sklearn's model-selection tools index a single 2-D ``X``.
:class:`MultiviewWrapper` concatenates the views into one array and splits
them back, so any sklearn tool applies. The search classes here wrap their
:mod:`sklearn.model_selection` namesakes to do this automatically.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, ClassVar, cast

import numpy as np
import sklearn.model_selection as skms
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401

_PARAM_PREFIX = "estimator__"
_VIEW_PARAM_RE = re.compile(r"^(.+)__(\d+)$")


class MultiviewWrapper(BaseEstimator):
    """Adapt a multiview estimator to sklearn's single-``X`` API.

    Views are concatenated along the feature axis on the way in and split
    back on the way out, so ``cross_val_score``, ``Pipeline`` and the
    sklearn searches work unmodified.

    A per-view parameter can be set for one view with a ``name__<view>``
    suffix, e.g. ``estimator__c__0``; unset views keep their current value.
    This lets a grid such as ``{"c__0": [0.01, 0.1], "c__1": [0.1, 1.0]}``
    search the views independently.

    Args:
        estimator: A multiview estimator.
        n_features_per_view: Number of features in each view, in order.

    Attributes:
        estimator_: The fitted multiview estimator.

    Examples:
        >>> import numpy as np
        >>> from sklearn.model_selection import cross_val_score
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import MultiviewWrapper
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 5)), rng.standard_normal((50, 4))
        >>> wrapper = MultiviewWrapper(CCA(), n_features_per_view=[5, 4])
        >>> scores = cross_val_score(wrapper, np.hstack([X1, X2]), cv=3)
    """

    def __init__(
        self, estimator: BaseEstimator, n_features_per_view: list[int]
    ) -> None:
        self.estimator = estimator
        self.n_features_per_view = n_features_per_view

    def _split_views(self, X: np.ndarray) -> list[np.ndarray]:
        """Split a concatenated matrix back into individual views."""
        views = []
        start = 0
        for p in self.n_features_per_view:
            views.append(X[:, start : start + p])
            start += p
        return views

    def set_params(self, **params: Any) -> MultiviewWrapper:
        """Set parameters, including per-view ``estimator__<name>__<view>`` keys.

        Args:
            **params: Parameter names and values.

        Returns:
            self.
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
                name, current = (
                    match.group(1),
                    getattr(self.estimator, match.group(1), None),
                )
                if not hasattr(current, "set_params"):
                    per_view.setdefault(name, {})[int(match.group(2))] = value
                    continue
            direct[key] = value

        if direct:
            self.estimator.set_params(**direct)

        n_views = len(self.n_features_per_view)
        for name, overrides in per_view.items():
            bad = sorted(idx for idx in overrides if idx >= n_views)
            if bad:
                raise ValueError(
                    f"Per-view parameter '{name}' has index/indices {bad} but "
                    f"there are only {n_views} views."
                )
            current = getattr(self.estimator, name)
            values = list(current) if isinstance(current, list) else [current] * n_views
            for idx, value in overrides.items():
                values[idx] = value
            self.estimator.set_params(**{name: values})

    def fit(self, X: np.ndarray, y: None = None, **fit_params: Any) -> MultiviewWrapper:
        """Fit the wrapped estimator on the concatenated multiview data."""
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._split_views(X), **fit_params)
        return self

    def score(self, X: np.ndarray, y: None = None) -> float:
        """Mean canonical correlation over all latent dimensions."""
        return float(self.estimator_.score(self._split_views(X)))

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
    """``cv_results_`` with the ``estimator__`` prefix removed from its keys."""
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


def _unwrapped_refit(refit: Any) -> Any:
    """Hand a callable ``refit`` the unprefixed ``cv_results_`` users see."""
    if not callable(refit):
        return refit
    return lambda cv_results: refit(_unwrap_cv_results(cv_results))


def _copy_fitted_attrs(target: Any, inner: BaseEstimator) -> None:
    """Copy the fitted attributes of ``inner``, unprefixing the parameter names."""
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

    refit: bool | str | Callable[[dict[str, Any]], int]
    _inner_cv: BaseEstimator
    best_estimator_: BaseEstimator

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Transform multiview data with the best estimator.

        Raises:
            AttributeError: If fitted with ``refit=False``.
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


class _BaseMultiviewSearchCV(_MultiviewSearchMixin, BaseEstimator):
    """Shared ``fit``: wrap the estimator, run ``_inner_cv_cls``, copy results back."""

    estimator: BaseEstimator
    _inner_cv_cls: ClassVar[type[BaseEstimator]]

    def _fit(
        self,
        views: list[ArrayLike],
        y: None,
        inner_cv_kwargs: dict[str, Any],
        **fit_params: Any,
    ) -> _BaseMultiviewSearchCV:
        arrays = [np.asarray(v) for v in views]
        wrapped_estimator = MultiviewWrapper(
            estimator=self.estimator,
            n_features_per_view=[a.shape[1] for a in arrays],
        )
        self._inner_cv = self._inner_cv_cls(
            estimator=wrapped_estimator, **inner_cv_kwargs
        )
        self._inner_cv.fit(np.hstack(arrays), y, **fit_params)
        _copy_fitted_attrs(self, self._inner_cv)
        return self


class GridSearchCV(_BaseMultiviewSearchCV):
    """Exhaustive grid search over a multiview estimator's parameters.

    A multiview adapter for :class:`sklearn.model_selection.GridSearchCV`;
    parameter names are unprefixed, and ``name__<view>`` searches one view's
    value of a per-view parameter.

    Args:
        estimator: A multiview estimator.
        param_grid: Dict, or list of dicts, of parameter values to try.
        cv: Number of folds or a splitter. Default is 5.
        scoring: Scoring strategy; ``None`` uses the estimator's ``score``.
            Default is None.
        n_jobs: Number of parallel jobs. Default is None.
        refit: Whether to refit the best candidate on all data, or a callable
            choosing it from ``cv_results_``. Default is True.
        verbose: Verbosity level. Default is 0.
        pre_dispatch: Jobs dispatched during parallel execution. Default is
            ``"2*n_jobs"``.
        error_score: Score assigned when a fit fails. Default is ``np.nan``.
        return_train_score: Whether to include training scores. Default is
            False.

    Attributes:
        cv_results_: Per-candidate results, as in sklearn.
        best_estimator_: The refitted multiview estimator.
        best_params_: Parameters of the best candidate.
        best_score_: Mean cross-validated score of the best candidate.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import GridSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> gs = GridSearchCV(
        ...     RidgeCCA(), param_grid={"c__0": [0.0, 0.1], "c__1": [0.0, 0.5]}, cv=2
        ... ).fit([X1, X2])
        >>> sorted(gs.best_params_.items())
        [('c__0', 0.0), ('c__1', 0.5)]
    """

    _inner_cv_cls = skms.GridSearchCV

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
        *,
        cv: int | Any = 5,
        scoring: str | None = None,
        n_jobs: int | None = None,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
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
        """Run the search.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            **fit_params: Forwarded to the estimator's ``fit``.

        Returns:
            self.
        """
        inner_cv_kwargs = dict(
            param_grid=_wrap_param_space(self.param_grid),
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=_unwrapped_refit(self.refit),
            verbose=self.verbose,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )
        return cast("GridSearchCV", self._fit(views, y, inner_cv_kwargs, **fit_params))


class RandomizedSearchCV(_BaseMultiviewSearchCV):
    """Randomized search over a multiview estimator's parameters.

    A multiview adapter for :class:`sklearn.model_selection.RandomizedSearchCV`,
    with the same parameter naming as :class:`GridSearchCV`.

    Args:
        estimator: A multiview estimator.
        param_distributions: Dict, or list of dicts, of value lists or
            distributions with an ``rvs`` method.
        n_iter: Number of candidates sampled. Default is 10.
        cv: Number of folds or a splitter. Default is 5.
        scoring: Scoring strategy; ``None`` uses the estimator's ``score``.
            Default is None.
        n_jobs: Number of parallel jobs. Default is None.
        refit: Whether to refit the best candidate on all data, or a callable
            choosing it from ``cv_results_``. Default is True.
        verbose: Verbosity level. Default is 0.
        random_state: Seed for the parameter sampling. Default is None.
        pre_dispatch: Jobs dispatched during parallel execution. Default is
            ``"2*n_jobs"``.
        error_score: Score assigned when a fit fails. Default is ``np.nan``.
        return_train_score: Whether to include training scores. Default is
            False.

    Attributes:
        cv_results_: Per-candidate results, as in sklearn.
        best_estimator_: The refitted multiview estimator.
        best_params_: Parameters of the best candidate.
        best_score_: Mean cross-validated score of the best candidate.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import loguniform
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import RandomizedSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> rs = RandomizedSearchCV(
        ...     RidgeCCA(),
        ...     param_distributions={
        ...         "c__0": loguniform(1e-3, 1.0),
        ...         "c__1": loguniform(1e-3, 1.0),
        ...     },
        ...     n_iter=5,
        ...     cv=2,
        ...     random_state=0,
        ... ).fit([X1, X2])
        >>> sorted(rs.best_params_)
        ['c__0', 'c__1']
    """

    _inner_cv_cls = skms.RandomizedSearchCV

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any] | list[dict[str, Any]],
        *,
        n_iter: int = 10,
        cv: int | Any = 5,
        scoring: str | None = None,
        n_jobs: int | None = None,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
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
        """Run the search.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            **fit_params: Forwarded to the estimator's ``fit``.

        Returns:
            self.
        """
        inner_cv_kwargs = dict(
            param_distributions=_wrap_param_space(self.param_distributions),
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=_unwrapped_refit(self.refit),
            verbose=self.verbose,
            random_state=self.random_state,
            pre_dispatch=self.pre_dispatch,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
        )
        return cast(
            "RandomizedSearchCV", self._fit(views, y, inner_cv_kwargs, **fit_params)
        )


class HalvingGridSearchCV(_BaseMultiviewSearchCV):
    """Successive-halving grid search over a multiview estimator's parameters.

    A multiview adapter for :class:`sklearn.model_selection.HalvingGridSearchCV`,
    with the same parameter naming as :class:`GridSearchCV`.

    Args:
        estimator: A multiview estimator.
        param_grid: Dict, or list of dicts, of parameter values to try.
        factor: Candidate reduction and resource growth per round. Default is 3.
        resource: Resource grown between rounds. Default is ``"n_samples"``.
        max_resources: Maximum resource per candidate. Default is ``"auto"``.
        min_resources: Resource in the first round. Default is ``"exhaust"``.
        aggressive_elimination: Whether to eliminate candidates before
            resources can grow. Default is False.
        cv: Number of folds or a splitter. Default is 5.
        scoring: Scoring strategy; ``None`` uses the estimator's ``score``.
            Default is None.
        refit: Whether to refit the best candidate on all data. Default is True.
        error_score: Score assigned when a fit fails. Default is ``np.nan``.
        return_train_score: Whether to include training scores. Default is
            True.
        random_state: Seed for the per-round subsampling. Default is None.
        n_jobs: Number of parallel jobs. Default is None.
        verbose: Verbosity level. Default is 0.

    Attributes:
        cv_results_: Per-candidate results, as in sklearn.
        best_estimator_: The refitted multiview estimator.
        best_params_: Parameters of the best candidate.
        best_score_: Mean cross-validated score of the best candidate.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import HalvingGridSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> hgs = HalvingGridSearchCV(
        ...     RidgeCCA(),
        ...     param_grid={"c__0": [0.0, 0.1], "c__1": [0.0, 0.5]},
        ...     cv=2,
        ...     random_state=0,
        ... ).fit([X1, X2])
        >>> sorted(hgs.best_params_.items())
        [('c__0', 0.1), ('c__1', 0.0)]
    """

    _inner_cv_cls = skms.HalvingGridSearchCV

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
        *,
        factor: int | float = 3,
        resource: str = "n_samples",
        max_resources: int | str = "auto",
        min_resources: int | str = "exhaust",
        aggressive_elimination: bool = False,
        cv: int | Any = 5,
        scoring: str | None = None,
        refit: bool = True,
        error_score: float = np.nan,
        return_train_score: bool = True,
        random_state: int | Any = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.factor = factor
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        **fit_params: Any,
    ) -> HalvingGridSearchCV:
        """Run the search.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            **fit_params: Forwarded to the estimator's ``fit``.

        Returns:
            self.
        """
        inner_cv_kwargs = dict(
            param_grid=_wrap_param_space(self.param_grid),
            factor=self.factor,
            resource=self.resource,
            max_resources=self.max_resources,
            min_resources=self.min_resources,
            aggressive_elimination=self.aggressive_elimination,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        return cast(
            "HalvingGridSearchCV", self._fit(views, y, inner_cv_kwargs, **fit_params)
        )


class HalvingRandomSearchCV(_BaseMultiviewSearchCV):
    """Successive-halving randomized search over a multiview estimator's parameters.

    A multiview adapter for
    :class:`sklearn.model_selection.HalvingRandomSearchCV`, with the same
    parameter naming as :class:`GridSearchCV`.

    Args:
        estimator: A multiview estimator.
        param_distributions: Dict, or list of dicts, of value lists or
            distributions with an ``rvs`` method.
        n_candidates: Number of candidates sampled. Default is ``"exhaust"``.
        factor: Candidate reduction and resource growth per round. Default is 3.
        resource: Resource grown between rounds. Default is ``"n_samples"``.
        max_resources: Maximum resource per candidate. Default is ``"auto"``.
        min_resources: Resource in the first round. Default is ``"smallest"``.
        aggressive_elimination: Whether to eliminate candidates before
            resources can grow. Default is False.
        cv: Number of folds or a splitter. Default is 5.
        scoring: Scoring strategy; ``None`` uses the estimator's ``score``.
            Default is None.
        refit: Whether to refit the best candidate on all data. Default is True.
        error_score: Score assigned when a fit fails. Default is ``np.nan``.
        return_train_score: Whether to include training scores. Default is
            True.
        random_state: Seed for the sampling and subsampling. Default is None.
        n_jobs: Number of parallel jobs. Default is None.
        verbose: Verbosity level. Default is 0.

    Attributes:
        cv_results_: Per-candidate results, as in sklearn.
        best_estimator_: The refitted multiview estimator.
        best_params_: Parameters of the best candidate.
        best_score_: Mean cross-validated score of the best candidate.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import loguniform
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import HalvingRandomSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> hrs = HalvingRandomSearchCV(
        ...     RidgeCCA(),
        ...     param_distributions={"c": loguniform(1e-3, 1.0)},
        ...     cv=2,
        ...     random_state=0,
        ... ).fit([X1, X2])
    """

    _inner_cv_cls = skms.HalvingRandomSearchCV

    def __init__(
        self,
        estimator: BaseEstimator,
        param_distributions: dict[str, Any] | list[dict[str, Any]],
        *,
        n_candidates: int | str = "exhaust",
        factor: int | float = 3,
        resource: str = "n_samples",
        max_resources: int | str = "auto",
        min_resources: int | str = "smallest",
        aggressive_elimination: bool = False,
        cv: int | Any = 5,
        scoring: str | None = None,
        refit: bool = True,
        error_score: float = np.nan,
        return_train_score: bool = True,
        random_state: int | Any = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates
        self.factor = factor
        self.resource = resource
        self.max_resources = max_resources
        self.min_resources = min_resources
        self.aggressive_elimination = aggressive_elimination
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self,
        views: list[ArrayLike],
        y: None = None,
        **fit_params: Any,
    ) -> HalvingRandomSearchCV:
        """Run the search.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            **fit_params: Forwarded to the estimator's ``fit``.

        Returns:
            self.
        """
        inner_cv_kwargs = dict(
            param_distributions=_wrap_param_space(self.param_distributions),
            n_candidates=self.n_candidates,
            factor=self.factor,
            resource=self.resource,
            max_resources=self.max_resources,
            min_resources=self.min_resources,
            aggressive_elimination=self.aggressive_elimination,
            cv=self.cv,
            scoring=self.scoring,
            refit=self.refit,
            error_score=self.error_score,
            return_train_score=self.return_train_score,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        return cast(
            "HalvingRandomSearchCV", self._fit(views, y, inner_cv_kwargs, **fit_params)
        )
