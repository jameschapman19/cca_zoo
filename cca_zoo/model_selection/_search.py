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

:class:`GridSearchCV`, :class:`RandomizedSearchCV`, :class:`HalvingGridSearchCV`
and :class:`HalvingRandomSearchCV` below are thin convenience wrappers that do
this concatenation automatically and delegate the actual search to the
corresponding :mod:`sklearn.model_selection` class, so all of sklearn's search
machinery (parallelism, scoring, ``cv_results_``, successive-halving resource
allocation, multimetric support, ...) is reused rather than reimplemented.
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

    Examples:
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

        n_views = len(self.split_indices)
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


def one_standard_error(param: str) -> Callable[[dict[str, Any]], int]:
    r"""``refit`` rule: the simplest candidate within one standard error of the best.

    The one-standard-error rule of ``rpart`` and ``glmnet``'s ``lambda.1se``,
    as a callable for any search class's ``refit`` argument (sklearn calls it
    with ``cv_results_`` and refits the candidate whose index it returns).
    The best candidate has the highest mean test score; every candidate
    whose mean falls short of it by no more than one standard error is
    eligible, and the one with the smallest value of ``param`` wins. Taking
    the bare maximum instead is biased towards complex models whenever
    scores are noisy, since the largest of many noisy estimates sits high.

    The standard error is that of each candidate's *paired* per-split
    difference from the best candidate,
    $\operatorname{sd}_s(\text{score}_{s,c} - \text{score}_{s,\text{best}}) /
    \sqrt{n_\text{splits}}$, not of its raw per-split scores: a split that
    is simply harder (or on which a greedy model settles on a worse path)
    shifts every candidate alike, and that shared offset would otherwise
    inflate the error and pick far too simple a model.

    Args:
        param: Name of the searched parameter that orders candidates by
            complexity, smaller being simpler (e.g. ``"nprune"``), as it
            appears in ``param_grid`` (per-view names like ``"c__0"`` work
            too).

    Returns:
        A callable mapping ``cv_results_`` to the index of the chosen
        candidate, for :class:`GridSearchCV` and :class:`RandomizedSearchCV`
        (and :mod:`sklearn.model_selection`'s own). The successive-halving
        searches pick their final candidate themselves and never call
        ``refit``, so they reject it. With a single split there is no
        standard error to estimate, and the rule reduces to the best mean.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.gam import MARSCCA
        >>> from cca_zoo.model_selection import GridSearchCV, one_standard_error
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((100, 5))
        >>> X2 = rng.standard_normal((100, 5))
        >>> gs = GridSearchCV(
        ...     MARSCCA(),
        ...     {"nprune": [2, 4, 8]},
        ...     cv=3,
        ...     refit=one_standard_error("nprune"),
        ... ).fit([X1, X2])
        >>> gs.best_params_["nprune"] in (2, 4, 8)
        True
    """

    def rule(cv_results: dict[str, Any]) -> int:
        results = _unwrap_cv_results(cv_results)
        n_splits = sum(
            re.fullmatch(r"split\d+_test_score", key) is not None for key in results
        )
        scores = np.array([results[f"split{s}_test_score"] for s in range(n_splits)])
        best = int(np.argmin(results["rank_test_score"]))
        diff = scores - scores[:, [best]]
        # One split has no spread to estimate: the rule is then plain argmax.
        se = (
            diff.std(axis=0, ddof=1) / np.sqrt(n_splits)
            if n_splits > 1
            else np.zeros(diff.shape[1])
        )
        eligible = np.flatnonzero(diff.mean(axis=0) + se >= 0)
        values = results[f"param_{param}"]
        return int(min(eligible, key=lambda c: values[c]))

    return rule


def _reject_callable_refit(refit: Any) -> None:
    """Successive halving never calls a callable ``refit``: fail, don't ignore it."""
    if callable(refit):
        raise TypeError(
            "Successive-halving searches choose their final candidate "
            "themselves and never call a callable `refit` (such as "
            "one_standard_error); use GridSearchCV or RandomizedSearchCV."
        )


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

    refit: bool | str | Callable[[dict[str, Any]], int]
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


class _BaseMultiviewSearchCV(_MultiviewSearchMixin, BaseEstimator):
    """Shared ``fit`` body for the wrapped multiview search classes.

    Both :class:`GridSearchCV` and :class:`RandomizedSearchCV` do exactly
    the same three things in ``fit``: wrap the estimator with
    :class:`MultiviewWrapper`, hand it to the corresponding
    ``sklearn.model_selection`` search class (``_inner_cv_cls``), and copy
    the fitted attributes back with the ``estimator__`` prefix undone. That
    shared plumbing lives here; each subclass's own ``fit`` only supplies
    its search-specific constructor kwargs (``param_grid`` vs.
    ``param_distributions``, ``n_iter``, ``random_state``, ...) and keeps
    its own full docstring, since sklearn's own search classes likewise
    don't share a public docstring via inheritance.
    """

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
            split_indices=[a.shape[1] for a in arrays],
        )
        self._inner_cv = self._inner_cv_cls(
            estimator=wrapped_estimator, **inner_cv_kwargs
        )
        self._inner_cv.fit(np.hstack(arrays), y, **fit_params)
        _copy_fitted_attrs(self, self._inner_cv)
        return self


class GridSearchCV(_BaseMultiviewSearchCV):
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
        refit: Whether to refit the best estimator on the full dataset,
            or a callable choosing which candidate to refit from
            ``cv_results_`` (e.g. :func:`one_standard_error`). Default is
            ``True``.
        verbose: Verbosity level. Default is 0.
        pre_dispatch: Controls the number of jobs dispatched during
            parallel execution, forwarded to sklearn's ``GridSearchCV``.
        error_score: Value to assign to the score if fitting a candidate
            raises an exception, forwarded to sklearn's ``GridSearchCV``.
        return_train_score: If ``True``, ``cv_results_`` also includes
            training scores.

    Examples:
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

        A per-view estimator parameter (e.g. :class:`~cca_zoo.linear.rCCA`'s
        ridge ``c``) can be searched independently per view with a
        ``name__<view index>`` suffix in ``param_grid``:

        >>> from cca_zoo.linear import rCCA
        >>> gs = GridSearchCV(
        ...     rCCA(), param_grid={"c__0": [0.0, 0.1], "c__1": [0.0, 0.5]}, cv=2
        ... )
        >>> gs = gs.fit([X1, X2])
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
        inner_cv_kwargs = dict(
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
        return cast("GridSearchCV", self._fit(views, y, inner_cv_kwargs, **fit_params))


class RandomizedSearchCV(_BaseMultiviewSearchCV):
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
        refit: Whether to refit the best estimator on the full dataset,
            or a callable choosing which candidate to refit from
            ``cv_results_`` (e.g. :func:`one_standard_error`). Default is
            ``True``.
        verbose: Verbosity level. Default is 0.
        random_state: Controls the randomness of the parameter sampling.
        pre_dispatch: Controls the number of jobs dispatched during
            parallel execution, forwarded to sklearn's ``RandomizedSearchCV``.
        error_score: Value to assign to the score if fitting a candidate
            raises an exception, forwarded to sklearn's ``RandomizedSearchCV``.
        return_train_score: If ``True``, ``cv_results_`` also includes
            training scores.

    Examples:
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

        Per-view distributions use the same ``name__<view index>`` suffix
        as :class:`GridSearchCV`:

        >>> rs = RandomizedSearchCV(
        ...     rCCA(),
        ...     param_distributions={
        ...         "c__0": loguniform(1e-3, 1.0),
        ...         "c__1": loguniform(1e-3, 1.0),
        ...     },
        ...     n_iter=5,
        ...     cv=2,
        ...     random_state=0,
        ... )
        >>> rs = rs.fit([X1, X2])
        >>> sorted(rs.best_params_.keys())
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
        inner_cv_kwargs = dict(
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
        return cast(
            "RandomizedSearchCV", self._fit(views, y, inner_cv_kwargs, **fit_params)
        )


class HalvingGridSearchCV(_BaseMultiviewSearchCV):
    """Successive-halving grid search with cross-validation for multiview CCA models.

    Like :class:`GridSearchCV`, but candidates are evaluated on a growing
    subset of the training samples across rounds: most candidates are
    eliminated early on a small subset, and only the survivors are evaluated
    on progressively larger subsets, which is usually much cheaper than an
    exhaustive :class:`GridSearchCV` when the grid is large. A thin multiview
    adapter around :class:`sklearn.model_selection.HalvingGridSearchCV`,
    following the same :class:`MultiviewWrapper` pattern as
    :class:`GridSearchCV`; the "resource" being grown across rounds is a row
    count of the (already view-concatenated) training array, so the
    successive-halving mechanics need no multiview-specific handling.

    Args:
        estimator: A multiview CCA estimator (e.g.
            :class:`~cca_zoo.linear.CCA`).
        param_grid: Dictionary or list of dictionaries with parameter
            names as keys and lists of parameter settings as values.
        factor: The proportion of candidates eliminated (and resources
            multiplied by) at each round. Default is 3.
        resource: The resource grown between rounds, forwarded to sklearn's
            ``HalvingGridSearchCV``. Default is ``"n_samples"``.
        max_resources: The maximum amount of resource a candidate is
            allowed to use, forwarded to sklearn's ``HalvingGridSearchCV``.
            Default is ``"auto"``.
        min_resources: The minimum amount of resource a candidate is
            allowed to use, forwarded to sklearn's ``HalvingGridSearchCV``.
            Default is ``"exhaust"``.
        aggressive_elimination: Whether to eliminate candidates at the same
            rate even before there are enough resources to grow, forwarded
            to sklearn's ``HalvingGridSearchCV``. Default is ``False``.
        cv: Number of cross-validation folds or a cross-validation
            splitter.  Default is 5.
        scoring: Scoring strategy.  When ``None`` the estimator's
            default :meth:`score` method is used.
        refit: Whether to refit the best estimator on the full dataset.
            Unlike :class:`GridSearchCV`, not a callable: successive halving
            picks its final candidate itself and would ignore it. Default is
            ``True``.
        error_score: Value to assign to the score if fitting a candidate
            raises an exception, forwarded to sklearn's
            ``HalvingGridSearchCV``.
        return_train_score: If ``True``, ``cv_results_`` also includes
            training scores. Default is ``True``, matching sklearn's
            ``HalvingGridSearchCV``.
        random_state: Controls the pseudo-random subsampling of the
            training set that determines the candidates' resources at each
            round.
        n_jobs: Number of jobs to run in parallel. Default is ``None``
            (sequential).
        verbose: Verbosity level. Default is 0.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.model_selection import HalvingGridSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> hgs = HalvingGridSearchCV(
        ...     CCA(), param_grid={"latent_dimensions": [1, 2]}, cv=2
        ... )
        >>> hgs = hgs.fit([X1, X2])

        Per-view parameters use the same ``name__<view index>`` suffix as
        :class:`GridSearchCV`:

        >>> from cca_zoo.linear import rCCA
        >>> hgs = HalvingGridSearchCV(
        ...     rCCA(),
        ...     param_grid={"c__0": [0.0, 0.1], "c__1": [0.0, 0.5]},
        ...     cv=2,
        ...     random_state=0,
        ... )
        >>> hgs = hgs.fit([X1, X2])
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
        """Run successive-halving grid search with cross-validation.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
                All arrays must have the same number of rows.
            y: Ignored.
            **fit_params: Additional keyword arguments forwarded to the
                estimator's ``fit`` method during each fold.

        Returns:
            self: Fitted search object.
        """
        _reject_callable_refit(self.refit)
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
    """Successive-halving randomized search with cross-validation for multiview models.

    Combines :class:`RandomizedSearchCV`'s sampling of ``param_distributions``
    with :class:`HalvingGridSearchCV`'s successive-halving elimination: most
    sampled candidates are eliminated early on a small subset of the training
    samples, and only the survivors are evaluated on progressively larger
    subsets. A thin multiview adapter around
    :class:`sklearn.model_selection.HalvingRandomSearchCV`, following the
    same :class:`MultiviewWrapper` pattern as :class:`RandomizedSearchCV`.

    Args:
        estimator: A multiview CCA estimator (e.g.
            :class:`~cca_zoo.linear.CCA`).
        param_distributions: Dictionary (or list of dictionaries) with
            parameter names as keys and either a list of values to sample
            from, or a distribution (anything with a ``rvs`` method, e.g.
            ``scipy.stats.loguniform``).
        n_candidates: The number of candidate parameters to sample,
            forwarded to sklearn's ``HalvingRandomSearchCV``. Default is
            ``"exhaust"``.
        factor: The proportion of candidates eliminated (and resources
            multiplied by) at each round. Default is 3.
        resource: The resource grown between rounds, forwarded to sklearn's
            ``HalvingRandomSearchCV``. Default is ``"n_samples"``.
        max_resources: The maximum amount of resource a candidate is
            allowed to use, forwarded to sklearn's ``HalvingRandomSearchCV``.
            Default is ``"auto"``.
        min_resources: The minimum amount of resource a candidate is
            allowed to use, forwarded to sklearn's ``HalvingRandomSearchCV``.
            Default is ``"smallest"``.
        aggressive_elimination: Whether to eliminate candidates at the same
            rate even before there are enough resources to grow, forwarded
            to sklearn's ``HalvingRandomSearchCV``. Default is ``False``.
        cv: Number of cross-validation folds or a cross-validation
            splitter.  Default is 5.
        scoring: Scoring strategy.  When ``None`` the estimator's
            default :meth:`score` method is used.
        refit: Whether to refit the best estimator on the full dataset.
            Unlike :class:`GridSearchCV`, not a callable: successive halving
            picks its final candidate itself and would ignore it. Default is
            ``True``.
        error_score: Value to assign to the score if fitting a candidate
            raises an exception, forwarded to sklearn's
            ``HalvingRandomSearchCV``.
        return_train_score: If ``True``, ``cv_results_`` also includes
            training scores. Default is ``True``, matching sklearn's
            ``HalvingRandomSearchCV``.
        random_state: Controls both the randomness of the parameter
            sampling and the pseudo-random subsampling of the training set.
        n_jobs: Number of jobs to run in parallel. Default is ``None``
            (sequential).
        verbose: Verbosity level. Default is 0.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import loguniform
        >>> from cca_zoo.linear import rCCA
        >>> from cca_zoo.model_selection import HalvingRandomSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> hrs = HalvingRandomSearchCV(
        ...     rCCA(),
        ...     param_distributions={"c": loguniform(1e-3, 1.0)},
        ...     cv=2,
        ...     random_state=0,
        ... )
        >>> hrs = hrs.fit([X1, X2])

        Per-view distributions use the same ``name__<view index>`` suffix
        as :class:`RandomizedSearchCV`:

        >>> hrs = HalvingRandomSearchCV(
        ...     rCCA(),
        ...     param_distributions={
        ...         "c__0": loguniform(1e-3, 1.0),
        ...         "c__1": loguniform(1e-3, 1.0),
        ...     },
        ...     cv=2,
        ...     random_state=0,
        ... )
        >>> hrs = hrs.fit([X1, X2])
        >>> sorted(hrs.best_params_.keys())
        ['c__0', 'c__1']
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
        """Run successive-halving randomized search with cross-validation.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
                All arrays must have the same number of rows.
            y: Ignored.
            **fit_params: Additional keyword arguments forwarded to the
                estimator's ``fit`` method during each fold.

        Returns:
            self: Fitted search object.
        """
        _reject_callable_refit(self.refit)
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
