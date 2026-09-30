"""Hyperparameter search for multiview estimators.

sklearn's model-selection tools index a single 2-D ``X``. The search classes
here wrap their :mod:`sklearn.model_selection` namesakes, stacking the views
into one array for sklearn and splitting them back for the estimator.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import sklearn.model_selection as skms
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.utils import Tags
from sklearn.utils.validation import (
    _get_feature_names,
    check_array,
    check_is_fitted,
    validate_data,
)

_PARAM_PREFIX = "estimator__"
_Scoring = Callable[..., float] | dict[str, Callable[..., float]] | None
_VIEW_PARAM_RE = re.compile(r"^(.+)__(\d+)$")


class _MultiviewWrapper(TransformerMixin, BaseEstimator):
    """A multiview estimator as an sklearn estimator of the stacked views.

    Views are concatenated along the feature axis on the way in and split
    back on the way out. ``estimator__<name>__<view>`` sets one view's value
    of a per-view parameter, keeping the other views' values.

    Args:
        estimator: A multiview estimator.
        n_features_per_view: Number of features in each view, in order.

    Attributes:
        estimator_: The fitted multiview estimator.
        n_features_in_: Total number of features across the views.
    """

    def __init__(
        self, estimator: BaseEstimator, n_features_per_view: list[int]
    ) -> None:
        self.estimator = estimator
        self.n_features_per_view = n_features_per_view

    def _split_views(self, X: ArrayLike, reset: bool) -> list[Any]:
        """Split a stacked array back into its views.

        Only the shape and feature names are checked here; the estimator
        validates the views. A DataFrame is split into DataFrames, so each
        view keeps its feature names.
        """
        stacked: np.ndarray = check_array(X, dtype=None, ensure_all_finite=False)
        validate_data(self, X, reset=reset, skip_check_array=True)
        edges = np.cumsum(self._view_widths(stacked.shape[1]))
        if hasattr(X, "iloc"):
            starts = np.concatenate([[0], edges[:-1]])
            return [X.iloc[:, a:b] for a, b in zip(starts, edges)]
        return np.split(stacked, edges[:-1], axis=1)

    def _view_widths(self, n_features: int) -> list[int]:
        """The width of each view."""
        return list(self.n_features_per_view)

    def set_params(self, **params: Any) -> _MultiviewWrapper:
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
        params = self.estimator.get_params()
        direct: dict[str, Any] = {}
        per_view: dict[str, dict[int, Any]] = {}
        for key, value in inner_params.items():
            match = _VIEW_PARAM_RE.match(key)
            if match and not hasattr(params.get(match.group(1)), "set_params"):
                per_view.setdefault(match.group(1), {})[int(match.group(2))] = value
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
            # Read after the whole-model values are set, which these override.
            current = self.estimator.get_params()[name]
            values = list(current) if isinstance(current, list) else [current] * n_views
            for idx, value in overrides.items():
                values[idx] = value
            self.estimator.set_params(**{name: values})

    def __sklearn_tags__(self) -> Tags:
        """The transformer tags, with the estimator's preserved dtypes."""
        tags = super().__sklearn_tags__()
        inner = self.estimator.__sklearn_tags__().transformer_tags
        if inner is not None:
            tags.transformer_tags.preserves_dtype = inner.preserves_dtype
        return tags

    def fit(
        self, X: np.ndarray, y: None = None, **fit_params: Any
    ) -> _MultiviewWrapper:
        """Fit the wrapped estimator on the concatenated multiview data."""
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._split_views(X, reset=True), **fit_params)
        return self

    def score(self, X: np.ndarray, y: None = None) -> float:
        """The estimator's score on the views of ``X``."""
        check_is_fitted(self)
        return float(self.estimator_.score(self._split_views(X, reset=False)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Each view's scores, side by side, so the wrapper composes with Pipeline."""
        check_is_fitted(self)
        return np.hstack(self.estimator_.transform(self._split_views(X, reset=False)))


def _stack(views: list[ArrayLike]) -> Any:
    """The views side by side: a DataFrame if every view has feature names.

    sklearn indexes one 2-D ``X``; a pandas DataFrame carries each view's
    names through its splits, whatever DataFrame library the views came
    from. Without pandas the names are dropped.
    """
    arrays = [np.asarray(v) for v in views]
    names = [_get_feature_names(v) for v in views]
    if all(n is not None for n in names) and importlib.util.find_spec("pandas"):
        import pandas as pd

        return pd.DataFrame(
            np.hstack(arrays),
            columns=np.concatenate(names),
            index=getattr(views[0], "index", None),
        )
    return np.hstack(arrays)


def _wrap(
    estimator: BaseEstimator, views: list[ArrayLike]
) -> tuple[_MultiviewWrapper, Any]:
    """``estimator`` wrapped for sklearn, and the views stacked by :func:`_stack`."""
    wrapper = _MultiviewWrapper(estimator, [np.shape(v)[1] for v in views])
    return wrapper, _stack(views)


def _unwrapped_scoring(scoring: Any) -> Any:
    """Hand a callable scorer, or a dict of them, the estimator and its views."""
    if isinstance(scoring, dict):
        return {name: _unwrapped_scoring(scorer) for name, scorer in scoring.items()}
    if not callable(scoring):
        return scoring

    def score(wrapper: _MultiviewWrapper, X: np.ndarray, y: None = None) -> float:
        return float(scoring(wrapper.estimator_, wrapper._split_views(X, reset=False)))

    return score


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


class _MultiviewSearch:
    """Mixin running the sklearn-style search class it precedes on lists of views.

    ``class GridSearchCV(_MultiviewSearch, skms.GridSearchCV)`` inherits the
    upstream constructor, so its parameters, ``get_params`` and ``clone`` are
    the upstream search's own. ``fit`` runs an upstream instance on the
    stacked views, with the estimator wrapped, the parameter names prefixed
    and ``scoring`` and ``refit`` adapted, and copies its results back with
    the prefix removed and the fitted multiview model as ``best_estimator_``.
    """

    refit: Any
    best_estimator_: BaseEstimator
    _search: BaseEstimator

    def fit(
        self, views: list[ArrayLike], y: None = None, **fit_params: Any
    ) -> _MultiviewSearch:
        """Run the search.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.
            **fit_params: Forwarded to the estimator's ``fit``.

        Returns:
            self.
        """
        mro = type(self).__mro__
        upstream = mro[mro.index(_MultiviewSearch) + 1]
        params = self.get_params(deep=False)  # type: ignore[attr-defined]
        params["estimator"], X = _wrap(params["estimator"], views)
        for space in ("param_grid", "param_distributions"):
            if space in params:
                params[space] = _wrap_param_space(params[space])
        params["scoring"] = _unwrapped_scoring(params["scoring"])
        params["refit"] = _unwrapped_refit(params["refit"])
        self._search = upstream(**params)
        self._search.fit(X, y, **fit_params)
        _copy_fitted_attrs(self, self._search)
        return self

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
        return float(self._search.score(_stack(views), y))


class GridSearchCV(_MultiviewSearch, skms.GridSearchCV):
    """Exhaustive grid search over a multiview estimator's parameters.

    The multiview form of :class:`~sklearn.model_selection.GridSearchCV`, with its
    parameters, attributes and methods. ``fit``, ``transform`` and ``score``
    take a list of views; parameter names are the estimator's own, with
    ``name__<view>`` setting one view's value of a per-view parameter;
    ``best_estimator_`` is the fitted multiview estimator; and a ``scoring``
    callable is called as ``scoring(estimator, views)``.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import GridSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> grid = {"shrinkage__0": [0.0, 0.1], "shrinkage__1": [0.0, 0.5]}
        >>> gs = GridSearchCV(RidgeCCA(), param_grid=grid, cv=2).fit([X1, X2])
        >>> sorted(gs.best_params_.items())
        [('shrinkage__0', 0.0), ('shrinkage__1', 0.5)]
    """


class RandomizedSearchCV(_MultiviewSearch, skms.RandomizedSearchCV):
    """Randomized search over a multiview estimator's parameters.

    The multiview form of :class:`~sklearn.model_selection.RandomizedSearchCV`, with its
    parameters, attributes and methods. ``fit``, ``transform`` and ``score``
    take a list of views; parameter names are the estimator's own, with
    ``name__<view>`` setting one view's value of a per-view parameter;
    ``best_estimator_`` is the fitted multiview estimator; and a ``scoring``
    callable is called as ``scoring(estimator, views)``.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import loguniform
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import RandomizedSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((50, 5))
        >>> X2 = rng.standard_normal((50, 4))
        >>> space = {"shrinkage__0": loguniform(1e-3, 1.0)}
        >>> rs = RandomizedSearchCV(
        ...     RidgeCCA(), space, n_iter=5, cv=2, random_state=0
        ... ).fit([X1, X2])
        >>> sorted(rs.best_params_)
        ['shrinkage__0']
    """


class HalvingGridSearchCV(_MultiviewSearch, skms.HalvingGridSearchCV):
    """Successive-halving grid search over a multiview estimator's parameters.

    The multiview form of
    :class:`~sklearn.model_selection.HalvingGridSearchCV`, with its
    parameters, attributes and methods. ``fit``, ``transform`` and ``score``
    take a list of views; parameter names are the estimator's own, with
    ``name__<view>`` setting one view's value of a per-view parameter;
    ``best_estimator_`` is the fitted multiview estimator; and a ``scoring``
    callable is called as ``scoring(estimator, views)``.

    Examples:
        >>> import numpy as np
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import HalvingGridSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((60, 5))
        >>> X2 = rng.standard_normal((60, 4))
        >>> grid = {"shrinkage__0": [0.0, 0.1, 0.5]}
        >>> hgs = HalvingGridSearchCV(RidgeCCA(), grid, cv=2, random_state=0)
        >>> sorted(hgs.fit([X1, X2]).best_params_)
        ['shrinkage__0']
    """


class HalvingRandomSearchCV(_MultiviewSearch, skms.HalvingRandomSearchCV):
    """Successive-halving randomized search over a multiview model's parameters.

    The multiview form of
    :class:`~sklearn.model_selection.HalvingRandomSearchCV`, with its
    parameters, attributes and methods. ``fit``, ``transform`` and ``score``
    take a list of views; parameter names are the estimator's own, with
    ``name__<view>`` setting one view's value of a per-view parameter;
    ``best_estimator_`` is the fitted multiview estimator; and a ``scoring``
    callable is called as ``scoring(estimator, views)``.

    Examples:
        >>> import numpy as np
        >>> from scipy.stats import loguniform
        >>> from cca_zoo.linear import RidgeCCA
        >>> from cca_zoo.model_selection import HalvingRandomSearchCV
        >>> rng = np.random.default_rng(0)
        >>> X1 = rng.standard_normal((60, 5))
        >>> X2 = rng.standard_normal((60, 4))
        >>> space = {"shrinkage__0": loguniform(1e-3, 1.0)}
        >>> hrs = HalvingRandomSearchCV(RidgeCCA(), space, cv=2, random_state=0)
        >>> sorted(hrs.fit([X1, X2]).best_params_)
        ['shrinkage__0']
    """


if importlib.util.find_spec("optuna_integration") is not None:
    import optuna_integration

    class OptunaSearchCV(_MultiviewSearch, optuna_integration.OptunaSearchCV):
        """Optuna's search over a multiview estimator's parameters.

        The multiview form of :class:`optuna_integration.OptunaSearchCV`, with
        its parameters, attributes and methods. ``fit``, ``transform`` and
        ``score`` take a list of views; parameter names are the estimator's
        own, with ``name__<view>`` setting one view's value of a per-view
        parameter; ``best_estimator_`` is the fitted multiview estimator; and
        a ``scoring`` callable is called as ``scoring(estimator, views)``.
        Optuna's own records, such as ``study_``, keep the internal
        ``estimator__`` prefix. Requires the ``optuna`` extra.

        Examples:
            >>> import numpy as np
            >>> from optuna.distributions import FloatDistribution
            >>> from cca_zoo.linear import RidgeCCA
            >>> from cca_zoo.model_selection import OptunaSearchCV
            >>> rng = np.random.default_rng(0)
            >>> X1 = rng.standard_normal((50, 5))
            >>> X2 = rng.standard_normal((50, 4))
            >>> space = {"shrinkage__0": FloatDistribution(0.0, 1.0)}
            >>> search = OptunaSearchCV(
            ...     RidgeCCA(), space, n_trials=5, cv=2, random_state=0
            ... ).fit([X1, X2])
            >>> sorted(search.best_params_)
            ['shrinkage__0']
        """

        @property
        def best_params_(self) -> dict[str, Any]:
            """Parameters of the best trial, by the estimator's own names."""
            return {_unprefix(k): v for k, v in self.study_.best_params.items()}
