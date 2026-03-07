"""GridSearchCV wrapper for multiview CCA models."""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.model_selection as skms
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone


class _MultiviewWrapper(BaseEstimator):
    """Internal wrapper that makes a multiview estimator sklearn-compatible.

    Sklearn's :class:`sklearn.model_selection.GridSearchCV` requires an
    estimator whose ``fit`` and ``score`` methods accept ``(X, y)`` where
    X is a 2-D array.  This wrapper encodes the split indices of the
    multiview data into a single concatenated array and restores the views
    before forwarding calls to the underlying multiview estimator.

    Args:
        estimator: A fitted or unfitted multiview CCA estimator (e.g.
            :class:`~cca_zoo.linear.CCA`).
        split_indices: List of feature counts per view, used to split the
            concatenated array back into individual views.

    Example:
        >>> import numpy as np
        >>> from cca_zoo.linear import CCA
        >>> X1 = np.random.randn(50, 5)
        >>> X2 = np.random.randn(50, 4)
        >>> wrapper = _MultiviewWrapper(CCA(), split_indices=[5, 4])
        >>> wrapper.fit(np.hstack([X1, X2]))
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        split_indices: list[int],
    ) -> None:
        self.estimator = estimator
        self.split_indices = split_indices

    def _split_views(self, X: np.ndarray) -> list[np.ndarray]:
        """Split a concatenated matrix back into individual views.

        Args:
            X: Concatenated array of shape (n_samples, sum(split_indices)).

        Returns:
            List of arrays, one per view.
        """
        views = []
        start = 0
        for p in self.split_indices:
            views.append(X[:, start : start + p])
            start += p
        return views

    def fit(
        self, X: np.ndarray, y: None = None, **fit_params: Any
    ) -> _MultiviewWrapper:
        """Fit the wrapped estimator on the multiview data.

        Args:
            X: Concatenated views, shape (n_samples, sum(n_features)).
            y: Ignored.
            **fit_params: Additional keyword arguments forwarded to
                the estimator's ``fit`` method.

        Returns:
            self: Fitted wrapper.
        """
        self.estimator_ = clone(self.estimator)
        views = self._split_views(X)
        self.estimator_.fit(views, **fit_params)
        return self

    def score(self, X: np.ndarray, y: None = None) -> float:
        """Return the mean canonical correlation over all latent dimensions.

        Args:
            X: Concatenated views, shape (n_samples, sum(n_features)).
            y: Ignored.

        Returns:
            Scalar: mean of the per-dimension average pairwise correlations.
        """
        views = self._split_views(X)
        scores: np.ndarray = self.estimator_.score(views)
        return float(scores.mean())

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return parameters for this estimator.

        Args:
            deep: If True, also return the parameters of the wrapped
                estimator prefixed with ``estimator__``.

        Returns:
            Dictionary of parameter names to values.
        """
        params: dict[str, Any] = {
            "estimator": self.estimator,
            "split_indices": self.split_indices,
        }
        if deep:
            inner = self.estimator.get_params(deep=True)
            params.update({f"estimator__{k}": v for k, v in inner.items()})
        return params

    def set_params(self, **params: Any) -> _MultiviewWrapper:
        """Set parameters for this estimator.

        Args:
            **params: Parameters to set. Keys prefixed with
                ``estimator__`` are forwarded to the wrapped estimator.

        Returns:
            self: Updated wrapper.
        """
        inner_params = {
            k[len("estimator__") :]: v
            for k, v in params.items()
            if k.startswith("estimator__")
        }
        own_params = {
            k: v for k, v in params.items() if not k.startswith("estimator__")
        }
        if own_params:
            super().set_params(**own_params)
        if inner_params:
            self.estimator.set_params(**inner_params)
        return self


class GridSearchCV:
    """Grid search with cross-validation for multiview CCA models.

    Wraps :class:`sklearn.model_selection.GridSearchCV` to support the
    ``list[ArrayLike]`` interface of cca_zoo models.  Views are
    horizontally stacked before being passed to sklearn and split back
    inside the wrapped estimator.

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
        >>> gs.fit([X1, X2])
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: dict[str, list[Any]] | list[dict[str, list[Any]]],
        cv: int | Any = 5,
        scoring: str | None = None,
        n_jobs: int | None = None,
        refit: bool = True,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.verbose = verbose

    def _make_wrapped_param_grid(
        self,
        split_indices: list[int],
    ) -> dict[str, list[Any]] | list[dict[str, list[Any]]]:
        """Prefix all param_grid keys with ``estimator__``.

        Args:
            split_indices: Feature counts per view (used only to create
                the wrapper, not the grid itself).

        Returns:
            Updated parameter grid with ``estimator__`` prefixes.
        """
        if isinstance(self.param_grid, dict):
            return {f"estimator__{k}": v for k, v in self.param_grid.items()}
        return [
            {f"estimator__{k}": v for k, v in grid.items()} for grid in self.param_grid
        ]

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
        split_indices = [a.shape[1] for a in arrays]
        x_concat = np.hstack(arrays)

        wrapped_estimator = _MultiviewWrapper(
            estimator=self.estimator,
            split_indices=split_indices,
        )
        wrapped_grid = self._make_wrapped_param_grid(split_indices)

        self._inner_cv: skms.GridSearchCV = skms.GridSearchCV(
            estimator=wrapped_estimator,
            param_grid=wrapped_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=self.refit,
            verbose=self.verbose,
        )
        self._inner_cv.fit(x_concat, y, **fit_params)

        # Expose the most important attributes at the top level
        self.cv_results_: dict[str, Any] = self._inner_cv.cv_results_
        self.best_score_: float = self._inner_cv.best_score_
        # Strip the estimator__ prefix from best_params_
        raw_best: dict[str, Any] = self._inner_cv.best_params_
        self.best_params_: dict[str, Any] = {
            k[len("estimator__") :]: v for k, v in raw_best.items()
        }
        if self.refit:
            self.best_estimator_: BaseEstimator = (
                self._inner_cv.best_estimator_.estimator_
            )
        return self

    def score(self, views: list[ArrayLike], y: None = None) -> float:
        """Score the best estimator on held-out multiview data.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            Scalar: mean canonical correlation of the best estimator.
        """
        arrays = [np.asarray(v) for v in views]
        x_concat = np.hstack(arrays)
        return float(self._inner_cv.score(x_concat, y))
