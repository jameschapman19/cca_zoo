"""Abstract base class for all cca-zoo models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils import Tags
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._validation import validate_views
from cca_zoo.metrics._correlation import (
    average_pairwise_correlations as _average_pairwise_correlations,
)
from cca_zoo.metrics._correlation import pairwise_correlations as _pairwise_correlations


def _least_squares_map(scores: np.ndarray, data: np.ndarray) -> np.ndarray:
    """The (k, p) matrix ``B`` minimising ``||scores @ B - data||``."""
    mapping: np.ndarray = np.linalg.lstsq(scores, data, rcond=None)[0]
    return mapping


class BaseModel(BaseEstimator, ABC):
    """Base class for multiview CCA models.

    Subclasses implement :meth:`fit`. A linear model sets ``weights_``; a
    nonlinear model overrides :meth:`_transform_view`, its per-view encoder.
    ``transform``, ``predict``, ``inverse_transform``, ``score`` and
    ``feature_importances_`` are built on that encoder.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.

    Attributes:
        means_: Per-view feature means subtracted before fitting.
        n_features_in_: Number of features in each view.
        n_samples_: Number of training samples.
        n_views_: Number of views.
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "center": ["boolean"],
    }

    def __init__(self, n_components: int = 1, center: bool = True) -> None:
        self.n_components = n_components
        self.center = center

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, views: list[ArrayLike], y: None = None) -> BaseModel:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """

    # ------------------------------------------------------------------
    # Shared sklearn-compatible helpers
    # ------------------------------------------------------------------

    def _setup_fit(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Validate parameters and views, record their shapes and centre them."""
        self._validate_params()
        validated = validate_views(views, ensure_min_samples=2)
        self.n_views_: int = len(validated)
        self.n_features_in_: list[int] = [v.shape[1] for v in validated]
        self.n_samples_: int = validated[0].shape[0]
        if self.center:
            self.means_: list[np.ndarray] = [v.mean(axis=0) for v in validated]
            validated = [v - m for v, m in zip(validated, self.means_)]
        else:
            self.means_ = [np.zeros(p) for p in self.n_features_in_]
        # Retained for predict() and inverse_transform(), which regress the
        # training data on its own latent scores.
        self._views_fit_: list[np.ndarray] = validated
        return validated

    def _check_view(self, i: int, view: ArrayLike) -> np.ndarray:
        """View ``i`` as a validated array with the width seen in fit."""
        (checked,) = validate_views([view], min_views=1)
        if checked.shape[1] != self.n_features_in_[i]:
            raise ValueError(
                f"View {i} has {checked.shape[1]} features, but "
                f"{type(self).__name__} is expecting {self.n_features_in_[i]} "
                "features."
            )
        return checked

    def _check_views(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Validate views passed after fitting against the fitted shapes."""
        check_is_fitted(self)
        if len(views) != self.n_views_:
            raise ValueError(f"Expected {self.n_views_} views, got {len(views)}.")
        checked = [self._check_view(i, v) for i, v in enumerate(views)]
        if len({v.shape[0] for v in checked}) > 1:
            raise ValueError(
                "All views must have the same number of samples. "
                f"Got shapes: {[v.shape for v in checked]}."
            )
        return checked

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project each view into the latent space.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_samples, n_components).
        """
        validated = self._check_views(views)
        return [
            self._transform_view(i, v - self.means_[i]) for i, v in enumerate(validated)
        ]

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        """Latent scores of one centred view, shape (n_samples, n_components).

        The per-view encoder behind ``transform``, ``predict`` and
        ``inverse_transform``; nonlinear models override it. Defaults to the
        projection onto ``weights_``.
        """
        scores: np.ndarray = centred @ self.weights_[view]
        return scores

    @property
    def feature_importances_(self) -> list[np.ndarray]:
        """Each feature's share of its view's embedding, one array per view.

        Non-negative and summing to 1 within each view (all zeros if a view's
        embedding uses no feature). Linear models use ``Var(x_j) * sum_k w_jk**2``;
        GAMCCA the variance of each smooth, MARSCCA ``earth``'s ``evimp`` and
        the tree models split gain. Other models use the mean squared change in
        the view's scores when the feature is permuted, which equals twice the
        variance share for a linear or additive model.
        """
        check_is_fitted(self)
        importances = []
        for raw in self._feature_importances():
            raw = np.maximum(raw, 0.0)
            total = raw.sum()
            importances.append(raw / total if total > 0 else raw)
        return importances

    def _feature_importances(self) -> list[np.ndarray]:
        """Unnormalised per-view importances; see :attr:`feature_importances_`."""
        if type(self)._transform_view is BaseModel._transform_view:
            return [
                train.var(axis=0) * np.sum(w**2, axis=1)
                for train, w in zip(self._views_fit_, self.weights_)
            ]
        return self._permutation_importances()

    def _permutation_importances(self) -> list[np.ndarray]:
        """Mean squared change in each view's scores when a feature is permuted."""
        rng = np.random.default_rng(0)
        importances = []
        for i, train in enumerate(self._views_fit_):
            scores = self._transform_view(i, train)
            order = rng.permutation(len(train))
            changes = np.empty(train.shape[1])
            for j in range(train.shape[1]):
                permuted = train.copy()
                permuted[:, j] = train[order, j]
                changes[j] = np.mean(
                    np.sum((self._transform_view(i, permuted) - scores) ** 2, axis=1)
                )
            importances.append(changes)
        return importances

    def _shared_latent(self, observed: dict[int, np.ndarray]) -> np.ndarray:
        """Shared latent scores estimated from the observed centred views.

        The mean of their own scores; the probabilistic models use the posterior
        mean instead.
        """
        latent: np.ndarray = np.mean(
            [self._transform_view(i, v) for i, v in observed.items()], axis=0
        )
        return latent

    def inverse_transform(self, scores: list[ArrayLike]) -> list[np.ndarray]:
        """Map each view's latent scores back to that view's feature space.

        Each view is reconstructed from its own scores by a least-squares
        regression fitted on the training data, as in
        :meth:`sklearn.decomposition.PCA.inverse_transform`. To reconstruct a
        view from the other views, use :meth:`predict`.

        Args:
            scores: One array of shape (n_samples, n_components) per view.

        Returns:
            List of arrays, each of shape (n_samples, n_features_i).

        Raises:
            ValueError: If ``scores`` has the wrong length or width.
        """
        check_is_fitted(self)
        if len(scores) != self.n_views_:
            raise ValueError(
                f"Expected {self.n_views_} score arrays, got {len(scores)}."
            )
        arrays = [np.asarray(s) for s in scores]
        # Models that prune dimensions, such as GFA, record the number kept.
        n_components = getattr(self, "n_components_", self.n_components)
        for i, s in enumerate(arrays):
            if s.shape[1] != n_components:
                raise ValueError(
                    f"scores[{i}] has {s.shape[1]} columns, expected {n_components}."
                )
        return [
            s @ _least_squares_map(self._transform_view(i, train), train)
            + self.means_[i]
            for i, (s, train) in enumerate(zip(arrays, self._views_fit_))
        ]

    def fit_transform(self, views: list[ArrayLike], y: None = None) -> list[np.ndarray]:
        """Fit the model and transform the training views.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            List of arrays, each of shape (n_samples, n_components).
        """
        return self.fit(views, y).transform(views)

    def score(self, views: list[ArrayLike], y: None = None) -> float:
        """Mean canonical correlation between the views' latent scores.

        The average over latent dimensions of the mean pairwise correlation.
        Per-dimension values:
        ``average_pairwise_correlations(pairwise_correlations(model.transform(views)))``
        from :mod:`cca_zoo.metrics`.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            The mean canonical correlation.
        """
        per_dimension = _average_pairwise_correlations(
            _pairwise_correlations(self.transform(views))
        )
        return float(np.mean(per_dimension))

    def predict(self, views: list[ArrayLike | None]) -> list[np.ndarray]:
        """Reconstruct every view from the views that are observed.

        The shared latent scores are estimated from the observed views (their
        mean, or the posterior mean for probabilistic models) and mapped to each
        view by a least-squares regression fitted on the training data. Pass
        ``None`` for a view to reconstruct.

        Args:
            views: One entry per view: an array of shape (n_samples, n_features_i)
                or ``None``.

        Returns:
            List of arrays, each of shape (n_samples, n_features_i).

        Raises:
            ValueError: If ``views`` has the wrong length, no view is observed,
                or the observed views have inconsistent shapes.

        Examples:
            >>> import numpy as np
            >>> from cca_zoo.linear import CCA
            >>> rng = np.random.default_rng(0)
            >>> X1, X2 = rng.standard_normal((50, 10)), rng.standard_normal((50, 8))
            >>> model = CCA(n_components=2).fit([X1, X2])
            >>> model.predict([X1, None])[1].shape
            (50, 8)
        """
        check_is_fitted(self)
        if len(views) != self.n_views_:
            raise ValueError(
                f"Expected {self.n_views_} views (pass None for an "
                f"unobserved view), got {len(views)}."
            )
        observed = {
            i: self._check_view(i, v) for i, v in enumerate(views) if v is not None
        }
        if not observed:
            raise ValueError("At least one view must be observed to predict.")
        first_i = next(iter(observed))
        n_samples = observed[first_i].shape[0]
        for i, v in observed.items():
            if v.shape[0] != n_samples:
                raise ValueError(
                    "All observed views must have the same number of "
                    f"samples. Got shapes: "
                    f"{[(j, a.shape) for j, a in observed.items()]}."
                )
        latent = self._shared_latent(
            {i: v - self.means_[i] for i, v in observed.items()}
        )
        train_latent = self._shared_latent(dict(enumerate(self._views_fit_)))
        return [
            latent @ _least_squares_map(train_latent, train) + self.means_[i]
            for i, train in enumerate(self._views_fit_)
        ]

    # ------------------------------------------------------------------
    # Sklearn compatibility
    # ------------------------------------------------------------------

    def __sklearn_tags__(self) -> Tags:
        """Tags marking the multiview input, which sklearn's own checks cannot build."""
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.two_d_array = False
        tags._skip_test = True
        return tags
