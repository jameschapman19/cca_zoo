"""Abstract base class for all cca-zoo models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, ClassVar, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils import Tags, TransformerTags, check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._validation import validate_views
from cca_zoo.metrics._correlation import (
    average_pairwise_correlations as _average_pairwise_correlations,
)
from cca_zoo.metrics._correlation import pairwise_correlations as _pairwise_correlations

_Model = TypeVar("_Model", bound="BaseModel")

# Permutation importances are computed at fit on at most this many training
# rows, as sklearn's ``permutation_importance(max_samples=...)``, so that they
# add a small fraction to the fit of the models that need them.
_PERMUTATION_SAMPLES = 500


def _least_squares_map(scores: np.ndarray, data: np.ndarray) -> np.ndarray:
    """The (k, p) matrix ``B`` minimising ``||scores @ B - data||``."""
    mapping: np.ndarray = np.linalg.lstsq(scores, data, rcond=None)[0]
    return mapping


class BaseModel(BaseEstimator, ABC):
    """Base class for multiview CCA models.

    Subclasses implement :meth:`fit`, starting with :meth:`_setup_fit` and
    ending with :meth:`_finish_fit`. A linear model sets ``weights_``; a
    nonlinear model overrides :meth:`_transform_view`, its per-view encoder.
    ``transform``, ``predict``, ``inverse_transform``, ``score`` and
    ``feature_importances_per_view_`` are built on that encoder.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            Default is True.

    Attributes:
        means_: Per-view feature means subtracted before fitting.
        n_features_per_view_: Number of features in each view.
        n_samples_: Number of training samples.
        n_views_: Number of views.
        feature_importances_per_view_: Each feature's share of its view's
            embedding, one array per view. Non-negative and summing to 1
            within each view (all zeros if a view's embedding uses no
            feature). Linear models use ``Var(x_j) * sum_k w_jk**2``; GAMCCA
            the variance of each smooth, MARSCCA ``earth``'s ``evimp`` and the
            tree models split gain. Other models use the mean squared change
            in the view's scores when the feature is permuted, over at most
            500 training rows, which equals twice the variance share for a
            linear or additive model.
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "center": ["boolean"],
    }
    # Input dtypes the model fits and transforms in; others become float64.
    _preserved_dtypes: ClassVar[list[type]] = [np.float64]

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
        validated = validate_views(
            views, ensure_min_samples=2, dtype=self._preserved_dtypes
        )
        self.n_views_: int = len(validated)
        self.n_features_per_view_: list[int] = [v.shape[1] for v in validated]
        self.n_samples_: int = validated[0].shape[0]
        if self.center:
            self.means_: list[np.ndarray] = [v.mean(axis=0) for v in validated]
            validated = [v - m for v, m in zip(validated, self.means_)]
        else:
            self.means_ = [np.zeros(p) for p in self.n_features_per_view_]
        return validated

    def _finish_fit(self: _Model, views: list[np.ndarray]) -> _Model:
        """Record what later calls need from the centred training views.

        ``predict`` and ``inverse_transform`` regress the training views on
        their latent scores; the maps and the importances are computed here so
        that the fitted model does not keep the training data.
        """
        own_scores = [self._transform_view(i, v) for i, v in enumerate(views)]
        self._inverse_maps_ = [
            _least_squares_map(s, v) for s, v in zip(own_scores, views)
        ]
        latent = self._shared_latent(dict(enumerate(views)))
        self._predict_maps_ = [_least_squares_map(latent, v) for v in views]
        self.feature_importances_per_view_: list[np.ndarray] = []
        for raw in self._feature_importances(views):
            raw = np.maximum(raw, 0.0)
            total = raw.sum()
            self.feature_importances_per_view_.append(raw / total if total > 0 else raw)
        return self

    def _check_view(self, i: int, view: ArrayLike) -> np.ndarray:
        """View ``i`` as a validated array with the width seen in fit."""
        (checked,) = validate_views([view], min_views=1, dtype=self._preserved_dtypes)
        if checked.shape[1] != self.n_features_per_view_[i]:
            raise ValueError(
                f"View {i} has {checked.shape[1]} features, but "
                f"{type(self).__name__} is expecting "
                f"{self.n_features_per_view_[i]} features."
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

    def _feature_importances(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Unnormalised importances from the centred training views.

        See ``feature_importances_per_view_``.
        """
        if type(self)._transform_view is BaseModel._transform_view:
            return [
                v.var(axis=0) * np.sum(w**2, axis=1)
                for v, w in zip(views, self.weights_)
            ]
        return self._permutation_importances(views)

    def _permutation_importances(self, views: list[np.ndarray]) -> list[np.ndarray]:
        """Mean squared change in each view's scores when a feature is permuted.

        Computed on a random subset of at most ``_PERMUTATION_SAMPLES`` rows.
        """
        # A model without randomness of its own gets reproducible importances.
        rng = check_random_state(self.get_params().get("random_state", 0))
        rows = rng.permutation(len(views[0]))[:_PERMUTATION_SAMPLES]
        importances = []
        for i, view in enumerate(v[rows] for v in views):
            scores = self._transform_view(i, view)
            order = rng.permutation(len(view))
            changes = np.empty(view.shape[1])
            for j in range(view.shape[1]):
                permuted = view.copy()
                permuted[:, j] = view[order, j]
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
            s @ m + mean for s, m, mean in zip(arrays, self._inverse_maps_, self.means_)
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
        return [latent @ m + mean for m, mean in zip(self._predict_maps_, self.means_)]

    # ------------------------------------------------------------------
    # Sklearn compatibility
    # ------------------------------------------------------------------

    def __sklearn_tags__(self) -> Tags:
        """Tags marking the multiview input, which sklearn's own checks cannot build."""
        tags = super().__sklearn_tags__()
        tags.transformer_tags = TransformerTags(
            preserves_dtype=[np.dtype(t).name for t in self._preserved_dtypes]
        )
        tags.no_validation = True
        tags.input_tags.two_d_array = False
        tags._skip_test = True
        return tags
