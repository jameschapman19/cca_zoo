"""Abstract base class for all cca-zoo models."""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, ClassVar, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from sklearn import get_config
from sklearn.base import BaseEstimator
from sklearn.utils import Tags, TransformerTags, check_random_state
from sklearn.utils._array_api import (
    _convert_to_numpy,
    _is_numpy_namespace,
    device,
    get_namespace,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import (
    _check_sample_weight,
    _get_feature_names,
    check_is_fitted,
)

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


def _weighted_pseudo_rows(views: list[Any], sample_weight: Any) -> list[Any]:
    """Rows whose unweighted moments are the views' weighted moments.

    Each row is scaled by the square root of its weight, and one Householder
    reflection maps the square-root weights onto the constant vector. The
    rows' column sums, centred Gram and ``n - 1`` denominators then give the
    weighted mean and covariance with denominator ``sum(w) - 1``: for integer
    weights, those of the views with each row repeated. A model whose fit
    depends on the views only through their centred second moments is thereby
    fitted with the weights unchanged.
    """
    xp, _ = get_namespace(sample_weight)
    n = sample_weight.shape[0]
    root = xp.sqrt(sample_weight)
    reflector = root / xp.linalg.vector_norm(root) - 1.0 / math.sqrt(n)
    norm2 = float(xp.sum(reflector**2))
    scale = math.sqrt((n - 1) / (float(xp.sum(sample_weight)) - 1))
    rows = []
    for v in views:
        scaled = root[:, None] * v
        if norm2 > 0:
            scaled = scaled - reflector[:, None] * ((2 / norm2) * (reflector @ scaled))
        rows.append(xp.astype(scale * scaled, v.dtype))
    return rows


def _to_numpy(array: Any) -> np.ndarray:
    """``array`` as a numpy array, from any Array API namespace."""
    xp, _ = get_namespace(array)
    converted: np.ndarray = _convert_to_numpy(array, xp)
    return converted


def _least_squares_map(scores: Any, data: Any) -> Any:
    """The (k, p) matrix ``B`` minimising ``||scores @ B - data||``."""
    xp, _ = get_namespace(scores, data)
    return xp.linalg.pinv(scores) @ data


class BaseModel(BaseEstimator, ABC):
    """Base class for multiview CCA models.

    Subclasses implement :meth:`fit`, starting with :meth:`_setup_fit` and
    ending with :meth:`_fit_maps_and_importances`. A linear model sets ``weights_``; a
    nonlinear model overrides :meth:`_transform_view`, its per-view encoder.
    ``transform``, ``predict``, ``inverse_transform``, ``score`` and
    ``feature_importances_per_view_`` are built on that encoder.

    Args:
        n_components: Number of latent dimensions. Default is 1.
        center: Whether to centre each view.
            Default is True.

    Attributes:
        means_: Per-view feature means subtracted before fitting.
        n_features_per_view_: Number of features in each view.
        n_components_: Number of latent dimensions fitted: ``n_components``,
            or fewer where the model prunes dimensions or the data have too
            few, as sklearn's ``PCA.n_components_``.
        feature_names_per_view_: Each view's feature names, set when every
            view was fitted as a DataFrame with string column names.
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
    # Whether n_components is bounded by the narrowest view's features, as for
    # any model whose embedding is a projection of the features.
    _components_bounded_by_features: ClassVar[bool] = True
    # Whether fit and transform compute in the namespace of Array API inputs
    # (with sklearn's array_api_dispatch), rather than only with numpy.
    _supports_array_api: ClassVar[bool] = False

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

    def _setup_fit(
        self, views: list[ArrayLike], sample_weight: ArrayLike | None = None
    ) -> list[np.ndarray]:
        """Validate parameters and views, record their shapes and centre them.

        With ``sample_weight``, the views are centred on their weighted means
        and returned as :func:`_weighted_pseudo_rows`, for the models whose fit
        uses only the centred second moments.
        """
        self._validate_params()
        names = [_get_feature_names(v) for v in views]
        if all(n is not None for n in names):
            self.feature_names_per_view_: list[np.ndarray] = names
        elif hasattr(self, "feature_names_per_view_"):
            del self.feature_names_per_view_
        xp, _ = get_namespace(*views)
        if not (self._supports_array_api or _is_numpy_namespace(xp)):
            raise TypeError(
                f"{type(self).__name__} computes with numpy; pass numpy arrays, "
                "or use a model that supports the Array API (see the user guide)."
            )
        validated = validate_views(
            views, ensure_min_samples=2, dtype=self._preserved_dtypes
        )
        self.n_views_: int = len(validated)
        self.n_features_per_view_: list[int] = [v.shape[1] for v in validated]
        max_components = min(self.n_features_per_view_)
        if self._components_bounded_by_features and self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} must be at most "
                f"{max_components}, the number of features in the narrowest view."
            )
        self.n_samples_: int = validated[0].shape[0]
        weights = (
            None
            if sample_weight is None
            else _check_sample_weight(
                sample_weight, validated[0], ensure_non_negative=True
            )
        )
        if weights is not None and float(xp.sum(weights)) <= 1:
            raise ValueError(
                "sample_weight must sum to more than 1: a covariance needs more "
                f"than one sample's weight, got a sum of {float(xp.sum(weights))}."
            )
        if not self.center:
            self.means_: list[Any] = [
                xp.zeros(v.shape[1], dtype=v.dtype, device=device(v)) for v in validated
            ]
        elif weights is None:
            self.means_ = [xp.mean(v, axis=0) for v in validated]
        else:
            self.means_ = [
                xp.astype(
                    xp.sum(v * weights[:, None], axis=0) / xp.sum(weights), v.dtype
                )
                for v in validated
            ]
        validated = [v - m for v, m in zip(validated, self.means_)]
        if weights is not None:
            validated = _weighted_pseudo_rows(validated, weights)
        return validated

    def _fit_maps_and_importances(self, views: list[np.ndarray]) -> None:
        """Fit what ``predict``, ``inverse_transform`` and the importances need.

        The one pass over the centred training views after a model's own
        algorithm: least-squares maps from each view's scores, and from the
        shared latent, back to the views, and ``feature_importances_per_view_``.
        Computing them here means the fitted model keeps no training data.
        """
        own_scores = [self._transform_view(i, v) for i, v in enumerate(views)]
        self.n_components_: int = own_scores[0].shape[1]
        self._inverse_maps_ = [
            _least_squares_map(s, v) for s, v in zip(own_scores, views)
        ]
        latent = self._shared_latent(dict(enumerate(views)))
        self._predict_maps_ = [_least_squares_map(latent, v) for v in views]
        self.feature_importances_per_view_: list[Any] = []
        for raw in self._feature_importances(views):
            xp, _ = get_namespace(raw)
            positive = xp.where(raw > 0, raw, 0.0)
            total = float(xp.sum(positive))
            self.feature_importances_per_view_.append(
                positive / total if total > 0 else positive
            )

    def _check_view(self, i: int, view: ArrayLike) -> np.ndarray:
        """View ``i`` as a validated array with the width and names seen in fit."""
        self._check_feature_names(i, view)
        (checked,) = validate_views([view], min_views=1, dtype=self._preserved_dtypes)
        if checked.shape[1] != self.n_features_per_view_[i]:
            raise ValueError(
                f"View {i} has {checked.shape[1]} features, but "
                f"{type(self).__name__} is expecting "
                f"{self.n_features_per_view_[i]} features."
            )
        return checked

    def _check_feature_names(self, i: int, view: ArrayLike) -> None:
        """Check view ``i``'s feature names against fit's, as sklearn checks ``X``'s."""
        fitted = getattr(self, "feature_names_per_view_", None)
        names = _get_feature_names(view)
        name = type(self).__name__
        if fitted is None and names is not None:
            warnings.warn(
                f"View {i} has feature names, but {name} was fitted without "
                "feature names.",
                UserWarning,
                stacklevel=4,
            )
        elif fitted is not None and names is None:
            warnings.warn(
                f"View {i} does not have valid feature names, but {name} was "
                "fitted with feature names.",
                UserWarning,
                stacklevel=4,
            )
        elif (
            fitted is not None
            and names is not None
            and (len(names) != len(fitted[i]) or np.any(names != fitted[i]))
        ):
            raise ValueError(
                f"The feature names of view {i} should match those passed "
                f"during fit. Expected {list(fitted[i])}, got {list(names)}."
            )

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

    def transform(self, views: list[ArrayLike]) -> list[Any]:
        """Project each view into the latent space.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_samples, n_components), or of
            DataFrames under :meth:`set_output`.
        """
        return self._wrap_output(self._transform_arrays(views), views)

    def _transform_arrays(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """:meth:`transform` as arrays, whatever the output container."""
        validated = self._check_views(views)
        return [
            self._transform_view(i, v - self.means_[i]) for i, v in enumerate(validated)
        ]

    def get_feature_names_out(
        self, input_features: list[list[str]] | None = None
    ) -> list[np.ndarray]:
        """Names of each view's latent dimensions, ``<model><k>`` as in sklearn.

        Args:
            input_features: Ignored, unless given: then each view's must match
                the names seen in fit.

        Returns:
            One array of names per view, of length n_components.

        Raises:
            ValueError: If ``input_features`` differ from the fitted names.
        """
        check_is_fitted(self)
        if input_features is not None:
            fitted = getattr(self, "feature_names_per_view_", None)
            if fitted is None or any(
                len(given) != len(names) or np.any(np.asarray(given) != names)
                for given, names in zip(input_features, fitted, strict=True)
            ):
                raise ValueError("input_features do not match the fitted names.")
        prefix = type(self).__name__.lower()
        names = np.asarray(
            [f"{prefix}{k}" for k in range(self.n_components_)], dtype=object
        )
        return [names.copy() for _ in range(self.n_views_)]

    @validate_params(
        {"transform": [StrOptions({"default", "pandas", "polars"}), None]},
        prefer_skip_nested_validation=True,
    )
    def set_output(self: _Model, *, transform: str | None = None) -> _Model:
        """Set the container :meth:`transform` returns, as sklearn's ``set_output``.

        Args:
            transform: ``"pandas"`` or ``"polars"`` for one DataFrame per view,
                indexed as the input view when it is a pandas DataFrame;
                ``"default"`` for sklearn's global ``transform_output``
                configuration; None leaves the setting unchanged.

        Returns:
            self.
        """
        if transform is not None:
            self._sklearn_output_config = {"transform": transform}
        return self

    def _wrap_output(
        self, arrays: list[np.ndarray], views: list[ArrayLike]
    ) -> list[Any]:
        """Each view's scores in the container chosen by :meth:`set_output`."""
        container = getattr(self, "_sklearn_output_config", {}).get(
            "transform", "default"
        )
        if container == "default":
            container = get_config()["transform_output"]
        if container == "default":
            return arrays
        names = self.get_feature_names_out()
        arrays = [_to_numpy(z) for z in arrays]
        if container == "pandas":
            import pandas as pd

            return [
                pd.DataFrame(z, columns=n, index=getattr(v, "index", None))
                for z, n, v in zip(arrays, names, views)
            ]
        import polars as pl

        return [pl.DataFrame(z, schema=list(n)) for z, n in zip(arrays, names)]

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
        xp, _ = get_namespace(*views)
        if type(self)._transform_view is BaseModel._transform_view:
            return [
                xp.var(v, axis=0) * xp.sum(w**2, axis=1)
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

    def _shared_latent(self, observed: dict[int, Any]) -> Any:
        """Shared latent scores estimated from the observed centred views.

        The mean of their own scores; the probabilistic models use the posterior
        mean instead.
        """
        xp, _ = get_namespace(*observed.values())
        scores = [self._transform_view(i, v) for i, v in observed.items()]
        return xp.mean(xp.stack(scores), axis=0)

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
        xp, _ = get_namespace(*scores)
        arrays = [xp.asarray(s) for s in scores]
        for i, s in enumerate(arrays):
            if s.shape[1] != self.n_components_:
                raise ValueError(
                    f"scores[{i}] has {s.shape[1]} columns, "
                    f"expected {self.n_components_}."
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
        scores = [_to_numpy(z) for z in self._transform_arrays(views)]
        per_dimension = _average_pairwise_correlations(_pairwise_correlations(scores))
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
        for v in observed.values():
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
        tags.array_api_support = self._supports_array_api
        tags.no_validation = True
        tags.input_tags.two_d_array = False
        tags._skip_test = True
        return tags
