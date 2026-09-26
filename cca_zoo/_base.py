"""Abstract base class for all cca-zoo models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Integral
from typing import Any, ClassVar, cast

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
from cca_zoo.metrics._correlation import factor_loadings as _factor_loadings
from cca_zoo.metrics._correlation import pairwise_correlations as _pairwise_correlations


def _least_squares_map(scores: np.ndarray, data: np.ndarray) -> np.ndarray:
    """The (k, p) matrix ``B`` minimising ``||scores @ B - data||``."""
    mapping: np.ndarray = np.linalg.lstsq(scores, data, rcond=None)[0]
    return mapping


class BaseModel(BaseEstimator, ABC):
    """Abstract base class for all multiview CCA models.

    Subclasses must implement :meth:`fit`. A linear model sets
    ``weights_``; a nonlinear one overrides :meth:`_transform_view`, its
    per-view encoder. Every other public method (``transform``,
    ``inverse_transform``, ``fit_transform``, ``score``, ``predict``) is
    built here on that one encoder, so it behaves identically for every
    model.

    This class inherits from :class:`sklearn.base.BaseEstimator` so that
    ``get_params`` / ``set_params`` round-trip correctly and sklearn model
    selection utilities work out of the box.

    Constructor parameters are validated with sklearn's
    ``_parameter_constraints`` mechanism (see :meth:`_setup_fit`).
    Subclasses that add their own constructor parameters may extend
    ``_parameter_constraints`` by merging in ``BaseModel._parameter_constraints``;
    parameters with no declared constraint are left unvalidated, so this is
    always safe to skip.

    Args:
        latent_dimensions: Number of latent dimensions to fit. Default is 1.
        center: Whether to subtract per-view column means before fitting.
            The means are stored in ``means_`` and applied in ``transform``.
    """

    _parameter_constraints: ClassVar[dict[str, list[Any]]] = {
        "latent_dimensions": [Interval(Integral, 1, None, closed="left")],
        "center": ["boolean"],
    }

    def __init__(self, latent_dimensions: int = 1, center: bool = True) -> None:
        self.latent_dimensions = latent_dimensions
        self.center = center

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, views: list[ArrayLike], y: None = None) -> BaseModel:
        """Fit the model to multiview data.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
                All arrays must have the same number of rows.
            y: Ignored.  Present for scikit-learn API compatibility.

        Returns:
            self: Fitted estimator.

        Raises:
            ValueError: If fewer than 2 views are provided.
            ValueError: If views have inconsistent numbers of samples.
        """

    # ------------------------------------------------------------------
    # Shared sklearn-compatible helpers
    # ------------------------------------------------------------------

    def _setup_fit(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Validate constructor parameters and views, record metadata, centre.

        Args:
            views: Raw input views.

        Returns:
            Validated (and optionally centred) list of numpy arrays.

        Raises:
            sklearn.utils._param_validation.InvalidParameterError: If a
                constructor parameter violates its declared constraint
                (a ``ValueError`` subclass).
        """
        self._validate_params()
        validated = validate_views(views)
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Project views into the latent space using the fitted weights.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_samples, latent_dimensions).

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        validated = validate_views(views, min_views=self.n_views_)
        return [
            self._transform_view(i, v - self.means_[i]) for i, v in enumerate(validated)
        ]

    def _transform_view(self, view: int, centred: np.ndarray) -> np.ndarray:
        """One view's latent scores from its centred data, shape (n, k).

        The single place a model maps a view into the latent space:
        :meth:`transform`, :meth:`predict` and :meth:`inverse_transform` all
        go through it, so a nonlinear model overrides this alone. The
        default is the linear projection onto ``weights_``.
        """
        scores: np.ndarray = centred @ self.weights_[view]
        return scores

    def _shared_latent(self, observed: dict[int, np.ndarray]) -> np.ndarray:
        """Estimate of the shared latent score from whichever views are observed.

        The mean of the observed views' own scores; models with a joint
        posterior over the latent (the probabilistic ones) override it.

        Args:
            observed: Centred arrays keyed by view index.

        Returns:
            Array of shape (n_samples, k).
        """
        latent: np.ndarray = np.mean(
            [self._transform_view(i, v) for i, v in observed.items()], axis=0
        )
        return latent

    def inverse_transform(self, scores: list[ArrayLike]) -> list[np.ndarray]:
        """Approximately invert ``transform``, mapping latent scores back to each view.

        Each view is reconstructed from *that same view's own* latent
        score only, via a per-view loading matrix: the least-squares
        regression of that view's centred training data onto that view's
        own training latent score. This makes
        ``inverse_transform(transform(views))`` an approximate round trip
        of ``views`` (exact wherever ``latent_dimensions`` and each view's
        own encoder spans it exactly), mirroring
        :meth:`sklearn.decomposition.PCA.inverse_transform`.

        This is a different operation from :meth:`predict`: ``predict``
        combines the *observed* views' scores into one shared consensus
        estimate to reconstruct views you don't have, which is only
        possible once at least one other view actually is observed.
        ``inverse_transform`` never mixes information across views — it
        needs a view's own score to reconstruct that same view, so it
        cannot be used to impute a view you never transformed in the first
        place; use :meth:`predict` for that.

        Args:
            scores: List of length ``n_views_``, each an array of shape
                (n_samples, latent_dimensions) — typically the output of
                :meth:`transform`.

        Returns:
            List of length ``n_views_``, each an array of shape
            (n_samples, n_features_i): the reconstructed view.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            ValueError: If ``scores`` has the wrong length, or an entry has
                the wrong number of latent dimensions.

        Examples:
            >>> import numpy as np
            >>> from cca_zoo.linear import CCA
            >>> rng = np.random.default_rng(0)
            >>> X1 = rng.standard_normal((50, 10))
            >>> X2 = rng.standard_normal((50, 8))
            >>> model = CCA(latent_dimensions=2).fit([X1, X2])
            >>> scores = model.transform([X1, X2])
            >>> X1_approx, X2_approx = model.inverse_transform(scores)
            >>> X1_approx.shape
            (50, 10)
        """
        check_is_fitted(self)
        if len(scores) != self.n_views_:
            raise ValueError(
                f"Expected {self.n_views_} score arrays, got {len(scores)}."
            )
        arrays = [np.asarray(s) for s in scores]
        for i, s in enumerate(arrays):
            if s.shape[1] != self.latent_dimensions:
                raise ValueError(
                    f"scores[{i}] has {s.shape[1]} columns, expected "
                    f"latent_dimensions={self.latent_dimensions}."
                )
        return [
            s @ _least_squares_map(self._transform_view(i, train), train)
            + self.means_[i]
            for i, (s, train) in enumerate(zip(arrays, self._views_fit_))
        ]

    def fit_transform(self, views: list[ArrayLike], y: None = None) -> list[np.ndarray]:
        """Fit and then transform the training data.

        Equivalent to ``self.fit(views).transform(views)`` but may be more
        efficient for some subclasses.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            List of arrays, each of shape (n_samples, latent_dimensions).
        """
        return self.fit(views, y).transform(views)

    def score(self, views: list[ArrayLike], y: None = None) -> np.ndarray:
        """Return average pairwise canonical correlations for each dimension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            Array of shape ``(latent_dimensions,)`` with the average
            pairwise correlation for each canonical dimension.
        """
        return self.average_pairwise_correlations(views)

    def pairwise_correlations(self, views: list[ArrayLike]) -> np.ndarray:
        """Compute the full pairwise correlation matrix per latent dimension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            Array of shape ``(n_views, n_views, latent_dimensions)`` where
            entry ``[i, j, d]`` is the Pearson correlation between the
            d-th canonical variate of view i and view j.
        """
        transformed = self.transform(views)
        return _pairwise_correlations(transformed)

    def average_pairwise_correlations(self, views: list[ArrayLike]) -> np.ndarray:
        """Return the mean off-diagonal pairwise correlation per dimension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            Array of shape ``(latent_dimensions,)`` with the average
            off-diagonal pairwise correlation for each canonical dimension.
        """
        corrs = self.pairwise_correlations(views)  # (n_views, n_views, k)
        return _average_pairwise_correlations(corrs)

    @property
    def weights(self) -> list[np.ndarray]:
        """Weight matrices post-fit, one per view.

        Shape is ``(n_features_i, latent_dimensions)`` for each view.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        return cast(list[np.ndarray], self.weights_)

    def get_factor_loadings(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Compute canonical factor loadings for each view.

        A loading is the Pearson correlation between an original feature and a
        canonical variate.  Loadings indicate which original variables drive
        each canonical direction.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of arrays, each of shape (n_features_i, latent_dimensions),
            where entry ``[j, d]`` is the correlation between feature j of
            view i and the d-th canonical variate of view i.
        """
        validated = validate_views(views)
        transformed = self.transform(views)
        return _factor_loadings(validated, transformed)

    def predict(self, views: list[ArrayLike | None]) -> list[np.ndarray]:
        """Reconstruct every view from whichever views are observed.

        ``transform`` maps data to the shared latent space; ``predict`` maps
        back the other way, from the latent space to each view's original
        feature space. Pass ``None`` for any view you want reconstructed —
        typically one you don't have, but you can also ask for a view you
        *did* supply, as a diagnostic (its own self-reconstruction).

        The shared latent score is estimated from the observed views only:
        the mean of their own latent scores, or, for the probabilistic
        models, the posterior mean given them. Each requested view is then
        reconstructed as that score against a per-view loading matrix: the
        least-squares regression of that view's centred training data onto
        the training data's own shared latent score. This works the same
        for every model, linear or not, since it only needs the model's
        encoder.

        This regression-based reconstruction is deliberate rather than the
        simpler ``score @ weights.T``: for CCA (unlike PLS), that simpler
        formula is only a correct inverse of ``transform`` when the data
        happens to be pre-whitened, since the true forward map needs an
        extra view-covariance factor that isn't recovered from ``weights_``
        alone (see the discussion on
        https://github.com/jameschapman19/cca_zoo/issues/182). Regressing
        on the training data sidesteps that entirely, at the cost of a
        fitted model retaining its own (centred) training views.

        See also :meth:`inverse_transform`, which reconstructs a view from
        that same view's own score (no cross-view imputation) — the
        appropriate choice when you already have every view's scores and
        just want to invert ``transform``.

        Args:
            views: List of length ``n_views_``. Each entry is either an
                array of shape (n_samples, n_features_i) or ``None`` for a
                view to reconstruct from the others. All non-``None``
                entries must have the same number of samples.

        Returns:
            List of length ``n_views_``: every view's reconstruction, each
            of shape (n_samples, n_features_i) with the same ``n_samples``
            as the observed view(s) passed in.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            ValueError: If ``views`` has the wrong length, every entry is
                ``None``, the observed views have inconsistent numbers of
                samples, or an observed view has the wrong number of
                features.

        Examples:
            >>> import numpy as np
            >>> from cca_zoo.linear import CCA
            >>> rng = np.random.default_rng(0)
            >>> X1 = rng.standard_normal((50, 10))
            >>> X2 = rng.standard_normal((50, 8))
            >>> model = CCA(latent_dimensions=2).fit([X1, X2])
            >>> X2_pred = model.predict([X1, None])[1]
            >>> X2_pred.shape
            (50, 8)
        """
        check_is_fitted(self)
        if len(views) != self.n_views_:
            raise ValueError(
                f"Expected {self.n_views_} views (pass None for an "
                f"unobserved view), got {len(views)}."
            )
        observed = {i: np.asarray(v) for i, v in enumerate(views) if v is not None}
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
            if v.shape[1] != self.n_features_in_[i]:
                raise ValueError(
                    f"View {i} has {v.shape[1]} features, expected "
                    f"{self.n_features_in_[i]}."
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
        """Return sklearn tags, corrected for this class's non-standard ``fit``.

        ``BaseModel`` subclasses deliberately don't conform to sklearn's
        standard estimator interface: ``fit``/``transform``/``score`` take a
        *list* of per-view arrays, not a single 2-D ``X``, so sklearn's own
        input validation and common estimator checks don't apply. This is
        surfaced honestly via tags rather than left to silently mismatch.

        Returns:
            Tags: sklearn tags with ``no_validation`` and ``_skip_test`` set,
            and ``input_tags.two_d_array`` cleared since a bare 2-D array is
            not a valid input on its own.
        """
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.two_d_array = False
        tags._skip_test = True
        return tags
