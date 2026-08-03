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


class BaseModel(BaseEstimator, ABC):
    """Abstract base class for all multiview CCA models.

    Subclasses must implement :meth:`fit`.  All other public methods
    (``transform``, ``fit_transform``, ``score``, ``pairwise_correlations``,
    ``average_pairwise_correlations``, ``get_factor_loadings``) are provided
    here using the ``weights_`` attribute set by ``fit``.

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
        validated = validate_views(views)
        centred = [v - m for v, m in zip(validated, self.means_)]
        return [v @ w for v, w in zip(centred, self.weights_)]

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
        # Stack: shape (n_views, n_samples, k)
        T = np.stack(transformed, axis=0)
        # Centre per view per dimension
        T = T - T.mean(axis=1, keepdims=True)
        # Normalise per view per dimension
        norms = np.sqrt((T**2).sum(axis=1, keepdims=True))
        T_norm = T / np.where(norms > 1e-12, norms, 1.0)
        # Correlation via einsum over samples
        corrs: np.ndarray = np.einsum("isd,jsd->ijd", T_norm, T_norm)
        return corrs

    def average_pairwise_correlations(self, views: list[ArrayLike]) -> np.ndarray:
        """Return the mean off-diagonal pairwise correlation per dimension.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            Array of shape ``(latent_dimensions,)`` with the average
            off-diagonal pairwise correlation for each canonical dimension.
        """
        corrs = self.pairwise_correlations(views)  # (n_views, n_views, k)
        n_views = corrs.shape[0]
        # Sum all off-diagonal entries (exclude self-correlations on diagonal)
        off_diag_sum: np.ndarray = corrs.sum(axis=(0, 1)) - sum(
            corrs[i, i, :] for i in range(n_views)
        )
        n_pairs = n_views * (n_views - 1)
        result: np.ndarray = off_diag_sum / n_pairs
        return result

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
        loadings = []
        for v, t in zip(validated, transformed):
            v_c = v - v.mean(axis=0)
            t_c = t - t.mean(axis=0)
            # Covariance between features and variates
            cov = v_c.T @ t_c / (v.shape[0] - 1)  # (p, k)
            std_v = np.maximum(v_c.std(axis=0, ddof=1), 1e-12)  # (p,)
            std_t = np.maximum(t_c.std(axis=0, ddof=1), 1e-12)  # (k,)
            loadings.append(cov / np.outer(std_v, std_t))
        return loadings

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
