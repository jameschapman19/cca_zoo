"""A generic per-view sklearn transformer adapter for multiview pipelines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._validation import validate_views


class PerViewTransformer(BaseEstimator):
    """Apply an sklearn transformer independently to each view.

    Wraps any sklearn transformer (``StandardScaler``, ``SimpleImputer``,
    ``PCA``, ``KernelCenterer``, ...) so it can be used as a preprocessing
    step ahead of a cca_zoo multiview estimator: a fresh clone of
    ``transformer`` is fit on each view separately, so features -- and
    fitted state such as a scaler's mean or an imputer's fill value -- are
    never shared across views.

    ``fit``/``transform`` take and return a ``list[ArrayLike]`` of views,
    the same convention every cca_zoo estimator uses, so
    :class:`PerViewTransformer` composes directly with
    ``sklearn.pipeline.Pipeline`` and a cca_zoo multiview estimator as its
    final step -- no dedicated multiview ``Pipeline`` class is needed.

    Args:
        transformer: An sklearn transformer instance, cloned once per view
            (e.g. ``StandardScaler()``), or a list with one transformer
            instance per view for heterogeneous preprocessing (e.g.
            ``[StandardScaler(), SimpleImputer()]`` when only one view has
            missing values).

    Example:
        >>> import numpy as np
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.decomposition import PCA
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.preprocessing import PerViewTransformer
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 20)), rng.standard_normal((50, 15))
        >>> pipe = Pipeline([
        ...     ("scale", PerViewTransformer(StandardScaler())),
        ...     ("pca", PerViewTransformer(PCA(n_components=5))),
        ...     ("cca", CCA(latent_dimensions=2)),
        ... ])
        >>> scores = pipe.fit_transform([X1, X2])
    """

    def __init__(self, transformer: BaseEstimator | list[BaseEstimator]) -> None:
        self.transformer = transformer

    def _transformers_for(self, n_views: int) -> list[BaseEstimator]:
        """Resolve ``transformer`` to one instance per view, broadcasting a scalar."""
        if isinstance(self.transformer, list):
            if len(self.transformer) != n_views:
                raise ValueError(
                    "A list of transformers must have one entry per view: got "
                    f"{len(self.transformer)} transformers for {n_views} views."
                )
            return self.transformer
        return [self.transformer] * n_views

    def fit(self, views: list[ArrayLike], y: None = None) -> PerViewTransformer:
        """Fit an independent clone of ``transformer`` on each view.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored. Present for scikit-learn API compatibility.

        Returns:
            self: Fitted transformer.
        """
        validated = validate_views(views, ensure_all_finite=False)
        self.transformers_: list[BaseEstimator] = [
            clone(t).fit(v)
            for t, v in zip(self._transformers_for(len(validated)), validated)
        ]
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Transform each view with its own fitted transformer.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).

        Returns:
            List of transformed views, one per fitted transformer.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
        """
        check_is_fitted(self)
        validated = validate_views(
            views, min_views=len(self.transformers_), ensure_all_finite=False
        )
        return [t.transform(v) for t, v in zip(self.transformers_, validated)]

    def fit_transform(self, views: list[ArrayLike], y: None = None) -> list[np.ndarray]:
        """Fit and then transform the training data.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i).
            y: Ignored.

        Returns:
            List of transformed views, one per fitted transformer.
        """
        return self.fit(views, y).transform(views)

    def inverse_transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Invert each view's transform via its own fitted transformer.

        Args:
            views: List of arrays, each of shape (n_samples, n_features_i)
                in the *transformed* space -- typically the output of
                :meth:`transform`.

        Returns:
            List of views mapped back to their original feature space.

        Raises:
            sklearn.exceptions.NotFittedError: If ``fit`` has not been called.
            AttributeError: If a view's transformer has no ``inverse_transform``.
        """
        check_is_fitted(self)
        validated = validate_views(
            views, min_views=len(self.transformers_), ensure_all_finite=False
        )
        return [t.inverse_transform(v) for t, v in zip(self.transformers_, validated)]

    def __sklearn_tags__(self) -> Tags:
        """Return sklearn tags, corrected for this class's non-standard ``fit``.

        Like :class:`~cca_zoo._base.BaseModel`, ``fit``/``transform`` take a
        *list* of per-view arrays, not a single 2-D ``X``, so sklearn's own
        input validation and common estimator checks don't apply.
        """
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.two_d_array = False
        tags._skip_test = True
        return tags
