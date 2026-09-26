"""Per-view sklearn transformers for multiview pipelines."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from cca_zoo._utils._validation import validate_views


class PerViewTransformer(BaseEstimator):
    """Apply an sklearn transformer to each view separately.

    Fits a clone per view, taking and returning a list of views, so it
    chains with multiview estimators in a :class:`~sklearn.pipeline.Pipeline`.

    Args:
        transformer: A transformer, cloned per view, or one per view.

    Attributes:
        transformers_: The fitted transformer of each view.

    Examples:
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from cca_zoo.linear import CCA
        >>> from cca_zoo.preprocessing import PerViewTransformer
        >>> rng = np.random.default_rng(0)
        >>> X1, X2 = rng.standard_normal((50, 20)), rng.standard_normal((50, 15))
        >>> pipe = Pipeline([
        ...     ("scale", PerViewTransformer(StandardScaler())),
        ...     ("pca", PerViewTransformer(PCA(n_components=5))),
        ...     ("cca", CCA(n_components=2)),
        ... ])
        >>> Z1, Z2 = pipe.fit_transform([X1, X2])
    """

    def __init__(self, transformer: BaseEstimator | list[BaseEstimator]) -> None:
        self.transformer = transformer

    def _transformers_for(self, n_views: int) -> list[BaseEstimator]:
        """One transformer per view."""
        if isinstance(self.transformer, list):
            if len(self.transformer) != n_views:
                raise ValueError(
                    "A list of transformers must have one entry per view: got "
                    f"{len(self.transformer)} transformers for {n_views} views."
                )
            return self.transformer
        return [self.transformer] * n_views

    def fit(self, views: list[ArrayLike], y: None = None) -> PerViewTransformer:
        """Fit the model.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            self.
        """
        validated = validate_views(views, ensure_all_finite=False)
        self.transformers_: list[BaseEstimator] = [
            clone(t).fit(v)
            for t, v in zip(self._transformers_for(len(validated)), validated)
        ]
        return self

    def transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Transform each view with its fitted transformer.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.

        Returns:
            The transformed views.
        """
        check_is_fitted(self)
        validated = validate_views(
            views, min_views=len(self.transformers_), ensure_all_finite=False
        )
        return [t.transform(v) for t, v in zip(self.transformers_, validated)]

    def fit_transform(self, views: list[ArrayLike], y: None = None) -> list[np.ndarray]:
        """Fit, then transform the training views.

        Args:
            views: Arrays of shape (n_samples, n_features_i), one per view.
            y: Ignored.

        Returns:
            The transformed views.
        """
        return self.fit(views, y).transform(views)

    def inverse_transform(self, views: list[ArrayLike]) -> list[np.ndarray]:
        """Map transformed views back to their original feature spaces.

        Args:
            views: Transformed views.

        Returns:
            The views in their original feature spaces.
        """
        check_is_fitted(self)
        validated = validate_views(
            views, min_views=len(self.transformers_), ensure_all_finite=False
        )
        return [t.inverse_transform(v) for t, v in zip(self.transformers_, validated)]

    def __sklearn_tags__(self) -> Tags:
        """Tags marking list-of-views input, which sklearn's checks cannot validate."""
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.two_d_array = False
        tags._skip_test = True
        return tags
