"""PerViewTransformer: an sklearn transformer fitted to each view separately."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cca_zoo.linear import CCA
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.preprocessing import PerViewTransformer


def test_each_view_gets_its_own_fit(two_views: list[np.ndarray]) -> None:
    """One transformer is cloned per view; a list gives each view its own."""
    scaled = PerViewTransformer(StandardScaler()).fit_transform(two_views)
    for view in scaled:
        np.testing.assert_allclose(view.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(view.std(axis=0), 1.0, atol=1e-10)
    mixed = PerViewTransformer([SimpleImputer(), PCA(n_components=3)])
    x1 = two_views[0].copy()
    x1[0, 0] = np.nan
    out = mixed.fit_transform([x1, two_views[1]])
    assert not np.isnan(out[0]).any() and out[1].shape == (50, 3)
    with pytest.raises(ValueError, match="one entry per view"):
        PerViewTransformer([StandardScaler()]).fit(two_views)


def test_inverse_transform(two_views: list[np.ndarray]) -> None:
    """Inverse transforms each view, raising where its transformer cannot."""
    scaler = PerViewTransformer(StandardScaler())
    for original, back in zip(
        two_views, scaler.inverse_transform(scaler.fit_transform(two_views))
    ):
        np.testing.assert_allclose(back, original, atol=1e-10)
    with pytest.raises(ValueError, match="add_indicator"):
        PerViewTransformer(SimpleImputer()).fit(two_views).inverse_transform(two_views)


def test_pipelines_need_no_multiview_class(two_views: list[np.ndarray]) -> None:
    """Sklearn's Pipeline chains view transformers into a model, and can be searched."""
    pipe = Pipeline([("scale", PerViewTransformer(StandardScaler())), ("cca", CCA())])
    search = GridSearchCV(pipe, param_grid={"cca__n_components": [1, 2]}, cv=2)
    assert len(search.fit(two_views).transform(two_views)) == 2


def test_transform_needs_the_fitted_number_of_views(
    two_views: list[np.ndarray],
) -> None:
    """A view without a fitted transformer is an error, not dropped."""
    transformer = PerViewTransformer(StandardScaler()).fit(two_views)
    with pytest.raises(ValueError, match="Expected 2 views"):
        transformer.transform([*two_views, two_views[0]])
