"""Tests for cca_zoo.preprocessing.PerViewTransformer."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cca_zoo.linear import CCA
from cca_zoo.model_selection import GridSearchCV
from cca_zoo.preprocessing import PerViewTransformer

# ---------------------------------------------------------------------------
# Basic fit/transform, one transformer broadcast to every view
# ---------------------------------------------------------------------------


def test_fit_transform_shapes(two_views: list[np.ndarray]) -> None:
    """Transform preserves sample count and returns one array per view."""
    pvt = PerViewTransformer(StandardScaler())
    out = pvt.fit(two_views).transform(two_views)
    assert len(out) == len(two_views)
    for original, transformed in zip(two_views, out):
        assert transformed.shape == original.shape


def test_each_view_scaled_independently(two_views: list[np.ndarray]) -> None:
    """Each view's own mean/scale is used, not a shared one across views."""
    pvt = PerViewTransformer(StandardScaler()).fit(two_views)
    out = pvt.transform(two_views)
    for transformed in out:
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-10)
    # Each view kept its own feature dimensionality (proof views aren't pooled).
    assert pvt.transformers_[0].mean_.shape == (two_views[0].shape[1],)
    assert pvt.transformers_[1].mean_.shape == (two_views[1].shape[1],)


def test_fit_transform_matches_fit_then_transform(two_views: list[np.ndarray]) -> None:
    """fit_transform is equivalent to fit(...).transform(...)."""
    a = PerViewTransformer(StandardScaler()).fit_transform(two_views)
    b = PerViewTransformer(StandardScaler()).fit(two_views).transform(two_views)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_transform_before_fit_raises(two_views: list[np.ndarray]) -> None:
    """Transform without fit raises NotFittedError."""
    with pytest.raises(NotFittedError):
        PerViewTransformer(StandardScaler()).transform(two_views)


# ---------------------------------------------------------------------------
# A distinct transformer per view
# ---------------------------------------------------------------------------


def test_per_view_transformer_list(two_views: list[np.ndarray]) -> None:
    """A list of transformers applies each entry to the matching view."""
    pvt = PerViewTransformer([StandardScaler(), PCA(n_components=3)])
    out = pvt.fit(two_views).transform(two_views)
    assert out[0].shape == two_views[0].shape
    assert out[1].shape == (two_views[1].shape[0], 3)


def test_per_view_transformer_list_wrong_length_raises(
    two_views: list[np.ndarray],
) -> None:
    """A transformer list with the wrong length raises a clear ValueError."""
    pvt = PerViewTransformer([StandardScaler(), StandardScaler(), StandardScaler()])
    with pytest.raises(ValueError, match="one entry per view"):
        pvt.fit(two_views)


def test_heterogeneous_missing_data(two_views: list[np.ndarray]) -> None:
    """SimpleImputer on one view, StandardScaler on the other."""
    x1, x2 = two_views
    x1_missing = x1.copy()
    x1_missing[0, 0] = np.nan
    pvt = PerViewTransformer([SimpleImputer(), StandardScaler()])
    out = pvt.fit_transform([x1_missing, x2])
    assert not np.isnan(out[0]).any()


# ---------------------------------------------------------------------------
# inverse_transform
# ---------------------------------------------------------------------------


def test_inverse_transform_round_trips(two_views: list[np.ndarray]) -> None:
    """inverse_transform undoes transform for a reversible transformer."""
    pvt = PerViewTransformer(StandardScaler())
    transformed = pvt.fit_transform(two_views)
    recovered = pvt.inverse_transform(transformed)
    for original, back in zip(two_views, recovered):
        np.testing.assert_allclose(original, back, atol=1e-10)


def test_inverse_transform_without_support_raises(two_views: list[np.ndarray]) -> None:
    """A transformer whose own inverse_transform refuses surfaces that error.

    E.g. SimpleImputer without ``add_indicator=True``.
    """
    pvt = PerViewTransformer(SimpleImputer()).fit(two_views)
    with pytest.raises(ValueError, match="add_indicator"):
        pvt.inverse_transform(two_views)


# ---------------------------------------------------------------------------
# sklearn compatibility: clone, get_params/set_params, Pipeline, GridSearchCV
# ---------------------------------------------------------------------------


def test_clonable() -> None:
    """PerViewTransformer round-trips through sklearn's clone."""
    pvt = PerViewTransformer(StandardScaler())
    cloned = clone(pvt)
    assert cloned is not pvt
    assert isinstance(cloned.transformer, StandardScaler)


def test_composes_with_sklearn_pipeline(two_views: list[np.ndarray]) -> None:
    """PerViewTransformer chains with sklearn's own Pipeline and a CCA model.

    No dedicated multiview Pipeline class is needed: as long as every step
    preserves the `list[ArrayLike]` views convention on fit/transform,
    sklearn.pipeline.Pipeline works unmodified.
    """
    pipe = Pipeline(
        [
            ("scale", PerViewTransformer(StandardScaler())),
            ("pca", PerViewTransformer(PCA(n_components=3))),
            ("cca", CCA(latent_dimensions=2)),
        ]
    )
    scores = pipe.fit_transform(two_views)
    assert len(scores) == 2
    assert scores[0].shape == (two_views[0].shape[0], 2)


def test_pipeline_with_multiview_grid_search(two_views: list[np.ndarray]) -> None:
    """A PerViewTransformer -> CCA pipeline works as GridSearchCV's estimator."""
    pipe = Pipeline(
        [
            ("scale", PerViewTransformer(StandardScaler())),
            ("cca", CCA()),
        ]
    )
    gs = GridSearchCV(pipe, param_grid={"cca__latent_dimensions": [1, 2]}, cv=2)
    gs.fit(two_views)
    assert gs.best_params_["cca__latent_dimensions"] in [1, 2]


# ---------------------------------------------------------------------------
# Three-view model
# ---------------------------------------------------------------------------


def test_three_views(three_views: list[np.ndarray]) -> None:
    """Works with more than two views."""
    pvt = PerViewTransformer(StandardScaler())
    out = pvt.fit_transform(three_views)
    assert len(out) == 3
