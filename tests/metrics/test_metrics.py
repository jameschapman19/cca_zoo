"""The correlation and redundancy metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA
from cca_zoo.metrics import (
    adequacy_coefficient,
    average_pairwise_correlations,
    factor_loadings,
    pairwise_correlations,
    redundancy_index,
    total_redundancy,
    variance_explained,
)


@pytest.fixture
def fitted(correlated_views: list[np.ndarray]) -> tuple[list, list]:
    """Views and their CCA scores."""
    return correlated_views, CCA(n_components=2).fit(correlated_views).transform(
        correlated_views
    )


def test_correlations_are_pearsons(fitted: tuple[list, list]) -> None:
    """Every correlation and loading is the np.corrcoef value."""
    views, scores = fitted
    corrs = pairwise_correlations(scores)
    loadings = factor_loadings(views, scores)
    for d in range(2):
        expected = np.corrcoef(scores[0][:, d], scores[1][:, d])[0, 1]
        assert corrs[0, 1, d] == pytest.approx(expected)
        assert average_pairwise_correlations(corrs)[d] == pytest.approx(expected)
        for view, score, loading in zip(views, scores, loadings):
            for j in range(view.shape[1]):
                assert loading[j, d] == pytest.approx(
                    np.corrcoef(view[:, j], score[:, d])[0, 1]
                )
    np.testing.assert_allclose(corrs, corrs.transpose(1, 0, 2))
    np.testing.assert_allclose(corrs[[0, 1], [0, 1]], 1.0)


def test_redundancy_is_bounded_by_adequacy(fitted: tuple[list, list]) -> None:
    """Redundancy with itself is a view's adequacy; with another, at most that."""
    views, scores = fitted
    loadings = factor_loadings(views, scores)
    adequacy = np.stack(adequacy_coefficient(loadings))
    redundancy = redundancy_index(loadings, pairwise_correlations(scores))
    assert np.all((adequacy >= 0) & (adequacy <= 1))
    np.testing.assert_allclose(redundancy[[0, 1], [0, 1]], adequacy)
    assert np.all((redundancy >= 0) & (redundancy <= adequacy[:, None] + 1e-10))
    np.testing.assert_allclose(total_redundancy(redundancy), redundancy.sum(axis=-1))


def test_variance_explained_is_the_r_squared_of_each_dimension_alone() -> None:
    """Each fraction is the R squared of regressing the view on one score."""
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((200, 2))
    view = scores @ rng.standard_normal((2, 5)) + rng.standard_normal((200, 5))
    explained = variance_explained([view], [scores])
    assert explained.shape == (1, 2)
    centred = view - view.mean(axis=0)
    for k in range(2):
        z = scores[:, k] - scores[:, k].mean()
        fitted = np.outer(z, z @ centred / (z @ z))
        assert explained[0, k] == pytest.approx(np.sum(fitted**2) / np.sum(centred**2))


def test_variance_explained_skips_the_samples_a_view_lacks() -> None:
    """NaN rows, as of a sample missing the view, do not enter the fraction."""
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((100, 1))
    view = scores @ rng.standard_normal((1, 4)) + rng.standard_normal((100, 4))
    expected = variance_explained([view[20:]], [scores[20:]])
    view[:20] = np.nan
    scores = scores.copy()
    scores[:20] = np.nan
    np.testing.assert_allclose(variance_explained([view], [scores]), expected)
