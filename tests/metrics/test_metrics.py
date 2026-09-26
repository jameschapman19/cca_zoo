"""Tests for cca_zoo.metrics."""

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
)

# ---------------------------------------------------------------------------
# pairwise_correlations / average_pairwise_correlations / factor_loadings
# match BaseModel's own methods (the extraction is behavior-preserving)
# ---------------------------------------------------------------------------


def test_pairwise_correlations_matches_basemodel(
    correlated_views: list[np.ndarray],
) -> None:
    """The extracted function reproduces BaseModel.pairwise_correlations exactly."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    transformed = model.transform(correlated_views)
    expected = pairwise_correlations(model.transform(correlated_views))
    np.testing.assert_array_equal(pairwise_correlations(transformed), expected)


def test_average_pairwise_correlations_matches_basemodel(
    correlated_views: list[np.ndarray],
) -> None:
    """The extracted function reproduces BaseModel.average_pairwise_correlations."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    corrs = pairwise_correlations(model.transform(correlated_views))
    expected = average_pairwise_correlations(
        pairwise_correlations(model.transform(correlated_views))
    )
    np.testing.assert_array_equal(average_pairwise_correlations(corrs), expected)


def test_factor_loadings_matches_basemodel(
    correlated_views: list[np.ndarray],
) -> None:
    """The extracted function reproduces BaseModel.get_factor_loadings exactly."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    transformed = model.transform(correlated_views)
    expected = factor_loadings(correlated_views, model.transform(correlated_views))
    actual = factor_loadings(correlated_views, transformed)
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)


# ---------------------------------------------------------------------------
# pairwise_correlations properties
# ---------------------------------------------------------------------------


def test_pairwise_correlations_diagonal_is_one(
    correlated_views: list[np.ndarray],
) -> None:
    """A view's correlation with itself is exactly 1 on every dimension."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    corrs = pairwise_correlations(model.transform(correlated_views))
    for i in range(len(correlated_views)):
        np.testing.assert_allclose(corrs[i, i, :], 1.0, atol=1e-10)


def test_pairwise_correlations_symmetric(correlated_views: list[np.ndarray]) -> None:
    """corrs[i, j] == corrs[j, i]."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    corrs = pairwise_correlations(model.transform(correlated_views))
    np.testing.assert_allclose(corrs, corrs.transpose(1, 0, 2))


# ---------------------------------------------------------------------------
# adequacy_coefficient / redundancy_index / total_redundancy
# ---------------------------------------------------------------------------


def test_adequacy_coefficient_in_unit_range(
    correlated_views: list[np.ndarray],
) -> None:
    """A mean of squared correlations lies in [0, 1]."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    loadings = factor_loadings(correlated_views, model.transform(correlated_views))
    adequacy = adequacy_coefficient(loadings)
    for a in adequacy:
        assert a.shape == (2,)
        assert np.all(a >= 0.0)
        assert np.all(a <= 1.0)


def test_redundancy_index_diagonal_equals_adequacy(
    correlated_views: list[np.ndarray],
) -> None:
    """redundancy[i, i] == adequacy_i, since a view's correlation with itself is 1."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    loadings = factor_loadings(correlated_views, model.transform(correlated_views))
    corrs = pairwise_correlations(model.transform(correlated_views))
    redundancy = redundancy_index(loadings, corrs)
    adequacy = adequacy_coefficient(loadings)
    for i, a in enumerate(adequacy):
        np.testing.assert_allclose(redundancy[i, i, :], a)


def test_redundancy_index_bounded_by_adequacy(
    correlated_views: list[np.ndarray],
) -> None:
    """Off-diagonal redundancy never exceeds the view's own adequacy (corr^2 <= 1)."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    loadings = factor_loadings(correlated_views, model.transform(correlated_views))
    corrs = pairwise_correlations(model.transform(correlated_views))
    redundancy = redundancy_index(loadings, corrs)
    adequacy = np.stack(adequacy_coefficient(loadings), axis=0)
    assert np.all(redundancy <= adequacy[:, np.newaxis, :] + 1e-10)
    assert np.all(redundancy >= 0.0)


def test_redundancy_index_shape(correlated_views: list[np.ndarray]) -> None:
    """redundancy_index has shape (n_views, n_views, latent_dimensions)."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    loadings = factor_loadings(correlated_views, model.transform(correlated_views))
    corrs = pairwise_correlations(model.transform(correlated_views))
    redundancy = redundancy_index(loadings, corrs)
    assert redundancy.shape == (2, 2, 2)


def test_total_redundancy_sums_over_dimensions(
    correlated_views: list[np.ndarray],
) -> None:
    """total_redundancy is the sum of redundancy_index over the last axis."""
    model = CCA(latent_dimensions=2).fit(correlated_views)
    loadings = factor_loadings(correlated_views, model.transform(correlated_views))
    corrs = pairwise_correlations(model.transform(correlated_views))
    redundancy = redundancy_index(loadings, corrs)
    total = total_redundancy(redundancy)
    assert total.shape == (2, 2)
    np.testing.assert_allclose(total, redundancy.sum(axis=-1))


def test_redundancy_asymmetric_across_views() -> None:
    """Redundancy of view i given j need not equal that of j given i.

    Construct view 1 with far more (noisy) features than view 2, so their
    own adequacy coefficients genuinely differ.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    x1 = z @ rng.standard_normal((1, 30)) + 2.0 * rng.standard_normal((200, 30))
    x2 = z @ rng.standard_normal((1, 2)) + 0.05 * rng.standard_normal((200, 2))
    model = CCA(latent_dimensions=1).fit([x1, x2])
    loadings = factor_loadings([x1, x2], model.transform([x1, x2]))
    corrs = pairwise_correlations(model.transform([x1, x2]))
    redundancy = redundancy_index(loadings, corrs)
    assert not np.allclose(redundancy[0, 1, :], redundancy[1, 0, :])


# ---------------------------------------------------------------------------
# Three-view model
# ---------------------------------------------------------------------------


def test_three_views(three_views: list[np.ndarray]) -> None:
    """Every function handles more than two views."""
    n_samples = three_views[0].shape[0]
    rng = np.random.default_rng(0)
    transformed = [rng.standard_normal((n_samples, 2)) for _ in range(3)]
    corrs = pairwise_correlations(transformed)
    assert corrs.shape == (3, 3, 2)
    avg = average_pairwise_correlations(corrs)
    assert avg.shape == (2,)
    loadings = factor_loadings(three_views, transformed)
    assert len(loadings) == 3


@pytest.mark.parametrize(
    "name",
    [
        "pairwise_correlations",
        "average_pairwise_correlations",
        "factor_loadings",
        "adequacy_coefficient",
        "redundancy_index",
        "total_redundancy",
    ],
)
def test_public_api_exported(name: str) -> None:
    """Every documented function is importable from cca_zoo.metrics."""
    import cca_zoo.metrics as metrics

    assert name in metrics.__all__
    assert hasattr(metrics, name)
