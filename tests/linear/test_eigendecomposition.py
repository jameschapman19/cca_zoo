"""The closed-form linear models, checked against CCA and each other."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import (
    CCA,
    CCAR3,
    ECCA,
    GCCA,
    GRCCA,
    MCCA,
    GraphicalLassoCCA,
    PartialCCA,
    RidgeCCA,
)
from tests._helpers import canonical_correlations


def _two_factor_views(n: int = 500) -> list[np.ndarray]:
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, 2)) * [1.0, 0.6]
    return [
        z @ rng.standard_normal((2, p)) + 0.5 * rng.standard_normal((n, p))
        for p in (8, 6)
    ]


def test_cca_variates_are_uncorrelated_and_ordered() -> None:
    """Each view's canonical variates are uncorrelated, strongest first."""
    views = _two_factor_views()
    model = CCA(n_components=2).fit(views)
    for z in model.transform(views):
        np.testing.assert_allclose(np.corrcoef(z.T), np.eye(2), atol=1e-6)
    corrs = canonical_correlations(model, views)
    assert corrs[0] > corrs[1]


@pytest.mark.parametrize(
    "model",
    [
        RidgeCCA(n_components=2, shrinkage=0.0),
        MCCA(n_components=2),
        GRCCA(n_components=2, shrinkage=0.0),
        CCAR3(n_components=2),
        CCAR3(n_components=2, highdim=False, ledoit_wolf=False),
        ECCA(n_components=2),
    ],
    ids=["RidgeCCA", "MCCA", "GRCCA", "CCAR3", "CCAR3-lowdim", "ECCA"],
)
def test_reduces_to_cca_without_regularisation(model: object) -> None:
    """With no penalty, each generalisation recovers CCA's correlations."""
    views = _two_factor_views()
    np.testing.assert_allclose(
        canonical_correlations(model.fit(views), views),
        canonical_correlations(CCA(n_components=2).fit(views), views),
        atol=1e-3,
    )


def test_ccar3_penalty_drops_whole_features(two_views: list[np.ndarray]) -> None:
    """The row-group penalty removes some features from every component."""
    weights = (
        CCAR3(n_components=2, alpha=0.5, ledoit_wolf=False).fit(two_views).weights_[0]
    )
    dropped = ~weights.any(axis=1)
    assert dropped.any() and not dropped.all()
    assert not CCAR3(alpha=1e3).fit(two_views).weights_[0].any()


@pytest.mark.parametrize(
    "model",
    [MCCA(), GCCA(), GraphicalLassoCCA(alpha=0.5)],
    ids=lambda m: type(m).__name__,
)
def test_fits_more_features_than_samples(model: object) -> None:
    """Singular covariances are floored rather than inverted."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((30, 1))
    views = [
        z @ rng.standard_normal((1, p)) + 0.5 * rng.standard_normal((30, p))
        for p in (50, 40)
    ]
    assert model.fit(views).score(views) > 0.3


def test_ecca_penalty_drops_features(two_views: list[np.ndarray]) -> None:
    """The entrywise penalty removes features whose coefficient rows it zeroes."""
    weights = ECCA(n_components=2, alpha=0.5, tol=1e-8).fit(two_views).weights_[0]
    dropped = ~weights.any(axis=1)
    assert dropped.any() and not dropped.all()


def test_partial_cca_removes_a_confound() -> None:
    """Conditioned on a dominant confound, the shared signal is recovered."""
    rng = np.random.default_rng(0)
    z, confound = rng.standard_normal((200, 2)), rng.standard_normal((200, 1))
    views = [
        z @ rng.standard_normal((2, 6))
        + 5.0 * confound @ rng.standard_normal((1, 6))
        + 0.1 * rng.standard_normal((200, 6))
        for _ in range(2)
    ]
    model = PartialCCA(n_components=2).fit(views, partials=confound)
    z1, z2 = model.transform(views, partials=confound)
    assert all(np.corrcoef(z1[:, d], z2[:, d])[0, 1] > 0.5 for d in range(2))
    with pytest.raises(ValueError, match="partials"):
        PartialCCA().fit(views)


def test_grcca_weights_are_on_the_original_features(
    two_views: list[np.ndarray],
) -> None:
    """Group penalties augment the features internally, not in weights_."""
    groups = [np.arange(v.shape[1]) % 3 for v in two_views]
    model = GRCCA(shrinkage=[0.5, 0.0]).fit(two_views, feature_groups=groups)
    assert [w.shape[0] for w in model.weights_] == [10, 8]
    with pytest.warns(UserWarning, match="feature_groups"):
        GRCCA(shrinkage=0.5).fit(two_views)
