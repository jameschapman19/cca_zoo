"""Tests for nonparametric (kernel-based) CCA methods: KCCA, KGCCA, KTCCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import CCA
from cca_zoo.nonparametric import KCCA, KGCCA, KTCCA

ALL_KERNEL_MODELS = [KCCA, KGCCA, KTCCA]

# Only KTCCA has a random_state parameter
_KERNEL_MODELS_WITH_RANDOM_STATE = {KTCCA}


def _make_kernel_model(
    ModelClass: type,
    n_components: int = 1,
    c: float = 0.1,
    **kwargs: object,
) -> object:
    """Construct a kernel model, passing random_state=0 only if supported."""
    if ModelClass in _KERNEL_MODELS_WITH_RANDOM_STATE:
        return ModelClass(n_components=n_components, c=c, random_state=0, **kwargs)
    return ModelClass(n_components=n_components, c=c, **kwargs)


# ---------------------------------------------------------------------------
# Per-view kernel specification
# ---------------------------------------------------------------------------


def test_kcca_per_view_kernel(two_views_small: list[np.ndarray]) -> None:
    """KCCA accepts per-view kernel specification as a list."""
    model = KCCA(n_components=1, c=0.1, kernel=["linear", "rbf"]).fit(two_views_small)
    result = model.transform(two_views_small)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_kcca_linear_kernel_matches_cca(correlated_views: list[np.ndarray]) -> None:
    """KCCA with a linear kernel recovers the same correlations as CCA (small c)."""
    k = 2
    s_cca = CCA(n_components=k).fit(correlated_views).score(correlated_views)
    s_kcca = (
        KCCA(n_components=k, kernel="linear", c=1e-4)
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(s_kcca, s_cca, atol=1e-3)


def test_kcca_regularisation_reduces_correlation(
    correlated_views: list[np.ndarray],
) -> None:
    """Higher regularisation c gives lower (or equal) training correlation."""
    s_low = (
        KCCA(n_components=1, kernel="linear", c=1e-4)
        .fit(correlated_views)
        .score(correlated_views)
    )
    s_high = (
        KCCA(n_components=1, kernel="linear", c=1.0)
        .fit(correlated_views)
        .score(correlated_views)
    )
    assert s_low >= s_high - 1e-6


@pytest.mark.parametrize("ModelClass", ALL_KERNEL_MODELS)
def test_transform_centres_like_fit(ModelClass: type) -> None:
    """Transforming the training data reproduces the fit-time kernel scores.

    The kernel is formed against the centred training views, so new data
    must be centred the same way; data far from the origin makes an
    uncentred transform collapse an RBF kernel to zero.
    """
    from sklearn.metrics import pairwise_kernels

    rng = np.random.default_rng(0)
    views = [5.0 + rng.standard_normal((40, 3)) for _ in range(2)]
    model = _make_kernel_model(ModelClass, kernel="rbf").fit(views)
    for i, scores in enumerate(model.transform(views)):
        kernel = pairwise_kernels(model.train_views_[i], metric="rbf")
        np.testing.assert_allclose(scores, kernel @ model.weights_[i], atol=1e-10)
