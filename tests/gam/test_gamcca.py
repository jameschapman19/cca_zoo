"""GAMCCA: an additive P-spline encoder per view, as mgcv's s(x, bs="ps")."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.gam import GAMCCA
from cca_zoo.linear import RidgeCCA


def test_recovers_a_smooth_nonmonotonic_relationship() -> None:
    """Linear CCA cannot relate z to z**2, which are uncorrelated; GAMCCA can."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(1000)
    views = [
        np.column_stack([f(z) + 0.3 * rng.standard_normal(1000) for _ in range(5)])
        for f in (np.asarray, np.square)
    ]
    train, test = [v[:500] for v in views], [v[500:] for v in views]
    gam = GAMCCA().fit(train).score(test)
    assert gam > 0.9
    assert gam > RidgeCCA(c=0.3).fit(train).score(test) + 0.5


def test_k_sets_each_views_basis_size(two_views_small: list[np.ndarray]) -> None:
    """Each view's basis has its own size k, as in mgcv."""
    model = GAMCCA(k=[6, 12]).fit(two_views_small)
    assert [enc.n_splines_ for enc in model.encoders_] == [6, 12]


def test_large_sp_makes_that_views_smooths_linear(
    correlated_views: list[np.ndarray],
) -> None:
    """A huge smoothing parameter removes the penalised differences of one view only."""
    model = GAMCCA(sp=[1e-3, 1e6]).fit(correlated_views)
    wiggle = [
        np.linalg.norm(enc.penalty_factor_ @ enc.coef_) / np.linalg.norm(scores)
        for enc, scores in zip(model.encoders_, model.transform(correlated_views))
    ]
    assert wiggle[1] < 1e-2 * wiggle[0]


def test_m_sets_spline_and_penalty_order(two_views_small: list[np.ndarray]) -> None:
    """m=(1, 1) is mgcv's quadratic spline with a first-difference penalty."""
    encoder = GAMCCA(k=8, m=(1, 1)).fit(two_views_small).encoders_[0]
    assert encoder._spline.degree == 2
    assert encoder.penalty_factor_.shape == (7 * encoder.p, 8 * encoder.p)
    with pytest.raises(ValueError, match="too small"):
        GAMCCA(k=4, m=(3, 2)).fit(two_views_small)


def test_shape_functions_sum_to_the_encoding(two_views_small: list[np.ndarray]) -> None:
    """The additive terms of a view sum to its encoding."""
    model = GAMCCA().fit(two_views_small)
    view = two_views_small[0]
    total = sum(model.shape_function(0, j, view[:, j]) for j in range(view.shape[1]))
    np.testing.assert_allclose(total, model.transform(two_views_small)[0], atol=1e-6)
