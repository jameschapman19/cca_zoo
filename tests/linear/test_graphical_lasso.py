"""GraphicalLassoCCA: MCCA with a sparse-precision covariance estimate per view."""

from __future__ import annotations

import numpy as np

from cca_zoo.linear import MCCA, GraphicalLassoCCA


def _offdiagonal_nonzeros(precision: np.ndarray) -> int:
    return int(np.sum(np.abs(precision[~np.eye(len(precision), dtype=bool)]) > 1e-8))


def test_small_alpha_is_mcca(two_views_small: list[np.ndarray]) -> None:
    """With almost no penalty the fit is MCCA's."""
    np.testing.assert_allclose(
        GraphicalLassoCCA(alpha=0.01).fit(two_views_small).score(two_views_small),
        MCCA(pca=False).fit(two_views_small).score(two_views_small),
        atol=1e-2,
    )


def test_alpha_per_view_sparsifies_that_precision(
    two_views_small: list[np.ndarray],
) -> None:
    """An alpha above the cross-validated one sparsifies that view's precision."""
    precision = GraphicalLassoCCA(alpha=[None, 2.0]).fit(two_views_small).precision_
    assert _offdiagonal_nonzeros(precision[1]) < _offdiagonal_nonzeros(precision[0])


def test_alpha_is_the_same_in_any_units(two_views_small: list[np.ndarray]) -> None:
    """Rescaling a view's features rescales its covariance, not the fit."""
    scaled = [
        two_views_small[0] * np.arange(1, 1 + two_views_small[0].shape[1]),
        two_views_small[1],
    ]
    np.testing.assert_allclose(
        GraphicalLassoCCA(alpha=0.2).fit(scaled).score(scaled),
        GraphicalLassoCCA(alpha=0.2).fit(two_views_small).score(two_views_small),
    )
