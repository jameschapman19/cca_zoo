"""Tests for cca_zoo.linear._graphical_lasso_cca (MCCA with a sparse-precision B)."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from sklearn.covariance import GraphicalLasso

from cca_zoo.linear import MCCA, GraphicalLassoCCA


def _make_model(n_components: int = 1, **kwargs: object) -> GraphicalLassoCCA:
    return GraphicalLassoCCA(n_components=n_components, **kwargs)


# ---------------------------------------------------------------------------
# Correctness: B is really built from the graphical-lasso covariance
# ---------------------------------------------------------------------------


def test_build_b_matches_manual_graphical_lasso(
    two_views_small: list[np.ndarray],
) -> None:
    """_build_B matches independently-fit GraphicalLasso covariances."""
    alpha = 0.2
    model = _make_model(alpha=alpha)
    views_ = model._setup_fit(two_views_small)
    B = model._build_B(views_, c=[0.0, 0.0])

    manual_covs = [
        GraphicalLasso(alpha=alpha, assume_centered=True).fit(v).covariance_
        for v in views_
    ]
    expected = np.asarray(block_diag(*manual_covs)) / len(views_)
    np.testing.assert_allclose(B, expected, atol=1e-8)


def test_larger_alpha_gives_sparser_precision(
    two_views_small: list[np.ndarray],
) -> None:
    """Increasing alpha drives more off-diagonal precision entries to (near) zero."""
    model_light = _make_model(alpha=1e-4).fit(two_views_small)
    model_heavy = _make_model(alpha=2.0).fit(two_views_small)

    def n_nonzero_offdiag(prec: np.ndarray, tol: float = 1e-8) -> int:
        mask = ~np.eye(prec.shape[0], dtype=bool)
        return int(np.sum(np.abs(prec[mask]) > tol))

    for light, heavy in zip(model_light.precision_, model_heavy.precision_):
        assert n_nonzero_offdiag(heavy) <= n_nonzero_offdiag(light)


def test_small_alpha_close_to_plain_mcca(two_views_small: list[np.ndarray]) -> None:
    """A small alpha gives near-unregularised results, close to plain MCCA.

    Uses well-conditioned (full-rank, i.i.d.) views rather than the
    ``correlated_views`` fixture: that fixture's covariance is close to
    rank-2 (features built from a 2-dimensional latent factor plus tiny
    noise), which pushes ``GraphicalLasso``'s own coordinate-descent solver
    into a genuine non-convergence edge case at small ``alpha`` -- unrelated
    to this class's own logic.
    """
    gl_model = GraphicalLassoCCA(n_components=1, alpha=0.01).fit(two_views_small)
    mcca_model = MCCA(n_components=1, c=0.0, pca=False).fit(two_views_small)

    gl_corr = gl_model.score(two_views_small)
    mcca_corr = mcca_model.score(two_views_small)
    np.testing.assert_allclose(gl_corr, mcca_corr, atol=1e-2)


def test_per_view_alpha_list(two_views_small: list[np.ndarray]) -> None:
    """A per-view alpha list applies a different penalty to each view."""
    model = _make_model(alpha=[1e-4, 2.0]).fit(two_views_small)

    def n_nonzero_offdiag(prec: np.ndarray, tol: float = 1e-8) -> int:
        mask = ~np.eye(prec.shape[0], dtype=bool)
        return int(np.sum(np.abs(prec[mask]) > tol))

    assert n_nonzero_offdiag(model.precision_[1]) <= n_nonzero_offdiag(
        model.precision_[0]
    )


def test_alpha_none_uses_cv(two_views_small: list[np.ndarray]) -> None:
    """alpha=None auto-selects a penalty per view via GraphicalLassoCV."""
    model = _make_model(alpha=None).fit(two_views_small)
    assert len(model.precision_) == 2


# ---------------------------------------------------------------------------
# High-dimensional data (p > n): the regime this class targets
# ---------------------------------------------------------------------------


def test_fits_in_high_dimensional_regime() -> None:
    """Fits cleanly and scores finitely when features outnumber samples."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((30, 1))
    x1 = z @ rng.standard_normal((1, 50)) + 0.5 * rng.standard_normal((30, 50))
    x2 = z @ rng.standard_normal((1, 40)) + 0.5 * rng.standard_normal((30, 40))

    model = GraphicalLassoCCA(n_components=1, alpha=0.5, max_iter=500).fit([x1, x2])
    assert model.score([x1, x2]) > 0.3
