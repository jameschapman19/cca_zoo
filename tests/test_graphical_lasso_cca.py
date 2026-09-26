"""Tests for cca_zoo.linear._graphical_lasso_cca (MCCA with a sparse-precision B)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag
from sklearn.covariance import GraphicalLasso

from cca_zoo.linear import MCCA, GraphicalLassoCCA


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> GraphicalLassoCCA:
    return GraphicalLassoCCA(latent_dimensions=latent_dimensions, **kwargs)


# ---------------------------------------------------------------------------
# fit completes / shapes
# ---------------------------------------------------------------------------


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = _make_model()
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model()
    fitted = model.fit(three_views_small)
    assert fitted is model


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """Weights are real (p_i, k) arrays and transform(v) == centred(v) @ weights."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    weights = model.weights_
    assert len(weights) == 2
    for w, v in zip(weights, two_views_small):
        assert w.shape == (v.shape[1], k)

    transformed = model.transform(two_views_small)
    for v, w, t, mean in zip(two_views_small, weights, transformed, model.means_):
        np.testing.assert_allclose((v - mean) @ w, t, atol=1e-8)


def test_weights_not_fitted_raises() -> None:
    """Transform before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = GraphicalLassoCCA()
    with pytest.raises(NotFittedError):
        model.transform([np.ones((3, 2)), np.ones((3, 2))])


def test_covariance_and_precision_shapes(three_views_small: list[np.ndarray]) -> None:
    """covariance_/precision_ are one symmetric (p_i, p_i) array per view."""
    model = _make_model().fit(three_views_small)
    assert len(model.covariance_) == 3
    assert len(model.precision_) == 3
    for v, cov, prec in zip(three_views_small, model.covariance_, model.precision_):
        p = v.shape[1]
        assert cov.shape == (p, p)
        assert prec.shape == (p, p)
        np.testing.assert_allclose(cov, cov.T)
        np.testing.assert_allclose(prec, prec.T)


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
    gl_model = GraphicalLassoCCA(latent_dimensions=1, alpha=0.01).fit(two_views_small)
    mcca_model = MCCA(latent_dimensions=1, c=0.0, pca=False).fit(two_views_small)

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

    model = GraphicalLassoCCA(latent_dimensions=1, alpha=0.5, max_iter=500).fit(
        [x1, x2]
    )
    assert model.score([x1, x2]) > 0.3


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = GraphicalLassoCCA(latent_dimensions=2, c=0.1, alpha=0.05, mode="cd")
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_alpha_raises() -> None:
    """A negative alpha is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        GraphicalLassoCCA(alpha=-1.0)._validate_params()


def test_invalid_mode_raises() -> None:
    """An unrecognised solver mode is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        GraphicalLassoCCA(mode="not-a-mode")._validate_params()


def test_pca_not_exposed() -> None:
    """No pca constructor parameter -- this always solves in feature space."""
    with pytest.raises(TypeError):
        GraphicalLassoCCA(pca=True)  # type: ignore[call-arg]
