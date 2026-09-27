"""Tests for eigendecomposition-based linear CCA methods.

Covers CCA, RidgeCCA, PLS, MCCA, GCCA, TCCA.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import (
    CCA,
    CCAR3,
    ECCA,
    GRCCA,
    MCCA,
    PartialCCA,
    RidgeCCA,
)
from tests._helpers import canonical_correlations

# ---------------------------------------------------------------------------
# CCA correlated views: score should be high
# ---------------------------------------------------------------------------


def test_cca_perfect_correlation_identical_views() -> None:
    """CCA on identical views (X1 == X2) should give correlation == 1.0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 5))
    model = CCA(n_components=3).fit([x, x])
    s = model.score([x, x])
    np.testing.assert_allclose(s, 1.0, atol=1e-6)


def test_cca_correlations_are_decreasing(correlated_views: list[np.ndarray]) -> None:
    """Canonical correlations are returned in non-increasing order."""
    s = canonical_correlations(
        CCA(n_components=2).fit(correlated_views), correlated_views
    )
    assert s[0] >= s[1] - 1e-10


def test_rcca_zero_regularisation_matches_cca(
    correlated_views: list[np.ndarray],
) -> None:
    """RCCA with c=0 should give the same correlations as CCA."""
    s_cca = CCA(n_components=2).fit(correlated_views).score(correlated_views)
    s_rcca = (
        RidgeCCA(n_components=2, c=0.0).fit(correlated_views).score(correlated_views)
    )
    np.testing.assert_allclose(s_rcca, s_cca, atol=1e-6)


# ---------------------------------------------------------------------------
# Mathematical properties / correctness
# ---------------------------------------------------------------------------


def test_mcca_two_views_matches_cca(correlated_views: list[np.ndarray]) -> None:
    """MCCA with two views is equivalent to CCA (same canonical correlations)."""
    k = 2
    s_cca = CCA(n_components=k).fit(correlated_views).score(correlated_views)
    s_mcca = MCCA(n_components=k).fit(correlated_views).score(correlated_views)
    np.testing.assert_allclose(s_mcca, s_cca, atol=1e-6)


def test_cca_canonical_variates_are_uncorrelated() -> None:
    """CCA canonical variates are orthogonal across dimensions."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 10))
    k = 3
    model = CCA(n_components=k).fit([x, x])
    Z = model.transform([x, x])
    for z in Z:
        corr = np.corrcoef(z.T)
        off_diag = corr - np.eye(k)
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# PartialCCA
# ---------------------------------------------------------------------------


def test_partial_cca_requires_partials(two_views: list[np.ndarray]) -> None:
    """PartialCCA.fit raises ValueError when partials is not provided."""
    with pytest.raises(ValueError, match="partials"):
        PartialCCA(n_components=1).fit(two_views)


def test_partial_cca_transform_without_partials_falls_back(
    two_views: list[np.ndarray],
) -> None:
    """PartialCCA.transform without partials falls back to a plain projection."""
    rng = np.random.default_rng(1)
    partials = rng.standard_normal((50, 3))
    model = PartialCCA(n_components=1).fit(two_views, partials=partials)
    result = model.transform(two_views)
    assert len(result) == 2


def test_partial_cca_removes_confound_effect() -> None:
    """PartialCCA recovers a shared signal even when a strong confound dominates."""
    rng = np.random.default_rng(0)
    n = 200
    z = rng.standard_normal((n, 2))
    confound = rng.standard_normal((n, 1))
    x1 = (
        z @ rng.standard_normal((2, 6))
        + confound @ rng.standard_normal((1, 6)) * 5.0
        + 0.1 * rng.standard_normal((n, 6))
    )
    x2 = (
        z @ rng.standard_normal((2, 6))
        + confound @ rng.standard_normal((1, 6)) * 5.0
        + 0.1 * rng.standard_normal((n, 6))
    )
    model = PartialCCA(n_components=2).fit([x1, x2], partials=confound)
    z1, z2 = model.transform([x1, x2], partials=confound)
    corrs = np.array(
        [np.corrcoef(z1[:, d], z2[:, d])[0, 1] for d in range(z1.shape[1])]
    )
    assert np.all(corrs > 0.5), (
        f"Expected residual correlation after deconfounding, got {corrs}"
    )


# ---------------------------------------------------------------------------
# GRCCA
# ---------------------------------------------------------------------------


def test_grcca_weights_shape_matches_original_features(
    two_views: list[np.ndarray],
) -> None:
    """GRCCA weights_ operate on the original (un-augmented) feature space."""
    rng = np.random.default_rng(2)
    groups1 = rng.integers(0, 3, size=two_views[0].shape[1])
    groups2 = rng.integers(0, 3, size=two_views[1].shape[1])
    model = GRCCA(n_components=1, c=[0.5, 0.0]).fit(
        two_views, feature_groups=[groups1, groups2]
    )
    for w, view in zip(model.weights_, two_views):
        assert w.shape == (view.shape[1], 1)


def test_grcca_zero_c_matches_mcca(two_views: list[np.ndarray]) -> None:
    """GRCCA with c=0 reduces to plain MCCA."""
    k = 2
    s_grcca = GRCCA(n_components=k, c=0.0).fit(two_views).score(two_views)
    s_mcca = MCCA(n_components=k, pca=False).fit(two_views).score(two_views)
    np.testing.assert_allclose(s_grcca, s_mcca, atol=1e-6)


def test_grcca_default_feature_groups_warns_when_c_nonzero(
    two_views: list[np.ndarray],
) -> None:
    """GRCCA warns when c>0 but no feature_groups are provided."""
    with pytest.warns(UserWarning, match="feature_groups"):
        GRCCA(n_components=1, c=0.5).fit(two_views)


# ---------------------------------------------------------------------------
# CCAR3
# ---------------------------------------------------------------------------


def test_ccar3_highdim_zero_penalty_matches_lowdim(
    correlated_views: list[np.ndarray],
) -> None:
    """With alpha=0, the highdim solver converges to the closed-form B."""
    k = 2
    s_lowdim = (
        CCAR3(n_components=k, highdim=False, ledoit_wolf=False)
        .fit(correlated_views)
        .score(correlated_views)
    )
    s_highdim = (
        CCAR3(
            n_components=k,
            highdim=True,
            ledoit_wolf=False,
            alpha=0.0,
            tol=1e-8,
        )
        .fit(correlated_views)
        .score(correlated_views)
    )
    np.testing.assert_allclose(s_highdim, s_lowdim, atol=1e-4)


def test_ccar3_row_sparse_rrr_reaches_the_true_optimum() -> None:
    """`_row_sparse_rrr` matches an independently-derived global optimum.

    Regression test for a real bug: an earlier hand-rolled ADMM solver for
    this same (convex) row-group-lasso problem had its B-update's linear
    system missing a factor of 2 on the smooth term's gradient (it solved
    `(X^T X / n + rho I) B = ...` where the true gradient of `(1/n)||Y -
    XB||^2` needs `(2/n) X^T X`), which silently doubled the effective
    penalty relative to what `alpha` documents. It converged smoothly and
    passed every existing (qualitative) sparsity test, so only comparing
    its objective value against an independently-implemented solver
    exposed it: it landed on a different, worse-objective stationary point
    on every tested problem. `_row_sparse_rrr` now delegates to
    `sklearn.linear_model.MultiTaskLasso`, whose coordinate descent is
    correct by construction for this convex problem; this test pins that
    down against a from-scratch proximal-gradient (ISTA) solve of the exact
    same objective, independent of both `_row_sparse_rrr` and sklearn.
    """
    from cca_zoo.linear._ccar3 import _row_sparse_rrr

    rng = np.random.default_rng(0)
    n, p, q = 150, 200, 5
    X = rng.standard_normal((n, p))
    true_B = np.zeros((p, q))
    true_B[:10] = rng.standard_normal((10, q))
    Y = X @ true_B + 0.3 * rng.standard_normal((n, q))
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    alpha = 0.1

    def objective(B: np.ndarray) -> float:
        resid = Y - X @ B
        return float((resid**2).sum() / n + alpha * np.linalg.norm(B, axis=1).sum())

    # From-scratch ISTA: proximal gradient descent on the smooth term with a
    # fixed step size (1 / Lipschitz constant of its gradient), followed by
    # a group soft-threshold -- textbook, independent of the code under test.
    L = 2 * np.linalg.eigvalsh(X.T @ X).max() / n
    step = 1.0 / L
    B = np.zeros((p, q))
    for _ in range(20_000):
        grad = (2.0 / n) * X.T @ (X @ B - Y)
        candidate = B - step * grad
        row_norms = np.linalg.norm(candidate, axis=1)
        shrink = np.maximum(0.0, 1.0 - (alpha * step) / np.maximum(row_norms, 1e-30))
        B_next = candidate * shrink[:, None]
        if np.linalg.norm(B_next - B) < 1e-14:
            B = B_next
            break
        B = B_next

    B_fit = _row_sparse_rrr(X, Y, alpha=alpha, max_iter=10_000, tol=1e-10)

    np.testing.assert_allclose(objective(B_fit), objective(B), rtol=1e-6)


def test_ccar3_sparsity_zeroes_rows(two_views: list[np.ndarray]) -> None:
    """A moderate alpha drives some rows of the X weights to zero, not all."""
    model = CCAR3(
        n_components=2,
        highdim=True,
        alpha=0.5,
        ledoit_wolf=False,
        tol=1e-8,
    ).fit(two_views)
    row_norms = np.linalg.norm(model.weights_[0], axis=1)
    assert np.any(row_norms < 1e-3)
    assert np.any(row_norms > 1e-2)


# ---------------------------------------------------------------------------
# ECCA
# ---------------------------------------------------------------------------


def test_ecca_postprocessing_gives_unit_variance_variates(
    correlated_views: list[np.ndarray],
) -> None:
    """ECCA's canonical variates have unit sample variance in every component.

    This is the invariant `_postprocess_rrr_fit`'s whitening step is meant
    to guarantee, checked independently of CCAR3: unlike CCAR3, ECCA does
    not pre-whiten Y (the R reference's ecca() explicitly ignores its Sy
    argument -- see the class docstring), so it can't be cross-checked
    against CCAR3's closed form the way a whitened method could be; this
    checks the property the postprocessing step itself promises instead.
    """
    k = 2
    X, Y = correlated_views
    model = ECCA(n_components=k, alpha=0.0).fit([X, Y])
    Xz, Yz = model.transform([X, Y])
    np.testing.assert_allclose(Xz.var(axis=0, ddof=0), np.ones(k), atol=1e-2)
    np.testing.assert_allclose(Yz.var(axis=0, ddof=0), np.ones(k), atol=1e-2)


def test_ecca_entrywise_sparse_rrr_reaches_the_true_optimum() -> None:
    """`_entrywise_sparse_rrr` matches an independently-derived global optimum.

    Since the entrywise L1 penalty places no coupling between a row's
    entries, the problem separates exactly into one Lasso regression per
    column of `Y_tilde`. This pins that decomposition down against a
    from-scratch proximal-gradient (ISTA) solve of the joint objective,
    independent of both `_entrywise_sparse_rrr` and sklearn.
    """
    from cca_zoo.linear._ecca import _entrywise_sparse_rrr

    rng = np.random.default_rng(0)
    n, p, q = 150, 200, 5
    X = rng.standard_normal((n, p))
    true_B = np.zeros((p, q))
    true_B[:10] = rng.standard_normal((10, q))
    Y = X @ true_B + 0.3 * rng.standard_normal((n, q))
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    alpha = 0.1

    def objective(B: np.ndarray) -> float:
        resid = Y - X @ B
        return float((resid**2).sum() / n + alpha * np.abs(B).sum())

    L = 2 * np.linalg.eigvalsh(X.T @ X).max() / n
    step = 1.0 / L
    B = np.zeros((p, q))
    for _ in range(20_000):
        grad = (2.0 / n) * X.T @ (X @ B - Y)
        cand = B - step * grad
        B_next = np.sign(cand) * np.maximum(np.abs(cand) - alpha * step, 0.0)
        if np.linalg.norm(B_next - B) < 1e-14:
            B = B_next
            break
        B = B_next

    B_fit = _entrywise_sparse_rrr(X, Y, alpha=alpha, max_iter=10_000, tol=1e-10)

    np.testing.assert_allclose(objective(B_fit), objective(B), rtol=1e-6)


def test_ecca_penalty_drops_features(two_views: list[np.ndarray]) -> None:
    """A moderate alpha drops some features from every component, not all."""
    weights = ECCA(n_components=2, alpha=0.5, tol=1e-8).fit(two_views).weights_[0]
    dropped = ~weights.any(axis=1)
    assert dropped.any() and not dropped.all()


@pytest.mark.parametrize(
    "model",
    [CCAR3(), CCAR3(highdim=False, ledoit_wolf=False), ECCA()],
    ids=["CCAR3", "CCAR3-lowdim", "ECCA"],
)
@pytest.mark.parametrize("n_components", [1, 2])
def test_rrr_models_reduce_to_cca_without_a_penalty(
    model: CCAR3 | ECCA, n_components: int
) -> None:
    """With alpha=0 the reduced-rank regressions recover CCA's correlations."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal((500, 2)) * [1.0, 0.6]
    views = [
        z @ rng.standard_normal((2, p)) + 0.5 * rng.standard_normal((500, p))
        for p in (8, 6)
    ]
    model.set_params(n_components=n_components)
    np.testing.assert_allclose(
        canonical_correlations(model.fit(views), views),
        canonical_correlations(CCA(n_components).fit(views), views),
        atol=1e-3,
    )
