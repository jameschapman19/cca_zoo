"""Tests for GAMCCA.

Unlike TreeCCA, GAMCCA has no optional dependency (it is built entirely on
scikit-learn's SplineTransformer/Ridge, already required by cca_zoo), so
these tests run unconditionally.
"""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.gam import GAMCCA


def _make_model(n_components: int = 1, **kwargs: object) -> GAMCCA:
    return GAMCCA(n_components=n_components, **kwargs)


# ---------------------------------------------------------------------------
# Per-view parameters
# ---------------------------------------------------------------------------


def test_per_view_k_list(two_views_small: list[np.ndarray]) -> None:
    """A per-view k list gives each view its own basis dimension, as mgcv's k."""
    model = _make_model(k=[6, 12]).fit(two_views_small)
    assert [enc.n_splines_ for enc in model.encoders_] == [6, 12]


def test_per_view_sp_smooths_only_that_view(correlated_views: list[np.ndarray]) -> None:
    """A large sp flattens that view's smooths to lines; the other view stays free.

    The P-spline penalty acts on second differences of neighbouring
    coefficients, so a huge sp drives them to zero (a straight line per
    feature) rather than shrinking the coefficients themselves.
    """
    model = _make_model(sp=[1e-3, 1e6]).fit(correlated_views)
    wiggle = [
        np.linalg.norm(enc.penalty_factor_ @ enc.coef_) / np.linalg.norm(enc.predict())
        for enc in model.encoders_
    ]
    assert wiggle[1] < 1e-2 * wiggle[0]


def test_m_tuple_sets_spline_and_penalty_order(
    two_views_small: list[np.ndarray],
) -> None:
    """m=(order, penalty order) as mgcv's: quadratic splines, first differences."""
    model = _make_model(k=8, m=(1, 1)).fit(two_views_small)
    enc = model.encoders_[0]
    assert enc._spline.degree == 2
    # First differences: one fewer row than coefficients per feature.
    assert enc.penalty_factor_.shape == (7 * enc.p, 8 * enc.p)


def test_k_too_small_for_m_raises(two_views_small: list[np.ndarray]) -> None:
    """A cubic P-spline needs k >= 4; smaller k is refused with the reason."""
    with pytest.raises(ValueError, match="too small"):
        _make_model(k=4, m=(3, 2)).fit(two_views_small)


def test_fit_is_deterministic(correlated_views: list[np.ndarray]) -> None:
    """The closed-form fit has no random start: two fits agree up to sign."""
    a = _make_model(n_components=2).fit(correlated_views)
    b = _make_model(n_components=2).fit(correlated_views)
    for za, zb in zip(a.transform(correlated_views), b.transform(correlated_views)):
        np.testing.assert_allclose(np.abs(za), np.abs(zb), atol=1e-8)


def test_data_free_splines_are_filled_in_by_the_penalty() -> None:
    """Evenly spaced knots over a heavy-tailed feature leave splines with no data.

    P-splines keep them and let the difference penalty interpolate their
    coefficients; the fit must stay finite and smooth across the gap.
    """
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_t(1.5, n)
    views = [
        np.column_stack([z, rng.standard_normal(n)]),
        np.column_stack(
            [np.tanh(z) + 0.1 * rng.standard_normal(n), rng.standard_normal(n)]
        ),
    ]
    model = _make_model().fit(views)
    grid = np.linspace(z.min(), z.max(), 400)
    curve = model.shape_function(0, 0, grid)[:, 0]
    assert np.all(np.isfinite(curve))
    assert model.score(views) > 0.8


# ---------------------------------------------------------------------------
# shape_function
# ---------------------------------------------------------------------------


def test_shape_function_sums_to_prediction(two_views_small: list[np.ndarray]) -> None:
    """Summed shape_function terms reproduce the encoder's raw prediction.

    Summing every feature's shape_function at the training values should
    reproduce the encoder's raw (base-margin-free) training prediction.
    """
    model = _make_model(n_components=1).fit(two_views_small)
    view = two_views_small[0]
    total = sum(model.shape_function(0, j, view[:, j]) for j in range(view.shape[1]))
    np.testing.assert_allclose(total, model.encoders_[0].predict(), atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gamcca_outperforms_linear_and_tree_on_smooth_nonmonotonic_data() -> None:
    """GAMCCA beats RidgeCCA and TreeCCA on a held-out, smooth-but-nonlinear task.

    View 1 is a noisy linear copy of a shared latent factor ``z``; view 2 is
    a noisy linear copy of ``z ** 2`` — a smooth but *non-monotonic* (even)
    transform, chosen so that ``corr(z, z**2) ~ 0`` for symmetric ``z``.  No
    linear combination of view 1's raw features can align with view 2 (so
    ``RidgeCCA`` is expected to fail), while a per-view nonlinear encoder that
    (approximately) learns the "square" transform recovers near-perfect
    cross-view correlation. GAMCCA's B-spline basis represents a quadratic
    almost exactly and its closed-form fit is the global optimum, so it
    should beat TreeCCA's piecewise-constant approximation.

    Marked slow since it also requires TreeCCA's optional ``xgboost``
    dependency, not part of the base ``dev`` install.
    """
    pytest.importorskip("xgboost", reason="xgboost is not installed")
    from cca_zoo.linear import RidgeCCA
    from cca_zoo.tree import XGBoostCCA

    rng = np.random.default_rng(0)
    n_train, n_test, p, noise = 500, 500, 5, 0.3
    n = n_train + n_test
    z = rng.standard_normal(n)
    X1 = np.column_stack([z + noise * rng.standard_normal(n) for _ in range(p)])
    X2 = np.column_stack([z**2 + noise * rng.standard_normal(n) for _ in range(p)])
    X1_tr, X1_te = X1[:n_train], X1[n_train:]
    X2_tr, X2_te = X2[:n_train], X2[n_train:]

    gam = GAMCCA(n_components=1)
    gam_test = gam.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    tree = XGBoostCCA(n_components=1, random_state=0)
    tree_test = tree.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    rcca = RidgeCCA(n_components=1, c=[0.3, 0.3])
    rcca_test = rcca.fit([X1_tr, X2_tr]).score([X1_te, X2_te])

    assert gam_test > 0.9, (
        f"Expected GAMCCA to recover the relationship, got {gam_test}"
    )
    assert gam_test > tree_test, (
        f"Expected GAMCCA ({gam_test}) to beat TreeCCA ({tree_test})"
    )
    assert gam_test > rcca_test + 0.5, (
        f"Expected GAMCCA ({gam_test}) to clearly beat linear RidgeCCA ({rcca_test})"
    )
