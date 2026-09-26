"""Tests for MARSCCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cca_zoo.gam import GAMCCA, MARSCCA
from cca_zoo.gam._marscca import _evaluate_terms, _HingeScorer

# get_params/set_params roundtrip behaviour is exercised generically for
# every model in the package (including MARSCCA) by tests/test_sklearn_compat.py.


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data and returns self."""
    model = MARSCCA()
    assert model.fit(two_views_small) is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data, one encoder per view."""
    model = MARSCCA().fit(three_views_small)
    assert len(model.encoders_) == 3


@pytest.mark.parametrize("k", [1, 2])
def test_transform_shapes(two_views_small: list[np.ndarray], k: int) -> None:
    """Transform returns (n_samples, latent_dimensions) arrays on new data."""
    rng = np.random.default_rng(99)
    test_views = [rng.standard_normal((10, 5)), rng.standard_normal((10, 5))]
    model = MARSCCA(latent_dimensions=k).fit(two_views_small)
    for arr in model.transform(test_views):
        assert arr.shape == (10, k)


def test_transform_reproduces_training_embedding(
    two_views_small: list[np.ndarray],
) -> None:
    """Transform on the training data reproduces the fitted embeddings exactly."""
    model = MARSCCA(latent_dimensions=2, max_degree=2, cv=None).fit(two_views_small)
    for z, enc in zip(model.transform(two_views_small), model.encoders_):
        np.testing.assert_allclose(z, enc.predict(), atol=1e-10)


def test_score_values_in_range(two_views_small: list[np.ndarray]) -> None:
    """Score has shape (latent_dimensions,) with values in [-1, 1]."""
    s = MARSCCA(latent_dimensions=2).fit(two_views_small).score(two_views_small)
    assert s.shape == (2,)
    assert np.all(np.abs(s) <= 1.0 + 1e-9)


def test_center_false(two_views_small: list[np.ndarray]) -> None:
    """MARSCCA works with center=False."""
    model = MARSCCA(center=False).fit(two_views_small)
    assert len(model.transform(two_views_small)) == 2


# ---------------------------------------------------------------------------
# Basis growth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_terms", [1, 4, 7])
def test_max_terms_respected(
    correlated_views: list[np.ndarray], max_terms: int
) -> None:
    """No view's basis exceeds its max_terms budget, odd budgets included."""
    model = MARSCCA(max_terms=max_terms, cv=None).fit(correlated_views)
    for enc in model.encoders_:
        assert 1 <= len(enc.terms_) <= max_terms
        assert enc.coef_.shape == (len(enc.terms_), 1)


def test_per_view_max_terms(correlated_views: list[np.ndarray]) -> None:
    """A per-view max_terms list gives each view its own budget."""
    model = MARSCCA(max_terms=[4, 12], cv=None).fit(correlated_views)
    assert len(model.encoders_[0].terms_) == 4
    assert len(model.encoders_[1].terms_) == 12


def test_per_view_max_terms_wrong_length_raises(
    two_views_small: list[np.ndarray],
) -> None:
    """A per-view max_terms list must have one entry per view."""
    with pytest.raises(ValueError, match="max_terms"):
        MARSCCA(max_terms=[4, 4, 4]).fit(two_views_small)


def test_max_degree_one_is_additive(correlated_views: list[np.ndarray]) -> None:
    """max_degree=1 selects only single-hinge (additive) terms."""
    model = MARSCCA(max_degree=1, cv=None).fit(correlated_views)
    for enc in model.encoders_:
        assert all(len(term) == 1 for term in enc.terms_)


def test_max_degree_bounds_interaction_order(
    correlated_views: list[np.ndarray],
) -> None:
    """No term uses more factors than max_degree, nor a feature twice."""
    model = MARSCCA(max_degree=2, max_terms=30, cv=None).fit(correlated_views)
    for enc in model.encoders_:
        for term in enc.terms_:
            features = [f for f, _, _ in term]
            assert len(term) <= 2
            assert len(set(features)) == len(features)


def test_basis_columns_are_not_degenerate(correlated_views: list[np.ndarray]) -> None:
    """Every selected basis function is nonzero and linearly independent."""
    model = MARSCCA(max_degree=2, max_terms=20, cv=None).fit(correlated_views)
    for X, enc in zip(correlated_views, model.encoders_):
        basis = _evaluate_terms(X - X.mean(axis=0), enc.terms_)
        centred = basis - basis.mean(axis=0)
        assert np.linalg.matrix_rank(centred) == len(enc.terms_)


@pytest.mark.parametrize("n_basis", [0, 3])
def test_hinge_scorer_matches_direct_projection(n_basis: int) -> None:
    """Every fast-update score equals tr(G^T P_H G) from explicit columns.

    Grows the scorer's state in the order the forward pass does — basis
    columns, then parents (one vanishing on part of the sample, one with a
    disallowed feature), then more basis columns after those parents are
    cached — and scores against a gradient that is *not* orthogonal to the
    basis: every place where the suffix-sum algebra or the cache could
    silently change the ranking.
    """
    rng = np.random.default_rng(1)
    n, p, k = 60, 3, 2
    X = rng.standard_normal((n, p))
    scorer = _HingeScorer(X, n_candidate_knots=5)
    if n_basis:
        scorer.add_columns(rng.standard_normal((n, n_basis)), [None] * n_basis)
    new_parents = np.column_stack(
        [np.maximum(0.0, rng.standard_normal(n)), rng.random(n)]
    )
    restricted = np.array([True, False, True])
    scorer.add_columns(new_parents, [np.ones(p, dtype=bool), restricted])
    scorer.add_columns(rng.standard_normal((n, 1)), [None])

    parents = np.column_stack([np.ones(n), new_parents])
    allowed = np.column_stack(
        [np.ones(p, dtype=bool), np.ones(p, dtype=bool), restricted]
    )
    q = scorer.q
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-12)
    grad = rng.standard_normal((n, k))
    grad -= grad.mean(axis=0)

    def direct(m: int, j: int, t: float) -> float:
        u = parents[:, m]
        h = np.column_stack(
            [u * np.maximum(0, X[:, j] - t), u * np.maximum(0, t - X[:, j])]
        )
        h -= h.mean(axis=0)
        h -= q @ (q.T @ h)
        return float(np.trace(grad.T @ h @ np.linalg.solve(h.T @ h, h.T @ grad)))

    scores = np.array(
        [
            [
                [direct(m, j, t) if allowed[j, m] else -np.inf for m in range(3)]
                for j, t in enumerate(knots)
            ]
            for knots in scorer.knots
        ]
    )
    best, parent, j, knot, keep = scorer.best_pair(grad)
    r, j_expected, m_expected = np.unravel_index(np.argmax(scores), scores.shape)
    assert keep == (True, True)
    assert (parent, j) == (m_expected, j_expected)
    assert knot == scorer.knots[r, j]
    np.testing.assert_allclose(best, scores.max(), rtol=1e-9)


def test_identically_zero_hinges_are_degenerate() -> None:
    """A hinge that vanishes on every training sample is never a usable candidate.

    With feature 1 a copy of feature 0 and parent ``h(x0)``, the reflected
    hinge ``h(x0) * h(t - x1)`` is identically zero for every knot
    ``t <= 0``. Its squared norm comes out of the suffix-sum expansion as
    rounding noise rather than exactly zero; judged only against itself,
    noise passes, and a noise-over-noise score can then win the argmax.
    """
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    X = np.column_stack([x, x, rng.standard_normal(n)])
    scorer = _HingeScorer(X, n_candidate_knots=20)
    parent = np.maximum(0.0, x)
    scorer.add_columns(parent[:, None], [np.array([False, True, True])])

    ok_a, ok_b, ok_pair, *_ = scorer.gram()
    zero = scorer.knots[:, 1] <= 0  # knots where h(x0) * h(t - x1) == 0
    assert zero.any()
    assert not ok_b[zero, 1, 1].any()
    assert not ok_pair[zero, 1, 1].any()
    assert ok_a[zero, 1, 1].all()


def test_basis_functions_strings(two_views_small: list[np.ndarray]) -> None:
    """basis_functions reports raw-unit knots, one string per term."""
    views = [v + 10.0 for v in two_views_small]
    model = MARSCCA(max_terms=4, cv=None).fit(views)
    names = model.basis_functions(0)
    assert len(names) == len(model.encoders_[0].terms_)
    feature, knot, sign = model.encoders_[0].terms_[0][0]
    raw = knot + model.means_[0][feature]
    assert raw > 0  # views shifted by +10, so the knot prints as "x - t"
    expected = (
        f"h(x{feature} - {raw:.4g})" if sign > 0 else f"h({raw:.4g} - x{feature})"
    )
    assert names[0] == expected


def test_basis_functions_not_fitted_raises() -> None:
    """basis_functions before fitting raises NotFittedError."""
    with pytest.raises(NotFittedError):
        MARSCCA().basis_functions(0)


def test_weights_raises_not_implemented(two_views_small: list[np.ndarray]) -> None:
    """Accessing weights after fitting raises NotImplementedError."""
    model = MARSCCA().fit(two_views_small)
    with pytest.raises(NotImplementedError, match="basis_functions"):
        _ = model.weights


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    with pytest.raises(NotFittedError):
        _ = MARSCCA().weights


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """MARSCCA finds substantial correlation on views with shared structure."""
    s = MARSCCA().fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


def test_interactions_beat_additive_models_on_product_signal() -> None:
    """With max_degree=2, MARSCCA recovers a within-view interaction held out.

    View 2 is a noisy copy of ``a * b``, where ``a`` and ``b`` are two
    independent features of view 1. No additive encoder of view 1 can
    represent ``a * b``, so GAMCCA and additive MARSCCA are capped well
    below what a degree-2 MARS term reaches.
    """
    rng = np.random.default_rng(0)
    n, noise = 1000, 0.3
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    X1 = np.column_stack([a, b, rng.standard_normal((n, 3))])
    X2 = np.column_stack([a * b + noise * rng.standard_normal(n) for _ in range(5)])
    train = [X1[:500], X2[:500]]
    test = [X1[500:], X2[500:]]

    interaction = MARSCCA(max_degree=2).fit(train).score(test)[0]
    additive = MARSCCA(max_degree=1).fit(train).score(test)[0]
    gam = GAMCCA().fit(train).score(test)[0]

    assert interaction > 0.9, f"Expected MARSCCA(max_degree=2) > 0.9, got {interaction}"
    assert interaction > additive + 0.05
    assert interaction > gam + 0.05


def test_degree_three_recovers_three_way_interaction() -> None:
    """max_degree=3 builds order-3 terms and recovers a pure a*b*c signal held out."""
    rng = np.random.default_rng(0)
    n = 2000
    a, b, c = rng.standard_normal((3, n))
    X1 = np.column_stack([a, b, c, rng.standard_normal((n, 7))])
    X2 = np.column_stack([a * b * c + 0.3 * rng.standard_normal(n) for _ in range(5)])
    model = MARSCCA(max_degree=3, max_terms=40).fit([X1[:1000], X2[:1000]])
    assert max(len(term) for term in model.encoders_[0].terms_) == 3
    score = model.score([X1[1000:], X2[1000:]])[0]
    assert score > 0.9, f"Expected held-out correlation > 0.9, got {score}"


# ---------------------------------------------------------------------------
# Cross-validated pruning
# ---------------------------------------------------------------------------


def test_cv_none_keeps_full_forward_pass(correlated_views: list[np.ndarray]) -> None:
    """cv=None keeps every round and records no CV scores."""
    model = MARSCCA(max_terms=12, cv=None).fit(correlated_views)
    assert model.cv_scores_ is None
    assert len(model.encoders_[0].terms_) == 12


def test_cv_scores_shape_and_selected_round(correlated_views: list[np.ndarray]) -> None:
    """cv_scores_ is (cv, n_rounds) and the kept model is a forward-pass prefix."""
    full = MARSCCA(max_terms=12, cv=None).fit(correlated_views)
    pruned = MARSCCA(max_terms=12, cv=3).fit(correlated_views)
    assert pruned.cv_scores_ is not None
    assert pruned.cv_scores_.shape[0] == 3
    assert 1 <= pruned.n_rounds_ <= pruned.cv_scores_.shape[1]
    for p_enc, f_enc in zip(pruned.encoders_, full.encoders_):
        assert p_enc.terms_ == f_enc.terms_[: len(p_enc.terms_)]


def test_pruning_selects_small_model_on_pure_noise() -> None:
    """On independent views, pruning discards almost all of an oversized basis.

    Without pruning, 40 terms per view overfit pure noise into a large
    training correlation; the pruned model stays small and its training
    correlation stays near zero.
    """
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((300, 10)), rng.standard_normal((300, 5))]
    full = MARSCCA(max_degree=2, max_terms=40, cv=None).fit(views)
    pruned = MARSCCA(max_degree=2, max_terms=40).fit(views)
    assert len(full.encoders_[0].terms_) == 40
    assert len(pruned.encoders_[0].terms_) <= 10
    assert pruned.score(views)[0] < full.score(views)[0] - 0.3
