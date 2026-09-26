"""Tests for MARSCCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cca_zoo._utils._ey import ey_loss, penalised_basis_ey_closed_form
from cca_zoo.gam import GAMCCA, MARSCCA
from cca_zoo.gam._marscca import (
    _backward_path,
    _constrained_top_eigenvalues,
    _evaluate_terms,
    _HingeScorer,
    _knot_ranks,
    _max_knot_slots,
)

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
    model = MARSCCA(latent_dimensions=2, degree=2).fit(two_views_small)
    for z, enc in zip(model.transform(two_views_small), model.encoders_):
        np.testing.assert_allclose(z, enc.predict(), atol=1e-10)


def test_score_values_in_range(two_views_small: list[np.ndarray]) -> None:
    """Score is one float in [-1, 1]."""
    s = MARSCCA(latent_dimensions=2).fit(two_views_small).score(two_views_small)
    assert isinstance(s, float)
    assert abs(s) <= 1.0 + 1e-9


def test_center_false(two_views_small: list[np.ndarray]) -> None:
    """MARSCCA works with center=False."""
    model = MARSCCA(center=False).fit(two_views_small)
    assert len(model.transform(two_views_small)) == 2


# ---------------------------------------------------------------------------
# Basis growth
# ---------------------------------------------------------------------------


def test_per_view_parameters_accept_none_entries(
    correlated_views: list[np.ndarray],
) -> None:
    """Per-view lists work for every view-level parameter; None means the default.

    nk=[None, 6] grows view 0 to the default (20 for 10 features) and view 1
    to 6. minspan=[40, None] leaves view 0 a single allowed knot per feature
    (50 samples, endspan 10), while view 1 keeps Friedman's spacing.
    """
    model = MARSCCA(
        nk=[None, 6], minspan=[40, None], endspan=[None, 1], thresh=0.0
    ).fit(correlated_views)
    assert len(model.encoders_[1].terms_) == 6
    view0 = model.encoders_[0].terms_
    assert 1 <= len(view0) <= 20
    for feature in range(correlated_views[0].shape[1]):
        assert len({t for term in view0 for f, t, _ in term if f == feature}) <= 1


@pytest.mark.parametrize("name", ["degree", "nk", "alpha", "minspan", "endspan"])
def test_per_view_parameter_wrong_length_raises(
    two_views_small: list[np.ndarray], name: str
) -> None:
    """Every per-view parameter must have one entry per view."""
    with pytest.raises(ValueError, match=name):
        MARSCCA(**{name: [2, 2, 2]}).fit(two_views_small)


@pytest.mark.parametrize("nk", [1, 4, 7])
def test_nk_respected(correlated_views: list[np.ndarray], nk: int) -> None:
    """No view's basis exceeds its nk budget, odd budgets included."""
    model = MARSCCA(nk=nk).fit(correlated_views)
    for enc in model.encoders_:
        assert 1 <= len(enc.terms_) <= nk
        assert enc.coef_.shape == (len(enc.terms_), 1)


def test_per_view_nk(correlated_views: list[np.ndarray]) -> None:
    """A per-view nk list gives each view its own budget."""
    model = MARSCCA(nk=[4, 12], thresh=0.0).fit(correlated_views)
    assert len(model.encoders_[0].terms_) == 4
    assert len(model.encoders_[1].terms_) == 12


def test_thresh_stops_forward_pass_early(correlated_views: list[np.ndarray]) -> None:
    """A round that barely lowers the loss ends the forward pass (earth's thresh).

    thresh=0 always grows to nk; a huge thresh stops after the
    second round, the first whose improvement can be measured.
    """
    grown = MARSCCA(nk=30, thresh=0.0).fit(correlated_views)
    stopped = MARSCCA(nk=30, thresh=1e6).fit(correlated_views)
    assert [len(e.terms_) for e in grown.encoders_] == [30, 30]
    assert [len(e.terms_) for e in stopped.encoders_] == [4, 4]


def test_max_degree_one_is_additive(correlated_views: list[np.ndarray]) -> None:
    """degree=1 selects only single-hinge (additive) terms."""
    model = MARSCCA(degree=1).fit(correlated_views)
    for enc in model.encoders_:
        assert all(len(term) == 1 for term in enc.terms_)


def test_max_degree_bounds_interaction_order(
    correlated_views: list[np.ndarray],
) -> None:
    """No term uses more factors than max_degree, nor a feature twice."""
    model = MARSCCA(degree=2, nk=30).fit(correlated_views)
    for enc in model.encoders_:
        for term in enc.terms_:
            features = [f for f, _, _ in term]
            assert len(term) <= 2
            assert len(set(features)) == len(features)


def test_basis_columns_are_not_degenerate(correlated_views: list[np.ndarray]) -> None:
    """Every selected basis function is nonzero and linearly independent."""
    model = MARSCCA(degree=2, nk=20).fit(correlated_views)
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
    scorer = _HingeScorer(X)
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

    slots = scorer.knots.shape[0]
    usable = scorer._allowed
    assert usable.any(axis=0)[~allowed].sum() == 0  # disallowed features never scored
    scores = np.array(
        [
            [
                [
                    direct(m, j, scorer.knots[r, j, m]) if usable[r, j, m] else -np.inf
                    for m in range(3)
                ]
                for j in range(p)
            ]
            for r in range(slots)
        ]
    )
    best, parent, j, knot, keep = scorer.best_pairs(grad)[0]
    r, j_expected, m_expected = np.unravel_index(np.argmax(scores), scores.shape)
    assert keep == (True, True)
    assert (parent, j) == (m_expected, j_expected)
    assert knot == scorer.knots[r, j, parent]
    np.testing.assert_allclose(best, scores.max(), rtol=1e-9)


def test_best_pairs_ranked_and_distinct() -> None:
    """best_pairs returns the top-n candidates, best first, all distinct."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    scorer = _HingeScorer(X)
    grad = rng.standard_normal((100, 1))
    pairs = scorer.best_pairs(grad - grad.mean(), n=8)
    scores = [p[0] for p in pairs]
    assert len(pairs) == 8
    assert scores == sorted(scores, reverse=True)
    assert len({p[1:4] for p in pairs}) == 8
    assert pairs[0] == scorer.best_pairs(grad - grad.mean())[0]


def test_exact_loss_rescoring_picks_lowest_loss_candidate() -> None:
    """Among the top candidates, the one with the lowest exact loss is added.

    The exact loss here deliberately prefers the gradient's fourth choice,
    so the test fails if rescoring is skipped or its result ignored.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    grad = rng.standard_normal((100, 1))
    grad -= grad.mean()
    ranked = _HingeScorer(X).best_pairs(grad, n=5)
    _, _, j_want, knot_want, _ = ranked[3]

    def exact_loss(columns: np.ndarray) -> float:
        hinge = np.maximum(0.0, X[:, j_want] - knot_want)
        return 0.0 if np.allclose(columns[:, 0], hinge) else 1.0

    terms: list = []
    MARSCCA._add_best_pair(
        _HingeScorer(X),
        terms,
        [()],
        grad,
        max_degree=1,
        max_terms=20,
        exact_loss=exact_loss,
    )
    assert terms[0] == ((j_want, knot_want, 1),)


def test_default_minspan_caps_knots_and_zero_is_friedman() -> None:
    """Default minspan caps knots at 20 per feature; minspan=0 is Friedman's rule."""
    for n, p in [(200, 5), (3000, 10)]:
        capped = _knot_ranks(n, p, None, None, interaction=False)
        friedman = _knot_ranks(n, p, 0, None, interaction=False)
        assert len(capped) <= 20
        assert len(friedman) >= len(capped)
        assert set(capped) <= set(range(n))
    # Friedman's spacing for 3000 samples and 10 features: floor(18.3 / 2.5) = 7,
    # endspan floor(3 - log2(0.005)) = 10.
    assert np.array_equal(
        _knot_ranks(3000, 10, 0, None, interaction=False), np.arange(10, 2990, 7)
    )


@pytest.mark.parametrize(
    "spans", [(None, None), (0, None), (1, 0), (3, None), (None, 2), (0, 0)]
)
def test_knot_slots_cover_every_possible_parent(spans: tuple) -> None:
    """Slots per (feature, parent) are the most knots any support size allows.

    Friedman's minspan shrinks with the support, so a parent nonzero on
    slightly fewer samples than the constant can have more knots than it;
    sizing slots from the constant parent overflowed once.
    """
    minspan, endspan = spans
    for n, p in [(57, 3), (1000, 10)]:
        most = max(
            len(_knot_ranks(support, p, minspan, endspan, interaction))
            for support in range(1, n + 1)
            for interaction in (False, True)
        )
        assert _max_knot_slots(n, p, minspan, endspan) == most


def test_identically_zero_hinges_are_degenerate() -> None:
    """A hinge that vanishes on every training sample is never a usable candidate.

    With ``endspan=0`` a knot may sit at a feature's smallest value, where
    the reflected hinge ``h(t - x)`` is identically zero. Its squared norm
    comes out of the suffix-sum expansion as rounding noise rather than
    exactly zero; judged only against itself, noise passes, and a
    noise-over-noise score can then win the argmax. (With the default
    ``endspan``, earth's rule keeps knots away from the ends, so this
    cannot arise.)
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 3)) + 5.0
    scorer = _HingeScorer(X, minspan=1, endspan=0)
    ok_a, ok_b, ok_pair, *_ = scorer.gram()
    lowest = scorer.knots[0, :, 0] == X.min(axis=0)
    assert lowest.all()
    assert not ok_b[0, :, 0].any()
    assert not ok_pair[0, :, 0].any()
    assert ok_a[0, :, 0].all()


def test_basis_stays_well_conditioned_with_near_duplicate_features() -> None:
    """Near-duplicate features must not let a nearly-in-span pair through.

    View 2 is five noisy copies of one signal, so candidate hinges are
    often almost in the current span *and* almost parallel to each other.
    Judged only against each other, two such hinges once passed as a pair
    and made the basis singular (condition number ~1e16), which the
    closed-form refit's Cholesky factorisation then rejected.
    """
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(1)
    n = 600
    a, b, _, _ = rng.standard_normal((4, n))
    data = [
        np.column_stack([a, b, rng.standard_normal((n, 8))])[:300],
        np.column_stack([a * b + 0.5 * rng.standard_normal(n) for _ in range(5)])[:300],
    ]
    train, _ = next(KFold(5, shuffle=True, random_state=1).split(data[0]))
    views = [X[train] for X in data]
    model = MARSCCA(degree=2, nk=40, random_state=1).fit(views)
    for X, enc in zip(views, model.encoders_):
        basis = _evaluate_terms(X - X.mean(axis=0), enc.terms_)
        basis -= basis.mean(axis=0)
        assert np.linalg.cond(basis / np.linalg.norm(basis, axis=0)) < 1e8


def test_basis_functions_strings(two_views_small: list[np.ndarray]) -> None:
    """basis_functions reports raw-unit knots, one string per term."""
    views = [v + 10.0 for v in two_views_small]
    model = MARSCCA(nk=4).fit(views)
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


def test_weights_not_fitted_raises() -> None:
    """Transform before fitting raises NotFittedError."""
    with pytest.raises(NotFittedError):
        MARSCCA().transform([np.ones((3, 2)), np.ones((3, 2))])


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
    """With degree=2, MARSCCA recovers a within-view interaction held out.

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

    interaction = MARSCCA(degree=2).fit(train).score(test)
    additive = MARSCCA(degree=1).fit(train).score(test)
    gam = GAMCCA().fit(train).score(test)

    assert interaction > 0.9, f"Expected MARSCCA(degree=2) > 0.9, got {interaction}"
    assert interaction > additive + 0.05
    assert interaction > gam + 0.05


def test_degree_three_recovers_three_way_interaction() -> None:
    """degree=3 builds order-3 terms and recovers a pure a*b*c signal held out."""
    rng = np.random.default_rng(0)
    n = 2000
    a, b, c = rng.standard_normal((3, n))
    X1 = np.column_stack([a, b, c, rng.standard_normal((n, 7))])
    X2 = np.column_stack([a * b * c + 0.3 * rng.standard_normal(n) for _ in range(5)])
    model = MARSCCA(degree=3, nk=40).fit([X1[:1000], X2[:1000]])
    assert max(len(term) for term in model.encoders_[0].terms_) == 3
    score = model.score([X1[1000:], X2[1000:]])
    assert score > 0.9, f"Expected held-out correlation > 0.9, got {score}"


# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deflate", [False, True])
@pytest.mark.parametrize("repeated", [False, True])
def test_constrained_top_eigenvalues_match_direct_compression(
    deflate: bool, repeated: bool
) -> None:
    """Bisection on the inertia count equals eigvalsh of the compressed matrix.

    Covers a removal direction orthogonal to the top eigenvector (that
    eigenvalue must survive unchanged), a repeated top eigenvalue, and
    asking for more eigenvalues than the compressed matrix has.
    """
    import scipy.linalg

    rng = np.random.default_rng(0)
    for d in (2, 3, 7):
        lam = np.sort(rng.standard_normal(d))
        if repeated:
            lam[-2] = lam[-1]
        u = np.linalg.qr(rng.standard_normal((d, d)))[0]
        c = u @ np.diag(lam) @ u.T
        z = rng.standard_normal((4, d))
        if deflate:
            z[:, -1] = 0.0
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        got = _constrained_top_eigenvalues(lam, z, k=3)
        for row, zc in zip(got, z):
            complement = scipy.linalg.null_space((u @ zc)[None, :])
            want = np.linalg.eigvalsh(complement.T @ c @ complement)[::-1]
            np.testing.assert_allclose(row[: len(want)][:3], want[:3], atol=1e-12)
            assert np.all(np.isneginf(row[len(want) :]))


def test_backward_step_matches_brute_force_refits() -> None:
    """Every backward step deletes the column whose explicit refit loss is lowest.

    Checks the whole path down to one column per view: each deletion and the
    loss recorded after it against refitting every remaining candidate.
    """
    rng = np.random.default_rng(0)
    n, k = 200, 2
    z = rng.standard_normal((n, 2))
    bases = [
        z @ rng.standard_normal((2, d)) + rng.standard_normal((n, d)) for d in (5, 4)
    ]
    bases = [b - b.mean(axis=0) for b in bases]
    ridge = [0.1, 0.3]

    def refit_loss(reduced: list[np.ndarray]) -> float:
        coefs = penalised_basis_ey_closed_form(reduced, k, ridge)
        loss = ey_loss([b @ c for b, c in zip(reduced, coefs)])["objective"]
        return loss + 0.5 * sum(r * float(np.sum(c**2)) for r, c in zip(ridge, coefs))

    removed, path_loss = _backward_path(bases, k, ridge)
    stacked_view = np.repeat([0, 1], [5, 4])
    stacked_col = np.concatenate([np.arange(5), np.arange(4)])
    active = [np.ones(5, dtype=bool), np.ones(4, dtype=bool)]
    assert len(removed) == 9 - 2  # down to one column per view
    np.testing.assert_allclose(path_loss[0], refit_loss(bases), rtol=1e-9)
    for step, column in enumerate(removed):
        current = [b[:, a] for b, a in zip(bases, active)]
        candidates = {
            (i, c): refit_loss(
                [
                    np.delete(b, c, axis=1) if j == i else b
                    for j, b in enumerate(current)
                ]
            )
            for i in range(2)
            if current[i].shape[1] > 1
            for c in range(current[i].shape[1])
        }
        best = min(candidates, key=candidates.__getitem__)
        i = int(stacked_view[column])
        assert (
            i,
            int(np.flatnonzero(active[i]).tolist().index(stacked_col[column])),
        ) == best
        np.testing.assert_allclose(path_loss[step + 1], candidates[best], rtol=1e-9)
        active[i][stacked_col[column]] = False


def test_backward_pass_keeps_nprune_as_subset(
    correlated_views: list[np.ndarray],
) -> None:
    """Exactly nprune terms survive, at least one per view, all forward-pass terms."""
    full = MARSCCA(degree=2, nk=10).fit(correlated_views)
    for nprune in (2, 5, 13):
        pruned = MARSCCA(degree=2, nk=10, nprune=nprune).fit(correlated_views)
        assert sum(len(e.terms_) for e in pruned.encoders_) == nprune
        for p_enc, f_enc in zip(pruned.encoders_, full.encoders_):
            assert len(p_enc.terms_) >= 1
            assert set(p_enc.terms_) <= set(f_enc.terms_)


def test_backward_pass_noop_when_nprune_not_smaller(
    correlated_views: list[np.ndarray],
) -> None:
    """An nprune at or above the forward pass's size leaves the model unchanged."""
    full = MARSCCA(nk=6).fit(correlated_views)
    same = MARSCCA(nk=6, nprune=100).fit(correlated_views)
    for a, b in zip(full.encoders_, same.encoders_):
        assert a.terms_ == b.terms_
        np.testing.assert_allclose(a.coef_, b.coef_)


def test_too_few_samples_for_any_knot_raises_clearly() -> None:
    """With 15 samples, Friedman's endspan leaves no knot: a clear error, not a crash.

    Lowering endspan is the documented fix, and then the fit succeeds.
    """
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((15, 4)), rng.standard_normal((15, 3))]
    with pytest.raises(ValueError, match="could not place a single hinge"):
        MARSCCA().fit(views)
    model = MARSCCA(endspan=0).fit(views)
    assert all(len(e.terms_) >= 1 for e in model.encoders_)


def test_nprune_below_number_of_views_raises(
    correlated_views: list[np.ndarray],
) -> None:
    """Every view must keep a term, so nprune below n_views is an error."""
    with pytest.raises(ValueError, match="nprune"):
        MARSCCA(nk=6, nprune=1).fit(correlated_views)


def test_feature_importances_rank_the_interaction_features() -> None:
    """evimp-style importance puts the two interacting features first in view 1.

    Features no selected term uses get exactly zero.
    """
    rng = np.random.default_rng(0)
    n = 400
    a, b = rng.standard_normal((2, n))
    views = [
        np.column_stack([a, b, rng.standard_normal((n, 6))]),
        np.column_stack([a * b + 0.3 * rng.standard_normal(n) for _ in range(3)]),
    ]
    model = MARSCCA(degree=2, nk=16, nprune=10).fit(views)
    importances = model.feature_importances_
    assert [imp.shape for imp in importances] == [(8,), (3,)]
    assert set(np.argsort(importances[0])[-2:]) == {0, 1}
    for imp, enc, X in zip(importances, model.encoders_, views):
        np.testing.assert_allclose(imp.sum(), 1.0)
        used = {f for term in enc.terms_ for f, _, _ in term}
        for f in range(X.shape[1]):
            if f not in used:
                assert imp[f] == 0
