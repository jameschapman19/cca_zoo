"""Tests for MARSCCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cca_zoo._utils._ey import ey_loss, ridge_basis_ey_closed_form
from cca_zoo.gam import GAMCCA, MARSCCA
from cca_zoo.gam._marscca import (
    _backward_path,
    _constrained_top_eigenvalues,
    _evaluate_terms,
    _HingeScorer,
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
    model = MARSCCA(latent_dimensions=2, max_degree=2).fit(two_views_small)
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
    model = MARSCCA(max_terms=max_terms).fit(correlated_views)
    for enc in model.encoders_:
        assert 1 <= len(enc.terms_) <= max_terms
        assert enc.coef_.shape == (len(enc.terms_), 1)


def test_per_view_max_terms(correlated_views: list[np.ndarray]) -> None:
    """A per-view max_terms list gives each view its own budget."""
    model = MARSCCA(max_terms=[4, 12], thresh=0.0).fit(correlated_views)
    assert len(model.encoders_[0].terms_) == 4
    assert len(model.encoders_[1].terms_) == 12


def test_thresh_stops_forward_pass_early(correlated_views: list[np.ndarray]) -> None:
    """A round that barely lowers the loss ends the forward pass (earth's thresh).

    thresh=0 always grows to max_terms; a huge thresh stops after the
    second round, the first whose improvement can be measured.
    """
    grown = MARSCCA(max_terms=30, thresh=0.0).fit(correlated_views)
    stopped = MARSCCA(max_terms=30, thresh=1e6).fit(correlated_views)
    assert [len(e.terms_) for e in grown.encoders_] == [30, 30]
    assert [len(e.terms_) for e in stopped.encoders_] == [4, 4]


def test_per_view_max_terms_wrong_length_raises(
    two_views_small: list[np.ndarray],
) -> None:
    """A per-view max_terms list must have one entry per view."""
    with pytest.raises(ValueError, match="max_terms"):
        MARSCCA(max_terms=[4, 4, 4]).fit(two_views_small)


def test_max_degree_one_is_additive(correlated_views: list[np.ndarray]) -> None:
    """max_degree=1 selects only single-hinge (additive) terms."""
    model = MARSCCA(max_degree=1).fit(correlated_views)
    for enc in model.encoders_:
        assert all(len(term) == 1 for term in enc.terms_)


def test_max_degree_bounds_interaction_order(
    correlated_views: list[np.ndarray],
) -> None:
    """No term uses more factors than max_degree, nor a feature twice."""
    model = MARSCCA(max_degree=2, max_terms=30).fit(correlated_views)
    for enc in model.encoders_:
        for term in enc.terms_:
            features = [f for f, _, _ in term]
            assert len(term) <= 2
            assert len(set(features)) == len(features)


def test_basis_columns_are_not_degenerate(correlated_views: list[np.ndarray]) -> None:
    """Every selected basis function is nonzero and linearly independent."""
    model = MARSCCA(max_degree=2, max_terms=20).fit(correlated_views)
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
    scorer = _HingeScorer(X, n_candidate_knots=10)
    grad = rng.standard_normal((100, 1))
    pairs = scorer.best_pairs(grad - grad.mean(), n=8)
    scores = [p[0] for p in pairs]
    assert len(pairs) == 8
    assert scores == sorted(scores, reverse=True)
    assert len({p[1:4] for p in pairs}) == 8
    assert pairs[0] == scorer.best_pairs(grad - grad.mean())[0]


def test_exact_loss_rescoring_picks_lowest_loss_candidate() -> None:
    """Among the top n_rescore candidates, the one with the lowest exact loss is added.

    The exact loss here deliberately prefers the gradient's fourth choice,
    so the test fails if rescoring is skipped or its result ignored.
    """
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4))
    grad = rng.standard_normal((100, 1))
    grad -= grad.mean()
    ranked = _HingeScorer(X, n_candidate_knots=10).best_pairs(grad, n=5)
    _, _, j_want, knot_want, _ = ranked[3]

    def exact_loss(columns: np.ndarray) -> float:
        hinge = np.maximum(0.0, X[:, j_want] - knot_want)
        return 0.0 if np.allclose(columns[:, 0], hinge) else 1.0

    terms: list = []
    MARSCCA._add_best_pair(
        _HingeScorer(X, n_candidate_knots=10),
        terms,
        [()],
        grad,
        max_degree=1,
        max_terms=20,
        n_rescore=5,
        exact_loss=exact_loss,
    )
    assert terms[0] == ((j_want, knot_want, 1),)


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
    scorer = _HingeScorer(X, n_candidate_knots=200, minspan=1, endspan=0)
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
    model = MARSCCA(max_degree=2, max_terms=40, random_state=1).fit(views)
    for X, enc in zip(views, model.encoders_):
        basis = _evaluate_terms(X - X.mean(axis=0), enc.terms_)
        basis -= basis.mean(axis=0)
        assert np.linalg.cond(basis / np.linalg.norm(basis, axis=0)) < 1e8


def test_basis_functions_strings(two_views_small: list[np.ndarray]) -> None:
    """basis_functions reports raw-unit knots, one string per term."""
    views = [v + 10.0 for v in two_views_small]
    model = MARSCCA(max_terms=4).fit(views)
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
        coefs = ridge_basis_ey_closed_form(reduced, k, ridge)
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


def test_backward_pass_keeps_n_terms_as_subset(
    correlated_views: list[np.ndarray],
) -> None:
    """n_terms total survive, every view keeps one, all from the forward pass."""
    full = MARSCCA(max_degree=2, max_terms=10).fit(correlated_views)
    for n_terms in (2, 5, 13):
        pruned = MARSCCA(max_degree=2, max_terms=10, n_terms=n_terms).fit(
            correlated_views
        )
        assert sum(len(e.terms_) for e in pruned.encoders_) == n_terms
        for p_enc, f_enc in zip(pruned.encoders_, full.encoders_):
            assert len(p_enc.terms_) >= 1
            assert set(p_enc.terms_) <= set(f_enc.terms_)


def test_backward_pass_noop_when_n_terms_not_smaller(
    correlated_views: list[np.ndarray],
) -> None:
    """n_terms at or above the forward pass's size leaves the model unchanged."""
    full = MARSCCA(max_terms=6).fit(correlated_views)
    same = MARSCCA(max_terms=6, n_terms=100).fit(correlated_views)
    for a, b in zip(full.encoders_, same.encoders_):
        assert a.terms_ == b.terms_
        np.testing.assert_allclose(a.coef_, b.coef_)


def test_n_terms_below_number_of_views_raises(
    correlated_views: list[np.ndarray],
) -> None:
    """Every view must keep a term, so n_terms below n_views is an error."""
    with pytest.raises(ValueError, match="n_terms"):
        MARSCCA(max_terms=6, n_terms=1).fit(correlated_views)


def test_variable_importance_ranks_the_interaction_features() -> None:
    """evimp-style importance puts the two interacting features first in view 1.

    ``loss`` peaks at exactly 100 across views and gives unused features 0;
    ``nsubsets`` never exceeds the number of nested subsets.
    """
    rng = np.random.default_rng(0)
    n = 400
    a, b = rng.standard_normal((2, n))
    views = [
        np.column_stack([a, b, rng.standard_normal((n, 6))]),
        np.column_stack([a * b + 0.3 * rng.standard_normal(n) for _ in range(3)]),
    ]
    model = MARSCCA(max_degree=2, max_terms=16, n_terms=10).fit(views)
    loss = model.variable_importance("loss")
    nsubsets = model.variable_importance("nsubsets")
    assert [imp.shape for imp in loss] == [(8,), (3,)]
    assert max(float(imp.max()) for imp in loss) == pytest.approx(100.0)
    assert set(np.argsort(loss[0])[-2:]) == {0, 1}
    n_subsets = 10 - 2 + 1  # fitted size down to one term per view
    for imp_loss, imp_count, enc, X in zip(loss, nsubsets, model.encoders_, views):
        used = {f for term in enc.terms_ for f, _, _ in term}
        assert imp_count.max() <= n_subsets
        for f in range(X.shape[1]):
            if f not in used:
                assert imp_count[f] == 0
                assert imp_loss[f] == 0


def test_variable_importance_rejects_unknown_criterion(
    correlated_views: list[np.ndarray],
) -> None:
    """Only earth's two criteria are accepted."""
    model = MARSCCA(max_terms=4).fit(correlated_views)
    with pytest.raises(ValueError, match="criterion"):
        model.variable_importance("gcv")


def test_one_standard_error_search_prunes_pure_noise() -> None:
    """GridSearchCV over n_terms + one_standard_error keeps a small model on noise.

    Unpruned, 40 terms per view overfit pure noise into a large training
    correlation; the searched model stays small and near zero.
    """
    from cca_zoo.model_selection import GridSearchCV, one_standard_error

    rng = np.random.default_rng(0)
    views = [rng.standard_normal((300, 10)), rng.standard_normal((300, 5))]
    full = MARSCCA(max_degree=2, max_terms=40).fit(views)
    search = GridSearchCV(
        MARSCCA(max_degree=2, max_terms=40),
        {"n_terms": [2, 4, 8, 20, 80]},
        cv=5,
        refit=one_standard_error("n_terms"),
    ).fit(views)
    assert search.best_params_["n_terms"] <= 8
    assert search.score(views) < full.score(views)[0] - 0.3
