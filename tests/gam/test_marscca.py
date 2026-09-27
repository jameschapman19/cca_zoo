"""MARSCCA: earth's forward and backward passes with a CCA objective."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._ey import ey_loss, penalised_basis_ey_closed_form
from cca_zoo.gam import GAMCCA, MARSCCA
from cca_zoo.gam._marscca import _backward_path, _HingeScorer, _knot_ranks


def _interaction(n: int, order: int) -> list[np.ndarray]:
    """View 2 is a noisy product of the first ``order`` features of view 1."""
    rng = np.random.default_rng(0)
    factors = rng.standard_normal((order, n))
    product = np.prod(factors, axis=0)
    return [
        np.column_stack([*factors, rng.standard_normal((n, 5))]),
        np.column_stack([product + 0.3 * rng.standard_normal(n) for _ in range(5)]),
    ]


@pytest.mark.parametrize("order", [2, 3])
def test_degree_captures_an_interaction_of_that_order(order: int) -> None:
    """degree=d recovers a held-out d-way product that no additive model can."""
    views = _interaction(2000, order)
    train, test = [v[:1000] for v in views], [v[1000:] for v in views]
    model = MARSCCA(degree=order, nk=40, random_state=0).fit(train)
    assert max(len(term) for term in model.encoders_[0].terms_) == order
    assert model.score(test) > 0.9
    assert model.score(test) > GAMCCA().fit(train).score(test) + 0.05


def test_feature_importances_rank_the_interacting_features() -> None:
    """Earth's evimp puts the interacting features first and unused ones at zero."""
    views = _interaction(400, 2)
    model = MARSCCA(degree=2, nk=16, nprune=10, random_state=0).fit(views)
    importance = model.feature_importances_per_view_[0]
    assert set(np.argsort(importance)[-2:]) == {0, 1}
    used = {f for term in model.encoders_[0].terms_ for f, _, _ in term}
    assert all(importance[f] == 0 for f in range(importance.size) if f not in used)


@pytest.mark.parametrize("nk", [1, 4, 7])
def test_nk_caps_the_basis(correlated_views: list[np.ndarray], nk: int) -> None:
    """No view's basis exceeds nk terms."""
    model = MARSCCA(nk=nk, random_state=0).fit(correlated_views)
    assert all(1 <= len(e.terms_) <= nk for e in model.encoders_)


def test_thresh_stops_the_forward_pass(correlated_views: list[np.ndarray]) -> None:
    """thresh=0 grows to nk; a huge thresh stops after the first measurable round."""
    grown = MARSCCA(nk=30, thresh=0.0, random_state=0).fit(correlated_views)
    stopped = MARSCCA(nk=30, thresh=1e6, random_state=0).fit(correlated_views)
    assert [len(e.terms_) for e in grown.encoders_] == [30, 30]
    assert [len(e.terms_) for e in stopped.encoders_] == [4, 4]


def test_friedman_knot_spacing() -> None:
    """minspan=0 is Friedman's rule; the default caps knots at 20 per feature."""
    # 3000 samples, 10 features: minspan floor(18.3 / 2.5) = 7, endspan 10.
    friedman = _knot_ranks(3000, 10, 0, None, interaction=False)
    np.testing.assert_array_equal(friedman, np.arange(10, 2990, 7))
    assert len(_knot_ranks(3000, 10, None, None, interaction=False)) <= 20


def test_nprune_keeps_a_subset(correlated_views: list[np.ndarray]) -> None:
    """Exactly nprune forward-pass terms survive, at least one per view."""
    full = MARSCCA(degree=2, nk=10, random_state=0).fit(correlated_views)
    pruned = MARSCCA(degree=2, nk=10, nprune=5, random_state=0).fit(correlated_views)
    assert sum(len(e.terms_) for e in pruned.encoders_) == 5
    for p, f in zip(pruned.encoders_, full.encoders_):
        assert p.terms_ and set(p.terms_) <= set(f.terms_)
    with pytest.raises(ValueError, match="nprune"):
        MARSCCA(nk=6, nprune=1).fit(correlated_views)


def test_too_few_samples_for_a_knot_names_the_fix() -> None:
    """Earth's endspan leaves no knot in 15 samples; lowering it is the fix."""
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((15, 4)), rng.standard_normal((15, 3))]
    with pytest.raises(ValueError, match="lower endspan"):
        MARSCCA().fit(views)
    MARSCCA(endspan=0).fit(views)


def test_basis_functions_in_raw_units(two_views_small: list[np.ndarray]) -> None:
    """basis_functions prints each term's knot on the data's original scale."""
    views = [v + 10.0 for v in two_views_small]
    model = MARSCCA(nk=4, random_state=0).fit(views)
    feature, knot, sign = model.encoders_[0].terms_[0][0]
    raw = f"{knot + model.means_[0][feature]:.4g}"
    expected = f"h(x{feature} - {raw})" if sign > 0 else f"h({raw} - x{feature})"
    assert model.basis_functions(0)[0] == expected


def test_forward_scores_match_explicit_projections() -> None:
    """The fast hinge scores equal tr(G' P_H G) computed from explicit columns."""
    rng = np.random.default_rng(1)
    n, p = 60, 3
    X = rng.standard_normal((n, p))
    scorer = _HingeScorer(X)
    scorer.add_columns(rng.standard_normal((n, 2)), [None, None])
    q = scorer.q
    grad = rng.standard_normal((n, 2))
    grad -= grad.mean(axis=0)

    def explicit(j: int, knot: float) -> float:
        h = np.column_stack(
            [np.maximum(0, X[:, j] - knot), np.maximum(0, knot - X[:, j])]
        )
        h -= h.mean(axis=0)
        h -= q @ (q.T @ h)
        return float(np.trace(grad.T @ h @ np.linalg.solve(h.T @ h, h.T @ grad)))

    score, parent, j, knot, _ = scorer.best_pairs(grad)[0]
    candidates = [
        explicit(jj, scorer.knots[r, jj, 0])
        for r in range(scorer.knots.shape[0])
        for jj in range(p)
        if scorer._allowed[r, jj, 0]
    ]
    assert parent == 0
    np.testing.assert_allclose(score, max(candidates), rtol=1e-9)
    np.testing.assert_allclose(score, explicit(j, knot), rtol=1e-9)


def test_backward_pass_matches_brute_force() -> None:
    """Each backward step drops the column whose explicit refit loses least."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 2))
    bases = [
        z @ rng.standard_normal((2, d)) + rng.standard_normal((200, d)) for d in (5, 4)
    ]
    bases = [b - b.mean(axis=0) for b in bases]
    ridge = [0.1, 0.3]

    def refit_loss(columns: list[np.ndarray]) -> float:
        coefs = penalised_basis_ey_closed_form(columns, 2, ridge)
        loss = ey_loss([b @ c for b, c in zip(columns, coefs)])["objective"]
        return loss + 0.5 * sum(r * float(np.sum(c**2)) for r, c in zip(ridge, coefs))

    removed, path_loss = _backward_path(bases, 2, ridge)
    owner = np.repeat([0, 1], [5, 4])
    index = np.concatenate([np.arange(5), np.arange(4)])
    active = [list(range(5)), list(range(4))]
    for step, column in enumerate(removed):
        losses = {
            (i, c): refit_loss(
                [
                    b[:, [x for x in a if x != c or j != i]]
                    for j, (b, a) in enumerate(zip(bases, active))
                ]
            )
            for i in range(2)
            if len(active[i]) > 1
            for c in active[i]
        }
        best = min(losses, key=losses.__getitem__)
        assert (owner[column], index[column]) == best
        np.testing.assert_allclose(path_loss[step + 1], losses[best], rtol=1e-9)
        active[best[0]].remove(best[1])
