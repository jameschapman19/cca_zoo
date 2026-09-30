"""The sparse CCA models: what each penalty does to the weights."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import ey_loss
from cca_zoo.linear import MCCA
from cca_zoo.metrics import pairwise_correlations
from cca_zoo.sparse import (
    ADMMCCA,
    IPLSCCA,
    PMDCCA,
    SAR,
    ElasticNetCCA,
    MultiTaskElasticNetCCA,
    OrthogonalMatchingPursuitCCA,
    ParkhomenkoCCA,
    SpanCCA,
)


def _active_rows(w: np.ndarray) -> int:
    return int(np.sum(np.linalg.norm(w, axis=1) > 1e-10))


def _signal_and_noise_columns(n: int = 200, k: int = 1) -> list[np.ndarray]:
    """Each view: three columns per latent factor, then pure-noise columns."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, k))
    return [
        np.column_stack(
            [
                np.repeat(z, 3, axis=1) + 0.2 * rng.standard_normal((n, 3 * k)),
                rng.standard_normal((n, noise)),
            ]
        )
        for noise in (27, 17)
    ]


_PENALISED = [
    ElasticNetCCA(alpha=0.1, l1_ratio=0.9),
    MultiTaskElasticNetCCA(alpha=0.3, l1_ratio=0.9),
    IPLSCCA(alpha=0.3),
    PMDCCA(l1_bound=0.3),
    ParkhomenkoCCA(alpha=0.3),
    SpanCCA(span=3),
    ADMMCCA(alpha=1.0),
]


@pytest.mark.parametrize("model", _PENALISED, ids=lambda m: type(m).__name__)
def test_penalty_zeroes_some_weights(
    model: BaseModel, correlated_views: list[np.ndarray]
) -> None:
    """Each penalty zeroes some, but not all, of a view's weights."""
    model.set_params(random_state=0).fit(correlated_views)
    assert any(0.0 < np.isclose(w, 0.0).mean() < 1.0 for w in model.weights_)


@pytest.mark.parametrize("model", _PENALISED, ids=lambda m: type(m).__name__)
def test_a_penalty_means_the_same_at_any_sample_size(
    model: BaseModel, correlated_views: list[np.ndarray]
) -> None:
    """Stacking the data twice keeps the covariances, so it keeps the weights.

    Up to the 1% that the covariances' n - 1 divisor moves at n = 50.
    """
    model.set_params(random_state=0)
    once = model.fit(correlated_views).weights_
    twice = model.fit([np.vstack([v, v]) for v in correlated_views]).weights_
    for a, b in zip(once, twice):
        np.testing.assert_array_equal(a == 0, b == 0)
        np.testing.assert_allclose(np.abs(a), np.abs(b), atol=0.01)


@pytest.mark.parametrize("cls", [ElasticNetCCA, MultiTaskElasticNetCCA])
def test_alpha_per_view(cls: type, correlated_views: list[np.ndarray]) -> None:
    """A larger alpha makes that view, and only that view, sparser."""
    model = cls(n_components=2, alpha=[0.001, 1.0], l1_ratio=0.9, random_state=0)
    rows = [_active_rows(w) for w in model.fit(correlated_views).weights_]
    assert rows[1] < rows[0]


@pytest.mark.parametrize("cls", [ElasticNetCCA, MultiTaskElasticNetCCA])
def test_each_sweep_lowers_the_objective(
    cls: type, correlated_views: list[np.ndarray]
) -> None:
    """Coordinate descent never raises the penalised EY objective."""

    def objective(max_iter: int) -> float:
        model = cls(
            n_components=2, alpha=0.1, max_iter=max_iter, tol=1e-300, random_state=0
        )
        model.fit(correlated_views)
        return ey_loss(model.transform(correlated_views))["objective"]

    losses = [objective(i) for i in range(1, 6)]
    assert losses[-1] <= losses[0]


@pytest.mark.parametrize(
    ("alpha", "fits"), [(0.1, True), (1.0, True), (2.0, True), (4.0, False)]
)
def test_multitask_with_one_component_is_elasticnet(alpha: float, fits: bool) -> None:
    """With one component the row-group penalty is the elastic net's.

    Both avoid the all-zero local minimum until the penalty outweighs the fit.
    """
    views = _signal_and_noise_columns()
    kwargs = {"alpha": alpha, "max_iter": 500, "random_state": 0}
    group = MultiTaskElasticNetCCA(**kwargs).fit(views).weights_
    lasso = ElasticNetCCA(**kwargs).fit(views).weights_
    for g, e in zip(group, lasso):
        np.testing.assert_allclose(g, e, atol=0.02)
        assert g.any() == fits


def test_multitask_drops_a_feature_from_every_component(
    correlated_views: list[np.ndarray],
) -> None:
    """A row is active in all components or none."""
    model = MultiTaskElasticNetCCA(
        n_components=2, alpha=0.3, l1_ratio=0.9, random_state=0
    )
    for w in model.fit(correlated_views).weights_:
        active = np.abs(w) > 1e-10
        np.testing.assert_array_equal(active.any(axis=1), active.all(axis=1))


def test_positive_weights(correlated_views: list[np.ndarray]) -> None:
    """positive=True gives non-negative weights."""
    model = ElasticNetCCA(n_components=2, alpha=0.05, positive=True, random_state=0)
    assert all(np.all(w >= 0) for w in model.fit(correlated_views).weights_)


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(3, [3, 3]), ([2, 4], [2, 4]), (None, [2, 2]), (100, [10, 8])],
)
def test_omp_keeps_its_budget_of_features(
    correlated_views: list[np.ndarray],
    budget: int | list[int] | None,
    expected: list[int],
) -> None:
    """n_nonzero_coefs per view: a tenth, at least n_components, at most all."""
    model = OrthogonalMatchingPursuitCCA(
        n_components=2, n_nonzero_coefs=budget, random_state=0
    )
    assert [_active_rows(w) for w in model.fit(correlated_views).weights_] == expected
    for bad in ([3], [3, 1]):
        with pytest.raises(ValueError, match="n_nonzero_coefs"):
            model.set_params(n_nonzero_coefs=bad).fit(correlated_views)


@pytest.mark.parametrize("cls", [IPLSCCA, ADMMCCA])
def test_unpenalised_deflation_is_cca_on_every_component(cls: type) -> None:
    """At alpha=0 each deflated component is the next canonical pair."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal((300, 3)) * [3, 2, 1]
    views = [
        z @ rng.standard_normal((3, p)) + rng.standard_normal((300, p)) for p in (6, 5)
    ]
    model = cls(n_components=3, alpha=0.0, random_state=0).fit(views)
    np.testing.assert_allclose(
        pairwise_correlations(model.transform(views))[0, 1],
        pairwise_correlations(MCCA(3).fit(views).transform(views))[0, 1],
        atol=1e-3,
    )


def test_pmd_l1_bound_one_is_unconstrained(two_views: list[np.ndarray]) -> None:
    """Sparsity falls as l1_bound rises, and l1_bound=1 keeps every feature."""
    active = [
        sum(
            _active_rows(w)
            for w in PMDCCA(l1_bound=b, random_state=0).fit(two_views).weights_
        )
        for b in (0.3, 0.5, 0.7, 1.0)
    ]
    assert active == sorted(active)
    assert active[-1] == 18


def test_pmd_ignores_the_scale_of_the_data(two_views: list[np.ndarray]) -> None:
    """Tau bounds unit-norm weights, so rescaling the data changes nothing."""
    a = PMDCCA(l1_bound=0.5, random_state=0).fit(two_views).weights_
    b = PMDCCA(l1_bound=0.5, random_state=0).fit([v * 37.0 for v in two_views]).weights_
    for wa, wb in zip(a, b):
        np.testing.assert_allclose(np.abs(wa), np.abs(wb), atol=1e-6)


def test_sar_selects_the_signal_columns() -> None:
    """BIC keeps each factor's columns and drops noise, one component per factor."""
    x, y = _signal_and_noise_columns(k=2)
    model = SAR(n_components=2).fit([x, y])
    for w in model.weights_:
        assert np.all(np.abs(w[:6]).sum(axis=1) > 0)
        assert np.sum(w[6:] ** 2) < 0.1 * np.sum(w[:6] ** 2)
    zx, zy = model.transform([x, y])
    assert min(np.corrcoef(zx[:, d], zy[:, d])[0, 1] for d in range(2)) > 0.9
    assert abs(np.corrcoef(zx[:, 0], zx[:, 1])[0, 1]) < 0.3


def test_sar_selects_nothing_from_noise() -> None:
    """With no shared signal and ample samples, BIC chooses all-zero weights."""
    rng = np.random.default_rng(0)
    noise = [rng.standard_normal((1000, 10)), rng.standard_normal((1000, 8))]
    assert not any(w.any() for w in SAR(max_iter=50).fit(noise).weights_)


def test_admm_update_solves_the_papers_problem() -> None:
    """One ADMM block solve meets the KKT conditions of Suo et al. (2017).

    Their problem is max_w w'X't - alpha ||w||_1 subject to ||Xw|| <= 1.
    """
    rng = np.random.default_rng(1)
    X, target, alpha = rng.standard_normal((60, 15)), rng.standard_normal(60) * 0.5, 0.2
    model = ADMMCCA(admm_iter=20_000, tol=1e-14)
    step = 1.0 / np.linalg.norm(X, ord=2) ** 2
    c = X.T @ target
    w, _, _ = model._admm_block(X, c, alpha, step, np.zeros(15), *np.zeros((2, 60)))
    score = X @ w
    assert np.linalg.norm(score) > 0.99  # the constraint is active
    active = np.abs(w) > 1e-6
    grad = X.T @ score / np.linalg.norm(score)
    duals = (c[active] - alpha * np.sign(w[active])) / grad[active]
    np.testing.assert_allclose(duals, duals.mean(), rtol=1e-3)
    assert duals.mean() > 0
    assert np.all(np.abs(c[~active] - duals.mean() * grad[~active]) <= alpha + 1e-3)
