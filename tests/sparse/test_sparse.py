"""The sparse CCA models: what each penalty does to the weights."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._base import BaseModel
from cca_zoo._utils._ey import ey_loss
from cca_zoo.sparse import (
    ADMMCCA,
    PMDCCA,
    SAR,
    ElasticNetCCA,
    MultiTaskElasticNetCCA,
    OrthogonalMatchingPursuitCCA,
    ParkhomenkoCCA,
    SpanCCA,
    WaijenborgCCA,
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


@pytest.mark.parametrize(
    "model",
    [
        ElasticNetCCA(alpha=0.1, l1_ratio=0.9),
        MultiTaskElasticNetCCA(alpha=0.3, l1_ratio=0.9),
        PMDCCA(tau=0.3),
        ParkhomenkoCCA(tau=2.0),
        SpanCCA(span=3),
        ADMMCCA(tau=1.0),
        WaijenborgCCA(alpha=0.1, l1_ratio=1.0),
    ],
    ids=lambda m: type(m).__name__,
)
def test_penalty_zeroes_some_weights(
    model: BaseModel, correlated_views: list[np.ndarray]
) -> None:
    """Each penalty zeroes some, but not all, of a view's weights."""
    model.set_params(random_state=0).fit(correlated_views)
    assert any(0.0 < np.isclose(w, 0.0).mean() < 1.0 for w in model.weights_)


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


@pytest.mark.parametrize("alpha", [0.1, 1.0, 2.0])
def test_multitask_with_one_component_is_elasticnet(alpha: float) -> None:
    """With one component the row-group penalty is the elastic net's.

    At alpha=2 the all-zero weights are a local minimum both must avoid.
    """
    views = _signal_and_noise_columns()
    kwargs = {"alpha": alpha, "max_iter": 500, "random_state": 0}
    group = MultiTaskElasticNetCCA(**kwargs).fit(views).weights_
    lasso = ElasticNetCCA(**kwargs).fit(views).weights_
    for g, e in zip(group, lasso):
        np.testing.assert_allclose(g, e, atol=0.02)
        assert g.any()


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
    [(3, [3, 3]), ([2, 4], [2, 4]), (None, [1, 1]), (100, [10, 8])],
)
def test_omp_keeps_its_budget_of_features(
    correlated_views: list[np.ndarray],
    budget: int | list[int] | None,
    expected: list[int],
) -> None:
    """Exactly n_nonzero_coefs features per view, by default a tenth, capped at all."""
    model = OrthogonalMatchingPursuitCCA(
        n_components=2, n_nonzero_coefs=budget, random_state=0
    )
    assert [_active_rows(w) for w in model.fit(correlated_views).weights_] == expected


def test_pmd_tau_one_is_unconstrained(two_views: list[np.ndarray]) -> None:
    """Sparsity falls as tau rises, and tau=1 keeps every feature."""
    active = [
        sum(
            _active_rows(w)
            for w in PMDCCA(tau=tau, random_state=0).fit(two_views).weights_
        )
        for tau in (0.3, 0.5, 0.7, 1.0)
    ]
    assert active == sorted(active)
    assert active[-1] == 18


def test_pmd_ignores_the_scale_of_the_data(two_views: list[np.ndarray]) -> None:
    """Tau bounds unit-norm weights, so rescaling the data changes nothing."""
    a = PMDCCA(tau=0.5, random_state=0).fit(two_views).weights_
    b = PMDCCA(tau=0.5, random_state=0).fit([v * 37.0 for v in two_views]).weights_
    for wa, wb in zip(a, b):
        np.testing.assert_allclose(np.abs(wa), np.abs(wb), atol=1e-6)


def test_sar_selects_the_signal_columns() -> None:
    """BIC keeps each factor's columns and drops noise, one component per factor."""
    x, y = _signal_and_noise_columns(k=2)
    model = SAR(n_components=2, random_state=0).fit([x, y])
    for w in model.weights_:
        assert np.all(np.abs(w[:6]).sum(axis=1) > 0)
        assert np.sum(w[6:] ** 2) < 0.1 * np.sum(w[:6] ** 2)
    zx, zy = model.transform([x, y])
    assert min(np.corrcoef(zx[:, d], zy[:, d])[0, 1] for d in range(2)) > 0.9
    assert abs(np.corrcoef(zx[:, 0], zx[:, 1])[0, 1]) < 0.3


def test_sar_selects_nothing_from_noise(two_views: list[np.ndarray]) -> None:
    """With no shared signal, BIC chooses all-zero weights."""
    assert not any(
        w.any() for w in SAR(max_iter=50, random_state=0).fit(two_views).weights_
    )


def test_admm_update_solves_the_papers_problem() -> None:
    """One ADMM block solve meets the KKT conditions of Suo et al. (2017).

    Their problem is max_w w'X't - tau ||w||_1 subject to ||Xw|| <= 1.
    """
    rng = np.random.default_rng(1)
    X, target, tau = rng.standard_normal((60, 15)), rng.standard_normal(60) * 0.5, 0.2
    model = ADMMCCA(tau=tau, max_iter=1, admm_iter=20_000, tol=1e-14, random_state=0)
    w = [np.zeros(15), np.array([1.0])]
    model._fit_single([X, target[:, None]], w, 0)
    score = X @ w[0]
    assert np.linalg.norm(score) > 0.99  # the constraint is active
    active = np.abs(w[0]) > 1e-6
    c, grad = X.T @ target, X.T @ score / np.linalg.norm(score)
    duals = (c[active] - tau * np.sign(w[0][active])) / grad[active]
    np.testing.assert_allclose(duals, duals.mean(), rtol=1e-3)
    assert duals.mean() > 0
    assert np.all(np.abs(c[~active] - duals.mean() * grad[~active]) <= tau + 1e-3)
