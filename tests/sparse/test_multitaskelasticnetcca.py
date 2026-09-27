"""Tests for MultiTaskElasticNetCCA."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.sparse import ElasticNetCCA, MultiTaskElasticNetCCA

# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_objective_decreases_monotonically(
    correlated_views: list[np.ndarray],
) -> None:
    """Every coordinate-descent sweep does not increase the penalised EY objective."""
    from cca_zoo._utils._ey import _group_penalty, ey_loss

    objs = []
    for n_iter in range(1, 8):
        model = MultiTaskElasticNetCCA(
            n_components=2,
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=n_iter,
            tol=1e-300,
            random_state=0,
        )
        model.fit(correlated_views)
        reps = model.transform(correlated_views)
        n_views = len(correlated_views)
        penalty = _group_penalty(
            model.weights_, [model.alpha] * n_views, [model.l1_ratio] * n_views
        )
        objs.append(ey_loss(reps)["objective"] + penalty)
    assert np.all(np.diff(objs) <= 1e-8), objs


def test_row_sparsity_is_joint_across_components(
    correlated_views: list[np.ndarray],
) -> None:
    """A feature's row is either active in every component or in none."""
    model = MultiTaskElasticNetCCA(
        n_components=2, alpha=0.3, l1_ratio=0.9, random_state=0
    )
    model.fit(correlated_views)
    for w in model.weights_:
        active_per_component = np.abs(w) > 1e-10  # (p, k) boolean
        # For every row, either all components are active or none are.
        row_any = active_per_component.any(axis=1)
        row_all = active_per_component.all(axis=1)
        np.testing.assert_array_equal(row_any, row_all)


def test_higher_alpha_increases_sparsity(
    correlated_views: list[np.ndarray],
) -> None:
    """Increasing alpha (with l1_ratio > 0) should not decrease row sparsity."""
    n_active_rows = []
    for alpha in [0.001, 0.1, 1.0]:
        model = MultiTaskElasticNetCCA(
            n_components=2, alpha=alpha, l1_ratio=0.9, random_state=0
        )
        model.fit(correlated_views)
        n_active_rows.append(
            sum(int(np.sum(np.linalg.norm(w, axis=1) > 1e-10)) for w in model.weights_)
        )
    assert n_active_rows[0] >= n_active_rows[1] >= n_active_rows[2]


def test_per_view_alpha_list_gives_sparser_penalised_view(
    correlated_views: list[np.ndarray],
) -> None:
    """A per-view alpha list applies a stronger penalty to only one view."""
    model = MultiTaskElasticNetCCA(
        n_components=2, alpha=[0.001, 1.0], l1_ratio=0.9, random_state=0
    ).fit(correlated_views)
    n_active_rows = [
        int(np.sum(np.linalg.norm(w, axis=1) > 1e-10)) for w in model.weights_
    ]
    assert n_active_rows[1] < n_active_rows[0]


@pytest.mark.parametrize("alpha", [0.1, 1.0, 2.0])
def test_one_component_matches_elasticnetcca(alpha: float) -> None:
    """With one component the row-group penalty is the elastic net's.

    At alpha=2 the all-zero weights are a local minimum that both solvers
    must avoid: the signal gives a lower penalised objective.
    """
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 1))
    views = [
        z @ rng.standard_normal((1, p)) + 0.3 * rng.standard_normal((200, p))
        for p in (6, 5)
    ]
    kwargs = {"alpha": alpha, "max_iter": 500, "random_state": 0}
    group = MultiTaskElasticNetCCA(**kwargs).fit(views)
    lasso = ElasticNetCCA(**kwargs).fit(views)
    for g, e in zip(group.weights_, lasso.weights_):
        np.testing.assert_allclose(g, e, atol=0.02)
    assert all(w.any() for w in group.weights_)
