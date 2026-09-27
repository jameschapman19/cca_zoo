"""Tests for ElasticNetCCA."""

from __future__ import annotations

import numpy as np

from cca_zoo.sparse import ElasticNetCCA

# ---------------------------------------------------------------------------
# Correctness / optimality
# ---------------------------------------------------------------------------


def test_objective_decreases_monotonically(
    correlated_views: list[np.ndarray],
) -> None:
    """Every coordinate-descent sweep strictly lowers the penalised EY objective."""
    from cca_zoo._utils._ey import ey_loss

    objs = []
    for n_iter in range(1, 11):
        model = ElasticNetCCA(
            n_components=1,
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=n_iter,
            tol=1e-300,
            random_state=0,
        )
        model.fit(correlated_views)
        reps = model.transform(correlated_views)
        penalty = sum(
            model.alpha * model.l1_ratio * np.sum(np.abs(w))
            + 0.5 * model.alpha * (1 - model.l1_ratio) * np.sum(w**2)
            for w in model.weights_
        )
        objs.append(ey_loss(reps)["objective"] + penalty)
    assert np.all(np.diff(objs) <= 1e-9), objs


def test_higher_alpha_increases_sparsity(
    correlated_views: list[np.ndarray],
) -> None:
    """Increasing alpha (with l1_ratio > 0) should not decrease sparsity."""
    n_nonzero = []
    for alpha in [0.001, 0.1, 1.0]:
        model = ElasticNetCCA(n_components=1, alpha=alpha, l1_ratio=0.9, random_state=0)
        model.fit(correlated_views)
        n_nonzero.append(sum((np.abs(w) > 1e-10).sum() for w in model.weights_))
    assert n_nonzero[0] >= n_nonzero[1] >= n_nonzero[2]


def test_per_view_alpha_list_gives_sparser_penalised_view(
    correlated_views: list[np.ndarray],
) -> None:
    """A per-view alpha list applies a stronger penalty to only one view."""
    model = ElasticNetCCA(
        n_components=1, alpha=[0.001, 1.0], l1_ratio=0.9, random_state=0
    ).fit(correlated_views)
    n_nonzero = [int((np.abs(w) > 1e-10).sum()) for w in model.weights_]
    assert n_nonzero[1] < n_nonzero[0]


# ---------------------------------------------------------------------------
# positive constraint
# ---------------------------------------------------------------------------


def test_positive_constraint_yields_nonnegative_weights(
    correlated_views: list[np.ndarray],
) -> None:
    """positive=True yields weights with no negative entries."""
    model = ElasticNetCCA(
        n_components=2, alpha=0.05, l1_ratio=0.5, positive=True, random_state=0
    )
    model.fit(correlated_views)
    for w in model.weights_:
        assert np.all(w >= -1e-10)
