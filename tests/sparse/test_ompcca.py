"""Tests for OrthogonalMatchingPursuitCCA."""

from __future__ import annotations

import numpy as np

from cca_zoo.sparse import OrthogonalMatchingPursuitCCA

# ---------------------------------------------------------------------------
# Correctness / cardinality
# ---------------------------------------------------------------------------


def test_active_feature_count_matches_budget(
    correlated_views: list[np.ndarray],
) -> None:
    """Each view has exactly n_nonzero_coefs active (nonzero-row) features."""
    budget = 3
    model = OrthogonalMatchingPursuitCCA(
        n_components=2, n_nonzero_coefs=budget, random_state=0
    )
    model.fit(correlated_views)
    for w in model.weights_:
        n_active = int(np.sum(np.linalg.norm(w, axis=1) > 1e-10))
        assert n_active == budget


def test_per_view_budget_list(correlated_views: list[np.ndarray]) -> None:
    """A list of per-view budgets is honoured independently per view."""
    budgets = [2, 4]
    model = OrthogonalMatchingPursuitCCA(
        n_components=1, n_nonzero_coefs=budgets, random_state=0
    )
    model.fit(correlated_views)
    for w, budget in zip(model.weights_, budgets):
        n_active = int(np.sum(np.abs(w.ravel()) > 1e-10))
        assert n_active == budget


def test_budget_exceeding_n_features_is_capped(
    correlated_views: list[np.ndarray],
) -> None:
    """A budget larger than a view's feature count is capped, not an error."""
    n_features = correlated_views[0].shape[1]
    model = OrthogonalMatchingPursuitCCA(
        n_components=1, n_nonzero_coefs=n_features + 100, random_state=0
    )
    model.fit(correlated_views)
    n_active = int(np.sum(np.abs(model.weights_[0].ravel()) > 1e-10))
    assert n_active <= n_features


def test_default_n_nonzero_coefs_is_ten_percent(
    correlated_views: list[np.ndarray],
) -> None:
    """With n_nonzero_coefs=None, each view defaults to max(1, n_features // 10)."""
    model = OrthogonalMatchingPursuitCCA(n_components=1, random_state=0)
    model.fit(correlated_views)
    for w, v in zip(model.weights_, correlated_views):
        expected = max(1, v.shape[1] // 10)
        n_active = int(np.sum(np.abs(w.ravel()) > 1e-10))
        assert n_active == expected
