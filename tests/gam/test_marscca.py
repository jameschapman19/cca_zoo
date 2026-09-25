"""Tests for MARSCCA."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cca_zoo.gam import GAMCCA, MARSCCA
from cca_zoo.gam._marscca import _evaluate_terms

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
    model = MARSCCA(max_terms=[4, 12]).fit(correlated_views)
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


def test_basis_functions_strings(two_views_small: list[np.ndarray]) -> None:
    """basis_functions reports raw-unit knots, one string per term."""
    views = [v + 10.0 for v in two_views_small]
    model = MARSCCA(max_terms=4).fit(views)
    names = model.basis_functions(0)
    assert len(names) == len(model.encoders_[0].terms_)
    feature, knot, sign = model.encoders_[0].terms_[0][0]
    raw = f"{knot + model.means_[0][feature]:.4g}"
    expected = f"h(x{feature} - {raw})" if sign > 0 else f"h({raw} - x{feature})"
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
