"""Tests for cca_zoo.linear._projection_pursuit_cca (robust CCA, projection pursuit)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import MCCA, ProjectionPursuitCCA
from cca_zoo.linear._projection_pursuit_cca import (
    _angles_to_unit_vector,
    mcd_projection_index,
    spearman_projection_index,
)


def _make_model(latent_dimensions: int = 1, **kwargs: object) -> ProjectionPursuitCCA:
    return ProjectionPursuitCCA(
        latent_dimensions=latent_dimensions,
        n_restarts=2,
        max_iter=30,
        random_state=0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# _angles_to_unit_vector
# ---------------------------------------------------------------------------


def test_angles_to_unit_vector_is_unit_norm() -> None:
    """The recovered vector always has unit norm, for any angles."""
    rng = np.random.default_rng(0)
    for p in range(1, 6):
        theta = rng.uniform(0, 2 * np.pi, size=max(p - 1, 0))
        a = _angles_to_unit_vector(theta, p)
        assert a.shape == (p,)
        np.testing.assert_allclose(np.linalg.norm(a), 1.0, atol=1e-10)


def test_angles_to_unit_vector_p1_is_scalar_one() -> None:
    """A 1-dimensional view needs no angle: the direction is just [1]."""
    np.testing.assert_allclose(_angles_to_unit_vector(np.zeros(0), 1), [1.0])


# ---------------------------------------------------------------------------
# projection indices
# ---------------------------------------------------------------------------


def test_spearman_index_perfect_for_monotone_scores() -> None:
    """Spearman index is 1 for a perfectly monotone (if nonlinear) relationship."""
    u = np.arange(20.0)
    v = u**3
    assert spearman_projection_index(u, v) == pytest.approx(1.0)


def test_spearman_index_zero_for_constant_input() -> None:
    """Spearman index falls back to 0 (not NaN) when one side is constant."""
    u = np.zeros(10)
    v = np.arange(10.0)
    assert spearman_projection_index(u, v) == 0.0


def test_mcd_index_matches_sign_of_relationship() -> None:
    """MCD index is positive for a positively-correlated bivariate scatter."""
    rng = np.random.default_rng(0)
    u = rng.standard_normal(200)
    v = u + 0.1 * rng.standard_normal(200)
    idx = mcd_projection_index(u, v, support_fraction=0.75, random_state=0)
    assert idx > 0.9


# ---------------------------------------------------------------------------
# fit completes / shapes
# ---------------------------------------------------------------------------


def test_two_view_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes on two-view data without error."""
    model = _make_model()
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_three_view_fit_completes(three_views_small: list[np.ndarray]) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model()
    fitted = model.fit(three_views_small)
    assert fitted is model


def test_mcd_projection_index_fit_completes(two_views_small: list[np.ndarray]) -> None:
    """Fit completes with the (more expensive) MCD projection index too."""
    model = _make_model(projection_index="mcd")
    fitted = model.fit(two_views_small)
    assert fitted is model


def test_weights_shapes_and_matches_transform(
    two_views_small: list[np.ndarray],
) -> None:
    """Weights are real (p_i, k) arrays and transform(v) == centred(v) @ weights."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    weights = model.weights_
    assert len(weights) == 2
    for w, v in zip(weights, two_views_small):
        assert w.shape == (v.shape[1], k)

    transformed = model.transform(two_views_small)
    for v, w, t, mean in zip(two_views_small, weights, transformed, model.means_):
        np.testing.assert_allclose((v - mean) @ w, t, atol=1e-8)


def test_weights_not_fitted_raises() -> None:
    """Transform before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = ProjectionPursuitCCA()
    with pytest.raises(NotFittedError):
        model.transform([np.ones((3, 2)), np.ones((3, 2))])


def test_directions_are_unit_norm(two_views_small: list[np.ndarray]) -> None:
    """Each fitted per-dimension weight vector has unit norm."""
    model = _make_model(latent_dimensions=2).fit(two_views_small)
    for w in model.weights_:
        norms = np.linalg.norm(w, axis=0)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)


# ---------------------------------------------------------------------------
# Correctness on clean data
# ---------------------------------------------------------------------------


def test_finds_correlation_on_correlated_views(
    correlated_views: list[np.ndarray],
) -> None:
    """On clean data ProjectionPursuitCCA still finds real correlation."""
    model = ProjectionPursuitCCA(latent_dimensions=1, n_restarts=5, random_state=0)
    s = model.fit(correlated_views).score(correlated_views)
    assert np.all(s > 0.5), f"Expected substantial correlation, got {s}"


# ---------------------------------------------------------------------------
# The headline scenario: robust to a handful of extreme outliers
# ---------------------------------------------------------------------------


def _make_outlier_contaminated_data(
    seed: int, n_train: int, n_test: int, p1: int, p2: int, contam_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two views sharing one real latent factor, with a few extreme outlier rows."""
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(p1)
    w1 /= np.linalg.norm(w1)
    w2 = rng.standard_normal(p2)
    w2 /= np.linalg.norm(w2)

    def clean(n: int) -> tuple[np.ndarray, np.ndarray]:
        t = rng.standard_normal(n)
        X = np.outer(t, w1) + 0.3 * rng.standard_normal((n, p1))
        Y = np.outer(t, w2) + 0.3 * rng.standard_normal((n, p2))
        return X, Y

    x_test, y_test = clean(n_test)

    n_bad = int(round(contam_frac * n_train))
    n_good = n_train - n_bad
    x_good, y_good = clean(n_good)
    x_bad = rng.standard_normal((n_bad, p1)) * 15
    y_bad = rng.standard_normal((n_bad, p2)) * 15
    x_train = np.vstack([x_good, x_bad])
    y_train = np.vstack([y_good, y_bad])

    perm = rng.permutation(n_train)
    return x_train[perm], y_train[perm], x_test, y_test


def _held_out_corr(
    model: MCCA | ProjectionPursuitCCA, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    z1, z2 = model.transform([x_test, y_test])
    return abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])


def test_robust_to_extreme_outliers_unlike_mcca() -> None:
    """A handful of huge, unrelated-across-views rows wreck plain MCCA but not this."""
    x_train, y_train, x_test, y_test = _make_outlier_contaminated_data(
        seed=0, n_train=150, n_test=200, p1=4, p2=3, contam_frac=0.2
    )

    mcca_corr = _held_out_corr(
        MCCA(latent_dimensions=1, c=0.1).fit([x_train, y_train]), x_test, y_test
    )
    pp_model = ProjectionPursuitCCA(
        latent_dimensions=1, n_restarts=5, random_state=0
    ).fit([x_train, y_train])
    pp_corr = _held_out_corr(pp_model, x_test, y_test)

    assert pp_corr > mcca_corr + 0.2


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = ProjectionPursuitCCA(
        latent_dimensions=2,
        projection_index="mcd",
        mcd_support_fraction=0.8,
        n_restarts=3,
        random_state=0,
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_projection_index_raises() -> None:
    """An unrecognised projection_index is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ProjectionPursuitCCA(projection_index="bogus")._validate_params()


def test_invalid_n_restarts_raises() -> None:
    """n_restarts below 1 is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ProjectionPursuitCCA(n_restarts=0)._validate_params()


def test_invalid_mcd_support_fraction_raises() -> None:
    """mcd_support_fraction outside (0, 1] is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ProjectionPursuitCCA(mcd_support_fraction=1.5)._validate_params()
