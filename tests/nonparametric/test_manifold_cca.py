"""Tests for cca_zoo.nonparametric._manifold_cca (transductive manifold CCA)."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo.linear import MCCA
from cca_zoo.nonparametric import ManifoldCCA
from cca_zoo.nonparametric._manifold_cca import (
    _centering_matrix,
    _floor_min_eig,
    _laplacian_operator,
    _lle_operator,
)


def _make_model(method: str = "laplacian", **kwargs: object) -> ManifoldCCA:
    kwargs.setdefault("n_neighbors", 8)
    return ManifoldCCA(method=method, **kwargs)


# ---------------------------------------------------------------------------
# Operator building blocks
# ---------------------------------------------------------------------------


def test_centering_matrix_rows_and_columns_sum_to_zero() -> None:
    """The centering matrix annihilates the all-ones vector."""
    C = _centering_matrix(10)
    np.testing.assert_allclose(C @ np.ones(10), 0, atol=1e-10)
    np.testing.assert_allclose(np.ones(10) @ C, 0, atol=1e-10)
    np.testing.assert_allclose(C, C.T)


def test_laplacian_operator_is_symmetric_psd(
    two_views_small: list[np.ndarray],
) -> None:
    """The graph Laplacian is symmetric and positive semi-definite."""
    L = _laplacian_operator(
        two_views_small[0], n_neighbors=8, affinity="rbf", gamma=None
    )
    np.testing.assert_allclose(L, L.T, atol=1e-10)
    assert np.linalg.eigvalsh(L).min() > -1e-8


def test_lle_operator_is_symmetric_psd(two_views_small: list[np.ndarray]) -> None:
    """The LLE reconstruction operator is symmetric and positive semi-definite."""
    M = _lle_operator(two_views_small[0], n_neighbors=8, reg=1e-3)
    np.testing.assert_allclose(M, M.T, atol=1e-10)
    assert np.linalg.eigvalsh(M).min() > -1e-8


def test_lle_operator_near_zero_for_the_ones_vector(
    two_views_small: list[np.ndarray],
) -> None:
    """A constant embedding is (almost) perfectly reconstructed by any weights."""
    M = _lle_operator(two_views_small[0], n_neighbors=8, reg=1e-3)
    n = two_views_small[0].shape[0]
    ones = np.ones(n)
    assert (ones @ M @ ones) / n < 1e-6


def test_floor_min_eig_raises_the_floor() -> None:
    """_floor_min_eig shifts a matrix's spectrum up to at least eps."""
    M = np.diag([-1.0, 0.0, 2.0])
    floored = _floor_min_eig(M, eps=0.5)
    assert np.linalg.eigvalsh(floored).min() >= 0.5 - 1e-10


# ---------------------------------------------------------------------------
# fit / transform completes, shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["laplacian", "lle"])
def test_two_view_fit_completes(
    method: str, two_views_small: list[np.ndarray]
) -> None:
    """Fit completes on two-view data without error, for both operator types."""
    model = _make_model(method=method).fit(two_views_small)
    assert hasattr(model, "weights_")


@pytest.mark.parametrize("method", ["laplacian", "lle"])
def test_three_view_fit_completes(
    method: str, three_views_small: list[np.ndarray]
) -> None:
    """Fit completes on three-view data without error."""
    model = _make_model(method=method, n_neighbors=6).fit(three_views_small)
    assert len(model.weights_) == 3


def test_weights_shapes(two_views_small: list[np.ndarray]) -> None:
    """weights_[i] is (n_train_samples, k), the training embedding itself."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    n = two_views_small[0].shape[0]
    for w in model.weights:
        assert w.shape == (n, k)


def test_transform_shapes_on_new_data(
    two_views_small: list[np.ndarray], two_views_test: list[np.ndarray]
) -> None:
    """Transform on unseen data returns (n_test, k) arrays via the extrapolator."""
    k = 2
    model = _make_model(latent_dimensions=k).fit(two_views_small)
    test_views = [v[:, :5] for v in two_views_test]  # match two_views_small's width
    transformed = model.transform(test_views)
    assert len(transformed) == 2
    for t in transformed:
        assert t.shape == (test_views[0].shape[0], k)


def test_transform_on_training_data_matches_weights_reasonably(
    two_views_small: list[np.ndarray],
) -> None:
    """Transform on the training data roughly recovers the fitted embedding.

    Not an exact match -- the KernelRidge extrapolator is a smoothed fit of
    the training embedding, not an interpolant -- but it should correlate
    strongly with it dimension-by-dimension.
    """
    model = _make_model(extrapolator_alpha=1e-3).fit(two_views_small)
    transformed = model.transform(two_views_small)
    for w, t in zip(model.weights, transformed):
        corr = np.corrcoef(w[:, 0], t[:, 0])[0, 1]
        assert abs(corr) > 0.9


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = ManifoldCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_independent_views_give_near_zero_training_correlation() -> None:
    """Independent random views shouldn't fake up cross-view correlation."""
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.standard_normal((n, 8))
    x2 = rng.standard_normal((n, 6))
    for method in ["laplacian", "lle"]:
        model = _make_model(method=method).fit([x1, x2])
        z1, z2 = model.weights
        corr = abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])
        assert corr < 0.3, f"{method}: unexpectedly high spurious correlation {corr}"


def _spiral_views(
    t: np.ndarray, rng: np.random.Generator, noise: float = 0.03
) -> tuple[np.ndarray, np.ndarray]:
    """Two different injective spiral embeddings of a shared 1-D coordinate t.

    Both views are smooth, injective functions of t, but with different
    angular frequency/phase, so there is no linear map from one view's raw
    ambient coordinates to the other's -- exactly the regime manifold
    methods (which only use *local* neighbourhood structure) are built for,
    and plain linear CCA is not.
    """
    x1 = np.stack([t * np.cos(t), t * np.sin(t)], axis=1)
    x2 = np.stack([t * np.cos(2 * t + 1), t * np.sin(2 * t + 1)], axis=1)
    x1 = x1 + noise * rng.standard_normal(x1.shape)
    x2 = x2 + noise * rng.standard_normal(x2.shape)
    return x1, x2


def test_laplacian_beats_linear_mcca_on_a_shared_nonlinear_spiral() -> None:
    """On a shared-but-nonlinearly-embedded coordinate, linear CCA underperforms.

    Two spirals sharing one latent coordinate t, wound at different angular
    frequencies: no linear combination of one spiral's (x, y) coordinates
    correlates well with the other's, but a k-NN graph on either spiral
    still respects the shared ordering of t -- laplacian ManifoldCCA should
    recover substantially more held-out correlation than plain MCCA.
    """
    rng = np.random.default_rng(0)
    n_train, n_test = 200, 200
    t_train = np.sort(rng.uniform(0.5, 4 * np.pi, n_train))
    t_test = np.sort(rng.uniform(0.5, 4 * np.pi, n_test))

    x1_tr, x2_tr = _spiral_views(t_train, rng)
    x1_te, x2_te = _spiral_views(t_test, rng)

    linear_corr = MCCA(latent_dimensions=1, c=0.1, pca=False).fit(
        [x1_tr, x2_tr]
    ).score([x1_te, x2_te])[0]
    manifold_corr = ManifoldCCA(
        method="laplacian", n_neighbors=10, latent_dimensions=1, extrapolator_alpha=0.1
    ).fit([x1_tr, x2_tr]).score([x1_te, x2_te])[0]

    assert manifold_corr > linear_corr + 0.3, (
        f"expected laplacian ({manifold_corr:.2f}) to clearly beat "
        f"linear MCCA ({linear_corr:.2f}) on the spiral data"
    )


# ---------------------------------------------------------------------------
# sklearn compatibility spot-checks
# ---------------------------------------------------------------------------


def test_clone_and_get_params_roundtrip() -> None:
    """clone()/get_params() round-trip correctly (sklearn BaseEstimator contract)."""
    from sklearn.base import clone

    model = ManifoldCCA(
        latent_dimensions=2, method="lle", n_neighbors=6, lle_reg=1e-2
    )
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_invalid_method_raises() -> None:
    """An unrecognised method is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ManifoldCCA(method="isomap")._validate_params()


def test_invalid_n_neighbors_raises() -> None:
    """n_neighbors below 1 is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ManifoldCCA(n_neighbors=0)._validate_params()


def test_invalid_affinity_raises() -> None:
    """An unrecognised affinity is rejected by parameter validation."""
    from sklearn.utils._param_validation import InvalidParameterError

    with pytest.raises(InvalidParameterError):
        ManifoldCCA(affinity="not-an-affinity")._validate_params()
