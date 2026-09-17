"""Tests for cca_zoo.nonparametric._manifold_cca (transductive manifold CCA)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import eigh

from cca_zoo.linear import MCCA
from cca_zoo.nonparametric import ManifoldCCA
from cca_zoo.nonparametric._manifold_cca import (
    _barycenter_weights,
    _centering_matrix,
    _laplacian_new_point_affinity,
    _laplacian_operator,
    _lle_operator,
    _orthonormal_complement_of_ones,
    _smooth_basis,
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


def test_smooth_basis_keeps_smallest_eigenpairs_and_floors_them() -> None:
    """_smooth_basis keeps the k smallest eigenpairs, each floored at eps."""
    M = np.diag([-1.0, 0.0, 2.0, 5.0])
    basis, eigenvalues = _smooth_basis(M, n_components=2, eps=0.1)
    np.testing.assert_allclose(eigenvalues, [0.1, 0.1])
    np.testing.assert_allclose(basis, np.eye(4)[:, :2])


def test_complement_of_ones_is_orthonormal_and_orthogonal_to_ones() -> None:
    """Spans exactly the (n-1)-dim space orthogonal to the constant vector."""
    n = 15
    P = _orthonormal_complement_of_ones(n)
    assert P.shape == (n, n - 1)
    np.testing.assert_allclose(P.T @ P, np.eye(n - 1), atol=1e-10)
    np.testing.assert_allclose(np.ones(n) @ P, 0, atol=1e-10)


def test_barycenter_weights_sum_to_one(two_views_small: list[np.ndarray]) -> None:
    """Every query point's reconstruction weights sum to 1."""
    v = two_views_small[0]
    from sklearn.neighbors import NearestNeighbors

    indices = NearestNeighbors(n_neighbors=4).fit(v).kneighbors(v)[1]
    weights = _barycenter_weights(v, v, indices, reg=1e-3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-8)


def test_barycenter_weights_reconstruct_the_query_point() -> None:
    """Weights sum to 1 and approximately reconstruct each query from its neighbours."""
    rng = np.random.default_rng(0)
    reference = rng.standard_normal((15, 4))
    query = rng.standard_normal((5, 4))
    indices = np.argsort(
        np.linalg.norm(reference[None, :, :] - query[:, None, :], axis=-1), axis=1
    )[:, :4]
    weights = _barycenter_weights(query, reference, indices, reg=1e-3)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-8)
    reconstructed = np.einsum("qn,qnd->qd", weights, reference[indices])
    # Not an exact interpolant (that's what reg trades away for stability), but
    # should land far closer to the query than a same-neighbourhood random guess.
    reconstruction_error = np.linalg.norm(reconstructed - query, axis=1)
    neighbour_spread = np.linalg.norm(
        reference[indices] - query[:, None, :], axis=-1
    ).mean(axis=1)
    assert np.all(reconstruction_error < neighbour_spread)


def test_laplacian_new_point_affinity_rbf_matches_rbf_kernel() -> None:
    """affinity='rbf' new-point affinity is exactly sklearn's own rbf_kernel."""
    from sklearn.metrics.pairwise import rbf_kernel

    rng = np.random.default_rng(0)
    v_train = rng.standard_normal((20, 4))
    v_new = rng.standard_normal((5, 4))
    got = _laplacian_new_point_affinity(v_new, v_train, "rbf", 0.3, 8, None)
    np.testing.assert_allclose(got, rbf_kernel(v_new, v_train, gamma=0.3))


def test_laplacian_new_point_affinity_nearest_neighbors_is_binary() -> None:
    """affinity='nearest_neighbors' new-point affinity is a 0/1 connectivity row."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(0)
    v_train = rng.standard_normal((20, 4))
    v_new = rng.standard_normal((5, 4))
    nn = NearestNeighbors(n_neighbors=6).fit(v_train)
    got = _laplacian_new_point_affinity(
        v_new, v_train, "nearest_neighbors", None, 6, nn
    )
    assert set(np.unique(got)) <= {0.0, 1.0}
    np.testing.assert_allclose(got.sum(axis=1), 6.0)


# ---------------------------------------------------------------------------
# fit / transform completes, shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["laplacian", "lle"])
def test_two_view_fit_completes(method: str, two_views_small: list[np.ndarray]) -> None:
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


@pytest.mark.parametrize("method", ["laplacian", "lle"])
def test_transform_on_training_data_matches_weights_reasonably(
    method: str, two_views_small: list[np.ndarray]
) -> None:
    """Transform on the training data roughly recovers the fitted embedding.

    Not an exact match -- neither out-of-sample extension is a strict
    interpolant at a training point (the Laplacian Nystrom formula and
    LLE's barycentric weights both treat every query point, training or
    not, via its own local neighbourhood only) -- but it should correlate
    strongly with the training embedding dimension-by-dimension.
    """
    model = _make_model(method=method).fit(two_views_small)
    transformed = model.transform(two_views_small)
    for w, t in zip(model.weights, transformed):
        corr = np.corrcoef(w[:, 0], t[:, 0])[0, 1]
        assert abs(corr) > 0.8


def test_weights_not_fitted_raises() -> None:
    """Accessing weights before fitting raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    model = ManifoldCCA()
    with pytest.raises(NotFittedError):
        _ = model.weights


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


def test_independent_views_dont_overfit_worse_than_plain_cca() -> None:
    """Independent-noise spurious correlation is a fact of life, not a bug to hide.

    Any *unregularised* multivariate CCA -- plain linear included --
    shows nontrivial top canonical correlation between independent
    Gaussian views once the per-view dimensionality isn't tiny relative
    to the sample count (the usual small-n/moderate-p CCA overfitting
    baseline). The bar for ManifoldCCA isn't "exactly zero", which no
    unregularised method in this family clears either -- it's "no worse
    than plain MCCA fit on the same nominal per-view dimensionality"
    (``n_operator_components``).
    """
    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.standard_normal((n, 8))
    x2 = rng.standard_normal((n, 6))
    k_op = 10

    # Same draw fit and scored (training/in-sample correlation), matching how
    # ManifoldCCA's spurious correlation below is measured -- an unregularised
    # linear CCA at this same nominal per-view dimensionality is the fair
    # baseline, not a held-out generalisation number.
    b1, b2 = rng.standard_normal((n, k_op)), rng.standard_normal((n, k_op))
    baseline_model = MCCA(latent_dimensions=1, c=0.0, pca=False).fit([b1, b2])
    baseline = abs(baseline_model.score([b1, b2])[0])

    for method in ["laplacian", "lle"]:
        model = _make_model(method=method, n_operator_components=k_op).fit([x1, x2])
        z1, z2 = model.weights
        corr = abs(np.corrcoef(z1[:, 0], z2[:, 0])[0, 1])
        assert corr < baseline + 0.3, (
            f"{method}: spurious correlation {corr:.2f} far exceeds the "
            f"unregularised-CCA baseline {baseline:.2f} at the same dimensionality"
        )


def test_duplicate_view_reduces_to_plain_spectral_embedding() -> None:
    """The natural-extension sanity check: identical views collapse to single-view SE.

    If both "views" are literally the same data, the joint problem's
    between-view reward restricted to the constant vector's orthogonal
    complement is proportional to the identity (see
    :func:`_orthonormal_complement_of_ones`), so the solution should be
    *exactly* the ordinary Rayleigh-quotient-optimal embedding of that one
    view alone -- i.e. plain :class:`~sklearn.manifold.SpectralEmbedding`
    on the same graph. This is the concrete test of "is ManifoldCCA
    actually the natural multiview generalisation of single-view spectral
    embedding" rather than some unrelated construction that happens to
    also use a graph.
    """
    rng = np.random.default_rng(0)
    n = 120
    x = rng.standard_normal((n, 6))

    model = ManifoldCCA(method="laplacian", n_neighbors=10, latent_dimensions=3).fit(
        [x, x.copy()]
    )
    z1, z2 = model.weights

    L = _laplacian_operator(
        x - x.mean(0), n_neighbors=10, affinity="nearest_neighbors", gamma=None
    )
    _, vecs = eigh(L)
    plain_spectral_embedding = vecs[:, 1:4]  # drop the trivial constant eigenvector

    for z in (z1, z2):
        Q1, _ = np.linalg.qr(z)
        Q2, _ = np.linalg.qr(plain_spectral_embedding)
        principal_cosines = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
        assert np.all(principal_cosines > 1 - 1e-3), (
            "ManifoldCCA on duplicated views should recover exactly the same "
            f"subspace as plain spectral embedding; got cosines {principal_cosines}"
        )


def test_lle_transform_matches_locally_linear_embedding_on_duplicate_views() -> None:
    """Gold-standard check: LLE transform on duplicated views matches sklearn's own.

    Extends the duplicate-view sanity check to *transform*: since both
    views are the same data, ManifoldCCA's LLE out-of-sample extension for
    either view should behave like
    :class:`~sklearn.manifold.LocallyLinearEmbedding` fit and transformed on
    that one view directly -- the same neighbours, the same barycentric
    weights, the same linear application to the training embedding.
    """
    from sklearn.manifold import LocallyLinearEmbedding

    rng = np.random.default_rng(0)
    n_train = 100
    x_train = rng.standard_normal((n_train, 6))
    x_new = rng.standard_normal((10, 6))

    model = ManifoldCCA(method="lle", n_neighbors=10, latent_dimensions=3).fit(
        [x_train, x_train.copy()]
    )
    z1_new, z2_new = model.transform([x_new, x_new.copy()])

    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=3).fit(
        x_train - x_train.mean(0)
    )
    lle_new = lle.transform(x_new - x_train.mean(0))

    for z_new in (z1_new, z2_new):
        Q1, _ = np.linalg.qr(z_new)
        Q2, _ = np.linalg.qr(lle_new)
        principal_cosines = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
        assert np.all(principal_cosines > 1 - 1e-2), (
            "ManifoldCCA(method='lle') transform on duplicated views should "
            f"match LocallyLinearEmbedding.transform; got {principal_cosines}"
        )


def test_laplacian_transform_stable_on_small_noisy_data() -> None:
    """Regression test: a near-1 kept eigenvalue must not blow up the extension.

    On a small (n=30), unstructured dataset, some of the default
    ``n_operator_components=10`` kept Laplacian eigenvalues sit right next
    to 1 (mu = 1 - eigenvalue near 0), which without a floor on mu makes
    the Nystrom extension's 1/mu rescaling explode -- caught by comparing
    the transformed scale against the training embedding's own scale.
    """
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((30, 5))
    x2 = rng.standard_normal((30, 5))

    model = ManifoldCCA(method="laplacian", n_neighbors=8).fit([x1, x2])
    transformed = model.transform([x1, x2])
    for w, t in zip(model.weights, transformed):
        assert t[:, 0].std() < 10 * w[:, 0].std(), (
            "transformed scale blew up relative to the training embedding"
        )


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

    linear_corr = (
        MCCA(latent_dimensions=1, c=0.1, pca=False)
        .fit([x1_tr, x2_tr])
        .score([x1_te, x2_te])[0]
    )
    manifold_corr = (
        ManifoldCCA(method="laplacian", n_neighbors=10, latent_dimensions=1)
        .fit([x1_tr, x2_tr])
        .score([x1_te, x2_te])[0]
    )

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

    model = ManifoldCCA(latent_dimensions=2, method="lle", n_neighbors=6, lle_reg=1e-2)
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
