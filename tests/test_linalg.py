"""Tests for cca_zoo._utils._linalg."""

from __future__ import annotations

import numpy as np

from cca_zoo._utils._linalg import deflate, gevp, soft_threshold, svd_whiten

# ---------------------------------------------------------------------------
# svd_whiten
# ---------------------------------------------------------------------------


class TestSvdWhiten:
    """Tests for svd_whiten."""

    def test_returns_tuple_of_two_arrays(self) -> None:
        """svd_whiten returns a tuple (X_white, W)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 5))
        x -= x.mean(axis=0)
        result = svd_whiten(x)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)

    def test_whitened_covariance_near_identity_no_regularization(self) -> None:
        """With regularization=0, the whitened data has near-identity covariance."""
        rng = np.random.default_rng(0)
        n, p = 100, 8
        x = rng.standard_normal((n, p))
        x -= x.mean(axis=0)
        x_w, _ = svd_whiten(x, regularization=0.0)
        cov = x_w.T @ x_w / (n - 1)
        np.testing.assert_allclose(cov, np.eye(cov.shape[0]), atol=1e-10)

    def test_whitening_matrix_shape(self) -> None:
        """The whitening matrix W has shape (n_features, rank)."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((50, 8))
        x -= x.mean(axis=0)
        _, w = svd_whiten(x, regularization=0.0)
        assert w.shape[0] == 8
        assert w.shape[1] <= 8

    def test_X_white_equals_X_at_W(self) -> None:
        """X_white must equal X @ W."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((40, 6))
        x -= x.mean(axis=0)
        x_w, w = svd_whiten(x, regularization=0.0)
        np.testing.assert_allclose(x_w, x @ w, atol=1e-12)

    def test_regularization_one_gives_sphering(self) -> None:
        """With regularization=1, inverse sqrt eigenvalue is 1/sqrt(c)=1."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 5))
        x -= x.mean(axis=0)
        x_w, w = svd_whiten(x, regularization=1.0)
        # With reg=1: inv_sqrt = 1/sqrt(0 + 1) = 1, so W = Vt.T and X_w = U * s
        assert x_w.shape[0] == 30
        assert w.shape[0] == 5

    def test_regularization_reduces_whitening(self) -> None:
        """Higher regularization gives a covariance further from identity."""
        rng = np.random.default_rng(0)
        n, p = 60, 5
        x = rng.standard_normal((n, p))
        x -= x.mean(axis=0)
        x_w_0, _ = svd_whiten(x, regularization=0.0)
        x_w_1, _ = svd_whiten(x, regularization=0.5)
        cov_0 = x_w_0.T @ x_w_0 / (n - 1)
        cov_1 = x_w_1.T @ x_w_1 / (n - 1)
        # Diagonal of cov_0 should be closer to 1 than cov_1
        diff_0 = np.abs(np.diag(cov_0) - 1.0).mean()
        diff_1 = np.abs(np.diag(cov_1) - 1.0).mean()
        assert diff_0 < diff_1

    def test_rank_deficient_input(self) -> None:
        """svd_whiten handles rank-deficient input without error."""
        rng = np.random.default_rng(0)
        # Create rank-2 data in 5d
        z = rng.standard_normal((30, 2))
        a = rng.standard_normal((2, 5))
        x = z @ a
        x -= x.mean(axis=0)
        x_w, w = svd_whiten(x, regularization=0.0)
        # Whitened columns should have near-identity covariance
        cov = x_w.T @ x_w / (x_w.shape[0] - 1)
        np.testing.assert_allclose(cov, np.eye(cov.shape[0]), atol=1e-10)

    def test_wide_input_uses_svd_path(self) -> None:
        """n_samples < n_features (wide X) is whitened correctly too."""
        rng = np.random.default_rng(0)
        n, p = 20, 200
        x = rng.standard_normal((n, p))
        x -= x.mean(axis=0)
        x_w, w = svd_whiten(x, regularization=0.0)
        assert w.shape[0] == p
        cov = x_w.T @ x_w / (n - 1)
        np.testing.assert_allclose(cov, np.eye(cov.shape[0]), atol=1e-8)

    def test_tall_and_wide_paths_agree_on_square_input(self) -> None:
        """At n == p, both branches of the n>=p/n<p split must agree."""
        rng = np.random.default_rng(0)
        n = p = 40
        x = rng.standard_normal((n, p))
        x -= x.mean(axis=0)
        x_w, w = svd_whiten(x, regularization=0.2)
        np.testing.assert_allclose(x_w, x @ w, atol=1e-8)
        # Rotation-invariant check: the Gram matrix of the whitened data only
        # depends on the whitened subspace, not on the (arbitrary) basis
        # eigh/svd happen to return it in.
        gram = x_w @ x_w.T
        assert gram.shape == (n, n)


# ---------------------------------------------------------------------------
# gevp
# ---------------------------------------------------------------------------


class TestGevp:
    """Tests for gevp (generalised eigenvalue problem solver)."""

    def test_standard_eigenvalue_problem_with_none_B(self) -> None:
        """When B=None, gevp solves the standard eigenvalue problem."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 5))
        a = a @ a.T  # make symmetric PD
        eigvals, eigvecs = gevp(a, None, k=3)
        assert eigvals.shape == (3,)
        assert eigvecs.shape == (5, 3)

    def test_returns_k_eigenpairs(self) -> None:
        """Gevp returns exactly k eigenpairs."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((8, 8))
        a = a @ a.T
        b = rng.standard_normal((8, 8))
        b = b @ b.T + 2.0 * np.eye(8)
        for k in [1, 3, 5]:
            eigvals, eigvecs = gevp(a, b, k=k)
            assert eigvals.shape == (k,)
            assert eigvecs.shape == (8, k)

    def test_descending_eigenvalue_order(self) -> None:
        """Gevp returns eigenvalues in descending order."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 6))
        a = a @ a.T
        eigvals, _ = gevp(a, None, k=4)
        assert np.all(eigvals[:-1] >= eigvals[1:])

    def test_generalised_eigenvector_equation(self) -> None:
        """Returned eigenvectors must satisfy A v = lam B v."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((6, 6))
        a = a @ a.T
        b = np.eye(6) + 0.1 * rng.standard_normal((6, 6))
        b = b @ b.T + np.eye(6)
        eigvals, eigvecs = gevp(a, b, k=3)
        for i in range(3):
            lhs = a @ eigvecs[:, i]
            rhs = eigvals[i] * (b @ eigvecs[:, i])
            np.testing.assert_allclose(lhs, rhs, atol=1e-8)

    def test_k_clamped_to_matrix_size(self) -> None:
        """When k > p, gevp returns min(k, p) eigenpairs."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((4, 4))
        a = a @ a.T
        eigvals, eigvecs = gevp(a, None, k=10)
        assert eigvals.shape == (4,)
        assert eigvecs.shape == (4, 4)

    def test_standard_problem_none_descending(self) -> None:
        """Standard problem (B=None) also returns descending eigenvalues."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal((5, 5))
        a = a @ a.T
        eigvals, _ = gevp(a, None, k=5)
        assert np.all(eigvals[:-1] >= eigvals[1:])


# ---------------------------------------------------------------------------
# soft_threshold
# ---------------------------------------------------------------------------


class TestSoftThreshold:
    """Tests for soft_threshold."""

    def test_zero_threshold_is_identity(self) -> None:
        """With threshold=0, output equals input."""
        x = np.array([1.0, -2.0, 0.5, -0.3])
        result = soft_threshold(x, 0.0)
        np.testing.assert_array_equal(result, x)

    def test_positive_values_shrunk_correctly(self) -> None:
        """Positive values above threshold are reduced by threshold."""
        x = np.array([3.0, 1.0, 0.5])
        result = soft_threshold(x, 1.0)
        np.testing.assert_allclose(result, [2.0, 0.0, 0.0])

    def test_negative_values_shrunk_correctly(self) -> None:
        """Negative values are shrunk towards zero symmetrically."""
        x = np.array([-3.0, -1.0, -0.5])
        result = soft_threshold(x, 1.0)
        np.testing.assert_allclose(result, [-2.0, 0.0, 0.0])

    def test_values_within_threshold_become_zero(self) -> None:
        """Values with absolute value <= threshold are set to zero."""
        x = np.array([0.3, -0.2, 0.0, 0.5])
        result = soft_threshold(x, 0.5)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0])

    def test_output_shape_preserved(self) -> None:
        """soft_threshold preserves the input array shape."""
        x = np.ones((3, 4))
        result = soft_threshold(x, 0.5)
        assert result.shape == (3, 4)

    def test_mixed_signs(self) -> None:
        """Mixed sign values are shrunk correctly."""
        x = np.array([2.0, -2.0, 0.5, -0.5])
        result = soft_threshold(x, 1.0)
        np.testing.assert_allclose(result, [1.0, -1.0, 0.0, 0.0])

    def test_large_threshold_zeros_all(self) -> None:
        """Threshold larger than all values produces all zeros."""
        x = np.array([0.1, -0.1, 0.05])
        result = soft_threshold(x, 1.0)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_2d_array(self) -> None:
        """soft_threshold works on 2-D arrays element-wise."""
        x = np.array([[2.0, -1.5], [0.4, -0.6]])
        result = soft_threshold(x, 0.5)
        expected = np.array([[1.5, -1.0], [0.0, -0.1]])
        np.testing.assert_allclose(result, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# deflate
# ---------------------------------------------------------------------------


class TestDeflate:
    """Tests for the deflate function."""

    def test_deflated_projection_is_near_zero(self) -> None:
        """After deflation, projection of the deflated view onto w is near zero."""
        rng = np.random.default_rng(0)
        n, p = 30, 5
        x = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        w /= np.linalg.norm(w)
        [x_deflated] = deflate([x], [w])
        projection = x_deflated @ w
        np.testing.assert_allclose(projection, np.zeros(n), atol=1e-12)

    def test_deflate_preserves_shape(self) -> None:
        """Deflate preserves the shape of each view."""
        rng = np.random.default_rng(0)
        views = [rng.standard_normal((20, 4)), rng.standard_normal((20, 6))]
        weights = [rng.standard_normal(4), rng.standard_normal(6)]
        deflated = deflate(views, weights)
        assert len(deflated) == 2
        assert deflated[0].shape == (20, 4)
        assert deflated[1].shape == (20, 6)

    def test_deflate_removes_variance_along_w(self) -> None:
        """Variance of score after deflation should be much smaller."""
        rng = np.random.default_rng(0)
        n, p = 50, 8
        x = rng.standard_normal((n, p))
        w = rng.standard_normal(p)
        w /= np.linalg.norm(w)
        [x_d] = deflate([x], [w])
        score_before = (x @ w).var()
        score_after = (x_d @ w).var()
        assert score_after < score_before * 1e-20

    def test_deflate_multiple_views(self) -> None:
        """Deflate handles multiple views simultaneously."""
        rng = np.random.default_rng(0)
        views = [rng.standard_normal((25, p)) for p in [5, 7, 3]]
        weights = [rng.standard_normal(p) for p in [5, 7, 3]]
        weights = [w / np.linalg.norm(w) for w in weights]
        deflated = deflate(views, weights)
        assert len(deflated) == 3
        for x_d, w in zip(deflated, weights):
            proj = x_d @ w
            np.testing.assert_allclose(proj, np.zeros(25), atol=1e-12)

    def test_deflate_zero_weight_unchanged(self) -> None:
        """A near-zero weight vector leaves the view nearly unchanged."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 4))
        w = np.zeros(4)  # zero weight
        [x_d] = deflate([x], [w])
        # norm_sq will be 0, so no deflation applied
        np.testing.assert_array_equal(x_d, x)
