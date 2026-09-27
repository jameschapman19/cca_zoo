"""The shared linear algebra in cca_zoo._utils._linalg."""

from __future__ import annotations

import numpy as np
import pytest

from cca_zoo._utils._linalg import (
    cross_moment_tensor,
    deflate,
    gevp,
    psd_inverse_sqrt,
    soft_threshold,
    svd_whiten,
)


def _centred(n: int, p: int, rank: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, rank or p)) @ (
        rng.standard_normal((rank, p)) if rank else np.eye(p)
    )
    return x - x.mean(axis=0)


@pytest.mark.parametrize(
    ("n", "p", "rank"), [(100, 8, None), (30, 5, 2), (20, 200, None)]
)
def test_svd_whiten_gives_identity_covariance(n: int, p: int, rank: int | None) -> None:
    """Unregularised whitening is exact for tall, rank-deficient and wide data."""
    x = _centred(n, p, rank)
    x_w, w = svd_whiten(x, regularization=0.0)
    np.testing.assert_allclose(x_w, x @ w, atol=1e-10)
    np.testing.assert_allclose(x_w.T @ x_w / (n - 1), np.eye(x_w.shape[1]), atol=1e-8)


def test_svd_whiten_regularization_shrinks_towards_the_raw_scale() -> None:
    """Regularisation leaves the whitened variances further from one."""
    x = _centred(60, 5)

    def distance_from_unit_variance(c: float) -> float:
        x_w, _ = svd_whiten(x, regularization=c)
        return float(np.abs(x_w.var(axis=0, ddof=1) - 1.0).mean())

    assert distance_from_unit_variance(0.0) < distance_from_unit_variance(0.5)


def test_gevp_solves_the_generalised_problem() -> None:
    """The top k pairs satisfy A v = lambda B v, in descending order."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((6, 6))
    a = a @ a.T
    b = rng.standard_normal((6, 6))
    b = b @ b.T + np.eye(6)
    eigvals, eigvecs = gevp(a, b, k=3)
    assert np.all(np.diff(eigvals) <= 0)
    np.testing.assert_allclose(a @ eigvecs, (b @ eigvecs) * eigvals, atol=1e-8)


def test_gevp_without_b_and_k_capped() -> None:
    """B=None is the standard problem; k beyond the size returns every pair."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    a = a @ a.T
    eigvals, eigvecs = gevp(a, None, k=10)
    np.testing.assert_allclose(eigvals, np.linalg.eigvalsh(a)[::-1], atol=1e-10)
    assert eigvecs.shape == (4, 4)


def test_soft_threshold() -> None:
    """Shrinks towards zero by the threshold, elementwise, zeroing the rest."""
    x = np.array([[2.0, -1.5, 0.4], [-0.6, 0.0, 3.0]])
    np.testing.assert_allclose(
        soft_threshold(x, 0.5), [[1.5, -1.0, 0.0], [-0.1, 0.0, 2.5]], atol=1e-15
    )
    np.testing.assert_array_equal(soft_threshold(x, 0.0), x)


def test_deflate_removes_each_views_projection() -> None:
    """Each deflated view is orthogonal to its weight; a zero weight changes nothing."""
    rng = np.random.default_rng(0)
    views = [rng.standard_normal((25, p)) for p in (5, 7, 3)]
    weights = [rng.standard_normal(5), rng.standard_normal(7), np.zeros(3)]
    deflated = deflate(views, weights)
    for x_d, w in zip(deflated[:2], weights[:2]):
        np.testing.assert_allclose(x_d @ w, 0.0, atol=1e-12)
    np.testing.assert_array_equal(deflated[2], views[2])


def test_psd_inverse_sqrt() -> None:
    """W C W = I, with a singular C lifted so its smallest eigenvalue is the floor."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((50, 8))
    cov = a.T @ a / 49
    w = psd_inverse_sqrt(cov, 1e-6)
    np.testing.assert_allclose(w @ cov @ w, np.eye(8), atol=1e-10)
    singular = rng.standard_normal((3, 6))
    singular = singular.T @ singular
    lifted = singular + (1e-3 - np.linalg.eigvalsh(singular)[0]) * np.eye(6)
    w = psd_inverse_sqrt(singular, 1e-3)
    np.testing.assert_allclose(w @ lifted @ w, np.eye(6), atol=1e-8)


def test_cross_moment_tensor() -> None:
    """Two views give X1' X2 / n; three give the mean of per-sample outer products."""
    rng = np.random.default_rng(0)
    x1, x2, x3 = (rng.standard_normal((30, p)) for p in (2, 3, 4))
    np.testing.assert_allclose(cross_moment_tensor([x1, x2]), x1.T @ x2 / 30)
    expected = np.mean(
        [np.multiply.outer(np.multiply.outer(a, b), c) for a, b, c in zip(x1, x2, x3)],
        axis=0,
    )
    np.testing.assert_allclose(cross_moment_tensor([x1, x2, x3]), expected)
